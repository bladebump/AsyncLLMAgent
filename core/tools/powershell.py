import asyncio
import os
from typing import Optional

from core.tools.base import BaseTool, CLIResult
from core.tools.errors import ToolError


_POWERSHELL_DESCRIPTION = """在Windows PowerShell中执行命令。
* 长时间运行的命令：对于可能无限期运行的命令，应该在后台运行并将输出重定向到文件，例如 command = `Start-Process -NoNewWindow python -ArgumentList "app.py" -RedirectStandardOutput server.log`。
* 交互式：如果命令返回退出码`-1`，这意味着进程尚未完成。助手必须向终端发送第二次调用，使用空的`command`（这将检索任何额外的日志），或者它可以发送额外的文本（将`command`设置为文本）到正在运行的进程的STDIN，或者它可以发送command=`ctrl+c`来中断进程。
* 超时：如果命令执行结果显示"Command timed out. Sending SIGINT to the process"，助手应该尝试在后台重新运行该命令。
"""


class _PowerShellSession:
    """一个PowerShell会话."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "powershell.exe -NoProfile -ExecutionPolicy Bypass"
    _output_delay: float = 0.2  # seconds
    _timeout: float = 120.0  # seconds
    _sentinel: str = "<<exit>>"

    def __init__(self):
        self._started = False
        self._timed_out = False

    async def start(self):
        if self._started:
            return

        self._process = await asyncio.create_subprocess_shell(
            "chcp 65001 && " + self.command,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """终止PowerShell会话."""
        if not self._started:
            raise ToolError("会话未启动.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str):
        """在PowerShell会话中执行命令."""
        if not self._started:
            raise ToolError("会话未启动.")
        if self._process.returncode is not None:
            return CLIResult(
                system="工具必须重新启动",
                error=f"PowerShell已退出，返回码为{self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"超时: PowerShell未在{self._timeout}秒内返回，必须重新启动",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process with sentinel to mark completion
        # 发送两次_sentinel标记，第一次可能在命令中，第二次作为命令结束标记
        ps_command = f"{command}; Write-Output '{self._sentinel}'\r\n"
        self._process.stdin.write(ps_command.encode())
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found twice
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = (
                        self._process.stdout._buffer.decode("utf-8", errors="replace")
                    )  # pyright: ignore[reportAttributeAccessIssue]
                    
                    # 检查_sentinel出现次数
                    sentinel_count = output.count(self._sentinel)
                    
                    # 只有当_sentinel出现两次时才结束
                    if sentinel_count >= 2:
                        # 找到第二个sentinel的位置，截取它之前的内容
                        first_sentinel_pos = output.find(self._sentinel)
                        second_sentinel_pos = output.find(self._sentinel, first_sentinel_pos + 1)
                        
                        # 提取到第二个sentinel之前的内容
                        output = output[:second_sentinel_pos]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"超时: PowerShell未在{self._timeout}秒内返回，必须重新启动",
            ) from None

        if output.endswith("\r\n"):
            output = output[:-2]

        error = (
            self._process.stderr._buffer.decode("utf-8", errors="replace")
        )  # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\r\n"):
            error = error[:-2]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)


class PowerShell(BaseTool):
    """一个用于执行PowerShell命令的工具"""

    name: str = "powershell"
    description: str = _POWERSHELL_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的PowerShell命令。可以为空以查看先前退出码为`-1`时的附加日志。可以发送`ctrl+c`来中断正在运行的进程。",
            },
        },
        "required": ["command"],
    }

    _session: Optional[_PowerShellSession] = None

    async def execute(
        self, command: str | None = None, restart: bool = False, **kwargs
    ) -> CLIResult:
        if restart:
            if self._session:
                self._session.stop()
            self._session = _PowerShellSession()
            await self._session.start()

            return CLIResult(system="工具已重新启动.")

        if self._session is None:
            self._session = _PowerShellSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("未提供命令.")

if __name__ == "__main__":
    async def main():
        powershell = PowerShell()
        try:
            rst = await powershell.execute("Get-ChildItem")
            print(rst)
        finally:
            if powershell._session:
                powershell._session.stop()
    
    asyncio.run(main()) 