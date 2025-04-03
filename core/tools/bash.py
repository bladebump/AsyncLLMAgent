import asyncio
import os
from typing import Optional

from core.tools.base import BaseTool, CLIResult
from core.tools.errors import ToolError


_BASH_DESCRIPTION = """在终端中执行bash命令。
* 长时间运行的命令：对于可能无限期运行的命令，应该在后台运行并将输出重定向到文件，例如 command = `python3 app.py > server.log 2>&1 &`。
* 交互式：如果bash命令返回退出码`-1`，这意味着进程尚未完成。助手必须向终端发送第二次调用，使用空的`command`（这将检索任何额外的日志），或者它可以发送额外的文本（将`command`设置为文本）到正在运行的进程的STDIN，或者它可以发送command=`ctrl+c`来中断进程。
* 超时：如果命令执行结果显示"Command timed out. Sending SIGINT to the process"，助手应该尝试在后台重新运行该命令。
"""


class _BashSession:
    """一个bash会话."""

    _started: bool
    _process: asyncio.subprocess.Process

    command: str = "/bin/bash"
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
            self.command,
            preexec_fn=os.setsid,
            shell=True,
            bufsize=0,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        self._started = True

    def stop(self):
        """终止bash会话."""
        if not self._started:
            raise ToolError("会话未启动.")
        if self._process.returncode is not None:
            return
        self._process.terminate()

    async def run(self, command: str):
        """在bash会话中执行命令."""
        if not self._started:
            raise ToolError("会话未启动.")
        if self._process.returncode is not None:
            return CLIResult(
                system="工具必须重新启动",
                error=f"bash已退出，返回码为{self._process.returncode}",
            )
        if self._timed_out:
            raise ToolError(
                f"超时: bash未在{self._timeout}秒内返回，必须重新启动",
            )

        # we know these are not None because we created the process with PIPEs
        assert self._process.stdin
        assert self._process.stdout
        assert self._process.stderr

        # send command to the process
        self._process.stdin.write(
            command.encode() + f"; echo '{self._sentinel}'\n".encode()
        )
        await self._process.stdin.drain()

        # read output from the process, until the sentinel is found
        try:
            async with asyncio.timeout(self._timeout):
                while True:
                    await asyncio.sleep(self._output_delay)
                    # if we read directly from stdout/stderr, it will wait forever for
                    # EOF. use the StreamReader buffer directly instead.
                    output = (
                        self._process.stdout._buffer.decode()
                    )  # pyright: ignore[reportAttributeAccessIssue]
                    if self._sentinel in output:
                        # strip the sentinel and break
                        output = output[: output.index(self._sentinel)]
                        break
        except asyncio.TimeoutError:
            self._timed_out = True
            raise ToolError(
                f"超时: bash未在{self._timeout}秒内返回，必须重新启动",
            ) from None

        if output.endswith("\n"):
            output = output[:-1]

        error = (
            self._process.stderr._buffer.decode()
        )  # pyright: ignore[reportAttributeAccessIssue]
        if error.endswith("\n"):
            error = error[:-1]

        # clear the buffers so that the next output can be read correctly
        self._process.stdout._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]
        self._process.stderr._buffer.clear()  # pyright: ignore[reportAttributeAccessIssue]

        return CLIResult(output=output, error=error)


class Bash(BaseTool):
    """一个用于执行bash命令的工具"""

    name: str = "bash"
    description: str = _BASH_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的bash命令。可以为空以查看先前退出码为`-1`时的附加日志。可以发送`ctrl+c`来中断正在运行的进程。",
            },
        },
        "required": ["command"],
    }

    _session: Optional[_BashSession] = None

    async def execute(
        self, command: str | None = None, restart: bool = False, **kwargs
    ) -> CLIResult:
        if restart:
            if self._session:
                self._session.stop()
            self._session = _BashSession()
            await self._session.start()

            return CLIResult(system="工具已重新启动.")

        if self._session is None:
            self._session = _BashSession()
            await self._session.start()

        if command is not None:
            return await self._session.run(command)

        raise ToolError("未提供命令.")


if __name__ == "__main__":
    bash = Bash()
    rst = asyncio.run(bash.execute("ls -l"))
    print(rst)