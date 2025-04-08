import asyncio
from typing import Optional
from core.tools.base import BaseTool, CLIResult
from core.tools.errors import ToolError


_BASH_DESCRIPTION = """在终端中执行bash命令。
* 长时间运行的命令：对于可能无限期运行的命令，应该在后台运行并将输出重定向到文件，例如 command = `python3 app.py > server.log 2>&1 &`。
"""


class _BashSession:
    """一个bash会话."""
    _timeout: float = 120.0  # seconds

    async def run(self, command: str):
        """在bash会话中执行命令."""
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=self._timeout)
            return CLIResult(output=stdout.decode(), error=stderr.decode())
        except asyncio.TimeoutError:
            proc.kill()
            raise ToolError("命令执行超时")


class Bash(BaseTool):
    """一个用于执行bash命令的工具"""

    name: str = "bash"
    description: str = _BASH_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要执行的bash命令。",
            },
        },
        "required": ["command"],
    }

    _session: Optional[_BashSession] = None

    async def execute(
        self, command: str | None = None, **kwargs
    ) -> CLIResult:

        if command is not None:
            return await self._session.run(command)

        raise ToolError("未提供命令.")


if __name__ == "__main__":
    bash = Bash()
    rst = asyncio.run(bash.execute("ls -l"))
    print(rst)