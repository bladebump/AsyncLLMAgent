class ToolError(Exception):
    """当工具遇到错误时引发。"""

    def __init__(self, message):
        self.message = message