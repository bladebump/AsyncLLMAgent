from pydantic import BaseModel
from typing import Literal, Any, Optional, Dict
from typing_extensions import Required, TypedDict

# 使用自定义基类
class Function(BaseModel):
    arguments: str
    """
    用于调用函数的参数,由模型以JSON格式生成。注意模型并不总是生成有效的JSON,
    并且可能会产生未在函数模式中定义的参数。在调用函数之前请在代码中验证这些参数。
    """

    name: str
    """要调用的函数名称。"""

class MessageToolCall(BaseModel):
    id: str
    """工具调用ID。"""

    function: Function
    """要调用的函数。"""

    type: Literal["function"]
    """工具类型,目前仅支持`function`。"""

class FunctionDefinition(TypedDict, total=False):
    name: Required[str]
    """要调用的函数名称。

    必须由a-z、A-Z、0-9或包含下划线和连字符组成，最大长度为64。
    """

    description: str
    """
    函数功能的描述，模型用它来决定何时以及如何调用函数。
    """

    parameters: Dict[str, object]
    """函数接受的参数，描述为JSON Schema对象。

    省略`parameters`定义了一个空参数列表的函数。
    """

    strict: Optional[bool]
    """生成函数调用时是否启用严格的模式遵循。
    """

class MessageToolParam(TypedDict, total=False):
    function: Required[FunctionDefinition]

    type: Required[Literal["function"]]
    """工具的类型。目前，仅支持`function`。"""

class Message(BaseModel):
    content: str
    """消息内容。"""

    reasoning_content: str | None = None
    """思考内容。"""

    role: str
    """消息角色,通常为"user"或"assistant"。"""

    tool_calls: list[MessageToolCall] | None = None
    """工具调用列表。"""
    
    agent_sender: str | None = None
    """如果是agent发送的消息,则该字段为agent的名称。"""

    def model_dump_json(self, **kwargs):
        """重写model_dump_json方法，默认排除None值"""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump_json(**kwargs)
    
    def model_dump(self, **kwargs):
        """重写model_dump方法，默认排除None值"""
        kwargs.setdefault("exclude_none", True)
        return super().model_dump(**kwargs)
