from typing import Any, List, Optional, Type, Union, get_args, get_origin
from pydantic import BaseModel, Field
from core.tools.base import BaseTool


class CreateChatCompletion(BaseTool):
    name: str = "create_chat_completion"
    description: str = (
        "创建一个结构化的完成，并指定输出格式。"
    )

    # Type mapping for JSON schema
    type_mapping: dict = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        dict: "object",
        list: "array",
    }
    response_type: Optional[Type] = None
    required: List[str] = Field(default_factory=lambda: ["response"])

    def __init__(self, response_type: Optional[Type] = str):
        """初始化时指定一个特定的响应类型。"""
        super().__init__()
        self.response_type = response_type
        self.parameters = self._build_parameters()

    def _build_parameters(self) -> dict:
        """根据响应类型构建参数模式。"""
        if self.response_type == str:
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "string",
                        "description": "应该传递给用户的响应文本。",
                    },
                },
                "required": self.required,
            }

        if isinstance(self.response_type, type) and issubclass(
            self.response_type, BaseModel
        ):
            schema = self.response_type.model_json_schema()
            return {
                "type": "object",
                "properties": schema["properties"],
                "required": schema.get("required", self.required),
            }

        return self._create_type_schema(self.response_type)

    def _create_type_schema(self, type_hint: Type) -> dict:
        """为给定的类型创建一个JSON模式。"""
        origin = get_origin(type_hint)
        args = get_args(type_hint)

        # 处理原始类型
        if origin is None:
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": self.type_mapping.get(type_hint, "string"),
                        "description": f"Response of type {type_hint.__name__}",
                    }
                },
                "required": self.required,
            }

        # 处理列表类型
        if origin is list:
            item_type = args[0] if args else Any
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "array",
                        "items": self._get_type_info(item_type),
                    }
                },
                "required": self.required,
            }

        # 处理字典类型
        if origin is dict:
            value_type = args[1] if len(args) > 1 else Any
            return {
                "type": "object",
                "properties": {
                    "response": {
                        "type": "object",
                        "additionalProperties": self._get_type_info(value_type),
                    }
                },
                "required": self.required,
            }

        # 处理联合类型
        if origin is Union:
            return self._create_union_schema(args)

        return self._build_parameters()

    def _get_type_info(self, type_hint: Type) -> dict:
        """获取单个类型的类型信息。"""
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return type_hint.model_json_schema()

        return {
            "type": self.type_mapping.get(type_hint, "string"),
            "description": f"Value of type {getattr(type_hint, '__name__', 'any')}",
        }

    def _create_union_schema(self, types: tuple) -> dict:
        """为联合类型创建模式。"""
        return {
            "type": "object",
            "properties": {
                "response": {"anyOf": [self._get_type_info(t) for t in types]}
            },
            "required": self.required,
        }

    async def execute(self, required: list | None = None, **kwargs) -> Any:
        """执行聊天完成，并进行类型转换。

        Args:
            required: 必需字段名称列表或None
            **kwargs: 响应数据

        Returns:
            基于response_type转换的响应
        """
        required = required or self.required

        # 处理required为列表的情况
        if isinstance(required, list) and len(required) > 0:
            if len(required) == 1:
                required_field = required[0]
                result = kwargs.get(required_field, "")
            else:
                # 返回多个字段作为字典
                return {field: kwargs.get(field, "") for field in required}
        else:
            required_field = "response"
            result = kwargs.get(required_field, "")

        # 类型转换逻辑
        if self.response_type == str:
            return result

        if isinstance(self.response_type, type) and issubclass(
            self.response_type, BaseModel
        ):
            return self.response_type(**kwargs)

        if get_origin(self.response_type) in (list, dict):
            return result  # 假设result已经是正确的格式

        try:
            return self.response_type(result)
        except (ValueError, TypeError):
            return result