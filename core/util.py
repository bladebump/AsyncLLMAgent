import inspect
from core.openai_types import MessageToolParam, FunctionDefinition
from mcp.types import Tool
from typing import List

def function_to_json(func) -> dict:
    """将Python函数转换为JSON可序列化的字典，描述函数的签名，包括其名称、描述和参数。"""
    type_map = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
        type(None): "null",
    }

    try:
        signature = inspect.signature(func)
    except ValueError as e:
        raise ValueError(
            f"Failed to get signature for function {func.__name__}: {str(e)}"
        )

    parameters = {}
    for param in signature.parameters.values():
        try:
            param_type = type_map.get(param.annotation, "string")
        except KeyError as e:
            raise KeyError(
                f"Unknown type annotation {param.annotation} for parameter {param.name}: {str(e)}"
            )
        parameters[param.name] = {"type": param_type}

    required = [
        param.name
        for param in signature.parameters.values()
        if param.default == inspect._empty
    ]

    return {
        "type": "function",
        "function": {
            "name": func.__name__,
            "description": func.__doc__ or "",
            "parameters": {
                "type": "object",
                "properties": parameters,
                "required": required,
            },
        },
    }

def mcp_tool_to_function(tools: list[Tool]) -> List[MessageToolParam]:
    """将MCP工具转换为OpenAI函数定义"""
    functions = []
    for tool in tools:
        functions.append(
            MessageToolParam(
                function=FunctionDefinition(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.inputSchema,
                ),
                type="function",
            )
        )
    return functions

if __name__ == "__main__":
    print(function_to_json(function_to_json))
