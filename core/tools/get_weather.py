from core.tools.base import BaseTool


_GET_WEATHER_DESCRIPTION = """获取天气信息"""


class GetWeather(BaseTool):
    name: str = "get_weather"
    description: str = _GET_WEATHER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "城市名称。",
            }
        },
        "required": ["city"],
    }

    async def execute(self, city: str) -> str:
        """获取天气信息"""
        return f"城市: {city} 是晴天"