from client import deepseek_v3_cot
import asyncio

def get_current_weather(location: str, unit: str = 'fahrenheit'):   
    """Get the current weather in a given location"""
    return f"The current weather in {location} is 70 degrees and cloudy."

async def test_function_calling():
    result = await deepseek_v3_cot.chat_with_functions(
        messages=[
            {
                'role': 'user',
                'content': 'What is the weather in Tokyo?'
            }
        ],
        functions=[get_current_weather]
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(test_function_calling())
