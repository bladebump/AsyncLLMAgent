import json
from typing import Any, List, Optional, Union
from core.agent.react import ReActAgent
from utils.log import logger
from core.schema import AgentState, Message, ToolCall, ToolChoice, AgentResultStream
from core.tools import ToolCollection
from core.llms import AsyncBaseChatCOTModel
from core.mem import AsyncMemory

SYSTEM_PROMPT = "你是一个可以执行工具调用的代理, 请根据用户的需求选择合适的工具, 并使用工具调用执行任务。你可以反复使用工具调用直到任务完成。"

class ToolCallAgent(ReActAgent):
    """用于处理工具/函数调用的基础代理类"""

    def __init__(
        self,
        name: str = "toolcall",
        llm: AsyncBaseChatCOTModel = None,
        memory: AsyncMemory = None,
        description: str = "一个可以执行工具调用的代理。",
        system_prompt: str = SYSTEM_PROMPT,
        state: AgentState = AgentState.IDLE,
        available_tools: Optional[ToolCollection] = None,
        tool_choices: str = ToolChoice.AUTO,
        max_steps: int = 30,
        max_observe: Optional[Union[int, bool]] = None,
        **kwargs
    ):
        super().__init__(
            name=name,
            llm=llm,
            memory=memory,
            description=description,
            system_prompt=system_prompt,
            state=state,
            max_steps=max_steps,
            **kwargs
        )
        
        # 初始化默认工具集合
        if available_tools is None:
            self.available_tools = ToolCollection()
        else:
            self.available_tools = available_tools
        self.tool_choices = tool_choices
            
        self.tool_calls = []
        self._current_base64_image = None
        self.max_observe = max_observe

    async def think(self) -> tuple[str, str]:
        """处理当前状态并决定下一步操作使用工具"""
        self.tool_calls = []
        if not await self.memory.has_system() and self.system_prompt:
            await self.memory.add_system(Message.system_message(self.system_prompt))

        if self.next_step_prompt:
            await self.memory.add(Message.user_message(self.next_step_prompt))

        thinking, content, tool_calls = await self.llm.chat(
            messages=self.memory.Messages,
            tools=self.available_tools.to_params(),
            tool_choice=self.tool_choices,
            stream=False,
        )
        self.tool_calls = tool_calls
        # 记录响应信息
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} 选择了 {len(tool_calls) if tool_calls else 0} 个工具"
        )
        if tool_calls:
            logger.info(
                f"🧰 正在准备工具: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 工具参数: {[call.function.arguments for call in tool_calls]}")

        assistant_msg = (
            Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
            if self.tool_calls
            else Message.assistant_message(content)
        )
        await self.memory.add(assistant_msg)
        return thinking, content, bool(tool_calls)

    async def act(self) -> str:
        """执行工具调用并处理其结果"""
        results = []
        for command in self.tool_calls:
            # 为每个工具调用重置base64_image
            self._current_base64_image = None
            result = await self.execute_tool(command)
            if self.max_observe:
                result = result[: self.max_observe]
            logger.info(f"🎯 工具 '{command.function.name}' 完成任务！结果: {result}")
            # 将工具响应添加到记忆中
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            await self.memory.add(tool_msg)
            results.append(result)

        return "\n\n".join(results)

    async def think_stream(self):
        """流式返回的think"""
        self.tool_calls = []
        if not await self.memory.has_system() and self.system_prompt:
            await self.memory.add_system(Message.system_message(self.system_prompt))

        gen = await self.llm.chat(
            messages=self.memory.Messages,
            tools=self.available_tools.to_params(),
            tool_choice=self.tool_choices,
            stream=True,
        )
        all_thinking = ""
        all_content = ""
        async for think, content, tool_calls in gen:
            if tool_calls:
                self.tool_calls = tool_calls
                logger.info(f"🧰 正在准备工具: {[call.function.name for call in tool_calls]}")
                logger.info(f"🔧 工具参数: {[call.function.arguments for call in tool_calls]}")
            all_thinking += think
            all_content += content
            yield AgentResultStream(thinking=all_thinking, content=all_content, tool_calls=tool_calls)

        if not tool_calls:
            self.state = AgentState.FINISHED

        assistant_msg = (
            Message.from_tool_calls(content=all_content, tool_calls=tool_calls)
            if tool_calls
            else Message.assistant_message(all_content)
        )
        await self.memory.add(assistant_msg)

    async def act_stream(self):
        """流式返回的act"""
        for command in self.tool_calls:
            # 为每个工具调用重置base64_image
            self._current_base64_image = None
            result = await self.execute_tool(command)
            if self.max_observe:
                result = result[: self.max_observe]
            logger.info(f"🎯 工具 '{command.function.name}' 完成任务！结果: {result}")
            # 将工具响应添加到记忆中
            tool_msg = Message.tool_message(
                content=result,
                tool_call_id=command.id,
                name=command.function.name,
                base64_image=self._current_base64_image,
            )
            await self.memory.add(tool_msg)
            yield AgentResultStream(thinking="", content=result, tool_calls=[])

    async def execute_tool(self, command: ToolCall) -> str:
        """执行单个工具调用，具有健壮的错误处理"""
        if not command or not command.function or not command.function.name:
            return "错误: 无效的命令格式"

        name = command.function.name
        if name not in self.available_tools.tool_map:
            return f"错误: 未知工具 '{name}'"

        try:
            # 解析参数
            args = json.loads(command.function.arguments or "{}")

            # 执行工具
            logger.info(f"🔧 激活工具: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)

            # 检查result是否是包含base64_image的ToolResult
            if hasattr(result, "base64_image") and result.base64_image:
                # 存储base64_image以供稍后在tool_message中使用
                self._current_base64_image = result.base64_image

            # 格式化结果以供显示
            observation = (
                f"执行命令 `{name}` 的观察结果:\n{str(result)}"
                if result
                else f"命令 `{name}` 完成但没有输出"
            )
            return observation
        except json.JSONDecodeError:
            error_msg = f"错误: 解析参数 {name}: 无效的JSON格式"
            logger.error(
                f"📝 Oops! 参数 '{name}' 不正确 - 无效的JSON, 参数:{command.function.arguments}"
            )
            return f"错误: {error_msg}"
        except Exception as e:
            error_msg = f"⚠️ 工具 '{name}' 遇到问题: {str(e)}"
            logger.exception(error_msg)
            return f"错误: {error_msg}"