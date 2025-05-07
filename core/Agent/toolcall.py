import json
from typing import Any, List, Optional, Union
from core.agent.react import ReActAgent
from utils.log import logger
from core.schema import AgentState, Message, ToolCall, ToolChoice, AgentResult
from core.tools import Terminate, ToolCollection
from core.llms.errors import TokenLimitExceeded
from core.llms import AsyncBaseChatCOTModel
from core.mem import AsyncMemory

TOOL_CALL_REQUIRED = "需要工具调用但未提供"
SYSTEM_PROMPT = "你是一个可以执行工具调用的代理, 请根据用户的需求选择合适的工具, 并使用工具调用执行任务。你可以反复使用工具调用直到任务完成。"
NEXT_STEP_PROMPT = (
    "如果你想停止交互，请使用`terminate`工具/函数调用。"
)


class ToolCallAgent(ReActAgent):
    """用于处理工具/函数调用的基础代理类"""

    def __init__(
        self,
        name: str = "toolcall",
        llm: AsyncBaseChatCOTModel = None,
        memory: AsyncMemory = None,
        description: str = "一个可以执行工具调用的代理。",
        system_prompt: str = SYSTEM_PROMPT,
        next_step_prompt: str = NEXT_STEP_PROMPT,
        state: AgentState = AgentState.IDLE,
        available_tools: Optional[ToolCollection] = None,
        tool_choices: str = ToolChoice.AUTO,
        special_tool_names: Optional[List[str]] = None,
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
            next_step_prompt=next_step_prompt,
            state=state,
            max_steps=max_steps,
            **kwargs
        )
        
        # 初始化默认工具集合
        if available_tools is None:
            self.available_tools = ToolCollection(Terminate())
        else:
            self.available_tools = available_tools
        self.tool_choices = tool_choices
        
        # 初始化特殊工具名称
        if special_tool_names is None:
            self.special_tool_names = [Terminate().name]
        else:
            self.special_tool_names = special_tool_names
            
        self.tool_calls = []
        self._current_base64_image = None
        self.max_observe = max_observe

    async def think(self) -> str:
        """处理当前状态并决定下一步操作使用工具"""
        if not await self.memory.has_system() and self.system_prompt:
            await self.memory.add_system(Message.system_message(self.system_prompt))

        if self.next_step_prompt:
            await self.memory.add(Message.user_message(self.next_step_prompt))

        response = await self.llm.chat_with_tools(
            messages=self.memory.Messages,
            tools=self.available_tools.to_params(),
            tool_choice=self.tool_choices,
        )

        self.tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
        content = response.content if response and response.content else ""

        # 记录响应信息
        logger.info(f"✨ {self.name}'s thoughts: {content}")
        logger.info(
            f"🛠️ {self.name} 选择了 {len(tool_calls) if tool_calls else 0} 个工具"
        )
        if tool_calls:
            logger.info(
                f"🧰 正在准备工具: {[call.function.name for call in tool_calls]}"
            )
            logger.info(f"🔧 工具参数: {tool_calls[0].function.arguments}")

        try:
            if response is None:
                raise RuntimeError("未从LLM收到响应")

            # 处理不同的tool_choices模式
            if self.tool_choices == ToolChoice.NONE:
                if tool_calls:
                    logger.warning(
                        f"🤔 嗯，{self.name} 尝试使用工具，但它们不可用！"
                    )
                if content:
                    await self.memory.add(Message.assistant_message(content))
                    return True
                return False

            # 创建并添加助手消息
            assistant_msg = (
                Message.from_tool_calls(content=content, tool_calls=self.tool_calls)
                if self.tool_calls
                else Message.assistant_message(content)
            )
            await self.memory.add(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # 将在act()中处理

            # 对于'auto'模式，如果没有任何命令但存在内容，则继续
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 嗯，{self.name} 的思考过程遇到了问题: {e}")
            await self.memory.add(Message.assistant_message(
                f"处理时遇到错误: {str(e)}"
            ))
            return False

    async def act(self) -> AgentResult:
        """执行工具调用并处理其结果"""
        if not self.tool_calls:
            # 如果没有任何命令，返回最后一条消息的内容
            messages = await self.memory.get_last_n_messages(1)
            return AgentResult("", result=messages[0].content or "没有内容或命令要执行")

        results = []
        for command in self.tool_calls:
            # 为每个工具调用重置base64_image
            self._current_base64_image = None

            result = await self.execute_tool(command)

            if self.max_observe:
                result = result[: self.max_observe]

            logger.info(
                f"🎯 工具 '{command.function.name}' 完成任务！结果: {result}"
            )

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

            # 处理特殊工具
            await self._handle_special_tool(name=name, result=result)

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

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """处理特殊工具执行和状态变化"""
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # 设置代理状态为完成
            logger.info(f"🏁 特殊工具 '{name}' 已完成任务!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """确定是否应该完成工具执行"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """检查工具名称是否在特殊工具列表中"""
        return name.lower() in [n.lower() for n in self.special_tool_names]