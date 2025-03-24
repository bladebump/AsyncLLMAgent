import abc
import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional, TypeVar, Union
from .items import RunItem
from core.openai_types import Message
from core.Agent.base import AsyncAgent
from pydantic import BaseModel
from utils.log import logger

T = TypeVar("T")

@dataclass
class RawTextEvent:
    """表示从LLM接收到的原始文本事件"""
    text: str
    """文本内容"""
    type: Literal["raw_text"] = "raw_text"

@dataclass
class AgentEvent:
    """表示Agent相关的事件"""
    name: str
    """Agent名称"""
    action: str
    """Agent执行的动作"""
    type: Literal["agent"] = "agent"

@dataclass
class ToolCallEvent:
    """表示工具调用事件"""
    tool_name: str
    """工具名称"""
    arguments: Dict[str, Any]
    """工具参数"""
    type: Literal["tool_call"] = "tool_call"

@dataclass
class ToolResultEvent:
    """表示工具调用结果事件"""
    tool_name: str
    """工具名称"""
    result: Any
    """工具调用结果"""
    type: Literal["tool_result"] = "tool_result"

# 定义StreamEvent类型
StreamEvent = Union[RawTextEvent, AgentEvent, ToolCallEvent, ToolResultEvent]

# 定义工具调用相关的数据结构
@dataclass
class ToolCall:
    """表示一个工具调用"""
    tool_name: str
    """工具名称"""
    arguments: Dict[str, Any]
    """工具参数"""
    result: Optional[Any] = None
    """工具调用结果，初始为None"""

# 队列完成标记
class QueueCompleteSentinel:
    """表示队列处理完成的标记"""
    pass

@dataclass
class RunResultBase(abc.ABC):
    input: str
    """输入"""

    new_items: list[RunItem]
    """在代理运行期间生成的新的项。这些包括新的消息、工具调用及其输出等。"""

    raw_responses: list[Message]
    """在代理运行期间由模型生成的原始LLM响应。"""

    final_output: Any
    """最后一个代理的输出。"""

    @property
    @abc.abstractmethod
    def last_agent(self) -> AsyncAgent:
        """The last agent that was run."""

    def to_input_list(self) -> list[Message]:
        """Creates a new input list, merging the original input with all the new items generated."""
        original_items: list[Message] = Message(role="user", content=self.input)
        new_items = [item.to_input_item() for item in self.new_items]

        return original_items + new_items


@dataclass
class RunResult(RunResultBase):
    _last_agent: AsyncAgent

    @property
    def last_agent(self) -> AsyncAgent:
        """The last agent that was run."""
        return self._last_agent


@dataclass
class RunResultStreaming(RunResultBase):
    """agent运行结果的流式处理. 
    
    流式处理方法会抛出以下异常:
    - MaxTurnsExceeded: 如果代理超过了最大回合数限制.
    - GuardrailTripwireTriggered: 如果触发了一个守卫线.
    """

    current_agent: AsyncAgent
    """当前正在运行的代理."""

    current_turn: int
    """当前回合数."""

    max_turns: int
    """代理可以运行的最大回合数."""

    final_output: Any
    """代理的最终输出. 在代理运行结束之前为None."""

    is_complete: bool = False
    """是否代理已经运行结束."""

    # 后台run_loop写入的队列
    _event_queue: asyncio.Queue[StreamEvent | QueueCompleteSentinel] = field(
        default_factory=asyncio.Queue, repr=False
    )

    # 存储我们正在等待的asyncio任务
    _run_impl_task: asyncio.Task[Any] | None = field(default=None, repr=False)
    _stored_exception: Exception | None = field(default=None, repr=False)

    @property
    def last_agent(self) -> AsyncAgent:
        """The last agent that was run. Updates as the agent run progresses, so the true last agent
        is only available after the agent run is complete.
        """
        return self.current_agent

    async def stream_events(self) -> AsyncIterator[StreamEvent]:
        """Stream deltas for new items as they are generated. We're using the types from the
        OpenAI Responses API, so these are semantic events: each event has a `type` field that
        describes the type of the event, along with the data for that event.

        This will raise:
        - A MaxTurnsExceeded exception if the agent exceeds the max_turns limit.
        - A GuardrailTripwireTriggered exception if a guardrail is tripped.
        """
        while True:
            self._check_errors()
            if self._stored_exception:
                logger.debug("Breaking due to stored exception")
                self.is_complete = True
                break

            if self.is_complete and self._event_queue.empty():
                break

            try:
                item = await self._event_queue.get()
            except asyncio.CancelledError:
                break

            if isinstance(item, QueueCompleteSentinel):
                self._event_queue.task_done()
                # Check for errors, in case the queue was completed due to an exception
                self._check_errors()
                break

            yield item
            self._event_queue.task_done()

        if self._trace:
            self._trace.finish(reset_current=True)

        self._cleanup_tasks()

        if self._stored_exception:
            raise self._stored_exception

    def _check_errors(self):
        if self.current_turn > self.max_turns:
            self._stored_exception = Exception(f"Max turns ({self.max_turns}) exceeded")

        # Check the tasks for any exceptions
        if self._run_impl_task and self._run_impl_task.done():
            exc = self._run_impl_task.exception()
            if exc and isinstance(exc, Exception):
                self._stored_exception = exc

    def _cleanup_tasks(self):
        if self._run_impl_task and not self._run_impl_task.done():
            self._run_impl_task.cancel()
