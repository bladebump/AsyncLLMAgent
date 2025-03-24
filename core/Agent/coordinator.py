from core.Agent.base import AsyncAgent
from core.llms.base import AsyncBaseLLMModel
from core.mem.base import AsyncMemory
from typing import Dict, Iterator, List, Union, Tuple, Callable
from core.openai_types import ChatCompletionMessageParam

coordinator_instruction = """
你是一个协调者，负责协调多个Agent之间的对话。
"""

class Coordinator(AsyncAgent):
    """
    一个协调者类，负责协调多个Agent之间的对话
    """
    
    def __init__(
        self, 
        agents: List[AsyncAgent],
        llm: AsyncBaseLLMModel = None, 
        memory: AsyncMemory = None,
        function_list: List[Callable] | None = None, 
        name: str | None = None, 
        instruction: str | None = None,
        max_rounds: int = 10,
        stream: bool = True, 
        **kwargs
    ):
        """
        初始化一个GroupChat

        Args:
            agents: 参与群聊的Agent列表
            llm: 这个GroupChat的LLM配置，如果为None则使用coordinator的LLM
            memory: 这个GroupChat的记忆
            function_list: 一个工具列表
            name: 这个GroupChat的名称
            max_rounds: 最大对话轮次
            stream: 是否流式输出
            kwargs: 其他潜在的参数
        """
        if instruction is None:
            instruction = coordinator_instruction
        else:
            instruction = f"{coordinator_instruction}\n{instruction}"

        super().__init__(
            function_list=function_list, 
            llm=llm, 
            memory=memory,
            name=name or "Coordinator", 
            instruction=instruction, 
            stream=stream, 
            **kwargs
        )
        
        self.agents = {agent.name: agent for agent in agents}
        self.max_rounds = max_rounds
        self.messages_history = []
        
    async def _run(self, prompt: str, messages: List[ChatCompletionMessageParam] | None = None, **kwargs) -> Union[Tuple[str, str], Iterator[Tuple[str, str]]]:
        """
        运行协调者

        Args:
            prompt: 用户的输入
            messages: 历史对话
            kwargs: 其他参数

        Returns:
            协调者的结果
        """
        # 初始化对话历史
        if messages is None:
            self.messages_history = [
                {"role": "system", "content": self.instruction},
                {"role": "user", "content": prompt}
            ]
        else:
            self.messages_history = messages
        
        # 开始协调
        current_round = 0
        final_answer = None
        
        while current_round < self.max_rounds:
            current_round += 1
            
            # 让协调者决定下一个发言的Agent
            coordinator_prompt = self._format_coordinator_prompt()
            coordinator_response = await self._call_llm(messages=self.messages_history, prompt=coordinator_prompt)
            
            # 解析协调者的响应，获取下一个发言的Agent和问题
            next_agent_name, next_prompt = self._parse_coordinator_response(coordinator_response)
            
            # 如果协调者决定结束对话，返回最终答案
            if next_agent_name.lower() == "final" or next_agent_name.lower() == "end":
                final_answer = next_prompt
                break
                
            # 获取下一个发言的Agent
            if next_agent_name not in self.agents:
                # 如果指定的Agent不存在，记录错误并继续
                error_msg = f"Agent '{next_agent_name}' 不存在，可用的Agent有: {list(self.agents.keys())}"
                self.messages_history.append({"role": "system", "content": error_msg})
                continue
                
            next_agent = self.agents[next_agent_name]
            
            # 让下一个Agent回答问题
            agent_response = await next_agent.run(prompt = next_prompt, messages=self.messages_history)
            
            # 将Agent的回答添加到对话历史
            if isinstance(agent_response, tuple):
                # 非流式响应
                _, content = agent_response
                self.messages_history.append({
                    "role": "assistant", 
                    "name": next_agent_name,
                    "content": content
                })
            else:
                # 流式响应，收集完整内容
                full_content = ""
                async for _, content in agent_response:
                    if content:
                        full_content += content
                
                self.messages_history.append({
                    "role": "assistant", 
                    "name": next_agent_name,
                    "content": full_content
                })
        
        # 如果达到最大轮次但没有最终答案，让协调者总结
        if final_answer is None:
            summary_prompt = "请总结以上对话，给出最终答案。"
            summary_response = await self.coordinator._run(summary_prompt)
            
            if isinstance(summary_response, tuple):
                _, final_answer = summary_response
            else:
                # 流式响应，收集完整内容
                final_answer = ""
                async for _, content in summary_response:
                    if content:
                        final_answer += content
        
        # 返回最终结果
        if self.stream:
            async def stream_result():
                yield "", final_answer
            return stream_result()
        else:
            return "", final_answer
            
    def _format_coordinator_prompt(self) -> str:
        """
        格式化协调者的提示
        
        Returns:
            协调者的提示
        """
        prompt = "当前对话历史:\n\n"
        
        for msg in self.messages_history:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt += f"系统: {content}\n\n"
            elif role == "user":
                prompt += f"用户: {content}\n\n"
            elif role == "assistant":
                name = msg.get("name", "未知")
                prompt += f"{name}: {content}\n\n"
        
        prompt += f"\n可用的Agent: {list(self.agents.keys())}\n\n"
        prompt += "请决定下一个应该发言的Agent，并给出要问的问题。\n"
        prompt += "回复格式: Agent名称: 问题\n"
        prompt += "如果对话应该结束，请回复: Final: 最终答案\n"
        
        return prompt
        
    def _parse_coordinator_response(self, response) -> Tuple[str, str]:
        """
        解析协调者的响应
        
        Args:
            response: 协调者的响应
            
        Returns:
            下一个发言的Agent名称和问题
        """
        if isinstance(response, tuple):
            _, content = response
        else:
            # 流式响应，收集完整内容
            content = ""
            for _, chunk in response:
                if chunk:
                    content += chunk
        
        # 解析响应
        try:
            parts = content.split(":", 1)
            if len(parts) == 2:
                agent_name = parts[0].strip()
                question = parts[1].strip()
                return agent_name, question
            else:
                # 如果格式不正确，默认让第一个Agent回答
                first_agent = list(self.agents.keys())[0]
                return first_agent, content
        except Exception as e:
            # 出错时默认让第一个Agent回答
            first_agent = list(self.agents.keys())[0]
            return first_agent, content 