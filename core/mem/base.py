from abc import ABC, abstractmethod
from core.schema import Message
from typing import List

class AsyncMemory(ABC):
    
    def __init__(self, messages: List[Message] = [], max_turn: int = 100, max_length: int = 100000):
        self.Messages = messages
        self.max_turn = max_turn
        self.max_length = max_length

    @abstractmethod
    async def add(self, message: Message):
        pass

    @abstractmethod
    async def add_system(self, message: Message):
        pass

    @abstractmethod
    async def has_system(self) -> bool:
        pass

    @abstractmethod
    async def search(self, query: str) -> List[Message]:
        pass

    @abstractmethod
    async def clear(self):
        pass
    
    async def get_last_n_messages(self, n: int) -> List[Message]:
        return self.Messages[-n:]

    @abstractmethod
    async def save(self):
        pass

    @abstractmethod
    async def load(self):
        pass

    def __len__(self):
        return len(self.Messages)

    @classmethod
    def load_from_history(cls, history: List[dict]):
        memory = cls()
        memory.Messages = [Message.from_history(message) for message in history]
        return memory
