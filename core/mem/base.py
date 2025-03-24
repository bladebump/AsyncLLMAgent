from abc import ABC, abstractmethod
from core.openai_types import Message
from typing import List

class AsyncMemory(ABC):
    @abstractmethod
    async def add(self, message: List[Message]):
        pass

    @abstractmethod
    async def search(self, query: str) -> str:
        pass

    @abstractmethod
    async def clear(self):
        pass
    
    @abstractmethod
    async def get_last_n_messages(self, n: int) -> str:
        pass

    @abstractmethod
    async def save(self):
        pass

    @abstractmethod
    async def load(self):
        pass
    
    @abstractmethod
    async def _summary(self) -> str:
        pass
