from typing import List
from core.schema import Message, Role
from .base import AsyncMemory

class ListMemory(AsyncMemory):
    async def add(self, message: Message):
        if len(self.Messages) < self.max_length:
            self.Messages.append(message)
        else:
            if self.Messages[0].role != Role.SYSTEM:
                self.Messages.pop(0)
            else:
                self.Messages.pop(1)
            self.Messages.append(message)
    
    async def add_system(self, message: Message):
        if self.Messages[0].role != Role.SYSTEM:
            self.Messages.insert(0, message)
        if len(self.Messages) > self.max_length:
            self.Messages.pop(1)
        
    async def has_system(self) -> bool:
        return self.Messages[0].role == Role.SYSTEM

    async def search(self, query: str) -> List[Message]:
        return self.Messages

    async def clear(self):
        self.Messages = []
    
    async def load_from_history(self, history: List[dict]):
        self.Messages = [Message.from_history(message) for message in history]

    async def save(self):
        raise NotImplementedError("ListMemory does not support saving")

    async def load(self):
        raise NotImplementedError("ListMemory does not support loading")
