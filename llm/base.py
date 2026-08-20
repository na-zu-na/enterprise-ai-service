from abc import ABC, abstractmethod

from schemas.embedding import RagResponse


class BaseLLM(ABC):
    @abstractmethod
    def generate(self,prompt:str,system_prompt:str)->str:
        pass

