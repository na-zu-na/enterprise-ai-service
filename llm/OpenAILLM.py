from openai import OpenAI

from core.config import Settings
from llm.base import BaseLLM
from schemas.embedding import RagResponse


class OpenAILLM(BaseLLM):
    def __init__(self):
        api_key=Settings.LLM_API_KEY
        base_url=Settings.LLM_BASE_URL

        self.model=Settings.LLM_MODEL
        self.client=OpenAI(api_key=api_key,timeout=60.0,base_url=base_url)

    def generate(self,prompt:str,system_prompt:str) ->str:
        response=self.client.responses.create(
            model=self.model,
            instructions=system_prompt,
            input=prompt
        )

        return response.output_text