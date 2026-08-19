from abc import abstractmethod


class BaseEmbedding:
    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """
        对单条文本进行向量化
        """
        pass

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        批量对文档文本进行向量化
        """
        pass

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """
        对用户查询进行向量化
        """
        pass