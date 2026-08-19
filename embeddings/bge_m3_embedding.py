from embeddings.base import BaseEmbedding
from FlagEmbedding import BGEM3FlagModel


class BgeM3Embedding(BaseEmbedding):
    def __init__(self):
        self.model = BGEM3FlagModel(
            "BAAI/bge-m3",
            use_fp16=False
        )

    def embed_text(self, text: str) -> list[float]:

        result = self.model.encode(
            [text],
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )

        return result["dense_vecs"][0].tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:

        result = self.model.encode(
            texts,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False
        )

        return result["dense_vecs"].tolist()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text(query)