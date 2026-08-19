from embeddings.base import BaseEmbedding
from embeddings.bge_m3_embedding import BgeM3Embedding


class EmbeddingFactory:
    _instances: dict[str,BaseEmbedding]={}

    @classmethod
    def get_embedding(cls,model_name: str = "bge-m3")->BaseEmbedding:
        if model_name not in cls._instances:
            if model_name == "bge-m3":
                cls._instances[model_name]=BgeM3Embedding()
            else:
                raise ValueError(f"Unsupported embedding model: {model_name}")

        return cls._instances[model_name]