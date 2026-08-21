from pydantic import BaseModel, Field, model_validator


class ChunkEmbeddingRequest(BaseModel):
    chunk_id:int
    content:str

class EmbeddingRequest(BaseModel):
    chunks:list[ChunkEmbeddingRequest]

class ChunkEmbeddingResponse(BaseModel):
    chunk_id:int
    embedding:list[float]

class VectorRetrievalRequest(BaseModel):
    query: str
    knowledge_base_ids: list[int]

    # Dense、BM25 各自取多少
    candidate_k: int = Field(
        default=30,
        ge=1,
        le=200,
    )

    # RRF 后送给 Reranker 的数量
    rrf_top_k: int = Field(
        default=20,
        ge=1,
        le=100,
    )

    # Reranker 后最终数量
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
    )

    @model_validator(mode="after")
    def validate_retrieval_limits(self):
        if self.top_k > self.rrf_top_k:
            raise ValueError(
                "top_k must be less than or equal to rrf_top_k"
            )

        if self.rrf_top_k > self.candidate_k:
            raise ValueError(
                "rrf_top_k must be less than or equal to candidate_k"
            )

        return self

class VectorRetrievalResponse(BaseModel):
    id:int
    document_id:int
    knowledge_base_id:int
    chunk_index:int
    content:str
    document_name:str

    section_title:str
    distance: float

class HybridRetrievalResponse(BaseModel):
    id: int
    document_id: int
    knowledge_base_id: int
    chunk_index: int
    content: str
    document_name: str
    section_title: str

    rrf_score: float

    dense_rank: int | None = None
    dense_distance: float | None = None

    bm25_rank: int | None = None
    bm25_score: float | None = None

    rerank_score: float | None = None
    rerank_rank: int | None = None

class RagResponse(BaseModel):
    answer:str
    citations:list[HybridRetrievalResponse]