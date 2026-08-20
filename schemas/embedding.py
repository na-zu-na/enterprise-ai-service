from pydantic import BaseModel, Field


class ChunkEmbeddingRequest(BaseModel):
    chunk_id:int
    content:str

class EmbeddingRequest(BaseModel):
    chunks:list[ChunkEmbeddingRequest]

class ChunkEmbeddingResponse(BaseModel):
    chunk_id:int
    embedding:list[float]

class VectorRetrievalRequest(BaseModel):
    query:str
    knowledge_base_ids: list[int]
    top_k:int=Field(default=5,ge=1,le=20)

class VectorRetrievalResponse(BaseModel):
    id:int
    document_id:int
    knowledge_base_id:int
    chunk_index:int
    content:str
    document_name:str

    section_title:str
    distance: float

class RagResponse(BaseModel):
    answer:str
    citations:list[VectorRetrievalResponse]