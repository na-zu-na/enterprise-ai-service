from pydantic import BaseModel


class ChunkEmbeddingRequest(BaseModel):
    chunk_id:int
    content:str

class EmbeddingRequest(BaseModel):
    chunks:list[ChunkEmbeddingRequest]

class ChunkEmbeddingResponse(BaseModel):
    chunk_id:int
    embedding:list[float]
