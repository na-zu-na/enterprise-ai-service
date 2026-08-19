from embeddings.factory import EmbeddingFactory
from schemas.embedding import EmbeddingRequest, ChunkEmbeddingResponse


class EmbeddingService:
    def __init__(self):
        self.embedding_model=EmbeddingFactory.get_embedding()

    def embed_chunks(self,request:EmbeddingRequest)->list[ChunkEmbeddingResponse]:
        texts=[
            chunk.content
            for chunk in request.chunks
        ]

        vectors=self.embedding_model.embed_documents(texts)

        result=[]
        for chunk,vector in zip(request.chunks,vectors):
            result.append(
                ChunkEmbeddingResponse(
                    chunk_id=chunk.chunk_id,
                    embedding=vector
                )
            )

        return result