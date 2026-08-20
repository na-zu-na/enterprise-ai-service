from sqlalchemy import func, select
from sqlalchemy.orm import Session

from embeddings.factory import EmbeddingFactory
from models.document_chunk import DocumentChunkEntity, KnowledgeDocumentEntity
from schemas.embedding import EmbeddingRequest, ChunkEmbeddingResponse, VectorRetrievalRequest, VectorRetrievalResponse


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

    def retrieve(self,request:VectorRetrievalRequest,db:Session)->list[VectorRetrievalResponse]:
        query_embedding=self.embedding_model.embed_query(request.query)

        if not request.knowledge_base_ids:
            return []

        distance=(
            DocumentChunkEntity.embedding
            .cosine_distance(query_embedding)
            .label('distance')
        )

        statement=(
            select(
                DocumentChunkEntity.id,
                DocumentChunkEntity.document_id,
                func.coalesce(
                    KnowledgeDocumentEntity.name,
                    ""
                ).label("document_name"),
                DocumentChunkEntity.knowledge_base_id,
                DocumentChunkEntity.chunk_index,
                DocumentChunkEntity.content,
                func.coalesce(
                    DocumentChunkEntity.section_title,
                    "",
                ).label("section_title"),
                distance,
            )
            .outerjoin(
                KnowledgeDocumentEntity,
                KnowledgeDocumentEntity.id==DocumentChunkEntity.document_id
            )
            .where(
                DocumentChunkEntity.knowledge_base_id.in_(request.knowledge_base_ids)
            )
            .order_by(distance)
            .limit(request.top_k)
        )

        rows=db.execute(statement).mappings().all()

        chunks=[
            VectorRetrievalResponse(**row)
            for row in rows
        ]

        return chunks