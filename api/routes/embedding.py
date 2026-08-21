from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.ApiResponse import ApiResponse
from db.session import get_db
from schemas.embedding import EmbeddingRequest, ChunkEmbeddingResponse, VectorRetrievalResponse, VectorRetrievalRequest
from services.embedding_service import EmbeddingService

router = APIRouter(
    prefix="/embedding",
    tags=["embedding"],
)

embedding_service = EmbeddingService()

@router.post("/documents")
def embedding_documents(
        request:EmbeddingRequest
)->ApiResponse[list[ChunkEmbeddingResponse]]:
    result = embedding_service.embed_chunks(request)
    return ApiResponse.success(result)

@router.post(
    "/retrieval",
    response_model=ApiResponse[list[VectorRetrievalResponse]],
)
def dense_retrieval(
        request:VectorRetrievalRequest,
        db: Session = Depends(get_db)
)->ApiResponse[list[VectorRetrievalResponse]]:
    result=embedding_service.retrieve_dense(request,db)
    return ApiResponse.success(result)
