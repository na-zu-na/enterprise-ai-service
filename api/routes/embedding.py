from fastapi import APIRouter

from core.ApiResponse import ApiResponse
from schemas.embedding import EmbeddingRequest, ChunkEmbeddingResponse
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