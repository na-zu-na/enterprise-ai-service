from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from core.ApiResponse import ApiResponse
from db.session import get_db
from schemas.embedding import RagResponse, VectorRetrievalRequest
from services.rag_service import RagService

router = APIRouter(
    prefix="/rag",
    tags=["rag"],
)

rag_service = RagService()

@router.post("/query",response_model=ApiResponse[RagResponse])
def rag_query(
        request:VectorRetrievalRequest,
        db:Session=Depends(get_db)
)-> ApiResponse[RagResponse]:
    result = rag_service.answer(request, db)
    return ApiResponse.success(result)