from fastapi import APIRouter

from core.ApiResponse import ApiResponse
from schemas.document import DocumentParseRequest, DocumentParseResponse
from services.document_service import DocumentService

router = APIRouter(
    prefix="/document",
    tags=["document"]
)

document_service=DocumentService()

@router.post(
    "/post",
    response_model=ApiResponse[DocumentParseResponse]
)
def parse_document(request:DocumentParseRequest):
    parsed=document_service.parse_document(request)
    document=DocumentParseResponse(
        document_id=parsed.document_id,
        title=parsed.title,
        status="SUCCESS",
        section_count=len(parsed.sections),
        char_count=len(parsed.content)
    )
    return ApiResponse.success(document)