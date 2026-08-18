from fastapi import APIRouter

from core.ApiResponse import ApiResponse
from schemas.document import DocumentParseRequest, DocumentParseResponse
from services.document_service import DocumentService

router = APIRouter(
    prefix="/document",
    tags=["document"],
)

document_service = DocumentService()


@router.post(
    "/parse",
    response_model=ApiResponse[list[DocumentParseResponse]],
)
def parse_document(
    request: DocumentParseRequest,
) -> ApiResponse[list[DocumentParseResponse]]:
    chunks = document_service.parse_document(request)
    response_chunks = [
        DocumentParseResponse.model_validate(chunk)
        for chunk in chunks
    ]

    return ApiResponse[list[DocumentParseResponse]].success(
        response_chunks
    )
