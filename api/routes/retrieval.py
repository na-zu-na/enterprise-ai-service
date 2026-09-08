from functools import lru_cache

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy import bindparam, text
from sqlalchemy.orm import Session

from clients.tools.get_access_token import get_access_token
from core.ApiResponse import ApiResponse
from db.session import get_db
from schemas.embedding import HybridRetrievalResponse, VectorRetrievalRequest
from schemas.retrieval import AccessibleKnowledgeBases, ChunkDetail, DocumentDetail, RetrieveRequest
from services.knowledge_access_service import get_accessible_knowledge_base_ids, get_document_detail

router = APIRouter(tags=["retrieval"])


@lru_cache(maxsize=1)
def get_hybrid_retrieval_service():
    # Load embedding resources only when retrieval is requested.
    from services.hybrid_retrieval_service import HybridRetrievalService
    return HybridRetrievalService()


@router.get("/knowledge-bases/accessible", response_model=ApiResponse[AccessibleKnowledgeBases])
def accessible_knowledge_bases(ids: list[int] = Depends(get_accessible_knowledge_base_ids)):
    return ApiResponse.success(AccessibleKnowledgeBases(knowledgeBaseIds=ids))


@router.post("/retrieval/retrieve", response_model=ApiResponse[list[HybridRetrievalResponse]])
def retrieve(
    request: RetrieveRequest,
    ids: list[int] = Depends(get_accessible_knowledge_base_ids),
    db: Session = Depends(get_db),
):
    if not set(request.knowledge_base_ids).issubset(ids):
        raise HTTPException(403, "请求包含无权访问的知识库")
    # top_k belongs to the old pipeline schema; it is unused by hybrid retrieval.
    internal_request = VectorRetrievalRequest(
        query=request.query,
        knowledge_base_ids=list(dict.fromkeys(request.knowledge_base_ids)),
        candidate_k=request.candidate_k,
        rrf_top_k=request.rrf_top_k,
        top_k=1,
    )
    return ApiResponse.success(get_hybrid_retrieval_service().retrieve(internal_request, db))


@router.get("/chunks/{chunk_id}", response_model=ApiResponse[ChunkDetail])
def chunk_detail(
    chunk_id: int = Path(gt=0),
    ids: list[int] = Depends(get_accessible_knowledge_base_ids),
    db: Session = Depends(get_db),
):
    row = db.execute(text("""
        SELECT c.id, c.document_id, c.knowledge_base_id, c.chunk_index,
               c.content, COALESCE(d.name, '') AS document_name,
               c.char_count, c.section_title, c.section_level, c.metadata,
               c.document_version, c.created_at, c.updated_at
        FROM document_chunk c
        JOIN knowledge_document d ON d.id = c.document_id
        WHERE c.id = :chunk_id AND c.deleted = FALSE AND d.deleted = FALSE
          AND c.knowledge_base_id = d.knowledge_base_id
          AND c.knowledge_base_id IN :ids
    """).bindparams(bindparam("ids", expanding=True)), {"chunk_id": chunk_id, "ids": ids}).mappings().first()
    if row is None:
        raise HTTPException(404, "chunk 不存在或无权访问")
    return ApiResponse.success(ChunkDetail.model_validate(row))


@router.get("/documents/{document_id}", response_model=ApiResponse[DocumentDetail])
def document_detail(
    document_id: int = Path(gt=0),
    access_token: str = Depends(get_access_token),
    ids: list[int] = Depends(get_accessible_knowledge_base_ids),
):
    detail = get_document_detail(document_id, access_token)
    if detail.knowledgeBaseId not in ids:
        raise HTTPException(404, "文档不存在或无权访问")
    return ApiResponse.success(detail)
