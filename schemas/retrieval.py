from datetime import datetime
from typing import Any, Annotated

from pydantic import BaseModel, ConfigDict, Field, model_validator


class RetrieveRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    query: str = Field(min_length=1)
    knowledge_base_ids: list[Annotated[int, Field(gt=0)]] = Field(
        alias="knowledgeBaseIds", min_length=1, max_length=200,
    )
    candidate_k: int = Field(default=30, ge=1, le=200, alias="candidateK")
    rrf_top_k: int = Field(default=20, ge=1, le=100, alias="rrfTopK")

    @model_validator(mode="after")
    def validate_limits(self):
        if not self.query.strip():
            raise ValueError("query must not be blank")
        if self.rrf_top_k > self.candidate_k:
            raise ValueError("rrf_top_k must be less than or equal to candidate_k")
        return self


class AccessibleKnowledgeBases(BaseModel):
    knowledgeBaseIds: list[int]


class ChunkDetail(BaseModel):
    id: int
    document_id: int
    knowledge_base_id: int
    chunk_index: int
    content: str
    document_name: str
    char_count: int
    section_title: str | None
    section_level: int | None
    metadata: dict[str, Any]
    document_version: int
    created_at: datetime
    updated_at: datetime


class DocumentDetail(BaseModel):
    # Preserve the established Spring document metadata contract.
    id: int
    knowledgeBaseId: int
    knowledgeBaseName: str | None = None
    name: str
    originalName: str | None = None
    fileType: str | None = None
    fileSize: int | None = None
    status: str | None = None
    version: int | None = None
    uploadedBy: int | None = None
    uploaderName: str | None = None
    errorMessage: str | None = None
    createdAt: datetime | None = None
    updatedAt: datetime | None = None
