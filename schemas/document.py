from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DocumentParseRequest(BaseModel):

    model_config = ConfigDict(
        validate_by_alias=True,
        validate_by_name=True,
    )

    document_id: int = Field(alias="documentId")

    knowledge_base_id: int = Field(
        alias="knowledgeBaseId"
    )

    name: str

    original_name: str = Field(
        alias="originalName"
    )

    file_type: str = Field(
        alias="fileType"
    )

    file_size: int = Field(
        alias="fileSize"
    )

    storage_path: str = Field(
        alias="storagePath"
    )

    version: int = 1

class DocumentParseResponse(BaseModel):

    model_config = ConfigDict(
        from_attributes=True,
        validate_by_alias=True,
        validate_by_name=True,
    )

    document_id: int = Field(
        alias="documentId"
    )

    knowledge_base_id: int = Field(
        alias="knowledgeBaseId"
    )
    chunk_index: int = Field(
        alias="chunkIndex"
    )
    content: str
    char_count: int = Field(
        default=0,
        alias="charCount"
    )

    metadata: dict[str, Any] = Field(default_factory=dict)
