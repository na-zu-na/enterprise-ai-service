from pydantic import BaseModel, Field


class DocumentParseRequest(BaseModel):

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

    model_config = {
        "populate_by_name": True
    }


class DocumentParseResponse(BaseModel):

    document_id: int = Field(
        alias="documentId"
    )

    status: str

    title: str | None = None

    char_count: int = Field(
        alias="charCount"
    )

    section_count: int = Field(
        alias="sectionCount"
    )

    model_config = {
        "populate_by_name": True
    }