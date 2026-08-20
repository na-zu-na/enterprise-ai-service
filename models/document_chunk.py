from pgvector.sqlalchemy import VECTOR
from sqlalchemy import BigInteger, Integer, Text
from sqlalchemy.orm import DeclarativeBase, mapped_column, Mapped


class Base(DeclarativeBase):
    pass


class DocumentChunkEntity(Base):
    __tablename__ = "document_chunk"

    id: Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
    )
    document_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    knowledge_base_id: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    section_title: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )
    embedding: Mapped[list[float]] = mapped_column(
        VECTOR(1024),
        nullable=False,
    )

class KnowledgeDocumentEntity(Base):
    __tablename__ = "knowledge_document"

    id:Mapped[int] = mapped_column(
        BigInteger,
        primary_key=True,
    )

    name: Mapped[str | None] = mapped_column(
        Text,
        nullable=False,
    )