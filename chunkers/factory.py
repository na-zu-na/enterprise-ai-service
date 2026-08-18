from chunkers.exceptions import DocumentChunkException
from chunkers.markdown_chunker import MarkdownDocumentChunker
from chunkers.text_chunker import TextDocumentChunker


class ChunkerFactory:
    @staticmethod
    def get_chunker(
            file_type: str,
            chunk_size: int = 800,
            chunk_overlap: int = 100
    ) -> TextDocumentChunker | MarkdownDocumentChunker:

        normalized_type = (
            file_type
            .strip()
            .lower()
            .lstrip(".")
        )

        if normalized_type == "txt":
            return TextDocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        if normalized_type in {
            "md",
            "markdown"
        }:
            return MarkdownDocumentChunker(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

        raise DocumentChunkException(
            f"Unsupported chunk file type: {file_type}"
        )