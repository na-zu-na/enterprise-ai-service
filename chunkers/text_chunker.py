from chunkers.base import BaseDocumentChunker
from chunkers.exceptions import InvalidChunkConfigException
from chunkers.models import DocumentChunk
from parsers.models import ParsedDocument


class TextDocumentChunker(BaseDocumentChunker):
    def __init__(self,chunk_size:int=800,chunk_overlap=100):
        if chunk_size <= 0:
            raise InvalidChunkConfigException(
                "chunk_size must be greater than 0"
            )

        if chunk_overlap < 0:
            raise InvalidChunkConfigException(
                "chunk_overlap cannot be negative"
            )

        if chunk_overlap >= chunk_size:
            raise InvalidChunkConfigException(
                "chunk_overlap must be smaller than chunk_size"
            )
        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap

    def chunk(self,document:ParsedDocument)->list[DocumentChunk]:
        content=document.content.strip()

        if not content:
            return []

        chunks: list[DocumentChunk]=[]

        start=0
        chunk_index=0

        while start<len(content):
            end=min(start+self.chunk_size,len(content))

            # 如果还没有到文档结尾，
            # 尝试在换行位置截断
            if end < len(content):
                newline_position = content.rfind("\n", start, end)

                if newline_position > start:
                    end = newline_position

            chunk_content=content[start:end].strip()
            if chunk_content:
                chunks.append(
                    DocumentChunk(
                        document_id=document.document_id,
                        knowledge_base_id=document.knowledge_base_id,
                        chunk_index=chunk_index,
                        content=chunk_content,
                        char_count=len(chunk_content),
                        metadata={
                            "file_type": document.file_type,
                            "document_name": document.name,
                            "version": document.version
                        }
                    )
                )
                chunk_index+=1

            if end>=len(content):
                break
            start=end-self.chunk_overlap

        return chunks
