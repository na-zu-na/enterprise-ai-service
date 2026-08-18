from chunkers.models import DocumentChunk
from chunkers.text_chunker import TextDocumentChunker
from parsers.models import ParsedDocument, ParsedSection


class MarkdownDocumentChunker:
    def __init__(self,chunk_size:int=800,chunk_overlap=100)->None:
        self.text_chunker=TextDocumentChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        self.chunk_size=chunk_size
        self.chunk_overlap=chunk_overlap

    def chunk(self,document:ParsedDocument)->list[DocumentChunk]:
        # 没有结构信息时，
        # 退化成普通文本 Chunk
        if not document.sections:
            return self.text_chunker.chunk(document)

        result:list[DocumentChunk]=[]

        chunk_index=0
        for section in document.sections:
            section_content=section.content
            if not section_content:
                continue

            # 短章节直接作为一个 Chunk
            if len(section_content)<=self.chunk_size:
                result.append(
                    self._create_chunk(
                        document=document,
                        section=section,
                        chunk_index=chunk_index,
                        content=section_content
                    )
                )

                chunk_index+=1
                continue

                # 长章节继续切分
            temp_document = ParsedDocument(
                document_id=document.document_id,
                knowledge_base_id=document.knowledge_base_id,

                name=document.name,
                original_name=document.original_name,

                file_type=document.file_type,
                file_size=document.file_size,
                storage_path=document.storage_path,

                version=document.version,

                title=document.title,
                content=section_content,

                sections=[],
                metadata=document.metadata
            )

            section_chunks=self.text_chunker.chunk(temp_document)
            for chunk in section_chunks:
                result.append(
                    self._create_chunk(
                        document=temp_document,
                        section=section,
                        chunk_index=chunk_index,
                        content=chunk.content
                    )
                )
                chunk_index+=1

        return result

    def _create_chunk(
            self,
            document: ParsedDocument,
            section: ParsedSection,
            content: str,
            chunk_index: int
    ) -> DocumentChunk:

        return DocumentChunk(
            document_id=document.document_id,
            knowledge_base_id=document.knowledge_base_id,

            chunk_index=chunk_index,

            content=content,

            char_count=len(content),

            metadata={
                "file_type": document.file_type,
                "document_name": document.name,
                "version": document.version,

                "section_title": section.title,
                "section_level": section.level,
                "page_number": section.page_number
            }
        )
