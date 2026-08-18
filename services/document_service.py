from chunkers.factory import ChunkerFactory
from chunkers.models import DocumentChunk
from parsers.factory import ParserFactory
from parsers.models import ParsedDocument, DocumentParseContext
from schemas.document import DocumentParseRequest


class DocumentService:
    def parse_document(self,request:DocumentParseRequest)-> list[DocumentChunk]:
        context = DocumentParseContext(
            document_id=request.document_id,
            knowledge_base_id=(
                request.knowledge_base_id
            ),
            name=request.name,
            original_name=(
                request.original_name
            ),
            file_type=request.file_type,
            file_size=request.file_size,
            storage_path=request.storage_path,
            version=request.version,
        )

        parser=ParserFactory.get_parser(request.file_type)
        chunker=ChunkerFactory.get_chunker(request.file_type)
        chunk_document=chunker.chunk(parser.parse(context))

        return chunk_document