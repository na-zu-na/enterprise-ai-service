from chunkers.factory import ChunkerFactory
from chunkers.models import DocumentChunk
from embeddings.factory import EmbeddingFactory
from parsers.factory import ParserFactory
from parsers.models import ParsedDocument, DocumentParseContext
from schemas.document import DocumentParseRequest
from schemas.embedding import EmbeddingRequest, ChunkEmbeddingRequest


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

        #解析
        parser=ParserFactory.get_parser(request.file_type)

        #chunk
        chunker=ChunkerFactory.get_chunker(request.file_type)
        chunk_document=chunker.chunk(parser.parse(context))
        if not chunk_document:
            return []

        texts=[
            chunk.content
            for chunk in chunk_document
        ]

        #完成embedding任务
        embedding_model=EmbeddingFactory.get_embedding("bge-m3")
        vectors=embedding_model.embed_documents(texts)
        if len(chunk_document)!=len(vectors):
            raise RuntimeError(
                "Embedding result count does not match chunk count"
            )
        for chunk,vector in zip(chunk_document,vectors):
            chunk.embedding=vector

        return chunk_document