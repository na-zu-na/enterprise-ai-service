from abc import ABC, abstractmethod
from typing import List

from chunkers.models import DocumentChunk
from parsers.models import ParsedDocument


class BaseDocumentChunker(ABC):
    @abstractmethod
    def chunk(
            self,
            document: ParsedDocument,
    )->List[DocumentChunk]:
        pass