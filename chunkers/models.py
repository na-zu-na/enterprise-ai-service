from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocumentChunk:
    document_id: int
    knowledge_base_id: int
    chunk_index: int
    content: str
    char_count:int=0

    metadata:dict[str,Any]=field(default_factory=dict)
