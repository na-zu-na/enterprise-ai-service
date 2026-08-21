from dataclasses import dataclass


@dataclass(slots=True)
class RetrievalCandidate:
    id: int
    document_id: int
    knowledge_base_id: int
    chunk_index: int
    content: str
    document_name: str
    section_title: str

    dense_distance: float | None = None
    bm25_score: float | None = None

    dense_rank: int | None = None
    bm25_rank: int | None = None

    rrf_score: float = 0.0
