from enum import Enum

from pydantic import BaseModel, Field, model_validator


class EvalCategory(str, Enum):
    KEYWORD = "keyword"
    SEMANTIC_REWRITE = "semantic_rewrite"
    EXACT_REFERENCE = "exact_reference"
    CROSS_CHUNK = "cross_chunk"
    NO_ANSWER = "no_answer"


class RetrievalEvalCase(BaseModel):
    id: str = Field(min_length=1)
    category: EvalCategory
    query: str = Field(min_length=1)
    knowledge_base_ids: list[int] = Field(min_length=1)
    expected_document_id: int | None = None
    expected_chunk_ids: list[int] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def validate_expected_chunks(self):
        if len(self.knowledge_base_ids) != len(set(self.knowledge_base_ids)):
            raise ValueError("knowledge_base_ids must not contain duplicates")
        if len(self.expected_chunk_ids) != len(set(self.expected_chunk_ids)):
            raise ValueError("expected_chunk_ids must not contain duplicates")

        is_no_answer = self.category == EvalCategory.NO_ANSWER
        if is_no_answer and self.expected_chunk_ids:
            raise ValueError("no_answer cases cannot have expected chunks")
        if not is_no_answer and not self.expected_chunk_ids:
            raise ValueError("answerable cases must have expected chunks")
        if not is_no_answer and self.expected_document_id is None:
            raise ValueError("answerable cases need expected_document_id")
        return self


class QueryMetrics(BaseModel):
    hit_at_k: float | None
    recall_at_k: float | None
    reciprocal_rank: float | None


class QueryEvalResult(BaseModel):
    case_id: str
    category: EvalCategory
    query: str
    expected_chunk_ids: list[int]
    retrieved_chunk_ids: list[int]
    metrics: QueryMetrics


class AggregateMetrics(BaseModel):
    evaluated_queries: int
    skipped_no_answer_queries: int
    hit_at_k: float
    recall_at_k: float
    mrr: float


class StrategyEvalResult(BaseModel):
    strategy: str
    top_k: int
    metrics: AggregateMetrics
    metrics_by_category: dict[str, AggregateMetrics]
    queries: list[QueryEvalResult]


class RetrievalEvalReport(BaseModel):
    dataset_path: str
    candidate_k: int
    rrf_top_k: int
    top_k: int
    strategies: list[StrategyEvalResult]

