from FlagEmbedding import FlagReranker

from core.config import Settings


class BgeReranker:
    def __init__(self) -> None:
        self.model = FlagReranker(
            Settings.RERANKER_MODEL,
            query_max_length=(
                Settings.RERANKER_QUERY_MAX_LENGTH
            ),
            max_length=(
                Settings.RERANKER_PASSAGE_MAX_LENGTH
            ),
            use_fp16=Settings.RERANKER_USE_FP16,
        )

    def compute_scores(
            self,
            query: str,
            passages: list[str],
    ) -> list[float]:
        if not passages:
            return []

        pairs=[
            (query,passage)
            for passage in passages
        ]

        scores = self.model.compute_score(
            pairs,
            normalize=True,
        )

        # 兼容 numpy.ndarray
        if hasattr(scores, "tolist"):
            scores = scores.tolist()

        # 兼容只有一个候选时返回标量
        if not isinstance(scores, (list, tuple)):
            return [float(scores)]

        return [
            float(score)
            for score in scores
        ]