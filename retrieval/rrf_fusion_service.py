from collections.abc import Sequence
from dataclasses import replace
from math import inf

from retrieval.models import RetrievalCandidate


class RrfFusionService:
    @staticmethod
    def fuse(
            *,
            dense_hits: Sequence[RetrievalCandidate],
            bm25_hits: Sequence[RetrievalCandidate],
            top_k: int,
            rank_constant: int = 60,
    ) -> list[RetrievalCandidate]:
        if top_k <= 0:
            return []

        if rank_constant < 1:
            raise ValueError(
                "rank_constant must be greater than 0"
            )

        dense_by_id = {hit.id: hit for hit in dense_hits}
        bm25_by_id = {hit.id: hit for hit in bm25_hits}

        dense_ranks: dict[int, int] = {}
        bm25_ranks: dict[int, int] = {}
        scores: dict[int, float] = {}

        for rank, hit in enumerate(dense_hits, start=1):
            dense_ranks.setdefault(hit.id, rank)
            scores[hit.id] = (
                scores.get(hit.id, 0.0)
                + 1.0 / (rank_constant + rank)
            )

        for rank, hit in enumerate(bm25_hits, start=1):
            bm25_ranks.setdefault(hit.id, rank)
            scores[hit.id] = (
                scores.get(hit.id, 0.0)
                + 1.0 / (rank_constant + rank)
            )

        fused: list[RetrievalCandidate] = []

        for chunk_id, rrf_score in scores.items():
            dense_hit = dense_by_id.get(chunk_id)
            bm25_hit = bm25_by_id.get(chunk_id)
            base_hit = dense_hit or bm25_hit

            if base_hit is None:
                continue

            fused.append(
                replace(
                    base_hit,
                    dense_distance=(
                        dense_hit.dense_distance
                        if dense_hit
                        else None
                    ),
                    bm25_score=(
                        bm25_hit.bm25_score
                        if bm25_hit
                        else None
                    ),
                    dense_rank=dense_ranks.get(chunk_id),
                    bm25_rank=bm25_ranks.get(chunk_id),
                    rrf_score=rrf_score,
                )
            )

        fused.sort(
            key=lambda hit: (
                -hit.rrf_score,
                min(
                    hit.dense_rank or inf,
                    hit.bm25_rank or inf,
                ),
                hit.id,
            )
        )

        return fused[:top_k]
