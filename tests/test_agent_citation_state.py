import unittest

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Overwrite

from agents.state import AgentState, merge_citations
from db.serialization import create_checkpoint_serializer
from schemas.embedding import HybridRetrievalResponse


def citation(chunk_id: int) -> HybridRetrievalResponse:
    return HybridRetrievalResponse(
        id=chunk_id,
        document_id=10,
        knowledge_base_id=20,
        chunk_index=chunk_id,
        content=f"内容 {chunk_id}",
        document_name="制度",
        section_title="章节",
        rrf_score=0.1,
    )


class AgentCitationStateTest(unittest.TestCase):
    def test_merge_citations_preserves_order_and_deduplicates_chunks(self):
        self.assertEqual(
            [item.id for item in merge_citations(
                [citation(1), citation(2)],
                [citation(2), citation(3)],
            )],
            [1, 2, 3],
        )

    def test_parallel_updates_are_merged_and_next_turn_can_clear_them(self):
        builder = StateGraph(AgentState)
        builder.add_node("search_a", lambda _state: {"citations": [citation(1)]})
        builder.add_node("search_b", lambda _state: {"citations": [citation(2)]})
        builder.add_edge(START, "search_a")
        builder.add_edge(START, "search_b")
        builder.add_edge("search_a", END)
        builder.add_edge("search_b", END)
        graph = builder.compile(
            checkpointer=InMemorySaver(
                serde=create_checkpoint_serializer(),
            )
        )
        config = {"configurable": {"thread_id": "citation-test"}}

        first = graph.invoke(
            {
                "messages": [],
                "knowledge_base_ids": [20],
                "citations": Overwrite(value=[]),
            },
            config,
        )
        self.assertEqual([item.id for item in first["citations"]], [1, 2])

        second = graph.invoke(
            {
                "messages": [],
                "knowledge_base_ids": [20],
                "citations": Overwrite(value=[]),
            },
            config,
        )
        self.assertEqual([item.id for item in second["citations"]], [1, 2])


if __name__ == "__main__":
    unittest.main()
