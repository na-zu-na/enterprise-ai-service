import importlib
import sys
import types
import unittest
from unittest.mock import patch


class _FakeModel:
    def __init__(self, *args, **kwargs):
        pass


class _FakeElasticsearch:
    def __init__(self, *args, **kwargs):
        pass


class ApplicationWiringTest(unittest.TestCase):
    def test_main_imports_and_registers_all_routes(self):
        flag_embedding = types.ModuleType("FlagEmbedding")
        flag_embedding.BGEM3FlagModel = _FakeModel
        flag_embedding.FlagReranker = _FakeModel

        elasticsearch = types.ModuleType("elasticsearch")
        elasticsearch.Elasticsearch = _FakeElasticsearch

        with patch.dict(
            sys.modules,
            {
                "FlagEmbedding": flag_embedding,
                "elasticsearch": elasticsearch,
            },
        ):
            sys.modules.pop("main", None)
            main = importlib.import_module("main")

        paths = set(main.app.openapi()["paths"])
        self.assertTrue({
            "/document/parse",
            "/embedding/documents",
            "/embedding/retrieval",
            "/rag/query",
            "/knowledge-bases/accessible",
            "/retrieval/retrieve",
            "/chunks/{chunk_id}",
            "/documents/{document_id}",
            "/agent/chat",
            "/agent/approvals/respond",
        }.issubset(paths))

    def test_get_time_is_a_single_valid_tool_with_fixed_timezone(self):
        from agents.tools.get_time import get_time

        value = get_time.invoke({})
        self.assertEqual(get_time.name, "get_time")
        self.assertEqual(value.isoformat(), "2026-09-07T09:00:00+08:00")
        self.assertEqual(value.tzinfo.key, "Asia/Singapore")


if __name__ == "__main__":
    unittest.main()
