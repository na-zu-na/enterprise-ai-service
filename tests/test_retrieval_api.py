import unittest
import sys
from datetime import datetime
from unittest.mock import Mock, patch

import requests
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from api.routes.retrieval import router, chunk_detail
from clients.spring_client import SpringUnauthorizedError
from db.session import get_db
from schemas.embedding import HybridRetrievalResponse


class RetrievalApiTest(unittest.TestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(router)
        self.db = Mock()
        self.app.dependency_overrides[get_db] = lambda: self.db
        self.client = TestClient(self.app)
        self.headers = {"Authorization": "Bearer test-user-token"}
        self.spring = Mock()
        self.spring.get.return_value.json.return_value = {"code": 200, "data": [1, 2]}
        patcher = patch("services.knowledge_access_service.get_spring_client", return_value=self.spring)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_authentication_required_on_all_routes(self):
        for path in ("/knowledge-bases/accessible", "/chunks/1", "/documents/1"):
            self.assertEqual(self.client.get(path).status_code, 401)
        response = self.client.post("/retrieval/retrieve", json={"query": "hello", "knowledgeBaseIds": [1]})
        self.assertEqual(response.status_code, 401)
        self.spring.get.assert_not_called()

    def test_accessible_ids_forward_token_and_deduplicate(self):
        self.spring.get.return_value.json.return_value["data"] = [2, 1, 2]
        response = self.client.get("/knowledge-bases/accessible", headers=self.headers)
        self.assertEqual(response.json()["data"], {"knowledgeBaseIds": [2, 1]})
        self.assertEqual(self.spring.get.call_args.kwargs["access_token"], "test-user-token")

    def test_permission_service_fails_closed(self):
        for payload in ({"code": 403}, {"code": 200, "data": None}, {"code": 200, "data": [True]}, {"code": 200, "data": ["1"]}):
            self.spring.get.return_value.json.return_value = payload
            response = self.client.get("/knowledge-bases/accessible", headers=self.headers)
            self.assertIn(response.status_code, (403, 502))
        for error, expected in ((SpringUnauthorizedError(), 401), (requests.Timeout(), 502)):
            self.spring.get.side_effect = error
            self.assertEqual(self.client.get("/knowledge-bases/accessible", headers=self.headers).status_code, expected)

    def test_retrieve_rejects_mixed_authorized_and_unauthorized_ids(self):
        with patch("api.routes.retrieval.get_hybrid_retrieval_service") as factory:
            response = self.client.post("/retrieval/retrieve", headers=self.headers,
                                        json={"query": "hello", "knowledgeBaseIds": [1, 99]})
            self.assertEqual(response.status_code, 403)
            factory.assert_not_called()

    def test_retrieve_returns_full_rrf_candidates_without_reranking(self):
        content = "完整正文" * 1000
        candidates = [HybridRetrievalResponse(id=i, document_id=1, knowledge_base_id=1,
                     chunk_index=i, content=content, document_name="文档", section_title="章节", rrf_score=0.03)
                      for i in (1, 2)]
        hybrid = Mock()
        hybrid.retrieve.return_value = candidates
        with patch("api.routes.retrieval.get_hybrid_retrieval_service", return_value=hybrid), \
                patch.dict(sys.modules, {"services.reranking_service": None,
                                         "services.retrieval_pipeline_service": None}):
            response = self.client.post("/retrieval/retrieve", headers=self.headers,
                json={"query": "hello", "knowledgeBaseIds": [1], "candidateK": 3, "rrfTopK": 2})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["data"]), 2)
        self.assertEqual(response.json()["data"][0]["content"], content)
        self.assertIsNone(response.json()["data"][0]["rerank_score"])
        self.assertEqual(hybrid.retrieve.call_args.args[0].rrf_top_k, 2)

    def test_retrieve_validates_limits_and_accepts_snake_case(self):
        for fields in ({"knowledgeBaseIds": []}, {"query": " "}, {"candidateK": 1, "rrfTopK": 2}, {"top_k": 5}):
            payload = {"query": "hello", "knowledgeBaseIds": [1]} | fields
            self.assertEqual(self.client.post("/retrieval/retrieve", headers=self.headers, json=payload).status_code, 422)
        with patch("api.routes.retrieval.get_hybrid_retrieval_service") as factory:
            factory.return_value.retrieve.return_value = []
            response = self.client.post("/retrieval/retrieve", headers=self.headers,
                json={"query": "hello", "knowledge_base_ids": [1], "candidate_k": 1, "rrf_top_k": 1})
        self.assertEqual(response.status_code, 200)

    def test_chunk_not_found_and_permission_filter(self):
        self.db.execute.return_value.mappings.return_value.first.return_value = None
        self.assertEqual(self.client.get("/chunks/99", headers=self.headers).status_code, 404)
        self.assertEqual(self.db.execute.call_args.args[1], {"chunk_id": 99, "ids": [1, 2]})

    def test_chunk_full_content_and_metadata(self):
        row = dict(id=1, document_id=2, knowledge_base_id=1, chunk_index=0,
                   content="正文" * 1000, document_name="文档", char_count=2000,
                   section_title=None, section_level=None, metadata={"page": 3},
                   document_version=1, created_at=datetime(2026, 1, 1), updated_at=datetime(2026, 1, 1))
        self.db.execute.return_value.mappings.return_value.first.return_value = row
        response = self.client.get("/chunks/1", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["metadata"], {"page": 3})
        self.assertEqual(response.json()["data"]["content"], row["content"])

    def test_chunk_sql_excludes_deleted_and_inaccessible_rows(self):
        engine = create_engine("sqlite://")
        self.addCleanup(engine.dispose)
        with Session(engine) as session:
            session.execute(text("CREATE TABLE knowledge_document (id INT, knowledge_base_id INT, name TEXT, deleted BOOLEAN)"))
            session.execute(text("""CREATE TABLE document_chunk (
                id INT, document_id INT, knowledge_base_id INT, chunk_index INT,
                content TEXT, char_count INT, section_title TEXT, section_level INT,
                metadata TEXT, document_version INT, created_at TEXT, updated_at TEXT,
                deleted BOOLEAN)"""))
            session.execute(text("INSERT INTO knowledge_document VALUES (1, 1, 'doc', FALSE)"))
            session.execute(text("""INSERT INTO document_chunk VALUES
                (1, 1, 1, 0, 'body', 4, NULL, NULL, '{}', 1, '2026-01-01', '2026-01-01', FALSE)"""))
            for ids, chunk_deleted, document_deleted in (([], False, False), ([2], False, False), ([1], True, False), ([1], False, True)):
                session.execute(text("UPDATE document_chunk SET deleted = :deleted"), {"deleted": chunk_deleted})
                session.execute(text("UPDATE knowledge_document SET deleted = :deleted"), {"deleted": document_deleted})
                with self.assertRaises(HTTPException) as error:
                    chunk_detail(chunk_id=1, ids=ids, db=session)
                self.assertEqual(error.exception.status_code, 404)

    def test_document_metadata_and_permission_check(self):
        def spring_get(path, **kwargs):
            data = [1, 2] if path.endswith("accessible-ids") else {"id": 3, "knowledgeBaseId": 1, "name": "文档", "fileType": "pdf"}
            response = Mock()
            response.json.return_value = {"code": 200, "data": data}
            return response
        self.spring.get.side_effect = spring_get
        response = self.client.get("/documents/3", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["data"]["fileType"], "pdf")
        with patch("api.routes.retrieval.get_document_detail", return_value=Mock(knowledgeBaseId=99)):
            self.assertEqual(self.client.get("/documents/3", headers=self.headers).status_code, 404)


if __name__ == "__main__":
    unittest.main()
