import unittest

from fastapi import HTTPException
from pydantic import ValidationError

from api.routes.agent import build_graph_config, get_authenticated_user_id
from schemas.agent import AgentChatRequest


class AgentApiTest(unittest.TestCase):
    def test_thread_is_scoped_to_authenticated_user(self) -> None:
        self.assertEqual(
            build_graph_config(42, 7)["configurable"]["thread_id"],
            "user:7:conversation:42",
        )

    def test_authenticated_user_header_is_required(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            get_authenticated_user_id(None)
        self.assertEqual(raised.exception.status_code, 401)

    def test_knowledge_base_ids_must_be_nonempty_and_positive(self) -> None:
        for knowledge_base_ids in ([], [0], [-1]):
            with self.subTest(knowledge_base_ids=knowledge_base_ids):
                with self.assertRaises(ValidationError):
                    AgentChatRequest(
                        message="query",
                        knowledge_base_ids=knowledge_base_ids,
                        conversation_id=1,
                    )


if __name__ == "__main__":
    unittest.main()
