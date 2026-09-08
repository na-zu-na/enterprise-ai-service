import os
import unittest

from clients.spring_client import get_spring_client
from core.config import Settings


@unittest.skipUnless(
    os.getenv("RUN_SPRING_INTEGRATION_TESTS", "").lower()
    in {"1", "true", "yes", "on"},
    "set RUN_SPRING_INTEGRATION_TESTS=true to run Spring integration tests",
)
class SpringIntegrationTest(unittest.TestCase):
    def test_query_tasks(self) -> None:
        if not Settings.TEST_ACCESS_TOKEN:
            self.fail("缺少 TEST_ACCESS_TOKEN 环境变量")

        response = get_spring_client().get(
            "/api/tasks",
            access_token=Settings.TEST_ACCESS_TOKEN,
        )

        self.assertLess(response.status_code, 400, response.text)


if __name__ == "__main__":
    unittest.main()
