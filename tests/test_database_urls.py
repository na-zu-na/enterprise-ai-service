import unittest

from db.urls import psycopg_connection_url, sqlalchemy_database_url


class DatabaseUrlTest(unittest.TestCase):
    def test_plain_postgresql_url_uses_psycopg3_in_sqlalchemy(self) -> None:
        self.assertEqual(
            sqlalchemy_database_url("postgresql://user:pass@db/app"),
            "postgresql+psycopg://user:pass@db/app",
        )

    def test_sqlalchemy_url_is_accepted_by_direct_psycopg(self) -> None:
        self.assertEqual(
            psycopg_connection_url("postgresql+psycopg://user:pass@db/app"),
            "postgresql://user:pass@db/app",
        )

    def test_missing_url_has_a_clear_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "DATABASE_URL"):
            sqlalchemy_database_url(None)


if __name__ == "__main__":
    unittest.main()
