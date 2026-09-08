import unittest
from pathlib import Path

from evals.retrieval_eval import DEFAULT_DATASET, load_dataset


class EvalDatasetTest(unittest.TestCase):
    def test_default_dataset_is_valid_and_has_expected_size(self):
        cases = load_dataset(DEFAULT_DATASET)
        self.assertGreaterEqual(len(cases), 20)
        self.assertLessEqual(len(cases), 50)
        self.assertEqual(len(cases), len({case.id for case in cases}))

    def test_missing_dataset_fails(self):
        with self.assertRaises(FileNotFoundError):
            load_dataset(Path("does-not-exist.json"))


if __name__ == "__main__":
    unittest.main()

