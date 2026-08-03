import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class CompleteReleaseTests(unittest.TestCase):
    def test_all_five_categories_are_runnable(self):
        expected = {
            "structural.jsonl": 1100,
            "motifs.jsonl": 600,
            "short_tactics.jsonl": 900,
            "position_judgement.jsonl": 500,
            "semantic.jsonl": 400,
        }
        total = 0
        for filename, count in expected.items():
            path = ROOT / "benchmark" / filename
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
            self.assertEqual(len(rows), count, filename)
            total += len(rows)
        self.assertEqual(total, 3500)

    def test_position_judgment_rows_use_canonical_fields(self):
        path = ROOT / "benchmark" / "position_judgement.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        self.assertEqual(len(rows), 500)
        for row in rows:
            self.assertTrue(row["task_id"].startswith("position_judgment_"))
            self.assertTrue(row["task_type"].startswith("position_judgment_"))
            self.assertEqual(row["task_category"], "Position Judgment")
            question = row["question"]
            self.assertIn("Analyze several moves ahead", question)
            self.assertEqual(question.count("Analyze step by step and explain your reasoning."), 1)
            self.assertIn("Example final answer:", question)


if __name__ == "__main__":
    unittest.main()
