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


if __name__ == "__main__":
    unittest.main()
