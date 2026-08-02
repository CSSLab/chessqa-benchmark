import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))

from rank_stability import source_disjoint_keys


class RankStabilityTests(unittest.TestCase):
    def test_source_disjoint_filter_uses_fen_and_puzzle_id(self):
        rows = {
            "a": {
                "task_id": "a",
                "task_category": "Structural",
                "input": "8/8/8/8/8/8/4K3/6k1 w - - 0 1",
                "metadata": {"puzzle_id": "shared"},
            },
            "b": {
                "task_id": "b",
                "task_category": "Motifs",
                "input": "8/8/8/8/8/8/4K3/6k1 w - - 7 20",
                "metadata": {"puzzle_id": "other"},
            },
            "c": {
                "task_id": "c",
                "task_category": "Short Tactics",
                "input": "8/8/8/8/8/8/3K4/6k1 w - - 0 1",
                "metadata": {"puzzle_id": "shared"},
            },
            "d": {
                "task_id": "d",
                "task_category": "Position Judgment",
                "input": "8/8/8/8/8/8/2K5/6k1 w - - 0 1",
                "metadata": {},
            },
        }
        self.assertEqual(source_disjoint_keys(rows), {"d"})


if __name__ == "__main__":
    unittest.main()
