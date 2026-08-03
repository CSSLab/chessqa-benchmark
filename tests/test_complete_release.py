import json
import re
import unittest
from collections import Counter
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
            metadata = row["metadata"]
            self.assertEqual(metadata["label_provenance"], "stockfish_fixed_nodes")
            self.assertEqual(metadata["engine_name"], "Stockfish 16")
            self.assertEqual(metadata["engine_nodes"], 100000)
            self.assertEqual(metadata["engine_threads"], 1)
            self.assertEqual(metadata["engine_hash_mb"], 128)
            self.assertRegex(metadata["engine_binary_sha256"], r"^[0-9a-f]{64}$")
            options = [-400, -200, 0, 200, 400]
            expected = min(options, key=lambda value: abs(value - metadata["actual_evaluation"]))
            self.assertEqual(row["correct_answer"], str(expected))

    def test_task_prompts_match_operational_targets(self):
        def rows(filename):
            path = ROOT / "benchmark" / filename
            return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

        semantic = rows("semantic.jsonl")
        base_counts = Counter(row["metadata"]["base_example_id"] for row in semantic)
        self.assertEqual(len(base_counts), 100)
        self.assertEqual(set(base_counts.values()), {4})
        for row in semantic:
            self.assertIn("source annotated-game record", row["question"])

        short = rows("short_tactics.jsonl")
        uci = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$")
        for row in short:
            self.assertIn("intended first solution move", row["question"])
            self.assertEqual(row["format_examples"], ["e2e4", "c2b1q"])
            self.assertTrue(all(uci.fullmatch(example) for example in row["format_examples"]))

        for row in rows("position_judgement.jsonl"):
            self.assertIn("reference Stockfish-family evaluation", row["question"])

        motifs = rows("motifs.jsonl")
        skewers = [row for row in motifs if row["task_type"] == "motifs_skewer"]
        self.assertEqual(len(skewers), 100)
        for row in skewers:
            self.assertIn("detector-defined skewer", row["question"])

        for filename in (
            "structural.jsonl",
            "motifs.jsonl",
            "short_tactics.jsonl",
            "position_judgement.jsonl",
            "semantic.jsonl",
        ):
            for row in rows(filename):
                self.assertEqual(
                    row["question"].count("CONTEXT_PLACEHOLDER"),
                    1,
                    row["task_id"],
                )
                self.assertEqual(
                    row["question"].count("FORMAT_EXAMPLE_PLACEHOLDER"),
                    1,
                    row["task_id"],
                )
                self.assertEqual(
                    row["question"].count("Analyze step by step and explain your reasoning."),
                    1,
                    row["task_id"],
                )


if __name__ == "__main__":
    unittest.main()
