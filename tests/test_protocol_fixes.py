import importlib.util
import json
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dataset"))
sys.path.insert(0, str(ROOT / "eval"))

from chessqa_protocol import accepted_answers, format_task_prompt, piece_context, score_response


def load_structural_module():
    path = ROOT / "dataset" / "01_structural.py"
    spec = importlib.util.spec_from_file_location("structural_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_motif_module():
    path = ROOT / "dataset" / "02_motifs.py"
    spec = importlib.util.spec_from_file_location("motif_generator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ProtocolFixTests(unittest.TestCase):
    def test_pinned_rook_retains_attacks(self):
        structural = load_structural_module()
        import chess

        board = chess.Board("8/8/4rkp1/1p2R2p/p1p1KP1P/P1P5/1P6/8 b - - 0 41")
        self.assertTrue(structural.is_pinned(board, chess.E5))
        self.assertEqual(
            sorted(structural.detect_capture_squares(board, "e5")),
            ["b5", "e6", "h5"],
        )

    def test_all_mate_in_one_moves_are_accepted(self):
        rows = [json.loads(line) for line in (ROOT / "benchmark" / "short_tactics.jsonl").read_text().splitlines()]
        task = next(row for row in rows if row["task_id"] == "short_tactics_theme_mateIn1_0004")
        self.assertEqual(accepted_answers(task), ["d5a2", "e2a2"])
        for answer in ("d5a2", "e2a2"):
            scored = score_response(task, f"FINAL ANSWER: {answer}", {}, 8192)
            self.assertTrue(scored["is_correct"])

    def test_motif_prompt_and_canonicalization(self):
        rows = [json.loads(line) for line in (ROOT / "benchmark" / "motifs.jsonl").read_text().splitlines()]
        by_id = {row["task_id"]: row for row in rows}

        pin_prompt = format_task_prompt(by_id["motifs_pin_0076"])
        self.assertIn("Identify all absolute pins", pin_prompt)

        double = by_id["motifs_double_check_0047"]
        self.assertEqual(accepted_answers(double), ["f2f1b, f2f1q"])
        self.assertTrue(score_response(double, "FINAL ANSWER: f2f1b, f2f1q", {}, 8192)["is_correct"])
        self.assertFalse(score_response(double, "FINAL ANSWER: f2f1", {}, 8192)["is_correct"])

        battery = by_id["motifs_battery_0001"]
        canonical = "a1>e1, c8>d8>f8, d3>e2, d8>e7, e1>e2"
        self.assertTrue(score_response(battery, f"FINAL ANSWER: {canonical}", {}, 8192)["is_correct"])

        motif = load_motif_module()
        import chess

        generated = motif.generate_double_check_task(
            chess.Board(double["input"]), {"double_check": 0}, "test"
        )
        self.assertEqual(generated.correct_answer, "f2f1b, f2f1q")

    def test_piece_context_does_not_leak_legal_moves(self):
        context = piece_context("8/8/8/8/8/8/4K3/6k1 w - - 0 1")
        self.assertIn("Piece arrangement", context)
        self.assertNotIn("Legal moves", context)

    def test_frozen_pinned_prompt_is_repaired_at_render_time(self):
        import chess

        rows = [json.loads(line) for line in (ROOT / "benchmark" / "structural.jsonl").read_text().splitlines()]
        task = next(
            row for row in rows
            if row["task_type"].endswith(("capture_squares", "control_squares", "protect_squares"))
            and (lambda board, square: board.is_pinned(board.color_at(square), square))(
                chess.Board(row["input"]), chess.parse_square(row["metadata"]["target_square"])
            )
        )
        prompt = format_task_prompt(task)
        self.assertIn("retain attacks by a pinned piece", prompt)
        self.assertNotIn("Exclude", prompt)

    def test_content_and_format_are_separate(self):
        task = {
            "question": "FINAL ANSWER: <answer>",
            "input": "8/8/8/8/8/8/4K3/6k1 w - - 0 1",
            "correct_answer": "A",
            "answer_type": "single",
        }
        scored = score_response(task, "A", {}, 100)
        self.assertFalse(scored["format_compliant"])
        self.assertFalse(scored["is_correct"])
        self.assertTrue(scored["content_correct"])

        trailing = score_response(task, "FINAL ANSWER: A\nextra prose", {}, 100)
        self.assertFalse(trailing["format_compliant"])
        self.assertTrue(trailing["content_correct"])

        extra_json = score_response(task, '{"answer": "A", "note": "extra"}', {}, 100, "json")
        self.assertFalse(extra_json["format_compliant"])
        self.assertTrue(extra_json["content_correct"])

    def test_ascii_representation_preserves_state(self):
        task = {
            "question": "You are given a chess position in FEN: 8/8/8/8/8/8/4K3/6k1 w - - 0 1",
            "input": "8/8/8/8/8/8/4K3/6k1 w - - 0 1",
        }
        prompt = format_task_prompt(task, representation="ascii")
        self.assertIn("a b c d e f g h", prompt)
        self.assertIn("White to move", prompt)


if __name__ == "__main__":
    unittest.main()
