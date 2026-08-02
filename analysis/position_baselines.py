#!/usr/bin/env python3
"""Chess-specific baselines for ChessQA Position Judgment.

The script never changes benchmark labels.  It evaluates material-only, a
small hand-written static evaluator, a cross-validated linear piece-square
model, and (optionally) a fixed-node Stockfish run against the five released
answer buckets.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import chess
import chess.engine
import numpy as np


OPTIONS = np.asarray([-400, -200, 0, 200, 400], dtype=float)
VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}


def load_items(path: Path) -> List[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def snap(score: float) -> int:
    return int(OPTIONS[np.argmin(np.abs(OPTIONS - score))])


def material_score(board: chess.Board) -> float:
    return float(sum(
        VALUES[piece.piece_type] * (1 if piece.color == chess.WHITE else -1)
        for piece in board.piece_map().values()
    ))


def _mobility(board: chess.Board, color: chess.Color) -> int:
    copy = board.copy(stack=False)
    copy.turn = color
    return copy.legal_moves.count()


def static_score(board: chess.Board) -> float:
    """Tiny transparent evaluator: material, mobility, center, bishop pair."""
    score = material_score(board)
    score += 4.0 * (_mobility(board, chess.WHITE) - _mobility(board, chess.BLACK))
    center = (chess.D4, chess.E4, chess.D5, chess.E5)
    for square in center:
        piece = board.piece_at(square)
        if piece:
            score += 18.0 if piece.color == chess.WHITE else -18.0
    for color, sign in ((chess.WHITE, 1), (chess.BLACK, -1)):
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            score += sign * 30.0
    return score


def piece_square_features(board: chess.Board) -> np.ndarray:
    """Signed 6x64 occupancy plus material/mobility summary features."""
    features = np.zeros(6 * 64 + 4, dtype=float)
    for square, piece in board.piece_map().items():
        sign = 1.0 if piece.color == chess.WHITE else -1.0
        features[(piece.piece_type - 1) * 64 + square] = sign
    features[-4] = 1.0
    features[-3] = material_score(board) / 100.0
    features[-2] = _mobility(board, chess.WHITE) / 40.0
    features[-1] = _mobility(board, chess.BLACK) / 40.0
    return features


def ridge_fit(x: np.ndarray, y: np.ndarray, penalty: float = 10.0) -> np.ndarray:
    identity = np.eye(x.shape[1])
    identity[-4, -4] = 0.0  # do not penalize intercept
    return np.linalg.solve(x.T @ x + penalty * identity, x.T @ y)


def cross_validated_linear(items: List[dict], folds: int, seed: int) -> List[int]:
    x = np.stack([piece_square_features(chess.Board(item["input"])) for item in items])
    y = np.asarray([float((item.get("metadata") or {})["actual_evaluation"]) for item in items])
    rng = np.random.default_rng(seed)
    assignment = np.arange(len(items)) % folds
    rng.shuffle(assignment)
    predictions = np.zeros(len(items), dtype=float)
    for fold in range(folds):
        train = assignment != fold
        test = ~train
        weights = ridge_fit(x[train], y[train])
        predictions[test] = x[test] @ weights
    return [snap(score) for score in predictions]


def fixed_engine_predictions(
    items: List[dict], path: str, nodes: int, threads: int, hash_mb: int
) -> Tuple[List[int], Dict[str, Any]]:
    engine = chess.engine.SimpleEngine.popen_uci(path)
    predictions = []
    try:
        engine.configure({"Threads": threads, "Hash": hash_mb})
        engine_info = dict(engine.id)
        for item in items:
            info = engine.analyse(chess.Board(item["input"]), chess.engine.Limit(nodes=nodes))
            score = info["score"].pov(chess.WHITE).score(mate_score=100000)
            predictions.append(snap(float(score)))
    finally:
        engine.quit()
    return predictions, {
        "binary": str(Path(path).resolve()),
        "engine": engine_info,
        "nodes_per_position": nodes,
        "threads": threads,
        "hash_mb": hash_mb,
    }


def summarize(items: List[dict], predictions: Iterable[int]) -> Dict[str, object]:
    by_type = defaultdict(list)
    confusion = defaultdict(lambda: defaultdict(int))
    correct = []
    disagreements = []
    for item, prediction in zip(items, predictions):
        gold = int(item["correct_answer"])
        hit = prediction == gold
        correct.append(hit)
        by_type[item["task_type"]].append(hit)
        confusion[str(gold)][str(prediction)] += 1
        if not hit:
            disagreements.append({
                "task_id": item["task_id"],
                "task_type": item["task_type"],
                "label": gold,
                "prediction": prediction,
            })
    return {
        "accuracy": float(np.mean(correct)),
        "correct": int(sum(correct)),
        "total": len(correct),
        "different": len(correct) - int(sum(correct)),
        "by_task": {key: float(np.mean(values)) for key, values in sorted(by_type.items())},
        "confusion": {
            gold: dict(sorted(predictions.items(), key=lambda pair: int(pair[0])))
            for gold, predictions in sorted(confusion.items(), key=lambda pair: int(pair[0]))
        },
        "disagreements": disagreements,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--stockfish")
    parser.add_argument("--nodes", type=int, default=5000)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=128)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    items = load_items(args.dataset)
    predictions = {
        "material": [snap(material_score(chess.Board(item["input"]))) for item in items],
        "hand_static": [snap(static_score(chess.Board(item["input"]))) for item in items],
        "linear_piece_square_cv": cross_validated_linear(items, args.folds, args.seed),
    }
    engine_run = None
    if args.stockfish:
        engine_predictions, engine_run = fixed_engine_predictions(
            items, args.stockfish, args.nodes, args.threads, args.hash_mb
        )
        predictions[f"stockfish_{args.nodes}_nodes"] = engine_predictions
    report = {
        "dataset": str(args.dataset),
        "n_items": len(items),
        "folds": args.folds,
        "seed": args.seed,
        "engine_run": engine_run,
        "baselines": {
            name: summarize(items, values) for name, values in predictions.items()
        },
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
