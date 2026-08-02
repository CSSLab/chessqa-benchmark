#!/usr/bin/env python3
"""Audit ChessQA source overlap and frozen-label risk cases."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import chess


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_benchmark(root: Path) -> Dict[str, list[dict]]:
    datasets = {path.stem: load_jsonl(path) for path in sorted(root.glob("*.jsonl"))}
    if not datasets:
        raise ValueError(f"No JSONL benchmark files under {root}")
    return datasets


def metadata(item: Mapping[str, Any]) -> Dict[str, Any]:
    value = item.get("metadata") or item.get("metadata_json") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {}
    return dict(value) if isinstance(value, Mapping) else {}


def fen4(value: str) -> str:
    return " ".join((value or "").split("|", 1)[0].strip().split()[:4])


def semantic_triple(item: Mapping[str, Any]) -> tuple[str, str, str]:
    meta = metadata(item)
    return (
        fen4(str(meta.get("fen_before") or item.get("input", ""))),
        str(meta.get("move_uci", "")),
        str(meta.get("correct_text", "")),
    )


def mating_moves(item: Mapping[str, Any]) -> list[str]:
    board = chess.Board(fen4(str(item.get("input", ""))))
    answers = []
    for move in list(board.legal_moves):
        board.push(move)
        if board.is_checkmate():
            answers.append(move.uci())
        board.pop()
    return sorted(answers)


def build_summary(datasets: Mapping[str, Sequence[dict]]) -> dict:
    rows = [(name, item) for name, items in datasets.items() for item in items]
    by_file = {}
    fen_sets = {}
    puzzle_sets = {}
    for name, items in datasets.items():
        fens = {fen4(str(item.get("input", ""))) for item in items}
        puzzle_ids = {
            str(metadata(item).get("puzzle_id"))
            for item in items
            if metadata(item).get("puzzle_id")
        }
        fen_sets[name] = fens
        puzzle_sets[name] = puzzle_ids
        by_file[name] = {
            "items": len(items),
            "unique_fen4": len(fens),
            "puzzle_ids": sum(bool(metadata(item).get("puzzle_id")) for item in items),
            "unique_puzzle_ids": len(puzzle_ids),
        }

    pairwise = []
    for left, right in itertools.combinations(sorted(datasets), 2):
        fen_overlap = len(fen_sets[left] & fen_sets[right])
        puzzle_overlap = len(puzzle_sets[left] & puzzle_sets[right])
        if fen_overlap or puzzle_overlap:
            pairwise.append({
                "left": left,
                "right": right,
                "fen4_overlap": fen_overlap,
                "puzzle_id_overlap": puzzle_overlap,
            })

    semantic = datasets.get("semantic", [])
    triples = {semantic_triple(item) for item in semantic}
    variants = defaultdict(set)
    for item in semantic:
        variants[str(item.get("task_type"))].add(semantic_triple(item))

    tactics = datasets.get("short_tactics", [])
    mate_items = [
        item for item in tactics if "mateIn1" in (metadata(item).get("themes") or [])
    ]
    mate_counts = [(item["task_id"], mating_moves(item)) for item in mate_items]

    pinned = defaultdict(lambda: {"pinned": 0, "pinned_sliders": 0})
    for item in datasets.get("structural", []):
        if item.get("task_type") not in {
            "structural_capture_squares",
            "structural_control_squares",
            "structural_protect_squares",
        }:
            continue
        meta = metadata(item)
        board = chess.Board(str(item["input"]))
        square = chess.parse_square(str(meta["target_square"]))
        piece = board.piece_at(square)
        if piece and board.is_pinned(piece.color, square):
            pinned[item["task_type"]]["pinned"] += 1
            if piece.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN):
                pinned[item["task_type"]]["pinned_sliders"] += 1

    all_fens = Counter(fen4(str(item.get("input", ""))) for _, item in rows)
    all_puzzles = Counter(
        str(metadata(item).get("puzzle_id"))
        for _, item in rows
        if metadata(item).get("puzzle_id")
    )
    task_groups = defaultdict(list)
    for _, item in rows:
        task_groups[str(item.get("task_type", "unknown"))].append(item)
    by_task = {}
    for task_type, items in sorted(task_groups.items()):
        puzzle_ids = [
            str(metadata(item).get("puzzle_id"))
            for item in items
            if metadata(item).get("puzzle_id")
        ]
        by_task[task_type] = {
            "items": len(items),
            "unique_fen4": len({fen4(str(item.get("input", ""))) for item in items}),
            "puzzle_ids": len(puzzle_ids),
            "unique_puzzle_ids": len(set(puzzle_ids)),
        }

    return {
        "items": len(rows),
        "unique_fen4": len(all_fens),
        "fen4_multiplicity": dict(sorted(Counter(all_fens.values()).items())),
        "unique_puzzle_ids": len(all_puzzles),
        "puzzle_id_multiplicity": dict(sorted(Counter(all_puzzles.values()).items())),
        "by_file": by_file,
        "by_task": by_task,
        "pairwise_overlap": pairwise,
        "semantic": {
            "items": len(semantic),
            "unique_state_move_comment_triples": len(triples),
            "unique_state_move_pairs": len({triple[:2] for triple in triples}),
            "unique_triples_by_variant": {
                key: len(value) for key, value in sorted(variants.items())
            },
        },
        "mate_in_one": {
            "tagged_items": len(mate_items),
            "items_with_multiple_legal_mates": sum(
                len(moves) > 1 for _, moves in mate_counts
            ),
            "multiple_answer_items": [
                {"task_id": task_id, "legal_mates": moves}
                for task_id, moves in mate_counts
                if len(moves) > 1
            ],
        },
        "pinned_structural_queries": dict(sorted(pinned.items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("summary",))
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = build_summary(load_benchmark(args.benchmark))
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
