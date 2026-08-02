#!/usr/bin/env python3
"""Generate Semantic shortcut-control task files without changing labels."""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path


CONTROLS = ("comments_only", "move_only", "fen_only", "shuffled_fen", "shuffled_move")


def load(path: Path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def derangement(values, rng):
    if len(values) < 2:
        raise ValueError("Need at least two unique values for a shuffled control")
    shifted = list(values)
    offset = rng.randrange(1, len(values))
    shifted = shifted[offset:] + shifted[:offset]
    return dict(zip(values, shifted))


def transform(item, control, fen_map, move_map):
    row = json.loads(json.dumps(item))
    meta = row.get("metadata") or {}
    old_fen = str(meta.get("fen_before") or row.get("input", ""))
    old_move = str(meta.get("move_uci", ""))
    question = str(row["question"])
    if control in ("comments_only", "move_only"):
        question = re.sub(r"^You are given a chess position in FEN:.*?\n", "", question, count=1)
    if control in ("comments_only", "fen_only"):
        question = re.sub(r"^A player makes the move:.*?\n", "", question, count=1, flags=re.MULTILINE)
    if control == "shuffled_fen":
        new_fen = fen_map[old_fen]
        question = question.replace(old_fen, new_fen, 1)
        row["input"] = new_fen
        meta["control_fen"] = new_fen
    if control == "shuffled_move":
        new_move = move_map[old_move]
        question = question.replace(old_move, new_move, 1)
        meta["control_move_uci"] = new_move
    row["question"] = question
    row["task_id"] = f"{row['task_id']}__{control}"
    row["task_type"] = f"{row['task_type']}__{control}"
    meta["semantic_control"] = control
    row["metadata"] = meta
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    rows = load(args.input)
    rng = random.Random(args.seed)
    fens = sorted({str((row.get("metadata") or {}).get("fen_before") or row.get("input", "")) for row in rows})
    moves = sorted({str((row.get("metadata") or {}).get("move_uci", "")) for row in rows})
    fen_map, move_map = derangement(fens, rng), derangement(moves, rng)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for control in CONTROLS:
        path = args.output_dir / f"semantic_{control}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(transform(row, control, fen_map, move_map), ensure_ascii=False) + "\n")
        print(path)


if __name__ == "__main__":
    main()
