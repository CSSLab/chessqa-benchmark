#!/usr/bin/env python3
"""Create a text-free Semantic manifest from authorized local task data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def manifest_row(row: dict) -> dict:
    metadata = row.get("metadata") or {}
    fen = str(metadata.get("fen_before") or row["input"])
    move = str(metadata["move_uci"])
    comment = str(metadata["correct_text"])
    return {
        "task_id": row["task_id"],
        "task_type": row["task_type"],
        "task_category": "Semantic",
        "fen_before": fen,
        "move_uci": move,
        "variant": str(metadata["variant"]),
        "base_example_id": digest("\0".join((fen, move, comment)))[:20],
        "correct_answer": row["correct_answer"],
        "source_comment_sha256": digest(comment),
        "option_sha256": [digest(str(option)) for option in metadata["options"]],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.source.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(manifest_row(row), sort_keys=True) + "\n")
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
