#!/usr/bin/env python3
"""Alternative ChessQA aggregations and source-cluster bootstrap intervals."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np


def canonical_category(value: Any) -> str:
    key = str(value or "unknown").strip().lower().replace("_", " ")
    return {
        "long term": "Position Judgment",
        "position judgement": "Position Judgment",
        "position judgment": "Position Judgment",
        "motif": "Motifs",
        "motifs": "Motifs",
        "semantic": "Semantic",
        "structural": "Structural",
        "tactic": "Short Tactics",
        "short tactics": "Short Tactics",
    }.get(key, str(value or "unknown"))


def canonical_task_type(value: Any) -> str:
    task_type = str(value or "unknown")
    for old, new in (
        ("position_judgement_", "position_judgment_"),
        ("judgement_", "position_judgement_"),
        ("motif_", "motifs_"),
        ("tactic_", "short_tactics_"),
    ):
        if task_type.startswith(old):
            normalized = new + task_type.removeprefix(old)
            return normalized.replace("position_judgement_", "position_judgment_", 1)
    return task_type


def canonical_item_key(row: Mapping[str, Any]) -> str:
    raw_task = str(row.get("task_type", "unknown"))
    task = canonical_task_type(raw_task)
    suffix = str(row["task_id"]).removeprefix(raw_task + "_")
    return f"{task}_{suffix}"


def load(path: Path) -> Dict[str, dict]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    loaded = {canonical_item_key(row): row for row in rows}
    if len(loaded) != len(rows):
        raise ValueError(f"Duplicate canonical task IDs in {path}")
    return loaded


def metadata(item: Mapping[str, Any]) -> dict:
    value = item.get("metadata") or item.get("metadata_json") or {}
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {}
    return value if isinstance(value, dict) else {}


def fen4(value: Any) -> str:
    fields = str(value or "").split("|", 1)[0].strip().split()
    return " ".join(fields[:4]) if len(fields) >= 4 else ""


def source_cluster(item: Mapping[str, Any]) -> str:
    """Cluster repeated views of one underlying source example together."""
    meta = metadata(item)
    if meta.get("base_example_id"):
        return "semantic:" + str(meta["base_example_id"])
    if meta.get("puzzle_id"):
        return "puzzle:" + str(meta["puzzle_id"])
    if meta.get("game_id"):
        return "game:" + str(meta["game_id"])
    if meta.get("position_id"):
        return "position:" + str(meta["position_id"])
    if canonical_category(item.get("task_category")) == "Semantic":
        raw = "|".join((str(meta.get("fen_before", item.get("input", ""))), str(meta.get("move_uci", ""))))
        return "semantic:" + hashlib.sha256(raw.encode()).hexdigest()[:16]
    return "item:" + str(item["task_id"])


def metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, float]:
    tasks, categories, sources = defaultdict(list), defaultdict(list), defaultdict(list)
    values = []
    for row in rows:
        value = bool((row.get("inference") or {}).get("is_correct", row.get("is_correct", False)))
        values.append(value)
        tasks[canonical_task_type(row.get("task_type"))].append(value)
        categories[canonical_category(row.get("task_category"))].append(value)
        sources[source_cluster(row)].append(value)
    return {
        "item_micro": mean(values),
        "task_macro": mean(mean(group) for group in tasks.values()),
        "category_macro": mean(mean(group) for group in categories.values()),
        "source_balanced": mean(mean(group) for group in sources.values()),
    }


def source_disjoint_keys(rows: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """Keep items whose FEN and PuzzleId occur in only one category."""
    fen_categories: Dict[str, set[str]] = defaultdict(set)
    puzzle_categories: Dict[str, set[str]] = defaultdict(set)
    for row in rows.values():
        category = canonical_category(row.get("task_category"))
        position = fen4(row.get("input"))
        puzzle_id = str(metadata(row).get("puzzle_id", ""))
        if position:
            fen_categories[position].add(category)
        if puzzle_id:
            puzzle_categories[puzzle_id].add(category)

    kept = set()
    for key, row in rows.items():
        position = fen4(row.get("input"))
        puzzle_id = str(metadata(row).get("puzzle_id", ""))
        fen_ok = not position or len(fen_categories[position]) == 1
        puzzle_ok = not puzzle_id or len(puzzle_categories[puzzle_id]) == 1
        if fen_ok and puzzle_ok:
            kept.add(key)
    return kept


def ranks(models: Sequence[str], values: Mapping[str, float]) -> Dict[str, int]:
    ordered = sorted(models, key=lambda model: (-values[model], model))
    return {model: index + 1 for index, model in enumerate(ordered)}


def spearman_rank_correlation(left: Sequence[int], right: Sequence[int]) -> float:
    """Pearson correlation of two already-ranked vectors (no SciPy dependency)."""
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Rank vectors must have the same length and at least two entries")
    return float(np.corrcoef(np.asarray(left, dtype=float), np.asarray(right, dtype=float))[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    loaded = {path.stem: load(path) for path in args.results}
    common = set.intersection(*(set(rows) for rows in loaded.values()))
    if not common:
        raise ValueError("Result files have no shared task_id values")
    loaded = {model: {key: rows[key] for key in common} for model, rows in loaded.items()}
    models = sorted(loaded)
    point = {model: metrics(list(rows.values())) for model, rows in loaded.items()}
    reference = next(iter(loaded.values()))
    disjoint_keys = source_disjoint_keys(reference)
    for model, rows in loaded.items():
        point[model]["source_disjoint_item_micro"] = metrics(
            [rows[key] for key in disjoint_keys]
        )["item_micro"]

    point_ranks = {}
    for metric_name in (
        "item_micro",
        "task_macro",
        "category_macro",
        "source_balanced",
        "source_disjoint_item_micro",
    ):
        values = {model: point[model][metric_name] for model in models}
        point_ranks[metric_name] = ranks(models, values)
    rank_stability = {}
    for left, right in (
        ("item_micro", "task_macro"),
        ("item_micro", "category_macro"),
        ("task_macro", "category_macro"),
        ("item_micro", "source_balanced"),
        ("item_micro", "source_disjoint_item_micro"),
    ):
        rho = spearman_rank_correlation(
            [point_ranks[left][model] for model in models],
            [point_ranks[right][model] for model in models],
        )
        rank_stability[f"{left}--{right}"] = float(rho)

    by_cluster = defaultdict(list)
    for task_id, row in reference.items():
        by_cluster[source_cluster(row)].append(task_id)
    cluster_ids = sorted(by_cluster)
    rng = np.random.default_rng(args.seed)
    draws = {model: defaultdict(list) for model in models}
    for _ in range(args.bootstrap):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        sampled_ids = [task_id for cluster in sampled for task_id in by_cluster[cluster]]
        for model in models:
            result = metrics([loaded[model][task_id] for task_id in sampled_ids])
            for metric_name in ("item_micro", "task_macro", "category_macro"):
                value = result[metric_name]
                draws[model][metric_name].append(value)
            cluster_values = [
                mean(
                    bool((loaded[model][task_id].get("inference") or {}).get(
                        "is_correct", loaded[model][task_id].get("is_correct", False)
                    ))
                    for task_id in by_cluster[cluster]
                )
                for cluster in sampled
            ]
            draws[model]["source_balanced"].append(mean(cluster_values))

    intervals = {}
    for model in models:
        intervals[model] = {}
        for metric_name, values in draws[model].items():
            low, high = np.quantile(values, [0.025, 0.975])
            intervals[model][metric_name] = {"low": float(low), "high": float(high)}

    report = {
        "n_models": len(models),
        "n_shared_items": len(common),
        "n_source_clusters": len(cluster_ids),
        "source_disjoint_items": len(disjoint_keys),
        "source_disjoint_items_by_category": {
            category: sum(
                canonical_category(reference[key].get("task_category")) == category
                for key in disjoint_keys
            )
            for category in sorted(
                {canonical_category(row.get("task_category")) for row in reference.values()}
            )
        },
        "point_estimates": point,
        "ranks": point_ranks,
        "rank_spearman": rank_stability,
        "cluster_bootstrap_replicates": args.bootstrap,
        "cluster_bootstrap_95_percent_intervals": intervals,
        "scope_note": "Intervals resample source clusters; they do not include endpoint run-to-run stochasticity.",
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
