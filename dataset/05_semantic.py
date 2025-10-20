import argparse
import json
import random
import re
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import tqdm
from sentence_transformers import SentenceTransformer

from utils import (
    seed_everything,
    save_tasks,
    construct_prompt,
    ChessQuestionAnsweringTask,
    FORMAT_EXAMPLES_MCQ
)


def normalize_comment_text(text: str) -> str:
    """Clean comment text."""
    if not text:
        return ""

    # Replace non-breaking space with regular space
    out = text.replace("\u00A0", " ")

    # Remove invisible Unicode characters
    out = re.sub(r"[\u200B-\u200F\u202A-\u202E]", "", out)

    # Replace chess figurines only if they exist
    figurine_replacements = {
        "\ue024": "K", "\ue025": "Q", "\ue026": "R",
        "\ue027": "B", "\ue028": "N", "\ue029": "P"
    }

    for figurine, letter in figurine_replacements.items():
        if figurine in out:
            out = out.replace(figurine, letter)

    return out.strip()


def _stage_bucket(move_number: int, opening_threshold: int = 12, middlegame_threshold: int = 30) -> str:
    """Get game phase bucket."""
    try:
        n = int(move_number)
    except Exception:
        n = 0
    if n <= opening_threshold:
        return "opening"
    if n <= middlegame_threshold:
        return "middlegame"
    return "endgame"


def _build_indices(items: List[Dict[str, Any]], cfg):
    """Build search indices."""
    by_keyword = {}
    by_piece = {}
    by_piece_stage = {}

    for i, it in enumerate(items):
        # Keywords
        kws = [str(k).strip().lower() for k in (it.get("keywords") or []) if str(k).strip()]
        for kw in kws:
            by_keyword.setdefault(kw, []).append(i)

        # Piece
        piece = str(it.get("move_piece") or "").strip().lower() or "unknown"
        by_piece.setdefault(piece, []).append(i)

        # Piece + stage
        buck = _stage_bucket(it.get("move_number", 0), cfg.opening_threshold, cfg.middlegame_threshold)
        by_piece_stage.setdefault((piece, buck), []).append(i)

    return by_keyword, by_piece, by_piece_stage


def _pick_random(pool: List[int], exclude: set, k: int) -> List[int]:
    """Pick k random items from pool excluding certain indices."""
    choices = [idx for idx in pool if idx not in exclude]
    if len(choices) <= k:
        random.shuffle(choices)
        return choices
    return random.sample(choices, k)


def _get_neighbors(idx: int, items: List[Dict[str, Any]], indices: Dict, k: int, strategy: str, cfg) -> List[int]:
    """Get neighbors using different strategies."""
    if strategy == "keyword":
        kws = [str(x).strip().lower() for x in (items[idx].get("keywords") or []) if str(x).strip()]
        cand = set()
        for kw in kws:
            cand.update(indices.get(kw, []))
        cand.discard(idx)
        return random.sample(list(cand), min(k, len(cand)))

    elif strategy == "piece":
        piece = str(items[idx].get("move_piece") or "").strip().lower() or "unknown"
        cand = [j for j in indices.get(piece, []) if j != idx]
        return random.sample(cand, min(k, len(cand)))

    elif strategy == "piece_stage":
        piece = str(items[idx].get("move_piece") or "").strip().lower() or "unknown"
        buck = _stage_bucket(items[idx].get("move_number", 0), cfg.opening_threshold, cfg.middlegame_threshold)
        cand = [j for j in indices.get((piece, buck), []) if j != idx]
        return random.sample(cand, min(k, len(cand)))

    return []


def _ensure_unique_options(base_opts: List[str], need: int, all_items: List[Dict], exclude_indices: set) -> List[str]:
    """Ensure we have enough unique options."""
    options = list(dict.fromkeys([o for o in base_opts if o]))  # Remove duplicates and empty

    if len(options) < need:
        # Fill with random options
        available = [i for i in range(len(all_items)) if i not in exclude_indices]
        needed = need - len(options)
        random_indices = random.sample(available, min(needed, len(available)))

        for idx in random_indices:
            text = all_items[idx].get("_comment_text", "")
            if text and text not in options:
                options.append(text)
                if len(options) >= need:
                    break

    return options[:need]


def _compute_embeddings(items: List[Dict[str, Any]], cfg) -> np.ndarray:
    """Compute or load cached embeddings for comment texts."""
    cache_dir = Path(cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "comment_embeddings.npy"
    map_path = cache_dir / "comment_embeddings.index.json"

    texts = [normalize_comment_text(it.get("cleaned_comment") or it.get("comment", "")) for it in items]

    # Try loading cached embeddings
    if emb_path.exists() and map_path.exists():
        try:
            with open(map_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            if meta.get("count") == len(texts) and meta.get("model") == cfg.embed_model:
                arr = np.load(str(emb_path))
                if arr.shape[0] == len(texts):
                    print(f"Loaded cached embeddings from {emb_path}")
                    return arr.astype(np.float32, copy=False)
        except Exception as e:
            print(f"Failed to load cached embeddings: {e}")

    # Must compute embeddings
    print(f"Computing embeddings using {cfg.embed_model}...")
    model = SentenceTransformer(cfg.embed_model)
    embs = []
    for i in range(0, len(texts), max(1, cfg.embed_batch)):
        batch = texts[i:i + cfg.embed_batch]
        vecs = model.encode(
            batch,
            batch_size=min(cfg.embed_batch, cfg.embed_max_batch),
            show_progress_bar=True,
            normalize_embeddings=True
        )
        embs.append(np.asarray(vecs, dtype=np.float32))

    arr = np.vstack(embs).astype(np.float32)

    # Cache results
    print(f"Saving embeddings to {emb_path}")
    np.save(str(emb_path), arr)
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(texts), "model": cfg.embed_model}, f)

    return arr


def _get_semantic_neighbors(idx: int, embeddings: np.ndarray, cfg, k: int = None) -> List[int]:
    """Get semantic neighbors using cosine similarity."""
    if embeddings is None:
        return []

    if k is None:
        k = cfg.num_distractors

    v = embeddings[idx]
    sims = v @ embeddings.T
    sims[idx] = -np.inf  # Exclude self

    if k < len(sims):
        idxs = np.argpartition(sims, -k)[-k:]
        idxs = idxs[np.argsort(sims[idxs])[::-1]]
    else:
        idxs = np.argsort(sims)[::-1]

    return [int(j) for j in idxs[:k].tolist()]


def generate_comment_mcq_task(item: Dict[str, Any], variant: str, options: List[str], idx: int) -> ChessQuestionAnsweringTask:
    """Generate a single MCQ comment task."""

    # Build context information
    fen_before = item.get("fen_before", "")
    move_uci = item.get("move_uci", "")

    prefix = f"You are given a chess position in FEN: {fen_before}\n"
    prefix += f"A player makes the move: {move_uci}\n"

    # Shuffle options for fairness and find correct letter
    correct_text = item["_comment_text"]
    shuffled_options = options[:]
    random.shuffle(shuffled_options)

    # Find the correct letter after shuffling
    correct_index = shuffled_options.index(correct_text)
    correct_letter = chr(65 + correct_index)  # A, B, C, D

    # Create multiple choice format
    option_text = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(shuffled_options)])

    task_description = f"Select the commentary that best describes this position and move.\n\nOptions:\n{option_text}\n"
    suffix = "Example format: FORMAT_EXAMPLE_PLACEHOLDER"

    full_question = construct_prompt(prefix, task_description, suffix).replace("CONTEXT_PLACEHOLDER", "")

    return ChessQuestionAnsweringTask(
        task_id=f"semantic_{variant}_{idx:04d}",
        task_type=f"semantic_{variant}",
        task_category="Semantic",
        input=fen_before,
        question=full_question,
        format_examples=FORMAT_EXAMPLES_MCQ,
        correct_answer=correct_letter,
        answer_type="single",
        metadata={
            "variant": variant,
            "options": shuffled_options,
            "correct_text": correct_text,
            "fen_before": item.get("fen_before"),
            "fen_after": item.get("fen_after"),
            "move_uci": item.get("move_uci"),
            "move_number": item.get("move_number"),
            "side_to_move": item.get("side_to_move"),
        }
    )


def find_comment_tasks_by_variant(items: List[Dict[str, Any]], variant: str, cfg, embeddings: np.ndarray) -> List[ChessQuestionAnsweringTask]:
    """Generate comment MCQ tasks for a specific variant."""

    # Build indices for neighbor finding
    by_keyword, by_piece, by_piece_stage = _build_indices(items, cfg)

    found = []
    found_counter = 0
    universe = list(range(len(items)))

    # Sample indices if needed
    if cfg.sample_size and cfg.sample_size > 0:
        total = len(items)
        k = min(cfg.sample_size, total)
        base_indices = random.sample(list(range(total)), k) if cfg.random_sample else list(range(k))
    else:
        base_indices = list(range(len(items)))

    for idx in tqdm.tqdm(base_indices):
        if found_counter >= cfg.N_sample_mcq:
            break

        item = items[idx]
        correct_text = item.get("_comment_text", "")
        if not correct_text:
            continue

        exclude = {idx}

        # Build options based on variant strategy
        if variant == "easy_random":
            distractor_indices = _pick_random(universe, exclude, cfg.num_distractors)
        elif variant == "keyword":
            distractor_indices = _get_neighbors(idx, items, by_keyword, cfg.num_distractors, "keyword", cfg)
            if len(distractor_indices) < cfg.num_distractors:
                more = _pick_random(universe, exclude | set(distractor_indices), cfg.num_distractors - len(distractor_indices))
                distractor_indices.extend(more)
        elif variant == "piece_stage":
            distractor_indices = _get_neighbors(idx, items, by_piece_stage, cfg.num_distractors, "piece_stage", cfg)
            if len(distractor_indices) < cfg.num_distractors:
                more = _get_neighbors(idx, items, by_piece, cfg.num_distractors - len(distractor_indices), "piece", cfg)
                distractor_indices.extend(more)
            if len(distractor_indices) < cfg.num_distractors:
                more = _pick_random(universe, exclude | set(distractor_indices), cfg.num_distractors - len(distractor_indices))
                distractor_indices.extend(more)
        elif variant == "embedding":
            distractor_indices = _get_semantic_neighbors(idx, embeddings, cfg)
            # Ensure we have enough indices
            if len(distractor_indices) < cfg.num_distractors:
                more = _pick_random(universe, exclude | set(distractor_indices), cfg.num_distractors - len(distractor_indices))
                distractor_indices.extend(more)
        else:
            # Default to random
            distractor_indices = _pick_random(universe, exclude, cfg.num_distractors)

        # Build options list
        options = [correct_text]
        for d_idx in distractor_indices[:cfg.num_distractors]:
            distractor_text = items[d_idx].get("_comment_text", "")
            if distractor_text and distractor_text != correct_text:
                options.append(distractor_text)

        # Ensure we have exactly num_options unique options
        options = _ensure_unique_options(options, cfg.num_options, items, exclude | set(distractor_indices))

        if len(options) == cfg.num_options and correct_text in options:
            task = generate_comment_mcq_task(item, variant, options, found_counter)
            found.append(task)
            found_counter += 1

    return found


def parse_args():
    parser = argparse.ArgumentParser(description="Generate comment MCQ tasks with multiple distractor strategies")

    # Input/Output paths
    parser.add_argument("--input", type=str, default="../../data/mid/comment_dataset.final.json",
                       help="Path to input comment dataset")
    parser.add_argument("--output_root", type=str, default="../../data/benchmark",
                       help="Output directory for generated tasks")
    parser.add_argument("--cache_dir", type=str, default="/chess-llm-benchmark/data/mid",
                       help="Directory for embedding cache")

    # Task generation parameters
    parser.add_argument("--N_sample_mcq", type=int, default=100,
                       help="Number of MCQ tasks per variant")
    parser.add_argument("--sample_size", type=int, default=0,
                       help="Number of items to sample for task generation (0 = all)")
    parser.add_argument("--random_sample", action="store_true", default=True,
                       help="Use random sampling instead of sequential")

    # Game phase thresholds
    parser.add_argument("--opening_threshold", type=int, default=12,
                       help="Move number threshold for opening phase")
    parser.add_argument("--middlegame_threshold", type=int, default=30,
                       help="Move number threshold for middlegame phase")

    # Embedding parameters
    parser.add_argument("--embed_model", type=str, default="Qwen/Qwen3-Embedding-8B",
                       help="Sentence transformer model for embeddings")
    parser.add_argument("--embed_batch", type=int, default=256,
                       help="Batch size for embedding computation")
    parser.add_argument("--embed_max_batch", type=int, default=64,
                       help="Maximum batch size for model encoding")

    # MCQ parameters
    parser.add_argument("--num_options", type=int, default=4,
                       help="Number of options per MCQ task")
    parser.add_argument("--num_distractors", type=int, default=3,
                       help="Number of distractor options per task")

    # Output parameters
    parser.add_argument("--output_filename", type=str, default="semantic.jsonl",
                       help="Output filename for generated tasks")

    # General parameters
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")

    return parser.parse_args()


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)

    # Load data
    print(f"Loading comment data from {cfg.input}")
    with open(cfg.input, 'r', encoding='utf-8') as f:
        items = json.load(f)

    # Normalize comment texts
    for item in items:
        txt = item.get("cleaned_comment") or item.get("comment", "")
        item["_comment_text"] = normalize_comment_text(txt)

    print(f"Loaded {len(items)} comment items")

    # Always compute/load embeddings for semantic variant
    embeddings = _compute_embeddings(items, cfg)

    # Generate tasks for each variant
    variants = ["easy_random", "keyword", "piece_stage", "embedding"]
    all_tasks = []

    for variant in variants:
        print(f"Generating {variant} tasks...")
        found = find_comment_tasks_by_variant(items, variant, cfg, embeddings)
        all_tasks.extend(found)
        print(f"Generated {len(found)} {variant} tasks")

    # Save all tasks to single file
    save_tasks(all_tasks, cfg.output_filename, cfg)
    print(f"Saved {len(all_tasks)} total tasks to {cfg.output_filename}")


if __name__ == "__main__":
    main()