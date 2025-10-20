#!/usr/bin/env python3
"""
judge_comments_vllm.py — Final relevance filter for cleaned comments using vLLM (offline).

Goal: Keep only comments that are explicitly about the given move/position in this game.

Strict judgement output: KEEP or DROP (uppercase, no extra text).

Signals to KEEP (non-exhaustive):
- Directly references the current move, pieces, or squares (SAN/UCI like Rae1, Qxh7+, e4e5, a1h8)
- Talks about the position (e.g., "kingside attack", "weak d6 pawn", "open c-file")
- Provides evaluation or reason related to this move/position

Signals to DROP:
- Generic aphorisms/quotes or meta commentary with no tie to the current move/position
- Player biography/psychology, event logistics, audience, stream, etc.
- Comments referencing unrelated games or people (should have been removed earlier)

Heuristic pre-filter: Before calling the LLM, quickly DROP comments that clearly lack any chess-specific signal
(no piece/square notation, too short, obvious generic phrases). You can disable this via --no-heuristics.

Usage:
  python scripts/judge_comments_vllm.py \
    --input ./datasets/comment_tasks/comment_dataset.cleaned.json \
    --output ./datasets/comment_tasks/comment_dataset.final.json \
    --model Qwen/Qwen3-4B --batch-size 256 --no-flashinfer --disable-thinking
"""

import argparse
import json
import os
import re
import time
import gc
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    from vllm import LLM, SamplingParams
except Exception:
    print("vLLM is required. Install with: pip install vllm")
    raise


def _setup_env(model_name: str,
               enable_prefix_caching: bool = True,
               attention_backend: str = None,
               use_flashinfer: bool | None = False) -> None:
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("VLLM_ALLOW_LONG_MAX_MODEL_LEN", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    if enable_prefix_caching:
        os.environ.setdefault("VLLM_ENABLE_PREFIX_CACHING", "1")
    else:
        os.environ["VLLM_ENABLE_PREFIX_CACHING"] = "0"
    if attention_backend and "VLLM_ATTENTION_BACKEND" not in os.environ:
        os.environ["VLLM_ATTENTION_BACKEND"] = attention_backend
    if "gemma-3" in (model_name or "").lower() and "VLLM_ATTENTION_BACKEND" not in os.environ:
        os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    if use_flashinfer is not None:
        os.environ["VLLM_USE_FLASHINFER"] = "1" if use_flashinfer else "0"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1" if use_flashinfer else "0"
        if not use_flashinfer:
            os.environ.setdefault("VLLM_FLASHINFER_CACHE_DISABLED", "1")


def load_items(path: Path, max_records: int = 0) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input must be a JSON array")
    if max_records and max_records > 0:
        data = data[:max_records]
    return data


def save_items(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


PIECE_WORDS = {
    "king", "queen", "rook", "bishop", "knight", "pawn",
    "kingside", "queenside", "file", "rank", "diagonal",
    "check", "checkmate", "mate", "castle", "castling",
    "gambit", "fork", "pin", "skewer", "x-ray", "xray",
    "tempo", "initiative", "zugzwang", "passed pawn", "outpost",
}


SAN_UCI_PATTERNS = [
    re.compile(r"\b[O0]-O(?:-O)?\b", re.I),           # O-O, O-O-O
    re.compile(r"\b[a-h][1-8][a-h][1-8]\b"),          # UCI move e2e4
    re.compile(r"\b[a-h]x?[a-h][1-8](?:[+#])?\b", re.I),  # SAN-like exd5, Qh4+, etc.
    re.compile(r"\b[QRBNK][a-h]?[1-8]?x?[a-h][1-8](?:[+#])?\b"),  # piece SAN moves
]


GENERIC_PHRASES = [
    "i will play", "we will draw", "as they say", "people say", "it is said",
    "in general", "generally speaking", "quote", "famous quote", "life", "luck",
]


def heuristic_is_relevant(text: str) -> bool:
    if not text:
        return False
    t = text.strip()
    # Length threshold
    if len(t) < 25:
        return False
    low = t.lower()
    # Obvious generic signals
    for p in GENERIC_PHRASES:
        if p in low:
            return False
    # SAN/UCI presence
    if any(p.search(t) for p in SAN_UCI_PATTERNS):
        return True
    # Chess word presence
    if any(w in low for w in PIECE_WORDS):
        return True
    # Square mention (at least one square)
    if re.search(r"\b[a-h][1-8]\b", t):
        return True
    return False


def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    comment = item.get("cleaned_comment") or item.get("comment", "")
    fen_before = item.get("fen_before", "")
    fen_after = item.get("fen_after", "")
    move_uci = item.get("move_uci", "")
    move_number = item.get("move_number", "")
    side_to_move = item.get("side_to_move", "")
    pgn_until = item.get("pgn_until_move", "")

    sys = (
        "You are a strict chess commentary relevance judge. "
        "Given a single comment and the exact game context (FENs, PGN so far, and the move), decide if the comment is directly useful to understand or evaluate the current move/position in THIS game. "
        "Output exactly one token: KEEP or DROP."
        "\nKEEP if and only if the comment provides actionable insight about the current move or position (e.g., mentions pieces/squares, tactical/positional ideas, consequences relevant to this position)."
        "\nDROP if the comment is generic quotes/aphorisms, meta or biography, audience/event chatter, or otherwise unrelated to the concrete move/position."
    )

    usr = (
        f"Comment: {comment}\n"
        f"Move UCI: {move_uci} | Move number: {move_number} | Side to move: {side_to_move}\n"
        f"FEN before: {fen_before}\n"
        f"FEN after: {fen_after}\n"
        f"PGN until move: {pgn_until}\n"
        "Answer strictly with KEEP or DROP."
    )

    # One tiny few-shot to anchor behavior
    ex_user = (
        "Comment: I will play 40 good moves. If my opponent plays 40 good moves too, we will draw.\n"
        "Move UCI: e2e4 | Move number: 1 | Side to move: white\n"
        "FEN before: startpos\n"
        "FEN after: <omitted>\n"
        "PGN until move: 1. e4\n"
        "Answer strictly with KEEP or DROP."
    )
    ex_assistant = "DROP"

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": ex_user},
        {"role": "assistant", "content": ex_assistant},
        {"role": "user", "content": usr},
    ]


def sanitize_label(text: str) -> str:
    if not text:
        return "DROP"
    t = text.strip().strip("` ")
    # remove <think> blocks if any
    t = re.sub(r"(?is)<think>.*?(</think>|$)", "", t).strip()
    # collapse whitespace
    t = re.sub(r"\s+", " ", t)
    # extract a label if present in text
    m = re.search(r"\b(KEEP|DROP)\b", t, re.I)
    if m:
        return m.group(1).upper()
    # fallback: first token upper
    tok = t.split()[0].upper() if t.split() else "DROP"
    return tok if tok in {"KEEP", "DROP"} else "DROP"


def main() -> int:
    # Compute default paths relative to repo root
    # script_dir = Path(__file__).parent
    # repo_root = script_dir.parent.parent
    # default_input = repo_root / "data" / "comment_tasks" / "comment_dataset.cleaned.json"
    # default_output = repo_root / "data" / "comment_tasks" / "comment_dataset.final.json"

    default_input = Path("../../data/mid/comment_dataset.cleaned.json")
    default_output = Path("../../data/mid/comment_dataset.final.json")
    ap = argparse.ArgumentParser(description="Judge relevance of cleaned comments using vLLM")
    ap.add_argument("--input", type=Path, default=default_input)
    ap.add_argument("--output", type=Path, default=default_output)
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--dtype", type=str, default=None, choices=[None, "auto", "float16", "bfloat16", "float32"], nargs='?')
    ap.add_argument("--attention-backend", type=str, default=None, choices=[None, "FLASH_ATTN", "XFORMERS"], nargs='?')
    gfi = ap.add_mutually_exclusive_group()
    gfi.add_argument("--use-flashinfer", dest="use_flashinfer", action="store_true")
    gfi.add_argument("--no-flashinfer", dest="use_flashinfer", action="store_false")
    ap.set_defaults(use_flashinfer=False)
    gthink = ap.add_mutually_exclusive_group()
    gthink.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    gthink.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    ap.set_defaults(enable_thinking=False)
    gheu = ap.add_mutually_exclusive_group()
    gheu.add_argument("--heuristics", dest="heuristics", action="store_true")
    gheu.add_argument("--no-heuristics", dest="heuristics", action="store_false")
    ap.set_defaults(heuristics=True)

    args = ap.parse_args()

    items = load_items(args.input, args.max_records)
    kept: List[Dict[str, Any]] = []

    # Heuristic prefilter
    heuristic_labels: List[str] = []
    to_judge: List[Dict[str, Any]] = []
    if args.heuristics:
        for it in items:
            text = it.get("cleaned_comment") or it.get("comment", "")
            if heuristic_is_relevant(text):
                to_judge.append(it)
                heuristic_labels.append("CANDIDATE")
            else:
                heuristic_labels.append("DROP")
        # Keep indices for mapping
        idx_map = [i for i, lab in enumerate(heuristic_labels) if lab == "CANDIDATE"]
    else:
        to_judge = items
        idx_map = list(range(len(items)))

    # Setup vLLM
    _setup_env(args.model, enable_prefix_caching=True, attention_backend=args.attention_backend,
               use_flashinfer=args.use_flashinfer)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    llm_kwargs: Dict[str, Any] = {
        "model": args.model,
        "max_model_len": int(args.max_model_len),
        "tensor_parallel_size": int(args.tensor_parallel_size),
        "gpu_memory_utilization": float(args.gpu_memory_utilization),
        "max_num_seqs": 1024,
    }
    if args.dtype and args.dtype != "auto":
        llm_kwargs["dtype"] = args.dtype

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(args.max_tokens))

    # Judge in batches
    labels: List[str] = [None] * len(to_judge)
    t0 = time.time()
    for i in range(0, len(to_judge), args.batch_size):
        batch = to_judge[i:i + args.batch_size]
        prompts = [build_messages(it) for it in batch]
        chat_kwargs: Dict[str, Any] = {}
        if args.enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": bool(args.enable_thinking)}
        try:
            outs = llm.chat(prompts, sp, **chat_kwargs)
        except Exception:
            outs = llm.chat(prompts, sp)
        for j, out in enumerate(outs):
            text = out.outputs[0].text if out.outputs and out.outputs[0] else ""
            labels[i + j] = sanitize_label(text)

    dt = time.time() - t0
    print(f"Judged {len(to_judge)} candidates in {dt:.1f}s")

    # Combine heuristic and LLM labels
    if args.heuristics:
        k = 0
        for idx, lab in enumerate(heuristic_labels):
            if lab == "DROP":
                continue
            heuristic_labels[idx] = labels[k]
            k += 1
        final_labels = heuristic_labels
    else:
        final_labels = labels

    # Decide and keep
    keeps = 0
    for i, it in enumerate(items):
        lab = final_labels[i] if i < len(final_labels) else "DROP"
        if lab == "KEEP":
            kept.append(it)
            keeps += 1

    save_items(args.output, kept)
    print(f"Saved final dataset: {args.output}")
    print(f"Input: {len(items)}, kept: {len(kept)}, dropped: {len(items) - len(kept)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

