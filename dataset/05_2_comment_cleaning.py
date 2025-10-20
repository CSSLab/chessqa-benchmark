#!/usr/bin/env python3
"""
clean_comments_vllm.py — Offline vLLM cleaner for chess comments.

Rules per comment (strict):
1) If the comment mentions either player of the game, replace those mentions with "White" or "Black".
   - Treat first name, last name, or full name mentions (case-insensitive) as the player.
   - Replace only the player mentions; do not change other content.
2) If the comment mentions any other person name that is neither the white nor black player, output SKIP.
3) Otherwise, return the original comment unchanged.

LLM must output ONLY one of:
  - the cleaned comment text
  - the unchanged original comment text
  - the exact token: SKIP

This script performs offline inference with vLLM.

Output format:
- Keeps all original fields.
- Adds "original_comment" (the source text) and "cleaned_comment" (the LLM result).
- Replaces "comment" field with the cleaned text for downstream usage.
 - Additionally normalizes legacy chess figurine glyphs (e.g., →Q, →R, →B, →N, →K), strips PGN/ChessBase tags like [%csl Rc4]/[%cal ...],
   removes numeric annotation glyphs like $1, and prunes stray Private-Use glyphs and zero-width chars.

Usage example:
  python scripts/clean_comments_vllm.py \
      --input ./datasets/comment_tasks/comment_dataset.json \
      --output ./datasets/comment_tasks/comment_dataset.cleaned.json \
      --model Qwen/Qwen3-4B \
      --batch-size 256 \
      --max-records 0
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import re
import gc
import time

from vllm import LLM, SamplingParams


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

    # FlashInfer toggles: default to OFF to avoid JIT build issues unless explicitly enabled
    if use_flashinfer is None:
        # Respect existing env if user set it; otherwise default OFF
        pass
    else:
        os.environ["VLLM_USE_FLASHINFER"] = "1" if use_flashinfer else "0"
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "1" if use_flashinfer else "0"
        if not use_flashinfer:
            os.environ.setdefault("VLLM_FLASHINFER_CACHE_DISABLED", "1")


def load_comments(path: Path, max_records: int = 0) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input file must be a JSON array of objects")
    if max_records and max_records > 0:
        return data[:max_records]
    return data


def save_comments(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def split_name(name: str) -> Tuple[str, str, List[str]]:
    """Return (first, last, tokens) from a name like 'Steinitz, William' or 'Paul Morphy'."""
    if not name:
        return "", "", []
    # Normalize separators and collapses
    cleaned = re.sub(r"[\\/]+", " ", name).strip()
    # Handle comma format: Last, First Middle
    if "," in cleaned:
        last, rest = [p.strip() for p in cleaned.split(",", 1)]
        first = rest.split()[0] if rest else ""
    else:
        parts = cleaned.split()
        if len(parts) == 1:
            first, last = parts[0], parts[0]
        else:
            first, last = parts[0], parts[-1]
    tokens = [t for t in re.split(r"[^A-Za-z']+", cleaned) if t]
    return first, last, tokens


def normalize_weird_symbols(text: str) -> str:
    """Replace private-use figurines with SAN letters; strip zero-width chars.

    Known mappings observed in dataset:
       → Q (Queen),  → R (Rook),  → B (Bishop),  → N (Knight),  → K (King)
    """
    if not text:
        return text
    rep = {
        "\ue025": "Q",  # sometimes serialized as U+E025 ()
        "\ue026": "R",  # ()
        "\ue027": "B",  # ()
        "\ue028": "N",  # ()
        "\ue024": "K",  # ()
        "": "Q",
        "": "R",
        "": "B",
        "": "N",
        "": "K",
    }
    out = text
    for k, v in rep.items():
        out = out.replace(k, v)
    # Remove zero-width and directional formatting chars
    out = re.sub(r"[\u200B-\u200F\u202A-\u202E]", "", out)
    # Normalize NBSP to space
    out = out.replace("\u00A0", " ")
    return out


def strip_pgn_markup(text: str) -> str:
    """Remove common PGN/ChessBase inline markup and stray private-use glyphs.

    Examples removed:
      - Square-bracket percent tags: [%csl Rc4], [%cal Ya1a8], [%eval 0.34], [%clk 1:23]
      - Numeric annotation glyphs: $1, $2, $3, ...
      - Any remaining Private Use Area glyphs (U+E000–U+F8FF) after figurine normalization
      - Excess whitespace left by removals
    """
    if not text:
        return text
    out = text
    # Remove ChessBase/PGN inline tags like [%csl Rc4], [%cal Ya1a8], [%eval 0.34]
    out = re.sub(r"\[%[^\]]*\]", "", out)
    # Remove numeric annotation glyphs like $1 $3, anywhere in text
    out = re.sub(r"\$\d+", "", out)
    # Reformat PGN move prefixes so they do not leak raw notation
    out = re.sub(r"(?<!\d)(\d+)\.{3,}(?=[A-Za-z])", r"\1 ", out)
    out = re.sub(r"(?<!\d)(\d+)\.{3,}(?=[0O])", r"\1 ", out)
    out = re.sub(r"(?<!\d)(\d+)\.(?=[A-Za-z])", r"\1 ", out)
    out = re.sub(r"(?<=\d)\.{3,}", "", out)
    out = re.sub(r"\.{3,}(?=[A-Za-z])", " ", out)
    out = re.sub(r"(?<!\S)(\d+)\.{3,}(?=\s)", r"\1", out)
    out = re.sub(r"(?<!\S)(\d+)\.(?=\s)", r"\1", out)
    out = re.sub(r"(?<!\S)\d+\.{3,}(?=$|[.,;:!?])", "", out)
    out = re.sub(r"(?<!\S)\d+\.(?=$|[.,;:!?])", "", out)
    out = re.sub(r"(?<=\()(\d+)\.{3,}(?=\s)", r"\1", out)
    out = re.sub(r"(?<=\()(\d+)\.(?=\s)", r"\1", out)
    out = re.sub(r"(?<=\()\d+\.{3,}(?=\)|$)", "", out)
    out = re.sub(r"(?<=\()\d+\.(?=\)|$)", "", out)
    # Remove stray trailing move counters left after stripping notation (e.g., '. 14' at sentence end)
    out = re.sub(r"([.!?])\s*\d+$", r"\1", out)
    out = re.sub(r"(\))\s*\d+$", r"\1", out)
    # Remove any remaining Private Use Area characters (except the ones already mapped earlier)
    # Do this after normalize_weird_symbols so known figurines become letters first.
    out = re.sub(r"[\ue000-\uf8ff]", "", out)
    # Collapse whitespace introduced by removals
    out = re.sub(r"\s+", " ", out).strip()
    return out


def _normalize_name_for_eq(name: str) -> str:
    """Normalize a name to lowercase letters-only for equality checks."""
    return re.sub(r"[^a-z]", "", (name or "").lower())


def _annotator_side(item: Dict[str, Any]) -> str | None:
    """Return 'White' or 'Black' if the annotator matches exactly that player; else None."""
    meta = item.get("meta", {}) or {}
    annot = meta.get("Annotator") or item.get("annotator") or ""
    white = item.get("white_player", "")
    black = item.get("black_player", "")
    a = _normalize_name_for_eq(annot)
    if a and a == _normalize_name_for_eq(white):
        return "White"
    if a and a == _normalize_name_for_eq(black):
        return "Black"
    return None


def build_messages(item: Dict[str, Any]) -> List[Dict[str, str]]:
    comment = item.get("comment", "").strip()
    comment = normalize_weird_symbols(comment)
    comment = strip_pgn_markup(comment)
    white = item.get("white_player", "").strip()
    black = item.get("black_player", "").strip()

    w_first, w_last, w_tokens = split_name(white)
    b_first, b_last, b_tokens = split_name(black)

    # Provide explicit tokens to help the model match partial mentions
    w_aliases = sorted({t for t in [white, w_first, w_last] + w_tokens if t}, key=lambda s: (-len(s), s.lower()))
    b_aliases = sorted({t for t in [black, b_first, b_last] + b_tokens if t}, key=lambda s: (-len(s), s.lower()))

    sys = (
        "You are a precise data cleaner for chess commentary. "
        "Given a comment and the players' names, output EXACTLY one of:"
        "\n- The cleaned comment text (no quotes)"
        "\n- The unchanged comment text (no quotes)"
        "\n- The single token: SKIP"
        "\nRules:"
        "\n1) If the comment mentions the WHITE player (any of the provided name variants), replace those mentions with 'White'."
        "\n2) If the comment mentions the BLACK player (any of the provided name variants), replace those mentions with 'Black'."
        "\n3) If the comment mentions ANY OTHER person name not matching these players, output exactly: SKIP."
        "\n   - Non-person capitalized terms (e.g., pieces like Queen/Rook, openings like Evans Gambit, places, events) are NOT grounds to skip."
        "\n4) Preserve all other text unchanged. No extra words, no explanations, no quotes, no newlines."
    )

    # Annotator pronoun special-case rule (only when annotator equals one of the players)
    side = _annotator_side(item)
    if side in ("White", "Black"):
        possessive = "White's" if side == "White" else "Black's"
        sys += (
            f"\n5) Special case: The annotator is the {side} player. "
            f"If first-person pronouns occur (I, me, my, mine, myself, I'm, I've, I'd, I'll), rewrite them to refer to {side}: "
            f"I/me/myself→{side}, my/mine→{possessive}, I'm→{side} is, I've→{side} has, I'd→{side} would, I'll→{side} will."
        )

    # few-shot examples
    few_shot_example_1 = (
        "Comment: A nice positional sacrifice. Ety will gain control over the light squares thanks to the exchange she has invested.\n"
        "White: Stefanova, Antoaneta\n"
        "Black: Kosintseva, Tatiana\n"
        "Output strictly one line: cleaned text or SKIP."
    )
    few_shot_example_2 = (
        "Comment: A huge blunder by Levon which loses instantly. I think he missed that the c8-bishop is unprotected.\n"
        "White: So, Wesley\n"
        "Black: Aronian, Levon\n"
        "Output strictly one line: cleaned text or SKIP."
    )
        
    usr = (
        f"Comment: {comment}\n"
        f"White player (full): {white}\n"
        f"Black player (full): {black}\n"
        f"Annotator: {(item.get('meta', {}) or {}).get('Annotator', '')}\n"
        f"Annotator-is-player: {side or 'No'}\n"
        "Output strictly one line: cleaned text or SKIP."
    )

    return [
        {"role": "system", "content": sys},
        {"role": "user", "content": few_shot_example_1},
        {"role": "assistant", "content": "SKIP"},
        {"role": "user", "content": few_shot_example_2},
        {"role": "assistant", "content": "A huge blunder by Black which loses instantly. I think he missed that the c8-bishop is unprotected."},
        {"role": "user", "content": usr},
    ]


def sanitize_llm_output(text: str) -> str:
    """Post-process to enforce the contract as strictly as possible.

    Strategy:
    - Strip code fences and remove any <think>...</think> blocks.
    - If any line equals SKIP, return SKIP.
    - Prefer the last quoted segment if present; otherwise the last non-empty line.
    - Remove leading labels like 'Cleaned:'/'Output:' etc and enclosing quotes.
    """
    if text is None:
        return "SKIP"

    t = text.strip()
    # Remove code fences/backticks
    t = re.sub(r"^```[\s\S]*?```$", lambda m: m.group(0).strip("`\n "), t, flags=re.M)
    t = t.strip("` ")

    # Remove <think> blocks (open-ended or closed)
    t = re.sub(r"(?is)<think>.*?(</think>|$)", "", t).strip()

    # Split by lines (keep order)
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if not lines:
        return "SKIP"

    # If any line is exactly SKIP
    for ln in lines:
        if ln.strip().upper() == "SKIP":
            return "SKIP"

    # Prefer the last quoted segment among all text
    quoted = re.findall(r'"([^"\n]+)"', t)
    candidate = None
    if quoted:
        candidate = quoted[-1].strip()
    else:
        candidate = lines[-1]

    # Remove leading labels like 'Cleaned:', 'Output:', etc.
    candidate = re.sub(r"^(Cleaned|Output|Result|Answer|Text|Comment)\s*[:：]\s*",
                       "", candidate, flags=re.I)

    # Collapse internal whitespace to one space
    candidate = re.sub(r"\s+", " ", candidate).strip()

    # Remove enclosing quotes if present
    if (candidate.startswith('"') and candidate.endswith('"')) or (
        candidate.startswith("'") and candidate.endswith("'")):
        candidate = candidate[1:-1].strip()

    return candidate if candidate else "SKIP"


def _sorted_aliases(name: str) -> List[str]:
    f, l, toks = split_name(name)
    aliases = sorted({t for t in [name, f, l] + toks if t}, key=lambda s: (-len(s), s.lower()))
    return aliases


def apply_alias_replacements(text: str, aliases: List[str], replacement: str) -> str:
    out = text
    for alias in aliases:
        # Word-ish boundaries: do not match inside larger alpha sequences
        pat = re.compile(rf"(?i)(?<![A-Za-z]){re.escape(alias)}(?![A-Za-z])")
        out = pat.sub(replacement, out)
    return out


def apply_pronoun_replacements(text: str, side: str) -> str:
    """Replace first-person pronouns with the player's side when applicable."""
    if not side or not text:
        return text
    repl_side = "White" if side.lower() == "white" else "Black"
    out = text
    # Contractions
    patterns = [
        (r"(?i)\bI'm\b", f"{repl_side} is"),
        (r"(?i)\bI've\b", f"{repl_side} has"),
        (r"(?i)\bI'll\b", f"{repl_side} will"),
        (r"(?i)\bI'd\b", f"{repl_side} would"),
    ]
    # Pronouns
    patterns += [
        (r"(?i)\bmyself\b", repl_side),
        (r"(?i)\bmy\b", f"{repl_side}'s"),
        (r"(?i)\bmine\b", f"{repl_side}'s"),
        (r"(?i)\bme\b", repl_side),
        (r"(?i)\bI\b", repl_side),
    ]
    for pat, rep in patterns:
        out = re.sub(pat, rep, out)
    return out


def perform_cleaning(records: List[Dict[str, Any]], model: str, batch_size: int, max_model_len: int,
                     max_tokens: int, tensor_parallel_size: int, gpu_mem_util: float,
                     dtype: str = None, attention_backend: str = None,
                     use_flashinfer: bool | None = False,
                     enable_thinking: bool | None = True) -> List[Tuple[Dict[str, Any], str]]:
    """Return list of tuples: (record, cleaned_text_or_SKIP)."""
    _setup_env(model_name=model, enable_prefix_caching=True, attention_backend=attention_backend,
               use_flashinfer=use_flashinfer)

    # Free CUDA caches before loading
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    llm_kwargs: Dict[str, Any] = {
        "model": model,
        "max_model_len": int(max_model_len),
        "tensor_parallel_size": int(tensor_parallel_size),
        "gpu_memory_utilization": float(gpu_mem_util),
        "max_num_seqs": 1024,
    }
    if dtype and dtype != "auto":
        llm_kwargs["dtype"] = dtype

    print("=== Initialize vLLM model ===")
    for k, v in llm_kwargs.items():
        print(f"{k}: {v}")
    llm = LLM(**llm_kwargs)

    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=int(max_tokens))

    outputs: List[Tuple[Dict[str, Any], str]] = []
    t0 = time.time()
    for i in range(0, len(records), batch_size):
        batch = records[i:i + batch_size]
        prompts = [build_messages(r) for r in batch]
        chat_kwargs: Dict[str, Any] = {}
        if enable_thinking is not None:
            chat_kwargs["chat_template_kwargs"] = {"enable_thinking": bool(enable_thinking)}
        try:
            outs = llm.chat(prompts, sampling, **chat_kwargs)
        except Exception:
            # Fallback without template kwargs (for models that don't accept it)
            outs = llm.chat(prompts, sampling)
        for rec, out in zip(batch, outs):
            text = out.outputs[0].text if out.outputs and out.outputs[0] else ""
            cleaned = sanitize_llm_output(text)
            # Post-ensure player substitutions, in case the LLM missed any exact alias surface forms
            w_aliases = _sorted_aliases(rec.get("white_player", ""))
            b_aliases = _sorted_aliases(rec.get("black_player", ""))
            if cleaned.upper() != "SKIP":
                cleaned = normalize_weird_symbols(cleaned)
                cleaned = strip_pgn_markup(cleaned)
                cleaned = apply_alias_replacements(cleaned, w_aliases, "White")
                cleaned = apply_alias_replacements(cleaned, b_aliases, "Black")
                # If annotator is exactly a player, normalize first-person pronouns
                side2 = _annotator_side(rec)
                if side2 in ("White", "Black"):
                    cleaned = apply_pronoun_replacements(cleaned, side2)
            outputs.append((rec, cleaned))
    dt = time.time() - t0
    print(f"Cleaning completed: {len(records)} records in {dt:.1f}s")
    return outputs


def main() -> int:

    default_input = Path("../../data/mid/comment_dataset.json")
    default_output = Path("../../data/mid/comment_dataset.cleaned.json")

    ap = argparse.ArgumentParser(description="Offline vLLM cleaning for chess comments")
    ap.add_argument("--input", type=Path, default=default_input, help="Input JSON array file of comment objects")
    ap.add_argument("--output", type=Path, default=default_output, help="Output JSON file for cleaned comments")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-30B-A3B-Instruct-2507", help="HuggingFace model ID or local path for vLLM")
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--max-records", type=int, default=0, help="Limit records for a quick run (0 = all)")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-tokens", type=int, default=128, help="Max new tokens for cleaning output")
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.80)
    ap.add_argument("--dtype", type=str, default=None, choices=[None, "auto", "float16", "bfloat16", "float32"], nargs='?')
    ap.add_argument("--attention-backend", type=str, default=None, choices=[None, "FLASH_ATTN", "XFORMERS"], nargs='?')
    gfi = ap.add_mutually_exclusive_group()
    gfi.add_argument("--use-flashinfer", dest="use_flashinfer", action="store_true")
    gfi.add_argument("--no-flashinfer", dest="use_flashinfer", action="store_false")
    ap.set_defaults(use_flashinfer=False)  # default OFF to bypass JIT
    # Thinking control (default off to avoid <think> blocks)
    gthink = ap.add_mutually_exclusive_group()
    gthink.add_argument("--enable-thinking", dest="enable_thinking", action="store_true")
    gthink.add_argument("--disable-thinking", dest="enable_thinking", action="store_false")
    ap.set_defaults(enable_thinking=False)

    args = ap.parse_args()

    records = load_comments(args.input, args.max_records)
    pairs = perform_cleaning(
        records=records,
        model=args.model,
        batch_size=args.batch_size,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_mem_util=args.gpu_memory_utilization,
        dtype=(None if args.dtype in (None, "auto") else args.dtype),
        attention_backend=args.attention_backend,
        use_flashinfer=args.use_flashinfer,
        enable_thinking=args.enable_thinking,
    )

    cleaned_items: List[Dict[str, Any]] = []
    skipped = 0
    for rec, out_text in pairs:
        if out_text.strip().upper() == "SKIP":
            skipped += 1
            continue
        # Replace the comment with cleaned text
        new_item = dict(rec)
        orig = rec.get("comment", "")
        new_item["original_comment"] = orig
        # Normalize symbols in the final output too (idempotent)
        final_clean = normalize_weird_symbols(out_text)
        final_clean = strip_pgn_markup(final_clean)
        new_item["cleaned_comment"] = final_clean
        new_item["comment"] = final_clean
        cleaned_items.append(new_item)

    save_comments(args.output, cleaned_items)
    print(f"Saved cleaned dataset: {args.output}")
    print(f"Records in: {len(records)}, kept: {len(cleaned_items)}, skipped: {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
