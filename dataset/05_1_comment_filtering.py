#!/usr/bin/env python3
"""
comment_dataset_creation.py - Extract a comment-based dataset from a PGN file.

For each comment found on the main line of a game, create an entry with:
  1. comment (raw text)
  2. move in UCI (the move the comment is attached to)
  3. pgn_until_move (SAN moves with move numbers up to and including this move)
  4. move_number (full-move index of the move that was just played)
  5. side_to_move ("white" or "black" after the move is played)
  6. keywords (matched from a provided keyword set; defaults to keys of all_themes.json)
  7. move_piece (piece type moved: pawn/knight/bishop/rook/queen/king)
  8. white_player (White tag)
  9. black_player (Black tag)
 10. meta (dict with game_id and selected headers)

Usage:
  python scripts/comment_dataset_creation.py \
    --pgn data/filtered_chessbase.pgn \
    --output datasets/comment_tasks/comment_dataset.json \
    --max-games 1000

Notes:
  - Only comments on the game mainline are extracted (variations ignored by default).
  - Root (pre-move) comments are skipped because they do not attach to a specific move.
  - Keyword matching normalizes text by removing non-alphanumeric chars and lowercasing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import chess
import chess.pgn
from tqdm import tqdm
import re

PIECE_NAME = {
    chess.PAWN: "pawn",
    chess.KNIGHT: "knight",
    chess.BISHOP: "bishop",
    chess.ROOK: "rook",
    chess.QUEEN: "queen",
    chess.KING: "king",
}

# A conservative, chess-specific default keyword set (no generic words).
KEYWORDS_DEFAULT: List[str] = [
    # Evaluation / judgment
    "advantage", "slight advantage", "clear advantage", "decisive advantage",
    "compensation", "initiative", "equality", "equalize", "unclear", "winning", "drawish",
    "brilliant", "inaccuracy", "mistake", "error", "blunder", "dubious", "only move", "good", "bad",

    # Tactics / dynamic play
    "attack", "counterattack", "counterstroke", "counterplay", "mate threat", "mating net",
    "checkmate", "perpetual check", "fork", "pin", "skewer", "double attack",
    "discovered attack", "discovered check", "x-ray", "deflection", "decoy", "interference",
    "clearance", "overloading", "overworked", "zwischenzug", "intermezzo", "trap",
    "trapped piece", "sacrifice", "exchange sacrifice", "underpromotion", "promotion",
    "smothered mate", "windmill", "quiet move", "forcing move", "tempo",
    "prophylaxis", "overprotection", "blockade",

    # Structure / strategy
    "outpost", "weak square", "weakness", "open file", "half-open file", "open diagonal",
    "battery", "rook lift", "seventh rank", "back rank", "space advantage",
    "centralization", "centralisation", "center", "centre", "control of the center",
    "control of the centre", "king safety", "exposed king", "castling",
    "opposite-side castling", "kingside", "king side", "queenside", "queen side",
    "bishop pair", "good bishop", "bad bishop", "opposite-colored bishops",
    "opposite-coloured bishops", "same-colored bishops", "same-coloured bishops",
    "color complex", "colour complex", "knight vs bishop", "domination",

    # Pawns / structure
    "isolated pawn", "iqp", "isolani", "backward pawn", "doubled pawns", "hanging pawns",
    "pawn majority", "pawn minority attack", "minority attack", "pawn chain",
    "pawn break", "breakthrough", "pawn storm", "passed pawn", "protected passer",
    "connected passers", "outside passer", "outside passed pawn", "distant passed pawn",

    # Opening
    "theory", "book move", "in book", "out of book", "move order",
    "transposition", "novelty", "gambit",

    # Endgame
    "endgame", "pawn ending", "rook ending", "bishop ending", "knight ending",
    "opposition", "distant opposition", "triangulation", "lucena", "philidor",
    "shoulder", "shouldering", "bridge-building", "corresponding squares", "fortress", "zugzwang",

    # Result / clock
    "resign", "resigns", "resignation", "draw", "repetition",
    "agreed draw", "time trouble", "zeitnot", "flag",
]

WORD_RE = re.compile(r"[A-Za-z0-9']+")

def count_words(s: str) -> int:
    """Count 'words' as alphanumeric (and apostrophe) tokens."""
    if not s:
        return 0
    return len(WORD_RE.findall(s))

def should_keep_comment(text: str, matched_keywords: List[str], min_words: int = 5) -> bool:
    """Keep only if comment has >= min_words and at least one matched keyword."""
    return bool(matched_keywords) and count_words(text) >= min_words

def _normalize_lines_to_keywords(s: str) -> List[str]:
    # From a newline-separated text file (supports comments with '#')
    kws = {
        line.strip() for line in s.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    # Sort longer phrases first (helps when you eventually add overlapping terms)
    return sorted(kws, key=lambda x: (-len(x), x))

def _flatten_json_keywords(data) -> List[str]:
    """
    Accepts:
      - list[str] -> as-is
      - dict[str, list[str] or dict with 'synonyms'] -> keys + synonyms
    """
    out: set[str] = set()
    if isinstance(data, list):
        out.update(str(x).strip() for x in data if str(x).strip())
    elif isinstance(data, dict):
        for k, v in data.items():
            k = str(k).strip()
            if k:
                out.add(k)
            # Allow {"canonical": ["syn1","syn2"]} OR {"canonical": {"synonyms":[...]}}
            if isinstance(v, list):
                out.update(str(x).strip() for x in v if str(x).strip())
            elif isinstance(v, dict) and "synonyms" in v and isinstance(v["synonyms"], list):
                out.update(str(x).strip() for x in v["synonyms"] if str(x).strip())
    return sorted(out, key=lambda x: (-len(x), x))

def load_keywords(keywords_path: Optional[Path]) -> List[str]:
    """Load keyword list from JSON (list or dict) or text file. Falls back to KEYWORDS_DEFAULT."""
    if keywords_path is None:
        # Default, already deduplicated & useful out-of-the-box
        return KEYWORDS_DEFAULT

    text = keywords_path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text:
        return KEYWORDS_DEFAULT

    # Try JSON first
    try:
        data = json.loads(text)
        kws = _flatten_json_keywords(data)
        if kws:
            return kws
    except json.JSONDecodeError:
        pass

    # Fallback: treat as newline-separated text
    kws = _normalize_lines_to_keywords(text)
    return kws if kws else KEYWORDS_DEFAULT


def normalize_text(s: str) -> str:
    """Normalize text by removing non-alphanumeric characters and lowercasing."""
    return "".join(ch for ch in s.lower() if ch.isalnum())


def find_keywords_in_comment(comment: str, keywords: Iterable[str]) -> List[str]:
    """
    Safer matching: normalize both comment and keyword to lowercase alphanumeric words,
    then match with word boundaries. Longer keywords are tried first.
    """
    if not comment:
        return []

    # Normalize to space-separated tokens
    def norm_words(s: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

    norm_comment = f" {norm_words(comment)} "  # pad to allow 'word boundary' with spaces

    matched: List[str] = []
    seen = set()

    # Try longer keywords first so 'distant passed pawn' matches before 'passed pawn'
    for kw in sorted(set(keywords), key=lambda x: (-len(x), x)):
        kw_norm = norm_words(kw)
        if not kw_norm:
            continue
        test = f" {kw_norm} "
        if test in norm_comment and kw_norm not in seen:
            matched.append(kw)
            seen.add(kw_norm)
    return matched

def format_pgn_until(san_moves: List[str]) -> str:
    """Format SAN moves with move numbers up to and including the last move.

    Example: ["e4", "e5", "Nf3"] -> "1. e4 e5 2. Nf3"
    """
    out: List[str] = []
    move_index = 0
    move_number = 1
    while move_index < len(san_moves):
        # White move
        out.append(f"{move_number}. {san_moves[move_index]}")
        move_index += 1
        if move_index < len(san_moves):
            # Black move on same move number
            out.append(san_moves[move_index])
            move_index += 1
        move_number += 1
    return " ".join(out)


def extract_comments_from_game(
    game: chess.pgn.Game,
    keywords: List[str],
) -> List[Dict]:
    """Extract comment entries from the mainline of a single game.

    Skips root node comments (pre-move) because they do not attach to a specific move.
    """
    entries: List[Dict] = []
    white_player = game.headers.get("White", "")
    black_player = game.headers.get("Black", "")
    game_id = game.headers.get("GameId") or game.headers.get("GameID") or ""

    # Prepare a board and SAN history to reconstruct PGN up to each move
    board = game.board()
    san_history: List[str] = []

    # We iterate mainline nodes, skipping the root (node.move is None)
    node = game
    while node.variations:
        next_node = node.variation(0)

        # Pre-push state: who moves and which move number this move belongs to
        pre_turn = board.turn
        pre_fullmove = board.fullmove_number

        move = next_node.move
        if move is None:
            break

        # Determine piece being moved before pushing
        piece = board.piece_at(move.from_square)
        move_piece = PIECE_NAME.get(piece.piece_type, "unknown") if piece else "unknown"

        # Record FEN before move, then SAN and push
        fen_before = board.fen()
        move_san = board.san(move)
        board.push(move)
        fen_after = board.fen()
        san_history.append(move_san)

        # Comment attached to this move
        raw_comment = (next_node.comment or "").strip()
        if raw_comment:
            # Move number refers to the move just played
            move_number = pre_fullmove  # full-move index of the move (white and black share the same number)
            side_to_move = "white" if board.turn == chess.WHITE else "black"  # after the move

            matched_keywords = find_keywords_in_comment(raw_comment, keywords)
            if should_keep_comment(raw_comment, matched_keywords):
                entry = {
                "comment": raw_comment,
                "move_uci": move.uci(),
                "pgn_until_move": format_pgn_until(san_history),
                "fen_before": fen_before,
                "fen_after": fen_after,
                "move_number": move_number,
                "side_to_move": side_to_move,
                "keywords": matched_keywords,
                "move_piece": move_piece,
                "white_player": white_player,
                "black_player": black_player,
                "meta": {
                    "game_id": game_id,
                    # Commonly useful headers (include if present)
                    **{k: v for k, v in game.headers.items() if k in {
                        "Event", "Site", "Date", "Round", "Result", "ECO", "PlyCount", "Annotator",
                        "EventDate", "EventType", "EventRounds", "EventCountry", "SourceTitle", "Source",
                        "SourceDate", "SourceVersion", "SourceVersionDate", "SourceQuality"
                        }}
                    }
                }
                entries.append(entry)

        # Continue down the mainline
        node = next_node

    return entries


def process_pgn(
    pgn_path: Path,
    output_path: Path,
    max_games: Optional[int] = None,
    keywords_path: Optional[Path] = None,
) -> Dict[str, int]:
    """Process a PGN file and write the comment dataset as a JSON array."""
    keywords = load_keywords(keywords_path)

    all_entries: List[Dict] = []
    games_processed = 0
    games_with_comments = 0

    with open(pgn_path, "r", encoding="utf-8", errors="ignore") as f:
        with tqdm(desc="Reading games", unit="game") as pbar:
            while True:
                if max_games is not None and games_processed >= max_games:
                    break
                game = chess.pgn.read_game(f)
                if game is None:
                    break
                games_processed += 1

                entries = extract_comments_from_game(game, keywords)
                if entries:
                    games_with_comments += 1
                    all_entries.extend(entries)

                pbar.update(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as out:
        json.dump(all_entries, out, ensure_ascii=False, indent=2)

    return {
        "games_processed": games_processed,
        "games_with_comments": games_with_comments,
        "entries": len(all_entries),
        "keywords_loaded": len(keywords),
    }


def main():
    p = argparse.ArgumentParser(description="Create a comment-based dataset from a PGN file")
    p.add_argument("--pgn", type=Path, default=Path("../../data/raw/filtered_chessbase.pgn"))
    p.add_argument("--output", type=Path, default=Path("../../data/mid/comment_dataset.json"))
    p.add_argument("--max-games", type=int, default=None, help="Optional limit on number of games to process")
    p.add_argument("--keywords", type=Path, default=None, help="Optional path to keywords file (json dict keys or text lines)")
    args = p.parse_args()

    stats = process_pgn(args.pgn, args.output, args.max_games, args.keywords)
    print(json.dumps({"ok": True, **stats, "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
