"""Representation, prompting, extraction, and scoring utilities for ChessQA."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from statistics import mean
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import chess


REPRESENTATIONS = ("fen", "ascii", "piece_list")
OUTPUT_PROTOCOLS = ("terminal", "json")
PROMPT_STYLES = ("analysis", "direct")

_POSITION_REASONING_RE = re.compile(
    r"\s*Think deeper about this position:.*?"
    r"How would a strong engine assess this position after calculating many moves deep\?\s*",
    flags=re.IGNORECASE | re.DOTALL,
)
_STEP_BY_STEP_RE = re.compile(
    r"\s*Analyze step by step and explain your reasoning\.\s*",
    flags=re.IGNORECASE,
)


def clean_fen(value: str) -> str:
    """Extract the FEN portion from a dataset ``input`` field."""
    return (value or "").split("|", 1)[0].strip()


def _board(value: str) -> chess.Board:
    return chess.Board(" ".join(clean_fen(value).split()))


def piece_list(value: str, include_state: bool = True) -> str:
    """Render a deterministic explicit piece list without legal-move hints."""
    board = _board(value)
    names = {
        chess.KING: "King",
        chess.QUEEN: "Queen",
        chess.ROOK: "Rook",
        chess.BISHOP: "Bishop",
        chess.KNIGHT: "Knight",
        chess.PAWN: "Pawn",
    }
    parts: List[str] = []
    for color, color_name in ((chess.WHITE, "White"), (chess.BLACK, "Black")):
        for piece_type in (
            chess.KING,
            chess.QUEEN,
            chess.ROOK,
            chess.BISHOP,
            chess.KNIGHT,
            chess.PAWN,
        ):
            squares = sorted(chess.square_name(sq) for sq in board.pieces(piece_type, color))
            if squares:
                parts.append(f"{color_name} {names[piece_type]}: {squares}")
    rendered = ", ".join(parts)
    if include_state:
        rendered += "\n" + state_fields(board)
    return rendered


def state_fields(board: chess.Board) -> str:
    """Render every non-placement FEN field needed to preserve game state."""
    fields = board.fen().split()
    turn = "White" if board.turn == chess.WHITE else "Black"
    ep = fields[3]
    return (
        f"State: {turn} to move; castling={fields[2]}; en-passant={ep}; "
        f"halfmove={fields[4]}; fullmove={fields[5]}"
    )


def ascii_board(value: str) -> str:
    """Render an oriented rank/file ASCII board plus all FEN state fields."""
    board = _board(value)
    lines = []
    for rank in range(7, -1, -1):
        cells = []
        for file_index in range(8):
            piece = board.piece_at(chess.square(file_index, rank))
            cells.append(piece.symbol() if piece else ".")
        lines.append(f"{rank + 1} " + " ".join(cells))
    lines.append("  a b c d e f g h")
    lines.append(state_fields(board))
    return "\n".join(lines)


def piece_context(value: str) -> str:
    """Context used by the historical FEN+Pieces ablation.

    Deliberately excludes legal moves; including them would leak answers for
    several Structural tasks and would not be a piece-arrangement ablation.
    """
    try:
        return f"Piece arrangement: {piece_list(value, include_state=False)}\n\n"
    except ValueError:
        return ""


def _replace_representation(question: str, value: str, representation: str) -> str:
    if representation == "fen":
        return question
    if representation not in REPRESENTATIONS:
        raise ValueError(f"Unknown representation {representation!r}")

    fen = clean_fen(value)
    board_text = ascii_board(fen) if representation == "ascii" else piece_list(fen)
    label = "an ASCII board" if representation == "ascii" else "an explicit piece list"

    replacements = (
        (f"in FEN: {fen}", f"as {label}:\n{board_text}"),
        (f"Initial FEN: {fen}", f"Initial position as {label}:\n{board_text}"),
    )
    rendered = question
    for old, new in replacements:
        if old in rendered:
            rendered = rendered.replace(old, new, 1)
            return rendered
    return rendered.replace(fen, f"{label}:\n{board_text}", 1)


def _select_example(examples: Any, one_based_index: int) -> str:
    if isinstance(examples, str):
        try:
            examples = json.loads(examples)
        except json.JSONDecodeError:
            examples = [examples]
    if not isinstance(examples, Sequence) or not examples:
        return ""
    index = max(0, int(one_based_index) - 1)
    return str(examples[index] if index < len(examples) else examples[0])


def repair_legacy_task_wording(task: Mapping[str, Any], question: str) -> str:
    """Repair frozen Structural prompts whose text contradicted their gold labels.

    The frozen answers already use ``Board.attacks``.  Older prompt templates,
    however, described legal destinations and instructed models to erase every
    attack by a pinned piece.  Rewriting those two lines at prompt-render time
    keeps the immutable item/result files intact while making the interface
    match the standard attack semantics used by the corrected generator.
    """
    task_type = str(task.get("task_type", ""))
    pin_marker = "For each pin, provide the key squares"
    if (
        task_type == "motifs_pin"
        and "absolute pin" not in question.lower()
        and pin_marker in question
    ):
        definition = (
            "Identify all absolute pins in this position. An absolute pin occurs "
            "when a piece cannot move because it would expose its own king to check. "
        )
        question = question.replace(pin_marker, definition + pin_marker, 1)
    kind = next(
        (name for name in ("capture_squares", "control_squares", "protect_squares") if task_type.endswith(name)),
        None,
    )
    if kind is None:
        return question

    meta = _metadata(task)
    piece = str(meta.get("piece", "specified piece"))
    square = str(meta.get("target_square", "the queried square"))
    descriptions = {
        "capture_squares": f"Find all enemy-occupied squares attacked by the {piece} on {square}.",
        "control_squares": f"Find all empty squares attacked (controlled) by the {piece} on {square}.",
        "protect_squares": f"Find all friendly-occupied squares (excluding the king) attacked and therefore protected by the {piece} on {square}.",
    }
    replacement = descriptions[kind]
    lines = question.splitlines()
    rendered: List[str] = []
    replaced = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Exclude ") and "pinned" in stripped:
            continue
        if not replaced and stripped.startswith("Find all squares that the"):
            rendered.extend((replacement, "Use the standard chess attack relation: retain attacks by a pinned piece."))
            replaced = True
        else:
            rendered.append(line)
    return "\n".join(rendered)


def format_task_prompt(
    task: Mapping[str, Any],
    *,
    add_piece_context: bool = False,
    representation: str = "fen",
    format_example_group: int = 1,
    output_protocol: str = "terminal",
    prompt_style: str = "analysis",
) -> str:
    """Format one task under controlled input, output, and prompt settings."""
    if output_protocol not in OUTPUT_PROTOCOLS:
        raise ValueError(f"Unknown output protocol {output_protocol!r}")
    if prompt_style not in PROMPT_STYLES:
        raise ValueError(f"Unknown prompt style {prompt_style!r}")
    question = str(task["question"])
    value = str(task.get("input", ""))
    context = piece_context(value) if add_piece_context else ""
    question = question.replace("CONTEXT_PLACEHOLDER", context)
    question = repair_legacy_task_wording(task, question)
    question = _replace_representation(question, value, representation)
    if prompt_style == "direct":
        question = _POSITION_REASONING_RE.sub("\n", question)
        question = _STEP_BY_STEP_RE.sub("\n", question)
    example = _select_example(task.get("format_examples", []), format_example_group)
    question = question.replace("FORMAT_EXAMPLE_PLACEHOLDER", example)
    if output_protocol == "json":
        question = re.sub(
            r"Finish with a single line formatted EXACTLY as:\s*\nFINAL ANSWER:\s*<answer>\s*\n?",
            'Return only a JSON object with one string field: {"answer": "<answer>"}.\n',
            question,
            flags=re.IGNORECASE,
        )
    return question


_TERMINAL_RE = re.compile(r"FINAL ANSWER:\s*(.+?)(?:\n|$)", re.IGNORECASE)
_TERMINAL_LINE_RE = re.compile(r"FINAL ANSWER:\s*(.+)", re.IGNORECASE)
_BOXED_RE = re.compile(r"(?:the\s+final\s+answer\s+is\s+)?\$?\\boxed\{([^}]+)\}\$?", re.IGNORECASE)


def extract_protocol_answer(response: str, output_protocol: str = "terminal") -> Tuple[str, bool]:
    """Strictly extract an answer and report protocol compliance."""
    text = response or ""
    if output_protocol == "json":
        try:
            obj = json.loads(text.strip())
        except (json.JSONDecodeError, TypeError):
            return "", False
        if not isinstance(obj, dict) or set(obj) != {"answer"} or not isinstance(obj["answer"], str):
            return "", False
        return obj["answer"].strip(), True

    lines = [line.strip() for line in text.rstrip().splitlines() if line.strip()]
    if not lines:
        return "", False
    match = _TERMINAL_LINE_RE.fullmatch(lines[-1])
    if not match:
        return "", False
    answer = match.group(1).strip()
    return answer, True


def extract_content_answer(response: str, output_protocol: str = "terminal") -> Tuple[str, bool]:
    """Extract chess content without requiring the requested wrapper format."""
    answer, ok = extract_protocol_answer(response, output_protocol)
    if ok:
        return answer, True

    text = (response or "").strip()
    # Try the other structured protocol, then common boxed-answer output.
    other = "json" if output_protocol == "terminal" else "terminal"
    answer, ok = extract_protocol_answer(text, other)
    if ok:
        return answer, True
    # A correctly marked answer followed by extra prose has valid content but
    # violates the terminal-line protocol.
    marked = list(_TERMINAL_RE.finditer(text))
    if marked:
        return marked[-1].group(1).strip().strip("*").strip(), True
    # Likewise, recover an answer from a JSON object with extra fields for the
    # content score while keeping strict JSON compliance false.
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        obj = None
    if isinstance(obj, dict) and isinstance(obj.get("answer"), (str, int, float)):
        return str(obj["answer"]).strip(), True
    boxed = list(_BOXED_RE.finditer(text))
    if boxed:
        return boxed[-1].group(1).strip(), True

    # A bare final line is valid chess content but is not format-compliant.
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", False
    candidate = re.sub(r"^(?:answer|final answer)\s*:\s*", "", lines[-1], flags=re.IGNORECASE)
    candidate = candidate.strip().strip("*").strip("`").strip()
    return (candidate, bool(candidate))


def _metadata(task: Mapping[str, Any]) -> Dict[str, Any]:
    metadata = task.get("metadata")
    if metadata is None:
        metadata = task.get("metadata_json")
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    return dict(metadata) if isinstance(metadata, Mapping) else {}


def legal_mating_moves(value: str) -> List[str]:
    board = _board(value)
    moves: List[str] = []
    for move in list(board.legal_moves):
        board.push(move)
        if board.is_checkmate():
            moves.append(move.uci())
        board.pop()
    return sorted(moves)


def legal_double_check_moves(value: str) -> List[str]:
    """Enumerate full UCI moves that leave at least two checking pieces."""
    board = _board(value)
    moves: List[str] = []
    for move in list(board.legal_moves):
        board.push(move)
        if board.is_check() and len(board.checkers()) >= 2:
            moves.append(move.uci())
        board.pop()
    return sorted(moves)


def _canonical_multi_item(task_type: str, item: str) -> str:
    rendered = item.strip().lower()
    if task_type == "motifs_battery" and ">" in rendered:
        rendered = ">".join(sorted(part.strip() for part in rendered.split(">")))
    return rendered


def _expected_multi_items(task: Mapping[str, Any]) -> set[str]:
    task_type = str(task.get("task_type", ""))
    if task_type == "motifs_double_check":
        try:
            moves = legal_double_check_moves(str(task.get("input", "")))
        except ValueError:
            moves = []
        if moves:
            return set(moves)
    return {
        _canonical_multi_item(task_type, item)
        for item in str(task.get("correct_answer", "")).split(",")
        if item.strip()
    }


def _normalize_single_answer(task: Mapping[str, Any], answer: str) -> str:
    """Normalize a single answer without discarding semantic FEN case."""
    rendered = answer.strip()
    task_type = str(task.get("task_type", ""))
    if task_type.startswith("structural_state_tracking"):
        return rendered
    return rendered.lower()


def accepted_answers(task: Mapping[str, Any]) -> List[str]:
    """Return accepted answers, expanding all mate-in-one solutions."""
    if str(task.get("answer_type", "single")) != "single":
        return [", ".join(sorted(_expected_multi_items(task)))]
    metadata = _metadata(task)
    configured = metadata.get("acceptable_answers") or task.get("acceptable_answers")
    answers = [str(task.get("correct_answer", ""))]
    if isinstance(configured, Sequence) and not isinstance(configured, str):
        answers.extend(str(item) for item in configured)

    themes = metadata.get("themes") or []
    task_type = str(task.get("task_type", ""))
    if "mateIn1" in themes or task_type.endswith("mateIn1"):
        try:
            answers.extend(legal_mating_moves(str(task.get("input", ""))))
        except ValueError:
            pass
    return sorted({_normalize_single_answer(task, answer) for answer in answers if answer.strip()})


def compare_answer(task: Mapping[str, Any], extracted: str) -> Tuple[bool, str]:
    answer_type = str(task.get("answer_type", "single"))
    normalized = _normalize_single_answer(task, extracted)
    if answer_type == "single":
        return (normalized in accepted_answers(task), "wrong_answer")

    task_type = str(task.get("task_type", ""))
    predicted = {
        _canonical_multi_item(task_type, item)
        for item in extracted.split(",") if item.strip()
    }
    expected = _expected_multi_items(task)
    if predicted == expected:
        return True, "correct"
    extra = predicted - expected
    missing = expected - predicted
    if extra and not missing:
        return False, "multi_extra_items"
    if missing and not extra:
        return False, "multi_missing_items"
    return False, "multi_false_items"


def score_response(
    task: Mapping[str, Any],
    response: str,
    usage: Mapping[str, Any] | None,
    max_tokens: int,
    output_protocol: str = "terminal",
) -> Dict[str, Any]:
    """Return strict score, content score, compliance, and error taxonomy."""
    usage = usage or {}
    extracted, compliant = extract_protocol_answer(response, output_protocol)
    content_extracted, content_parseable = extract_content_answer(response, output_protocol)
    content_correct = compare_answer(task, content_extracted)[0] if content_parseable else False

    completion_tokens = int(usage.get("completion_tokens", 0) or 0)
    if max_tokens and completion_tokens >= max_tokens * 0.98:
        strict_correct, error_type = False, "max_token_reached"
    elif not compliant:
        strict_correct, error_type = False, "format_error"
    else:
        strict_correct, detail = compare_answer(task, extracted)
        error_type = "correct" if strict_correct else detail

    return {
        "extracted": extracted,
        "extraction_successful": compliant,
        "format_compliant": compliant,
        "is_correct": strict_correct,
        "error_type": error_type,
        "content_extracted": content_extracted,
        "content_extraction_successful": content_parseable,
        "content_correct": content_correct,
        "accepted_answers": accepted_answers(task),
    }


def aggregate_scores(results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Compute micro, task-macro, category-macro, format, and content scores."""
    rows = list(results)
    if not rows:
        return {
            "item_micro_accuracy": 0.0,
            "task_macro_accuracy": 0.0,
            "category_macro_accuracy": 0.0,
            "format_compliance": 0.0,
            "content_accuracy": 0.0,
            "content_accuracy_given_parse": 0.0,
        }
    task_values: Dict[str, List[bool]] = defaultdict(list)
    category_values: Dict[str, List[bool]] = defaultdict(list)
    strict: List[bool] = []
    content: List[bool] = []
    parsed_content: List[bool] = []
    compliant: List[bool] = []
    for row in rows:
        inference = row.get("inference", {}) or {}
        value = bool(inference.get("is_correct", False))
        strict.append(value)
        content_value = bool(inference.get("content_correct", value))
        content.append(content_value)
        parseable = bool(inference.get("content_extraction_successful", inference.get("extraction_successful", False)))
        if parseable:
            parsed_content.append(content_value)
        compliant.append(bool(inference.get("format_compliant", inference.get("extraction_successful", False))))
        task_values[str(row.get("task_type", "unknown"))].append(value)
        category = str(row.get("task_category", "unknown")).replace("Judgement", "Judgment")
        category_values[category].append(value)
    return {
        "item_micro_accuracy": mean(strict),
        "task_macro_accuracy": mean(mean(values) for values in task_values.values()),
        "category_macro_accuracy": mean(mean(values) for values in category_values.values()),
        "format_compliance": mean(compliant),
        "content_accuracy": mean(content),
        "content_accuracy_given_parse": mean(parsed_content) if parsed_content else 0.0,
        "n_tasks": len(task_values),
        "n_categories": len(category_values),
    }


__all__ = [
    "OUTPUT_PROTOCOLS",
    "PROMPT_STYLES",
    "REPRESENTATIONS",
    "accepted_answers",
    "aggregate_scores",
    "ascii_board",
    "clean_fen",
    "compare_answer",
    "extract_content_answer",
    "extract_protocol_answer",
    "format_task_prompt",
    "legal_mating_moves",
    "legal_double_check_moves",
    "piece_context",
    "piece_list",
    "repair_legacy_task_wording",
    "score_response",
]
