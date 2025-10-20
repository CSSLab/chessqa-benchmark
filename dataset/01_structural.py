import chess
import chess.pgn
import json
import tqdm
import argparse
import random
from typing import List, Tuple, Optional, Dict, Any, Iterator
from utils import (
    seed_everything,
    read_puzzles,
    save_tasks,
    construct_prompt,
    get_piece_name,
    get_piece_arrangement,
    FORMAT_EXAMPLES_UCI_MOVE,
    FORMAT_EXAMPLES_SQUARES,
    FORMAT_EXAMPLES_PIECE,
    FORMAT_EXAMPLES_ARRANGEMENT,
    FORMAT_EXAMPLES_FEN,
    FORMAT_EXAMPLES_UCI_MOVE_SAME_START,
    ChessQuestionAnsweringTask
)

def detect_piece_arrangement(board: chess.Board) -> str:
    """Return the full piece arrangement of the board."""
    return get_piece_arrangement(board.fen())


def detect_legal_moves_piece(board: chess.Board, target_square: str) -> List[str]:
    legal_moves_per_square = {}

    for move in board.legal_moves:
        from_square = chess.square_name(move.from_square)
        if from_square not in legal_moves_per_square:
            legal_moves_per_square[from_square] = []
        legal_moves_per_square[from_square].append(move.uci())

    return legal_moves_per_square.get(target_square, [])


def detect_check_detection(board: chess.Board) -> List[str]:
    if not board.is_check():
        return []

    king_square = board.king(board.turn)
    if king_square is None:
        return []

    attackers = board.attackers(not board.turn, king_square)
    checking_pieces = []

    for attacker_square in attackers:
        piece = board.piece_at(attacker_square)
        if piece:
            piece_name = get_piece_name(piece)
            square_name = chess.square_name(attacker_square)
            checking_pieces.append(f"{piece_name} at {square_name}")

    return checking_pieces


def detect_check_in_1(board: chess.Board) -> List[str]:
    checking_moves = []

    for move in board.legal_moves:
        board_copy = board.copy()
        board_copy.push(move)
        if board_copy.is_check():
            checking_moves.append(move.uci())

    return checking_moves


def is_pinned(board: chess.Board, piece_square: int) -> bool:
    """Check if a piece is pinned to its king."""
    piece = board.piece_at(piece_square)
    if piece is None:
        return False

    king_square = board.king(piece.color)
    if king_square is None:
        return False

    # Check if piece is on same rank, file, or diagonal as king
    attacks_between = chess.SquareSet.between(piece_square, king_square)
    if not attacks_between:
        # Piece not aligned with king
        return False

    # Include the piece square itself in the line
    attacks_between.add(piece_square)

    # Check if there's a sliding piece (rook, bishop, queen) attacking along this line
    enemy_color = not piece.color

    # Check for rook/queen attacks on ranks and files
    if chess.square_rank(piece_square) == chess.square_rank(king_square) or \
       chess.square_file(piece_square) == chess.square_file(king_square):
        for attacker_square in board.pieces(chess.ROOK, enemy_color) | board.pieces(chess.QUEEN, enemy_color):
            if chess.SquareSet.between(attacker_square, king_square) & attacks_between == attacks_between:
                # Check if there are no other pieces between attacker and king
                between_squares = chess.SquareSet.between(attacker_square, king_square)
                between_squares.discard(piece_square)
                if all(board.piece_at(sq) is None for sq in between_squares):
                    return True

    # Check for bishop/queen attacks on diagonals
    if abs(chess.square_rank(piece_square) - chess.square_rank(king_square)) == \
       abs(chess.square_file(piece_square) - chess.square_file(king_square)):
        for attacker_square in board.pieces(chess.BISHOP, enemy_color) | board.pieces(chess.QUEEN, enemy_color):
            if chess.SquareSet.between(attacker_square, king_square) & attacks_between == attacks_between:
                # Check if there are no other pieces between attacker and king
                between_squares = chess.SquareSet.between(attacker_square, king_square)
                between_squares.discard(piece_square)
                if all(board.piece_at(sq) is None for sq in between_squares):
                    return True

    return False


def detect_capture_squares(board: chess.Board, target_square: str) -> List[str]:
    square_index = chess.parse_square(target_square)
    piece = board.piece_at(square_index)

    if piece is None:
        return []

    # If piece is pinned, it cannot capture anything
    if is_pinned(board, square_index):
        return []

    # Get squares this piece attacks
    attacks = board.attacks(square_index)
    capture_squares = []

    for attack_square in attacks:
        attacked_piece = board.piece_at(attack_square)
        if attacked_piece and attacked_piece.color != piece.color:  # Enemy piece
            capture_squares.append(chess.square_name(attack_square))

    return capture_squares


def detect_control_squares(board: chess.Board, target_square: str) -> List[str]:
    square_index = chess.parse_square(target_square)
    piece = board.piece_at(square_index)

    if piece is None:
        return []

    # If piece is pinned, it cannot control any squares
    if is_pinned(board, square_index):
        return []

    attacks = board.attacks(square_index)
    control_squares = []

    for attack_square in attacks:
        attacked_piece = board.piece_at(attack_square)
        if attacked_piece is None:
            control_squares.append(chess.square_name(attack_square))

    return control_squares


def detect_protect_squares(board: chess.Board, target_square: str) -> List[str]:
    square_index = chess.parse_square(target_square)
    piece = board.piece_at(square_index)

    if piece is None:
        return []

    # If piece is pinned, it cannot protect any squares
    if is_pinned(board, square_index):
        return []

    attacks = board.attacks(square_index)
    protect_squares = []

    for attack_square in attacks:
        attacked_piece = board.piece_at(attack_square)
        # Check if it's a friendly piece AND not the king (protecting king means check/checkmate)
        if attacked_piece and attacked_piece.color == piece.color and attacked_piece.piece_type != chess.KING:
            protect_squares.append(chess.square_name(attack_square))

    return protect_squares


def detect_legal_move_all(board: chess.Board) -> List[str]:
    """Return all legal moves in UCI format."""
    legal_moves = []
    for move in board.legal_moves:
        legal_moves.append(move.uci())
    return legal_moves



def generate_piece_arrangement_task(board: chess.Board, found_counter: Dict[str, int], puzzle_id: str) -> ChessQuestionAnsweringTask:
    correct_answer = detect_piece_arrangement(board)

    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    task_description = "Provide the complete piece arrangement of this position. List all pieces with their colors, types, and squares.\n"
    task_description += "Format: 'Color PieceType: [square1, square2, ...]', separated by commas and spaces for different piece types.\n"
    task_description += "List the pieces in the order of White pieces first (King, Queen, Rook, Bishop, Knight, Pawn) followed by Black pieces (King, Queen, Rook, Bishop, Knight, Pawn).\n"
    task_description += "If a piece type has no pieces on the board, skip it in the listing.\n"
    task_description += "List the squares for each piece type in alphabetical order.\n"
    suffix = "Example final answer: FORMAT_EXAMPLE_PLACEHOLDER"

    return ChessQuestionAnsweringTask(
        task_id=f"structural_piece_arrangement_{found_counter['structural_piece_arrangement']:04d}",
        task_type="structural_piece_arrangement",
        task_category="Structural",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_ARRANGEMENT,
        correct_answer=correct_answer,
        answer_type="single",
        metadata={
            "puzzle_id": puzzle_id
        }
    )


def generate_legal_move_piece_task(board: chess.Board, found_counter: Dict[str, int], puzzle_id: str) -> Optional[ChessQuestionAnsweringTask]:
    legal_moves_per_square = {}

    for move in board.legal_moves:
        from_square = chess.square_name(move.from_square)
        if from_square not in legal_moves_per_square:
            legal_moves_per_square[from_square] = []
        legal_moves_per_square[from_square].append(move.uci())

    if not legal_moves_per_square:
        return None

    target_square = random.choice(list(legal_moves_per_square.keys()))
    legal_moves = detect_legal_moves_piece(board, target_square)

    if not legal_moves:
        return None

    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    task_description = f"Find all legal moves for the piece on square {target_square}. List the moves in UCI format, separated by commas and spaces.\n"
    suffix = "Example final answer: FORMAT_EXAMPLE_PLACEHOLDER"

    correct_answer = ", ".join(sorted(legal_moves))

    return ChessQuestionAnsweringTask(
        task_id=f"structural_legal_move_piece_{found_counter['structural_legal_move_piece']:04d}",
        task_type="structural_legal_move_piece",
        task_category="Structural",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_UCI_MOVE_SAME_START,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={
            "puzzle_id": puzzle_id,
            "target_square": target_square,
            "n_moves": len(legal_moves)
        }
    )


def generate_legal_move_all_task(board: chess.Board, found_counter: Dict[str, int], puzzle_id: str) -> Optional[ChessQuestionAnsweringTask]:
    legal_moves = detect_legal_move_all(board)

    if not legal_moves:
        return None

    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    task_description = "Find all legal moves in this position. List the moves in UCI format, separated by commas and spaces.\n"
    suffix = "Example final answer: FORMAT_EXAMPLE_PLACEHOLDER"

    correct_answer = ", ".join(sorted(legal_moves))

    return ChessQuestionAnsweringTask(
        task_id=f"structural_legal_move_all_{found_counter['structural_legal_move_all']:04d}",
        task_type="structural_legal_move_all",
        task_category="Structural",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_UCI_MOVE,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={
            "puzzle_id": puzzle_id,
            "n_moves": len(legal_moves)
        }
    )


def generate_check_detection_task(board: chess.Board, found_counter: Dict[str, int], puzzle_id: str) -> Optional[ChessQuestionAnsweringTask]:
    checking_pieces = detect_check_detection(board)
    if not checking_pieces:
        return None

    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    task_description = "In this position, the side to move is in check. Identify the piece(s) that is delivering the check.\n"
    suffix = "List each checking piece with its color, type, and square (e.g., FORMAT_EXAMPLE_PLACEHOLDER). Separate multiple pieces with commas and spaces if applicable."

    correct_answer = ", ".join(checking_pieces)

    return ChessQuestionAnsweringTask(
        task_id=f"structural_check_detection_{found_counter['structural_check_detection']:04d}",
        task_type="structural_check_detection",
        task_category="Structural",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_PIECE,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={
            "puzzle_id": puzzle_id,
            "n_checkers": len(checking_pieces)
        }
    )


def generate_check_in_1_task(board: chess.Board, found_counter: Dict[str, int], puzzle_id: str) -> Optional[ChessQuestionAnsweringTask]:
    checking_moves = detect_check_in_1(board)
    if not checking_moves:
        return None

    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    task_description = "Find all moves that put the opponent in check. List the moves in UCI format, separated by commas and spaces.\n"
    suffix = "Example final answer: FORMAT_EXAMPLE_PLACEHOLDER"

    correct_answer = ", ".join(sorted(checking_moves))

    return ChessQuestionAnsweringTask(
        task_id=f"structural_check_in_1_{found_counter['structural_check_in_1']:04d}",
        task_type="structural_check_in_1",
        task_category="Structural",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_UCI_MOVE,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={
            "puzzle_id": puzzle_id,
            "n_checking_moves": len(checking_moves)
        }
    )


def generate_capture_squares_task(board: chess.Board, found_counter: Dict[str, int], puzzle_id: str) -> Optional[ChessQuestionAnsweringTask]:
    occupied_squares = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    occupied_squares = [sq for sq in occupied_squares if board.piece_at(sq).piece_type not in [chess.PAWN, chess.KING]]

    if not occupied_squares:
        return None

    target_square_index = random.choice(occupied_squares)
    target_square = chess.square_name(target_square_index)
    piece = board.piece_at(target_square_index)

    capture_squares = detect_capture_squares(board, target_square)

    piece_name = get_piece_name(piece)

    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    # task_description = f"Find all squares that the {piece_name} on {target_square} can capture (reachable squares that have opponent pieces)."
    # task_description += " Exclude captures if the piece is pinned to its king.\n"
    task_description = f"Find all squares that the {piece_name} on {target_square} can capture (i.e. every square that has an opponent piece such that the {piece_name} on {target_square} could legally move to that square and capture the piece).\n"
    task_description += f"Exclude captures if the {piece_name} on {target_square} is pinned to its king and thus cannot move.\n"
    suffix = "Example final answer: FORMAT_EXAMPLE_PLACEHOLDER"

    correct_answer = ", ".join(sorted(capture_squares))
    assert correct_answer, "There should be at least one capture square."

    return ChessQuestionAnsweringTask(
        task_id=f"structural_capture_squares_{found_counter['structural_capture_squares']:04d}",
        task_type="structural_capture_squares",
        task_category="Structural",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_SQUARES,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={
            "puzzle_id": puzzle_id,
            "target_square": target_square,
            "piece": piece_name,
            "n_captures": len(capture_squares)
        }
    )


def generate_control_squares_task(board: chess.Board, found_counter: Dict[str, int], puzzle_id: str) -> Optional[ChessQuestionAnsweringTask]:
    occupied_squares = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    occupied_squares = [sq for sq in occupied_squares if board.piece_at(sq).piece_type not in [chess.PAWN, chess.KING]]

    if not occupied_squares:
        return None

    target_square_index = random.choice(occupied_squares)
    target_square = chess.square_name(target_square_index)
    piece = board.piece_at(target_square_index)

    control_squares = detect_control_squares(board, target_square)

    piece_name = get_piece_name(piece)

    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    # task_description = f"Find all squares that the {piece_name} on {target_square} controls (reachable empty squares)."
    # task_description += " Exclude control if the piece is pinned to its king.\n"
    task_description = f"Find all squares that the {piece_name} on {target_square} controls (i.e. every empty square that the {piece_name} on {target_square} could legally move to, excluding squares occupied by any piece).\n"
    task_description += f"Exclude control if the {piece_name} on {target_square} is pinned to its king and thus cannot move.\n"
    suffix = "Example final answer: FORMAT_EXAMPLE_PLACEHOLDER"

    correct_answer = ", ".join(sorted(control_squares))
    assert correct_answer, "There should be at least one control square."

    return ChessQuestionAnsweringTask(
        task_id=f"structural_control_squares_{found_counter['structural_control_squares']:04d}",
        task_type="structural_control_squares",
        task_category="Structural",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_SQUARES,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={
            "puzzle_id": puzzle_id,
            "target_square": target_square,
            "piece": piece_name,
            "n_controlled": len(control_squares)
        }
    )


def generate_protect_squares_task(board: chess.Board, found_counter: Dict[str, int], puzzle_id: str) -> Optional[ChessQuestionAnsweringTask]:
    occupied_squares = [sq for sq in chess.SQUARES if board.piece_at(sq) is not None]
    occupied_squares = [sq for sq in occupied_squares if board.piece_at(sq).piece_type not in [chess.PAWN, chess.KING]]

    if not occupied_squares:
        return None

    target_square_index = random.choice(occupied_squares)
    target_square = chess.square_name(target_square_index)
    piece = board.piece_at(target_square_index)

    protect_squares = detect_protect_squares(board, target_square)

    piece_name = get_piece_name(piece)
    
    # You are given a chess position in FEN: r4r2/pb2ppkp/1p4p1/2pq4/8/1P1P4/P1PN1PPP/R2Q1RK1 w - - 0 15.
    # CONTEXT_PLACEHOLDERFind all squares that contain pieces that the Black Queen on d5 protects (i.e. every square that contains a piece such that the Black Queen on d5 could legally recapture if an enemy piece captured it, excluding the king since it can't be captured).
    # Exclude protection if Black Queen on d5 is pinned to its king and thus cannot move.
    # Analyze step by step and explain your reasoning.
    # Finish with a single line formatted EXACTLY as:
    # FINAL ANSWER: <answer>
    # Example final answer: FORMAT_EXAMPLE_PLACEHOLDER

    prefix = f"You are given a chess position in FEN: {board.fen()}.\n"
    task_description = f"Find all squares that contain pieces that the {piece_name} on {target_square} protects (i.e. every square that contains a piece such that the {piece_name} on {target_square} could legally recapture if an enemy piece captured it, excluding the king since it can't be captured).\n"
    task_description += f"Exclude protection if the {piece_name} on {target_square} is pinned to its king and thus cannot move.\n"
    suffix = "Example final answer: FORMAT_EXAMPLE_PLACEHOLDER"

    correct_answer = ", ".join(sorted(protect_squares))
    assert correct_answer, "There should be at least one protect square."

    return ChessQuestionAnsweringTask(
        task_id=f"structural_protect_squares_{found_counter['structural_protect_squares']:04d}",
        task_type="structural_protect_squares",
        task_category="Structural",
        input=board.fen(),
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_SQUARES,
        correct_answer=correct_answer,
        answer_type="multi",
        metadata={
            "puzzle_id": puzzle_id,
            "target_square": target_square,
            "piece": piece_name,
            "n_protected": len(protect_squares)
        }
    )


def read_pgn_games(pgn_path: str) -> Iterator[chess.pgn.Game]:
    with open(pgn_path, 'r', encoding='utf-8') as pgn_file:
        while True:
            game = chess.pgn.read_game(pgn_file)
            if game is None:
                break
            yield game


def extract_game_fragment(game: chess.pgn.Game, start_after_moves: int, track_moves: int) -> Tuple[str, List[str], str]:
    board = game.board()
    all_moves = []
    node = game
    while node.variations:
        node = node.variation(0)
        all_moves.append(node.move.uci())

    # Need at least start_after_moves + track_moves total moves
    if len(all_moves) < start_after_moves + track_moves:
        return None, None, None

    # Start tracking after start_after_moves
    temp_board = game.board()
    temp_node = game
    for i in range(start_after_moves):
        temp_node = temp_node.variation(0)
        temp_board.push(temp_node.move)

    start_fen = temp_board.fen()
    moves_uci = all_moves[start_after_moves:start_after_moves + track_moves]

    game_id = f"{game.headers.get('Event', 'unknown')}_{game.headers.get('Round', '1')}_{game.headers.get('White', 'w')}_{game.headers.get('Black', 'b')}"
    game_id = game_id.replace(' ', '_').replace('/', '_')
    return start_fen, moves_uci, game_id


def _apply_uci_moves(start_fen: str, moves_uci: List[str]) -> Tuple[str, List[str]]:
    board = chess.Board(start_fen)
    san_list = []
    for u in moves_uci:
        mv = chess.Move.from_uci(u)
        san_list.append(board.san(mv))
        board.push(mv)
    return board.fen(), san_list


def generate_fen_after_moves_task(game_id: str, start_fen: str, moves: List[str], task_subtype: str, found_counter: dict) -> ChessQuestionAnsweringTask:
    final_fen, san_list = _apply_uci_moves(start_fen, moves)
    moves_str = " ".join(moves)
    prefix = "Given an initial FEN and a sequence of UCI moves, apply the moves in order and output the exact resulting FEN.\n"
    task_description = f"Initial FEN: {start_fen}\nMoves (UCI): {moves_str}\n"
    suffix = "Example final answer: FORMAT_EXAMPLE_PLACEHOLDER"
    return ChessQuestionAnsweringTask(
        task_id=f"structural_state_tracking_{task_subtype}_{found_counter[f'structural_state_tracking_{task_subtype}']:04d}",
        task_type=f"structural_state_tracking_{task_subtype}",
        task_category="Structural",
        input=f"{start_fen} | {moves_str}",
        question=construct_prompt(prefix, task_description, suffix).replace("CONTEXT_PLACEHOLDER", ""),
        format_examples=FORMAT_EXAMPLES_FEN,
        correct_answer=final_fen,
        answer_type="single",
        metadata={"game_id": game_id, "moves_uci": moves, "moves_san": san_list, "track_length": len(moves)}
    )


def find_structural_tasks(unique_positions, data, cfg):
    found_counter = {
        "structural_piece_arrangement": 0,
        "structural_legal_move_piece": 0,
        "structural_legal_move_all": 0,
        "structural_check_detection": 0,
        "structural_check_in_1": 0,
        "structural_capture_squares": 0,
        "structural_protect_squares": 0,
        "structural_control_squares": 0,
        "structural_state_tracking_short": 0,
        "structural_state_tracking_mid": 0,
        "structural_state_tracking_long": 0
    }

    task_generators = {
        "structural_piece_arrangement": generate_piece_arrangement_task,
        "structural_legal_move_piece": generate_legal_move_piece_task,
        "structural_legal_move_all": generate_legal_move_all_task,
        "structural_check_detection": generate_check_detection_task,
        "structural_check_in_1": generate_check_in_1_task,
        "structural_capture_squares": generate_capture_squares_task,
        "structural_protect_squares": generate_protect_squares_task,
        "structural_control_squares": generate_control_squares_task
    }

    found = []
    for _, row in tqdm.tqdm(data.iterrows()):
        puzzle_id = row['PuzzleId']
        if puzzle_id in unique_positions:
            continue

        fen = row['FEN']
        board = chess.Board(fen)

        for task_type, task_generator in task_generators.items():
            if found_counter[task_type] >= cfg.N_sample:
                continue

            try:
                task = task_generator(board, found_counter, puzzle_id)
                if task:
                    found_counter[task_type] += 1
                    found.append(task)
                    unique_positions.add(puzzle_id)
                    print(f"Found {task.task_type} in puzzle {puzzle_id}, total found: {found_counter}")
                    break
            except Exception as e:
                continue

        if all(found_counter[t] >= cfg.N_sample for t in list(task_generators.keys())):
            break


    # State tracking subtasks: short (1-5), mid (6-10), long (11-15) moves
    state_tracking_configs = [
        ("short", 1, 5),
        ("mid", 6, 10),
        ("long", 11, 15)
    ]

    for game in read_pgn_games(cfg.pgn_path):
        # Check if all state tracking subtasks are complete
        if all(found_counter[f"structural_state_tracking_{subtype}"] >= cfg.N_sample for subtype, _, _ in state_tracking_configs):
            break

        for subtype, min_moves, max_moves in state_tracking_configs:
            if found_counter[f"structural_state_tracking_{subtype}"] >= cfg.N_sample:
                continue

            # Try different track lengths within the range
            track_moves = random.randint(min_moves, max_moves)
            start_fen, moves, game_id = extract_game_fragment(game, 30, track_moves)

            if moves:
                task = generate_fen_after_moves_task(game_id, start_fen, moves, subtype, found_counter)
                found_counter[f"structural_state_tracking_{subtype}"] += 1
                found.append(task)
                print(f"Found structural_state_tracking_{subtype} task in game {game_id} (tracking {len(moves)} moves), total found: {found_counter[f'structural_state_tracking_{subtype}']}")
                break  # Move to next game after finding one task

    return found


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle_path", type=str, default="../../data/raw/lichess_db_puzzle.csv")
    parser.add_argument("--pgn_path", type=str, default="../../data/raw/lichess_db_broadcast_2025-04.pgn")
    parser.add_argument("--output_root", type=str, default="../../data/benchmark")
    parser.add_argument("--N_sample", type=int, default=100)
    parser.add_argument("--min_moves", type=int, default=5)
    parser.add_argument("--max_moves", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    cfg = parse_args()
    seed_everything(cfg.seed)

    data = read_puzzles(cfg.puzzle_path)
    unique_positions = set()
    all_found = find_structural_tasks(unique_positions, data, cfg)
    print(f"Found {len(all_found)} total tasks")

    save_tasks(all_found, "structural.jsonl", cfg)


if __name__ == "__main__":
    main()