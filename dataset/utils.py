import chess
import random
import os
import numpy as np
import torch
import pandas as pd
import pdb
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional
import json

FORMAT_EXAMPLES_UCI_MOVE = ["e2e4, c2b1q",
                            "g1f3, a2a1q"]
FORMAT_EXAMPLES_UCI_MOVE_SAME_START = ["e2e3, e2e4",
                                       "d7d5, d7d6"]
FORMAT_EXAMPLES_LINE = ["d1>d7>d8, a2>e2>h2", 
                        "a5>e5>h5, h1>h4>h7"]
FORMAT_EXAMPLES_FORK = ["e5>e7-f6, f3>e1-g1", 
                        "e4>b1-g2, c4>b2-d6"]
FORMAT_EXAMPLES_BATTERY = ["b2>e5>h8, h1>h7", 
                           "h1>h4, a1>a4>a8"]
FORMAT_EXAMPLES_FEN = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", 
                       "r2qr1k1/1b1p1ppp/p2Q1n2/1p6/8/1BN2n2/PPP2PPP/R1B1R1K1 w - - 1 15"]
FORMAT_EXAMPLES_CENTIPAWN = ["400", 
                             "-200"]
FORMAT_EXAMPLES_SQUARES = ["e4, f5", 
                           "a1, b2, c3"]
FORMAT_EXAMPLES_PIECE = ["White Queen at e5", 
                         "Black Rook at d5"]
FORMAT_EXAMPLES_ARRANGEMENT = ["White King: ['e1'], White Queen: ['d1'], White Rook: ['a1', 'h1'], White Bishop: ['c1', 'f1'], White Knight: ['b1', 'g1'], White Pawn: ['a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2'], Black King: ['e8'], Black Queen: ['d8'], Black Rook: ['a8', 'h8'], Black Bishop: ['c8', 'f8'], Black Knight: ['b8', 'g8'], Black Pawn: ['a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7']",
                               "White King: ['g1'], White Queen: ['d6'], White Rook: ['a1', 'e1'], White Bishop: ['b3', 'c1'], White Knight: ['c3'], White Pawn: ['a2', 'b2', 'c2', 'f2', 'g2', 'h2'], Black King: ['g8'], Black Queen: ['d8'], Black Rook: ['a8', 'e8'], Black Bishop: ['b7'], Black Knight: ['f3', 'f6'], Black Pawn: ['a6', 'b5', 'd7', 'f7', 'g7', 'h7']"]
FORMAT_EXAMPLES_MCQ = ["A", "D"]


PIECE_NAMES = {
    (chess.PAWN, chess.WHITE): "White Pawn",
    (chess.KNIGHT, chess.WHITE): "White Knight", 
    (chess.BISHOP, chess.WHITE): "White Bishop",
    (chess.ROOK, chess.WHITE): "White Rook",
    (chess.QUEEN, chess.WHITE): "White Queen",
    (chess.KING, chess.WHITE): "White King",
    (chess.PAWN, chess.BLACK): "Black Pawn",
    (chess.KNIGHT, chess.BLACK): "Black Knight",
    (chess.BISHOP, chess.BLACK): "Black Bishop", 
    (chess.ROOK, chess.BLACK): "Black Rook",
    (chess.QUEEN, chess.BLACK): "Black Queen",
    (chess.KING, chess.BLACK): "Black King",
}


def get_piece_arrangement(fen):
    """Get the complete piece arrangement from a FEN string.

    Returns pieces in order: White pieces first (King, Queen, Rook, Bishop, Knight, Pawn)
    followed by Black pieces (King, Queen, Rook, Bishop, Knight, Pawn).
    Squares are listed in alphabetical order for each piece type.
    """
    board = chess.Board(fen)

    # Define piece order
    piece_order = ["King", "Queen", "Rook", "Bishop", "Knight", "Pawn"]
    colors = ["White", "Black"]

    pieces = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            color = "White" if piece.color == chess.WHITE else "Black"
            names = {1: "Pawn", 2: "Knight", 3: "Bishop", 4: "Rook", 5: "Queen", 6: "King"}
            key = f"{color} {names[piece.piece_type]}"
            if key not in pieces:
                pieces[key] = []
            pieces[key].append(chess.square_name(square))

    # Sort squares alphabetically for each piece type
    for key in pieces:
        pieces[key].sort()

    # Build arrangement string in specified order
    arrangement_parts = []
    for color in colors:
        for piece_type in piece_order:
            key = f"{color} {piece_type}"
            if key in pieces:
                arrangement_parts.append(f"{key}: {pieces[key]}")

    return ", ".join(arrangement_parts)


@dataclass
class ChessQuestionAnsweringTask:
    task_id: str
    task_type: str
    task_category: str
    input: str
    question: str
    format_examples: List[str]
    correct_answer: str
    answer_type: str
    metadata: Optional[dict]


def construct_prompt(prefix, task_description, suffix):
    
    question = prefix
    question += f"CONTEXT_PLACEHOLDER"
    question += task_description
    question += "Analyze step by step and explain your reasoning.\n"
    question += "Finish with a single line formatted EXACTLY as:\n"
    question += "FINAL ANSWER: <answer>\n"
    question += suffix
    
    return question


def make_pre_move(row: pd.Series) -> Tuple[str, str]:
    
    fen_before = row['FEN']
    moves = row['Moves'].split(' ')
    board = chess.Board(fen_before)
    pre_move = chess.Move.from_uci(moves[0])
    board.push(pre_move)
    fen_after = board.fen()
    
    return fen_after, moves[1]


def save_tasks(tasks, file_name, cfg):
    
    if not os.path.exists(cfg.output_root):
        os.makedirs(cfg.output_root)
    output_path = os.path.join(cfg.output_root, file_name)
    tasks_data = [asdict(task) for task in tasks]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for task in tasks_data:
            f.write(json.dumps(task, ensure_ascii=False) + '\n')


def get_piece_name(piece: chess.Piece) -> str:
    color_name = "White" if piece.color == chess.WHITE else "Black"
    piece_names = {
        chess.PAWN: "Pawn", chess.KNIGHT: "Knight", chess.BISHOP: "Bishop",
        chess.ROOK: "Rook", chess.QUEEN: "Queen", chess.KING: "King"
    }
    piece_name = piece_names[piece.piece_type]
    return f"{color_name} {piece_name}"


def fen_to_pieces(fen):

    board = chess.Board(fen)
    pieces = {}
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_key = PIECE_NAMES[(piece.piece_type, piece.color)]
            square_name = chess.square_name(square)

            if piece_key not in pieces:
                pieces[piece_key] = []
            pieces[piece_key].append(square_name)

    return pieces


def seed_everything(seed):

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def readable_num(num):
    if num >= 1e9:
        return f'{num / 1e9:.2f}B'
    elif num >= 1e6:
        return f'{num / 1e6:.2f}M'
    elif num >= 1e3:
        return f'{num / 1e3:.2f}K'
    else:
        return str(num)


def readable_time(elapsed_time):
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)

    if hours > 0:
        return f"{int(hours)}h {int(minutes)}m {seconds:.2f}s"
    elif minutes > 0:
        return f"{int(minutes)}m {seconds:.2f}s"
    else:
        return f"{seconds:.2f}s"



def read_puzzles(file_path):
    
    t_0 = time.time()
    df = pd.read_csv(file_path)
    t_1 = time.time()
    df = df.sample(frac=1).reset_index(drop=True)
    print(f"Time to read: {readable_time(t_1 - t_0)}", flush=True)
    print(f"Time to shuffle: {readable_time(time.time() - t_1)}", flush=True)
    
    return df
    
    