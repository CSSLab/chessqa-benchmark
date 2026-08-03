import chess
import chess.engine
import json
import random
import zstandard as zstd
import tqdm
import argparse
import hashlib
from typing import Dict, List, Tuple, Optional, Any, Set
from collections import defaultdict
from utils import (
    seed_everything,
    save_tasks,
    construct_prompt,
    FORMAT_EXAMPLES_CENTIPAWN,
    ChessQuestionAnsweringTask
)

EVAL_CATEGORIES = {
    "neutral": {"min": -50, "max": 50, "name": "neutral"},
    "winning": {"min": 350, "max": 450, "name": "winning"},
    "advantage": {"min": 150, "max": 250, "name": "advantage"},
    "disadvantage": {"min": -250, "max": -150, "name": "disadvantage"},
    "losing": {"min": -450, "max": -350, "name": "losing"}
}

OPTION_CATEGORIES = {
    "-400": "losing",
    "-200": "disadvantage",
    "0": "neutral",
    "200": "advantage",
    "400": "winning",
}


def _get_position_hash(fen: str) -> str:
    position_part = " ".join(fen.split()[:4])
    return hashlib.md5(position_part.encode()).hexdigest()[:16]

def _determine_evaluation_category(centipawns: int) -> Optional[str]:
    for category, bounds in EVAL_CATEGORIES.items():
        if bounds["min"] <= centipawns <= bounds["max"]:
            return category
    return None

def _find_best_option_set(target_eval: int) -> List[int]:
    # Use fixed options for all judgment tasks
    return [-400, -200, 0, 200, 400]

def _get_correct_answer(target_eval: int, options: List[int]) -> str:
    closest_option = min(options, key=lambda x: abs(x - target_eval))
    return str(closest_option)


def _parse_evaluation_line(line: str) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(line)
        fen = data.get('fen', '')
        evals = data.get('evals', [])

        if not fen or not evals:
            return None

        best_eval = max(evals, key=lambda e: e.get('depth', 0))
        depth = best_eval.get('depth', 0)
        knodes = best_eval.get('knodes', 0)

        pvs = best_eval.get('pvs', [])
        if not pvs:
            return None

        best_pv = pvs[0]
        centipawns = best_pv.get('cp', 0)
        line_moves = best_pv.get('line', '')

        if 'mate' in best_pv:
            return None

        position_id = _get_position_hash(fen)

        return {
            'fen': fen,
            'best_evaluation': centipawns,
            'depth': depth,
            'knodes': knodes,
            'best_line': line_moves,
            'position_id': position_id,
            'selection_provenance': 'lichess_evaluations',
        }

    except (ValueError, KeyError):
        return None

def _load_evaluations(data_path: str, max_evaluations: Optional[int] = None) -> List[Dict[str, Any]]:
    evaluations = []
    evaluations_loaded = 0

    try:
        with open(data_path, 'rb') as f:
            dctx = zstd.ZstdDecompressor()
            chunk_size = 1024 * 1024
            buffer = ""

            with dctx.stream_reader(f) as reader:
                while True:
                    chunk = reader.read(chunk_size)
                    if not chunk:
                        break

                    buffer += chunk.decode('utf-8')
                    lines = buffer.split('\n')
                    buffer = lines[-1]

                    lines_to_process = lines[:-1]
                    if max_evaluations:
                        remaining_needed = max_evaluations - evaluations_loaded
                        if remaining_needed <= 0:
                            break
                        lines_to_process = lines_to_process[:remaining_needed * 2]

                    for line in tqdm.tqdm(lines_to_process, desc=f"Processing evaluations (loaded: {evaluations_loaded})", leave=False):
                        if max_evaluations and evaluations_loaded >= max_evaluations:
                            break

                        evaluation = _parse_evaluation_line(line)
                        if evaluation:
                            evaluations.append(evaluation)
                            evaluations_loaded += 1

                    if max_evaluations and evaluations_loaded >= max_evaluations:
                        break

                if buffer.strip() and (not max_evaluations or evaluations_loaded < max_evaluations):
                    evaluation = _parse_evaluation_line(buffer)
                    if evaluation:
                        evaluations.append(evaluation)
                        evaluations_loaded += 1

    except Exception as e:
        raise Exception(f"Error loading evaluations: {e}")

    return evaluations


def _evaluate_with_fixed_engine(evaluations: List[Dict[str, Any]], cfg) -> List[Dict[str, Any]]:
    """Produce reference labels with one declared Stockfish configuration."""
    fixed = []
    engine = chess.engine.SimpleEngine.popen_uci(cfg.stockfish_path)
    try:
        options = {}
        if cfg.stockfish_threads:
            options['Threads'] = cfg.stockfish_threads
        if cfg.stockfish_hash_mb:
            options['Hash'] = cfg.stockfish_hash_mb
        if options:
            engine.configure(options)
        engine_name = engine.id.get('name', 'Stockfish')
        with open(cfg.stockfish_path, 'rb') as engine_binary:
            engine_binary_sha256 = hashlib.sha256(engine_binary.read()).hexdigest()

        for evaluation in tqdm.tqdm(evaluations, desc="Fixed-engine evaluation"):
            board = chess.Board(evaluation['fen'])
            info = engine.analyse(board, chess.engine.Limit(nodes=cfg.stockfish_nodes))
            score = info['score'].pov(chess.WHITE)
            centipawns = score.score(mate_score=100000)
            if centipawns is None:
                continue
            updated = dict(evaluation)
            updated.update({
                'best_evaluation': int(centipawns),
                'mate_in': score.mate(),
                'depth': info.get('depth'),
                'seldepth': info.get('seldepth'),
                'nodes': info.get('nodes'),
                'knodes': (info.get('nodes') or 0) // 1000,
                'best_line': ' '.join(move.uci() for move in info.get('pv', [])),
                'label_provenance': 'stockfish_fixed_nodes',
                'engine_name': engine_name,
                'engine_binary_sha256': engine_binary_sha256,
                'engine_nodes': cfg.stockfish_nodes,
                'engine_threads': cfg.stockfish_threads,
                'engine_hash_mb': cfg.stockfish_hash_mb,
            })
            fixed.append(updated)
    finally:
        engine.quit()
    return fixed

def generate_centipawn_eval_task(evaluation: Dict[str, Any], category: str, found_counter: Dict[str, int]) -> ChessQuestionAnsweringTask:
    task_id = f"position_judgment_{category}_{found_counter[category]:04d}"

    option_values = _find_best_option_set(evaluation['best_evaluation'])
    correct_answer = _get_correct_answer(evaluation['best_evaluation'], option_values)

    prefix = f"You are analyzing a chess position in FEN: {evaluation['fen']}.\n"
    task_description = "Estimate the reference Stockfish-family evaluation in centipawns (from White's perspective). Think deeper about this position: do not just evaluate the current board state. Consider the most likely moves for both sides and how the centipawn evaluation would change as the position develops. Analyze several moves ahead: what does the future of this position look like? How would a strong engine assess this position after calculating many moves deep?\n"
    suffix = f"Choose the closest evaluation from the following options: {', '.join(map(str, option_values))}.\nExample final answer: FORMAT_EXAMPLE_PLACEHOLDER"

    return ChessQuestionAnsweringTask(
        task_id=task_id,
        task_type=f"position_judgment_{category}",
        task_category="Position Judgment",
        input=evaluation['fen'],
        question=construct_prompt(prefix, task_description, suffix),
        format_examples=FORMAT_EXAMPLES_CENTIPAWN,
        correct_answer=correct_answer,
        answer_type="single",
        metadata={
            "position_id": evaluation['position_id'],
            "actual_evaluation": evaluation['best_evaluation'],
            "depth": evaluation['depth'],
            "knodes": evaluation['knodes'],
            "best_line": evaluation['best_line'],
            "category": category,
            "selection_category": category,
            "label_category": OPTION_CATEGORIES[correct_answer],
            "options": option_values,
            "selection_provenance": evaluation.get('selection_provenance'),
            "label_provenance": evaluation.get('label_provenance'),
            "engine_name": evaluation.get('engine_name'),
            "engine_binary_sha256": evaluation.get('engine_binary_sha256'),
            "engine_nodes": evaluation.get('engine_nodes'),
            "engine_threads": evaluation.get('engine_threads'),
            "engine_hash_mb": evaluation.get('engine_hash_mb'),
            "mate_in": evaluation.get('mate_in'),
            "seldepth": evaluation.get('seldepth'),
            "nodes": evaluation.get('nodes'),
        }
    )

def find_centipawn_eval_tasks(cfg):
    found_counter = {category: 0 for category in EVAL_CATEGORIES.keys()}
    found = []
    position_hashes = set()

    evaluations = _load_evaluations(cfg.data_path, cfg.max_evaluations if cfg.max_evaluations > 0 else None)
    eval_buckets = defaultdict(list)

    for evaluation in tqdm.tqdm(evaluations, desc="Categorizing evaluations"):
        if evaluation['position_id'] in position_hashes:
            continue
        position_hashes.add(evaluation['position_id'])

        category = _determine_evaluation_category(evaluation['best_evaluation'])
        if category is None:
            continue
        eval_buckets[category].append(evaluation)

    selected = []
    for category in EVAL_CATEGORIES:
        if len(eval_buckets[category]) < cfg.tasks_per_category:
            raise ValueError(
                f"Need {cfg.tasks_per_category} candidates for {category}, "
                f"found {len(eval_buckets[category])}"
            )
    for category, evaluations in eval_buckets.items():
        if not evaluations:
            continue

        random.shuffle(evaluations)
        selected_evaluations = evaluations[:cfg.tasks_per_category]
        for evaluation in selected_evaluations:
            candidate = dict(evaluation)
            candidate['selection_category'] = category
            selected.append(candidate)

    fixed_evaluations = _evaluate_with_fixed_engine(selected, cfg)
    if len(fixed_evaluations) != len(selected):
        raise ValueError("The fixed engine did not return a score for every selected position")

    for evaluation in fixed_evaluations:
        category = evaluation['selection_category']
        task = generate_centipawn_eval_task(evaluation, category, found_counter)
        found_counter[category] += 1
        found.append(task)
        print(f"Found judgment task in category {category}, total found: {found_counter}")

    return found

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="../../data/raw/lichess_db_eval.jsonl.zst")
    parser.add_argument("--output_root", type=str, default="../../data/benchmark")
    parser.add_argument("--max_evaluations", type=int, default=10000)
    parser.add_argument("--tasks_per_category", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stockfish_path", type=str, required=True,
                        help="Pinned Stockfish binary used to generate every reference label")
    parser.add_argument("--stockfish_nodes", type=int, default=100000)
    parser.add_argument("--stockfish_threads", type=int, default=1)
    parser.add_argument("--stockfish_hash_mb", type=int, default=128)
    return parser.parse_args()

def main():
    cfg = parse_args()
    seed_everything(cfg.seed)
    found = find_centipawn_eval_tasks(cfg)
    save_tasks(found, "position_judgement.jsonl", cfg)

if __name__ == "__main__":
    main()
