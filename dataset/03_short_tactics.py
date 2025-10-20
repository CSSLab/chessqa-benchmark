import json
import tqdm
import argparse
import pdb
from utils import (
    seed_everything, 
    read_puzzles, 
    make_pre_move,
    save_tasks,
    construct_prompt,
    FORMAT_EXAMPLES_UCI_MOVE,
    ChessQuestionAnsweringTask
)


def rating_to_level(rating):
    
    if rating <= 999:
        return "beginner"
    elif rating <= 1499:
        return "intermediate"
    elif rating <= 1999:
        return "advanced"
    else:
        return "expert"


def find_puzzles_by_rating(unique_puzzles, data, cfg):
    
    found = []
    found_counter = {"beginner": 0, "intermediate": 0, "advanced": 0, "expert": 0}
    
    for _, row in tqdm.tqdm(data.iterrows()):
        
        rating_deviation = row['RatingDeviation']
        popularity = row['Popularity']
        pv_length = len(row['Moves'].split(' '))
        
        if (rating_deviation > cfg.max_rating_deviation or
            popularity < cfg.min_popularity or
            pv_length > cfg.max_pv_length):
            continue
        
        puzzle_id = row['PuzzleId']
        if puzzle_id in unique_puzzles:
            continue
        
        fen, move = make_pre_move(row)
        task_name = rating_to_level(row['Rating'])
        
        if found_counter[task_name] < cfg.N_sample_rating:
            prefix = f"You are given a chess position in FEN: {fen}.\n"
            task_description = "Find the best move for the side to play.\n"
            suffix = "Use UCI notation (e.g., FORMAT_EXAMPLE_PLACEHOLDER) for the final answer."
            
            sample = ChessQuestionAnsweringTask(
                task_id = f"short_tactics_rating_{task_name}_{found_counter[task_name]:04d}",
                task_type = f"short_tactics_rating_{task_name}",
                task_category = "Short Tactics",
                input = fen,
                question = construct_prompt(prefix, task_description, suffix),
                format_examples = FORMAT_EXAMPLES_UCI_MOVE,
                correct_answer = move,
                answer_type="single",
                metadata = {
                    "puzzle_id": puzzle_id,
                    "rating": row['Rating'],
                    "themes": row['Themes'].split(' '),
                    "rating_deviation": rating_deviation,
                    "popularity": popularity,
                    "pv_length": pv_length,
                    "pv": row['Moves']
                }
            )
            found_counter[task_name] = found_counter.get(task_name, 0) + 1
            found.append(sample)
            unique_puzzles.add(puzzle_id)

        if all(count >= cfg.N_sample_rating for count in found_counter.values()):
            break
    
    print(found_counter, flush=True)
    
    return found, unique_puzzles
    

def find_puzzles_by_theme(unique_puzzles, data, cfg):
    
    with open(cfg.all_themes_path, 'r') as f:
        all_themes_to_include = set(json.load(f))
    
    all_themes_to_include = set(all_themes_to_include)
    found_counter = {theme: 0 for theme in all_themes_to_include}
    found = []
    
    for _, row in tqdm.tqdm(data.iterrows()):
        
        rating_deviation = row['RatingDeviation']
        popularity = row['Popularity']
        pv_length = len(row['Moves'].split(' '))
        
        if (rating_deviation > cfg.max_rating_deviation or
            popularity < cfg.min_popularity or
            pv_length > cfg.max_pv_length):
            continue
        
        puzzle_id = row['PuzzleId']
        if puzzle_id in unique_puzzles:
            continue
        
        fen, move = make_pre_move(row)
        puzzle_themes = set(row['Themes'].split(' '))
        intersecting_themes = puzzle_themes.intersection(all_themes_to_include)
        if not intersecting_themes:
            continue
        
        for primary_theme in intersecting_themes:
            if found_counter[primary_theme] < cfg.N_sample_theme:
                prefix = f"You are given a chess position in FEN: {fen}.\n"
                task_description = "Find the best move for the side to play.\n"
                suffix = "Use UCI notation (e.g., FORMAT_EXAMPLE_PLACEHOLDER) for the final answer."
                
                sample = ChessQuestionAnsweringTask(
                    task_id = f"short_tactics_theme_{primary_theme}_{found_counter[primary_theme]:04d}",
                    task_type = f"short_tactics_theme_{primary_theme}",
                    task_category = "Short Tactics",
                    input = fen,
                    question = construct_prompt(prefix, task_description, suffix),
                    format_examples = FORMAT_EXAMPLES_UCI_MOVE,
                    correct_answer = move,
                    answer_type="single",
                    metadata = {
                        "puzzle_id": puzzle_id,
                        "rating": row['Rating'],
                        "themes": row['Themes'].split(' '),
                        "primary_theme": primary_theme,
                        "rating_deviation": rating_deviation,
                        "popularity": popularity,
                        "pv_length": pv_length,
                        "pv": row['Moves']
                    }
                )
                found_counter[primary_theme] = found_counter.get(primary_theme, 0) + 1
                found.append(sample)
                unique_puzzles.add(puzzle_id)
                break

        if all(count >= cfg.N_sample_theme for count in found_counter.values()):
            break

    print(found_counter, flush=True)
    
    return found


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle_path", type=str, default="../../data/raw/lichess_db_puzzle.csv")
    parser.add_argument("--all_themes_path", type=str, default="../../data/info/all_themes_to_include.json")
    parser.add_argument("--output_root", type=str, default="../../data/benchmark")
    parser.add_argument("--N_sample_rating", type=int, default=100)
    parser.add_argument("--N_sample_theme", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_popularity", type=int, default=90)
    parser.add_argument("--max_pv_length", type=int, default=4)
    parser.add_argument("--max_rating_deviation", type=int, default=100)
    
    return parser.parse_args()


def main():

    cfg = parse_args()
    seed_everything(cfg.seed)
    data = read_puzzles(cfg.puzzle_path)
    
    unique_puzzles = set()
    found_rating, unique_puzzles = find_puzzles_by_rating(unique_puzzles, data, cfg)
    found_theme = find_puzzles_by_theme(unique_puzzles, data, cfg)
    found = found_rating + found_theme
    
    save_tasks(found, "short_tactics.jsonl", cfg)


if __name__ == "__main__":
    main()
