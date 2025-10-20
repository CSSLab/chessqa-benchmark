import tqdm
import pdb
import pandas as pd
from utils import read_puzzles
import argparse
from pathlib import Path
import json
import os

def find_all_themes(data, cfg):
    
    all_themes = {}
    for _, row in tqdm.tqdm(data.iterrows()):
        themes = row['Themes']
        if themes:
            themes_list = themes.split(' ')
            for theme in themes_list:
                all_themes[theme] = all_themes.get(theme, 0) + 1

    with open(os.path.join(cfg.output_root, 'info', 'all_themes.json'), 'w') as f:
        json.dump(all_themes, f, indent=4)
    
    return all_themes


def parse_args():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--puzzle_path", type=Path, default="../../data/raw/lichess_db_puzzle.csv")
    parser.add_argument("--output_root", type=Path, default="chess-llm-benchmark/data")
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def main():

    cfg = parse_args()
    puzzle_data = read_puzzles(cfg.puzzle_path)
    all_themes = find_all_themes(puzzle_data, cfg)
    print(f"Found {len(all_themes)} unique themes.")
    

if __name__ == "__main__":
    main()
    