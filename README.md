# ChessQA: Evaluating Large Language Models for Chess Understanding

ChessQA is a FEN-based, text-only chess QA benchmark with 50 tasks and 3,500 items across Structural (1,100), Motifs (600), Short Tactics (900), Position Judgment (500), and Semantic (400). The categories are complementary diagnostic views, not a difficulty hierarchy.

Prompt wording follows each operational target: Short Tactics asks for the Lichess puzzle's intended first solution move, Position Judgment asks for the reference Stockfish-family bucket, Semantic asks for the comment associated with the displayed record, and detector-defined Motif prompts state the implemented geometric relation directly.

All 3,500 items are included and can be evaluated with the same runner.

The evaluation code provides:

- Structural capture/control/protect labels use the standard attack relation. Pinned pieces retain attacked squares; pin status is audit metadata.
- Every legal mate-in-one move is accepted by enumerating mating moves from the supplied position.
- Motif prompts and scoring apply the absolute-pin definition, retain full UCI promotion suffixes, and canonicalize battery square ordering.
- Accuracy, Protocol compliance, and Content accuracy as separate fields.
- FEN, ASCII-board, and explicit-piece-list inputs can be evaluated under otherwise matched prompts.
- Overall reporting includes item-micro, equal-task macro, and equal-category macro scores.
- Source-balanced scores, source-cluster intervals, and source-disjoint slices.
- Semantic variants carry stable base-example IDs and the relevance judge defaults to the same FEN-before + move context shown to evaluated models.
- Position Judgment can be regenerated with a pinned Stockfish binary and fixed node budget; simple chess baselines are included.

## Repository layout

- `benchmark/`: the complete 3,500-item benchmark in five category files
- `dataset/`: dataset builders
- `eval/`: OpenRouter runner and shared protocol/scoring code
- `analysis/`: overlap/label audits, rank stability, Semantic controls, and Position Judgment baselines
- `tests/`: scoring, prompt, and dataset tests

## Install

```bash
pip install -r requirements.txt
```

Semantic construction additionally needs `sentence-transformers`; its offline cleaning/judging stages need vLLM. Set `OPENROUTER_API_KEY` for inference, or place `openrouter_api_key` in `keys/api_keys.json`.

## Run model evaluation

```bash
OPENROUTER_API_KEY=... python eval/run_openrouter.py \
  --dataset-root benchmark \
  --model anthropic/claude-3.5-haiku \
  --output-dir results \
  --workers 8
```

Controlled variants:

```bash
MODEL_ID=anthropic/claude-3.5-haiku

# Semantically equivalent board representations
python eval/run_openrouter.py --model "$MODEL_ID" --representation fen        --run-id fen-01
python eval/run_openrouter.py --model "$MODEL_ID" --representation ascii      --run-id ascii-01
python eval/run_openrouter.py --model "$MODEL_ID" --representation piece_list --run-id pieces-01

# Structured output and independent repeats
python eval/run_openrouter.py --model "$MODEL_ID" --output-protocol json --run-id json-01
python eval/run_openrouter.py --model "$MODEL_ID" --temperature 0.7 --run-id temp07-01
python eval/run_openrouter.py --model "$MODEL_ID" --prompt-style direct --run-id direct-01

# Provider reasoning setting; set the output ceiling explicitly
python eval/run_openrouter.py --model "$MODEL_ID" --enable-thinking --reasoning-effort medium \
  --max-tokens 32768 --run-id reasoning-medium-01
```

`--add-context` adds a piece list without legal moves. The default `--run-id new` adds a filename suffix, and each controlled condition receives its own output file.

Each result contains:

- `is_correct`: Accuracy under the requested answer protocol
- `format_compliant`: Protocol compliance
- `content_correct`: Content accuracy
- `accepted_answers`: all accepted single answers, including all legal mate-in-one moves

Resulting FEN answers are trimmed and compared case-sensitively because letter case encodes piece color. Other single-answer tasks retain case normalization.

Stats files include item-micro, task-macro, category-macro, Protocol compliance, and Content accuracy. `--eval-only` rescores the selected result file.

## Audit benchmark structure

```bash
python analysis/benchmark_audit.py summary benchmark
```

The full benchmark contains 2,803 unique four-field FEN positions and 1,644 unique Lichess PuzzleIds. Semantic contains 100 state--move--comment triples reused under four distractor strategies.

## Aggregation and cluster-aware uncertainty

```bash
python analysis/rank_stability.py results/model-a.jsonl results/model-b.jsonl \
  --bootstrap 1000 --output rank_stability.json
```

The analysis clusters shared PuzzleIds, Semantic base examples, Position Judgment position IDs, and Structural state-tracking games. It also reports source-balanced Accuracy and a slice that removes FENs or PuzzleIds shared across categories.

## Semantic shortcut controls

```bash
python analysis/semantic_controls.py benchmark/semantic.jsonl semantic_controls
```

This generates comments-only, move-only, FEN-only, shuffled-FEN, and shuffled-move views without changing options or labels.

The relevance filter uses observable context by default:

```bash
python dataset/05_3_comment_judging.py --judge-context observable
```

`--judge-context historical` is retained only as a diagnostic and must not be used as the sole answerability filter.

## Position Judgment provenance and baselines

The benchmark labels use the deepest available records in Lichess Evaluations. To create a fixed-engine reference set:

```bash
python dataset/04_position_judgement.py \
  --data_path data/raw/lichess_db_eval.jsonl.zst \
  --output_root generated/fixed_engine \
  --stockfish_path /absolute/path/to/stockfish \
  --stockfish_nodes 200000 --stockfish_threads 1 --stockfish_hash_mb 128
```

Evaluate material-only, transparent static, cross-validated material/piece-square, and optional shallow-engine baselines:

```bash
python analysis/position_baselines.py benchmark/position_judgement.jsonl
python analysis/position_baselines.py benchmark/position_judgement.jsonl \
  --stockfish /absolute/path/to/stockfish --nodes 100000 \
  --threads 1 --hash-mb 128 --output position_validation.json
```

## Rebuild datasets

Builders live in `dataset/01_structural.py` through `dataset/05_semantic.py`. Source defaults are documented by `--help`. Semantic samples one shared set of 100 base examples across four distractor strategies. Position Judgment task IDs use `judgment`; the data filename remains `position_judgement.jsonl`.

## Licensing and provenance

The repository's MIT license covers code. The benchmark release includes all 3,500 evaluation items. Lichess-derived items retain their source provenance, and the Semantic comments come from a purchased copy of ChessBase~17. Benchmark data retains the terms associated with its source material.
