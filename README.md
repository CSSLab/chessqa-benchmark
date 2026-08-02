# ChessQA: Evaluating Large Language Models for Chess Understanding

ChessQA is a text-only, tool-free chess QA benchmark with 50 tasks and 3,500 items across Structural (1,100), Motifs (600), Short Tactics (900), Position Judgment (500), and Semantic (400). The categories are complementary diagnostic views, not a curriculum or a strict difficulty hierarchy. “Chess understanding” is operationalized as performance under the stated QA interface; the score also reflects board-representation parsing and output serialization.

The frozen benchmark and reported model outputs are kept immutable. The current code corrects the interfaces and graders around those files without rewriting results:

- Structural capture/control/protect labels use the standard attack relation. Pinned pieces retain attacked squares; pin status is audit metadata.
- Every legal mate-in-one move is accepted, including for old JSONL rows that store only the Lichess move.
- Frozen Motif prompts/labels are repaired at evaluation time for the omitted absolute-pin definition, promotion suffixes in double checks, and battery square ordering; corrected builders emit these forms directly.
- Strict protocol accuracy, format compliance, and format-relaxed chess-content accuracy are separate fields.
- FEN, ASCII-board, and explicit-piece-list inputs can be evaluated under otherwise matched prompts.
- Overall reporting includes item-micro, equal-task macro, and equal-category macro scores.
- Semantic variants carry stable base-example IDs and the relevance judge defaults to the same FEN-before + move context shown to evaluated models.
- Position Judgment can be regenerated with a pinned Stockfish binary and fixed node budget; simple chess baselines are included.

## Repository layout

- `benchmark/`: frozen benchmark JSONL files
- `dataset/`: corrected dataset builders
- `eval/`: OpenRouter runner and shared protocol/scoring code
- `analysis/`: overlap/label audits, rank stability, Semantic controls, and Position Judgment baselines
- `tests/`: regression tests for the label and protocol fixes

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
# Semantically equivalent board representations
python eval/run_openrouter.py --representation fen        --run-id fen-01
python eval/run_openrouter.py --representation ascii      --run-id ascii-01
python eval/run_openrouter.py --representation piece_list --run-id pieces-01

# Structured output and independent repeats
python eval/run_openrouter.py --output-protocol json --run-id json-01
python eval/run_openrouter.py --temperature 0.7 --run-id temp07-01

# Provider reasoning setting; set the output ceiling explicitly
python eval/run_openrouter.py --enable-thinking --reasoning-effort medium \
  --max-tokens 32768 --run-id reasoning-medium-01
```

`--add-context` is the legacy “FEN + Pieces” ablation. It adds only piece arrangement, never legal moves. The default `--run-id new` adds a filename suffix, and explicit representation/protocol/run variants receive their own suffixes, so an ordinary run does not replace an unsuffixed frozen result.

Each result contains:

- `is_correct`: strict requested-protocol score (the paper’s primary protocol)
- `format_compliant`: whether the requested terminal line or JSON wrapper was followed
- `content_correct`: answer correctness after format-relaxed parsing
- `accepted_answers`: all accepted single answers, including all legal mate-in-one moves

Stats files include item-micro, task-macro, category-macro, format, and content scores. Use `--eval-only` to rescore an existing compatible result file; this is a write operation on that selected result file, so copy it first when preserving an archival artifact.

## Audit benchmark structure

```bash
python analysis/benchmark_audit.py summary benchmark
```

The frozen release contains 2,803 unique four-field FEN positions and 1,644 unique Lichess PuzzleIds. Semantic contains 100 unique state–move–gold-comment triples reused under four distractor strategies. The audit command recomputes these counts from the selected files.

## Aggregation and cluster-aware uncertainty

```bash
python analysis/rank_stability.py results/model-a.jsonl results/model-b.jsonl \
  --bootstrap 1000 --output rank_stability.json
```

The bootstrap clusters shared PuzzleIds, Semantic base examples, Position Judgment position IDs, and Structural state-tracking games. Its intervals capture dataset/source sampling, not API endpoint run-to-run variation; use distinct `--run-id` values for independent endpoint repeats.

## Semantic shortcut controls

```bash
python analysis/semantic_controls.py benchmark/semantic.jsonl semantic_controls
```

This generates comments-only, move-only, FEN-only, shuffled-FEN, and shuffled-move views without changing options or labels. The original file is the full-prompt condition.

The corrected relevance filter uses observable context by default:

```bash
python dataset/05_3_comment_judging.py --judge-context observable
```

`--judge-context historical` is retained only as a diagnostic and must not be used as the sole answerability filter.

## Position Judgment provenance and baselines

The frozen labels use the deepest available records in Lichess Evaluations, which aggregates heterogeneous Stockfish-family browser analyses. To create a controlled reference set:

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
  --stockfish /absolute/path/to/stockfish --nodes 5000
```

## Rebuild datasets

Builders live in `dataset/01_structural.py` through `dataset/05_semantic.py`. Source defaults are documented by `--help`. Semantic now samples one shared set of 100 base examples and verifies that every distractor strategy uses exactly that set. Generated Position Judgment task IDs use the spelling `judgment`; the frozen filename retains `position_judgement.jsonl` for backward compatibility.

## Licensing and provenance

The repository’s MIT license covers code, not third-party source text. Lichess-derived artifacts retain Lichess’s stated source license and provenance. ChessBase-derived commentary is not granted for redistribution by this repository: public redistributable packages should ship code, schemas, state–move records, identifiers, and hashes, while users with their own authorized ChessBase access regenerate comment strings locally.
