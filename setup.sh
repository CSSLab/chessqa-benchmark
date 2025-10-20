#!/bin/bash

# =============================================================================
# ChessQA Setup Script
# =============================================================================

set -e

echo "🏆 ChessQA - Setup"
echo "=================="
echo

# Python
if ! command -v python >/dev/null 2>&1; then
  echo "❌ Python is not installed. Please install Python 3.8+ first."
  exit 1
fi
echo "✅ Python found: $(python --version)"

# Dependencies
echo "📦 Installing Python dependencies..."
if python -m pip install -r requirements.txt; then
  echo "✅ Dependencies installed"
else
  echo "❌ Failed to install dependencies"
  exit 1
fi

# API keys (OpenRouter)
echo
echo "🔑 API key setup (OpenRouter)"
if [ -z "${OPENROUTER_API_KEY:-}" ]; then
  if [ -f "keys/openrouter.key" ]; then
    export OPENROUTER_API_KEY="$(cat keys/openrouter.key)"
    echo "✅ Loaded OPENROUTER_API_KEY from keys/openrouter.key"
  else
    echo "⚠️  OPENROUTER_API_KEY not set. To enable cloud inference via OpenRouter:"
    echo "   - Export key: export OPENROUTER_API_KEY=\"your_key\""
    echo "   - Or save it to keys/openrouter.key"
    echo "   Get a key: https://openrouter.ai/"
  fi
else
  echo "✅ Detected OPENROUTER_API_KEY in environment"
fi

# Data files
echo
echo "📁 Checking for source data (under data/raw)..."
need_help=0
if [ -f "data/raw/lichess_db_puzzle.csv" ]; then
  echo "✅ lichess_db_puzzle.csv"
else
  echo "⚠️  Missing data/raw/lichess_db_puzzle.csv (Lichess puzzles)"
  need_help=1
fi
if [ -f "data/raw/lichess_db_eval.jsonl.zst" ]; then
  echo "✅ lichess_db_eval.jsonl.zst"
else
  echo "⚠️  Missing data/raw/lichess_db_eval.jsonl.zst (engine evals for Position Judgment)"
  need_help=1
fi
if [ -f "data/raw/lichess_db_broadcast_2025-04.pgn" ]; then
  echo "✅ lichess_db_broadcast_2025-04.pgn"
else
  echo "ℹ️  Optional: data/raw/lichess_db_broadcast_2025-04.pgn (state tracking)"
fi
if [ "$need_help" -eq 1 ]; then
  echo "   Download from https://database.lichess.org/ and place files under data/raw/"
fi

# Optional: vLLM (used by comment cleaning/judging helpers)
echo
echo "🚀 Optional dependency check (vLLM)"
if python - <<'PY'
try:
  import vllm  # noqa: F401
  print('yes')
except Exception:
  pass
PY
then
  echo "✅ vLLM installed (used by comment cleaning/judging helpers)"
else
  echo "ℹ️  vLLM not installed (optional). Install with: pip install vllm"
fi

# Next steps
echo
echo "🎯 Next steps"
echo "-------------"
echo "1) Generate datasets (writes to data/benchmark):"
echo "   python code/dataset/01_structural.py --puzzle_path data/raw/lichess_db_puzzle.csv \\"
echo "     --pgn_path data/raw/lichess_db_broadcast_2025-04.pgn --output_root data/benchmark --N_sample 100"
echo "   python code/dataset/02_motifs.py --puzzle_path data/raw/lichess_db_puzzle.csv \\"
echo "     --output_root data/benchmark --N_sample 100"
echo "   python code/dataset/03_short_tactics.py --puzzle_path data/raw/lichess_db_puzzle.csv \\"
echo "     --all_themes_path data/info/all_themes_to_include.json --output_root data/benchmark \\
     --N_sample_rating 100 --N_sample_theme 25"
echo "   python code/dataset/04_position_judgement.py --data_path data/raw/lichess_db_eval.jsonl.zst \\"
echo "     --output_root data/benchmark --tasks_per_category 100 --max_evaluations 10000"
echo "   python code/dataset/05_semantic.py --input data/mid/comment_dataset.final.json \\"
echo "     --output_root data/benchmark --N_sample_mcq 100"
echo
echo "2) Run inference (OpenRouter):"
echo "   OPENROUTER_API_KEY=... python code/eval/run_openrouter.py \\"
echo "     --dataset-root data/benchmark --model anthropic/claude-3.5-haiku \\"
echo "     --output-dir results --workers 256"
echo
echo "3) Browse and plot results:"
echo "   python code/eval/browse_results.py --results-dir results"
echo "   python code/plot/plot_overall_error_breakdown.py --results-dir results --output-dir plots"
echo "   python code/plot/plot_2D.py --results-dir results --output-dir plots"
echo "   python code/plot/export_comprehensive_stats.py --results-dir results --output comprehensive_stats.json"
echo
echo "📚 For details and background, see README.md"
echo "✅ Setup completed"
