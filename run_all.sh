#!/usr/bin/env bash
# Full benchmark on a 16-core machine (uses 14 workers). Resumable: just run it again if interrupted.
# Usage: bash run_all.sh            (inside tmux/screen, with the venv activated, vina on PATH)
set -e
export CEREBRO_DOCK_WORKERS=14
W=14

echo "== 0. sanity =="; vina --version | head -1; python -m pytest -q 2>&1 | tail -1

echo "== 1. library + exhaustive docking (ground truth) =="
[ -f data/library/raw/zinc_subset.csv ] || python library/download_zinc.py --count 8500
python run_pipeline.py          # no --force: resumes if interrupted

echo "== 2. surrogate-screening comparison on the ground truth =="
python -m experiments.surrogate_eval --source csv --csv data/physics_engine/docking_results.csv \
    --bbb-col bbb_probability --seeds 5 --rounds 6 --tag _ground_truth

echo "== 3. DUD-E validation (needs data/dude/aces and data/dude/aofb, see RUN_ON_PC.md step 5) =="
if [ -d data/dude/aces ] && [ -d data/dude/aofb ]; then
  python -m experiments.dude_validation --targets aces aofb --custom config/validation_targets.json --workers $W
else
  echo "data/dude/ not found: running only the PDB-based validation (re-docking + 5-HT2A)"
  python -m experiments.dude_validation --targets --custom config/validation_targets.json --workers $W
fi

echo "== 4. BBB ensemble + conformal =="
python -m experiments.bbb_uncertainty

echo "ALL DONE. Send me: data/physics_engine/docking_results.csv, results/surrogate_eval/csv_ground_truth_summary.md, results/dude_validation/summary.md, results/bbb_uncertainty/summary.md"
