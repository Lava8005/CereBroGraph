# Full benchmark on a 16-core Windows machine (14 workers). Resumable: run it again if interrupted.
# Usage (venv activated, vina.exe on PATH):  .\run_all.ps1
$ErrorActionPreference = "Stop"
$env:CEREBRO_DOCK_WORKERS = "14"
$W = 14

Write-Host "== 0. sanity =="; vina --version | Select-Object -First 1; python -m pytest -q 2>&1 | Select-Object -Last 1

Write-Host "== 1. library + exhaustive docking (ground truth) =="
if (-not (Test-Path data/library/raw/zinc_subset.csv)) { python library/download_zinc.py --count 8500 }
python run_pipeline.py

Write-Host "== 2. surrogate-screening comparison on the ground truth =="
python -m experiments.surrogate_eval --source csv --csv data/physics_engine/docking_results.csv --bbb-col bbb_probability --seeds 5 --rounds 6 --tag _ground_truth

Write-Host "== 3. DUD-E validation =="
if ((Test-Path data/dude/aces) -and (Test-Path data/dude/aofb)) {
  python -m experiments.dude_validation --targets aces aofb --custom config/validation_targets.json --workers $W
} else {
  Write-Host "data/dude/ not found: running only the PDB-based validation (re-docking + 5-HT2A)"
  python -m experiments.dude_validation --targets --custom config/validation_targets.json --workers $W
}

Write-Host "== 4. BBB ensemble + conformal =="
python -m experiments.bbb_uncertainty

Write-Host "ALL DONE. Send me: data/physics_engine/docking_results.csv, results/surrogate_eval/csv_ground_truth_summary.md, results/dude_validation/summary.md, results/bbb_uncertainty/summary.md"
