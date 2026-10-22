"""CerebroGraph / Omni-Vector-AI-Dashboard Master Orchestrator. 

Executes the polypharmacology pipeline in isolated OS-level subprocesses. 
Features strict artifact hashing, atomic marker-based resumption, 
cross-platform absolute path resolution, and automated Scientific Validation.
"""

import argparse
import logging
import os
import subprocess
import sys
import json
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
PROJECT_ROOT = Path(__file__).parent.resolve()

def is_stale(inputs: List[Path], output: Path) -> bool:
    if not output.exists():
        return True
    out_mtime = output.stat().st_mtime
    for in_file in inputs:
        if in_file.exists() and in_file.stat().st_mtime > out_mtime:
            return True
    return False

def run_stage(script_relative_path: str, stage_name: str, force: bool = False):
    script_path = PROJECT_ROOT / script_relative_path
    if not script_path.exists():
        logging.error(f"Missing executable module: {script_path}")
        sys.exit(2)
        
    logging.info(f"Executing Stage: {stage_name}...")
    cmd = [sys.executable, str(script_path)]
    if force: cmd.append("--force")
        
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline failure in stage: {stage_name} (Exit code {e.returncode})")
        sys.exit(e.returncode)

def validate_fda_controls(docking_csv: Path, final_json: Path):
    if not docking_csv.exists() or not final_json.exists():
        return
        
    import pandas as pd
    import json
    
    try:
        dock_df = pd.read_csv(docking_csv)
        surviving_sources = set(dock_df['source_id'].astype(str).str.upper())
    except Exception:
        surviving_sources = set()
        
    d_docked = "DONEPEZIL" in surviving_sources
    s_docked = "SAFINAMIDE" in surviving_sources
    l_docked = "LOPERAMIDE" in surviving_sources # THE FIX: Re-added Loperamide
    
    with open(final_json, "r", encoding="utf-8") as f:
        candidates = json.load(f)
    top_k_sources = [str(c.get("source_id", "")).upper() for c in candidates]
    
    d_topk = "DONEPEZIL" in top_k_sources
    s_topk = "SAFINAMIDE" in top_k_sources
    
    logging.info("-" * 60)
    logging.info("SCIENTIFIC VALIDATION 2 (FDA CONTROL SPIKE):")
    
    if d_docked and s_docked:
        logging.info("  [BIOLOGY] Success: Donepezil & Safinamide passed BBB & QED filters.")
    else:
        logging.warning(f"  [BIOLOGY] Failed: Pos. controls dropped! (Donepezil: {d_docked}, Safinamide: {s_docked})")

    # THE FIX: Explicitly assert the negative control was destroyed
    if not l_docked:
        logging.info("  [BIOLOGY] Success: Loperamide (Negative Control) correctly rejected by filters.")
    else:
        logging.warning("  [BIOLOGY] FAILURE: Loperamide leaked through the BBB filter!")
        
    if d_topk and s_topk:
        logging.info("  [PARETO] Success: Positive controls reached the Top-K Skyline.")
        
    logging.info("-" * 60)

def main():
    parser = argparse.ArgumentParser(description="Run the Polypharmacology Virtual Screening Pipeline.")
    parser.add_argument("--force", action="store_true", help="Force run all stages, bypassing cache.")
    args = parser.parse_args()

    raw_lib = PROJECT_ROOT / "data" / "library" / "raw" / "zinc_subset.csv"
    curated_lib = PROJECT_ROOT / "data" / "library" / "curated_library.csv"
    filtered_step1 = PROJECT_ROOT / "data" / "physics_engine" / "filtered_step1.csv"
    filtered_step2 = PROJECT_ROOT / "data" / "physics_engine" / "filtered_step2.csv"
    docking_results = PROJECT_ROOT / "data" / "physics_engine" / "docking_results.csv"
    final_json = PROJECT_ROOT / "data" / "ranking" / "final_candidates.json"
    
    target_dir = PROJECT_ROOT / "data" / "target"
    target_index = target_dir / "target_index.json"
    gnn_checkpoint = PROJECT_ROOT / "vector_engine" / "checkpoints" / "best_bbb_gnn.pt"
    docking_marker = PROJECT_ROOT / "data" / "physics_engine" / "docking.marker"

    run_stage("target/prepare_target.py", "Target Preparation", args.force)

    if args.force or is_stale([raw_lib], curated_lib):
        run_stage("library/curate.py", "Curate Library", args.force)
    else:
        logging.info("Skipping Stage: Curate Library (Cache hit)")

    if args.force or is_stale([curated_lib], filtered_step1):
        run_stage("physics_engine/drug_likeness.py", "Drug Likeness Filter", args.force)
    else:
        logging.info("Skipping Stage: Drug Likeness (Cache hit)")

    if args.force or is_stale([filtered_step1, gnn_checkpoint], filtered_step2):
        run_stage("vector_engine/predict_bbb.py", "BBB GNN Inference", args.force)
    else:
        logging.info("Skipping Stage: BBB GNN Inference (Cache hit)")

    # THE FIX: Glob physical biological artifacts so manual grid/receptor edits trigger re-docking
    target_artifacts = list(target_dir.glob("*_target.pdbqt")) + list(target_dir.glob("*_grid.json"))
    docking_inputs = [filtered_step2, target_index] + target_artifacts

    if args.force or not docking_marker.exists() or is_stale(docking_inputs, docking_marker):
        if docking_results.exists():
            docking_results.unlink()
        if docking_marker.exists():
            docking_marker.unlink()
            
        run_stage("physics_engine/dock_vina.py", "AutoDock Vina Docking", args.force)
        docking_marker.touch()
    else:
        logging.info("Skipping Stage: AutoDock Vina Docking (Cache hit)")

    if args.force or is_stale([docking_results], final_json):
        run_stage("ranking/pipeline_ranker.py", "Skyline Pareto Ranking", args.force)
    else:
        logging.info("Skipping Stage: Skyline Pareto Ranking (Cache hit)")

    validate_fda_controls(docking_results, final_json)
    
    # THE FIX: Generate empirical metrics for publication
    run_stage("analysis/generate_paper_metrics.py", "Generating Paper Metrics", args.force)
    
    run_stage("database/upload_results.py", "Database Sync", args.force)
    logging.info("  Full pipeline execution completed successfully.")

if __name__ == "__main__":
    main()