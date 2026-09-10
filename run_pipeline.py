"""CerebroGraph / Omni-Vector-AI-Dashboard Master Orchestrator.

Executes the polypharmacology pipeline in isolated OS-level subprocesses.
Features caching/resumability and cross-platform absolute path resolution.
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Anchor absolute path to the directory containing this script
PROJECT_ROOT = Path(__file__).parent.resolve()

def is_stale(inputs: List[Path], output: Path) -> bool:
    """Determine if a stage needs to run based on file modification times."""
    if not output.exists():
        return True
    
    out_mtime = output.stat().st_mtime
    for in_file in inputs:
        if in_file.exists() and in_file.stat().st_mtime > out_mtime:
            return True
            
    return False

def run_stage(script_relative_path: str, stage_name: str, force: bool = False):
    """Execute a pipeline module in an isolated subprocess."""
    script_path = PROJECT_ROOT / script_relative_path
    
    if not script_path.exists():
        logging.error(f"Missing executable module: {script_path}")
        sys.exit(2)
        
    logging.info(f"Executing Stage: {stage_name}...")
    
    cmd = [sys.executable, str(script_path)]
    if force:
        cmd.append("--force")
        
    try:
        # CWD ensures imports inside the subprocess resolve correctly
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        logging.error(f"Pipeline failure in stage: {stage_name} (Exit code {e.returncode})")
        sys.exit(e.returncode)

def main():
    parser = argparse.ArgumentParser(description="Run the Polypharmacology Virtual Screening Pipeline.")
    parser.add_argument("--force", action="store_true", help="Force run all stages, bypassing cache.")
    args = parser.parse_args()

    # Define the data lineage and intermediate file artifacts
    raw_lib = PROJECT_ROOT / "data" / "library" / "raw" / "zinc_subset.csv"
    curated_lib = PROJECT_ROOT / "data" / "library" / "curated.csv"
    filtered_step1 = PROJECT_ROOT / "data" / "physics_engine" / "filtered_step1.csv"
    filtered_step2 = PROJECT_ROOT / "data" / "physics_engine" / "filtered_step2.csv"
    docking_results = PROJECT_ROOT / "data" / "physics_engine" / "docking_results.csv"
    final_json = PROJECT_ROOT / "data" / "ranking" / "final_candidates.json"

    # Stage 1: Target Preparation
    # Always run to check for new targets in cns_targets.json
    run_stage("target/prepare_target.py", "Target Preparation", args.force)

    # Stage 2: Library Curation
    if args.force or is_stale([raw_lib], curated_lib):
        run_stage("library/curate.py", "Curate Library", args.force)
    else:
        logging.info("Skipping Stage: Curate Library (Cache hit)")

    # Stage 3: Drug Likeness (QED & Lipinski)
    if args.force or is_stale([curated_lib], filtered_step1):
        run_stage("physics_engine/drug_likeness.py", "Drug Likeness Filter", args.force)
    else:
        logging.info("Skipping Stage: Drug Likeness (Cache hit)")

    # Stage 4: BBB Permeability (GNN Inference)
    if args.force or is_stale([filtered_step1], filtered_step2):
        run_stage("vector_engine/predict_bbb.py", "BBB GNN Inference", args.force)
    else:
        logging.info("Skipping Stage: BBB GNN Inference (Cache hit)")

    # Stage 5: AutoDock Vina (Threaded Docking)
    if args.force or is_stale([filtered_step2], docking_results):
        run_stage("physics_engine/dock_vina.py", "AutoDock Vina Docking", args.force)
    else:
        logging.info("Skipping Stage: AutoDock Vina Docking (Cache hit)")

    # Stage 6: Skyline Pareto Ranking
    if args.force or is_stale([docking_results], final_json):
        run_stage("ranking/pipeline_ranker.py", "Skyline Pareto Ranking", args.force)
    else:
        logging.info("Skipping Stage: Skyline Pareto Ranking (Cache hit)")

    # Stage 7: Database Sync
    run_stage("database/upload_results.py", "Database Sync", args.force)
    
    logging.info("✅ Full pipeline execution completed successfully.")

if __name__ == "__main__":
    main()