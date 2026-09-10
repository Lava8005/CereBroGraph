"""Database Synchronization Module.

Pushes the final ranked polypharmacology candidates to Supabase 
to drive the real-time React frontend dashboard. Features idempotent upserts
to support continuous pipeline re-runs without database corruption.
"""

import json
import logging
import sys
from pathlib import Path

# Dynamically anchor Python's module search path to the project root
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Safely import the resolved client (must use the renamed supabase_client.py)
from database.supabase_client import supabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def sync_to_cloud(
    input_json: Path = Path("data/ranking/final_candidates.json"),
    table_name: str = "polypharmacology_results"
) -> bool:
    """Read the Pareto skyline JSON payload and execute an idempotent upsert to Supabase."""
    if supabase is None:
        logging.error("Pipeline Aborted: Supabase client is offline due to missing credentials.")
        return False

    if not input_json.exists():
        logging.error(f"Missing ranking payload at {input_json}. Upstream ranking may have failed.")
        return False

    with open(input_json, "r", encoding="utf-8") as f:
        try:
            candidates = json.load(f)
        except json.JSONDecodeError:
            logging.error("Final payload is corrupted or invalid JSON.")
            return False

    if not candidates:
        logging.warning("No candidates found in the JSON payload. Skipping database sync.")
        return False

    logging.info(f"Preparing to sync {len(candidates)} candidates to Supabase table '{table_name}'...")

    try:
        # THE FIX: Use upsert with on_conflict to support continuous pipeline re-runs
        response = supabase.table(table_name).upsert(
            candidates, 
            on_conflict="candidate_code"
        ).execute()
        
        if not response.data or len(response.data) == 0:
            logging.error(
                "Supabase API accepted the request but inserted 0 rows. "
                "This is almost certainly a Row Level Security (RLS) block."
            )
            return False
            
        logging.info(f"✅ Database sync successful! {len(response.data)} records delivered to the cloud.")
        return True
        
    except Exception as e:
        logging.error(f"Supabase network/API rejection: {e}")
        return False


if __name__ == "__main__":
    # Execute the sync and capture the boolean result
    success = sync_to_cloud()
    
    # THE FIX: Propagate failures to the OS so the orchestrator halts immediately
    if not success:
        sys.exit(1)