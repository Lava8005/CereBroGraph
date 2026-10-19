"""Database Synchronization Module.

Uploads final pipeline candidates to Supabase using an idempotent UPSERT.
Keys off `candidate_code` to prevent duplication on multiple pipeline runs.
"""

import json
import logging
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from database.supabase_client import supabase

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def upload_final_candidates(
    input_json: Path = Path("data/ranking/final_candidates.json")
):
    if not input_json.exists():
        logging.error(f"Missing ranking output: {input_json}")
        return

    with open(input_json, "r", encoding="utf-8") as f:
        candidates = json.load(f)

    if not candidates:
        logging.warning("No candidates to upload.")
        return

    # Sanitize payload for Postgres Schema
    sanitized_payload = []
    for c in candidates:
        # Convert stringified dict back to object if necessary for JSONB column
        target_scores = c.get("target_scores", {})
        if isinstance(target_scores, str):
            try:
                target_scores = json.loads(target_scores.replace("'", '"'))
            except Exception:
                target_scores = {}

        sanitized_payload.append({
            "candidate_code": c.get("candidate_code", "UNKNOWN"),
            "source_id": str(c.get("source_id", "UNKNOWN")),
            "smiles": str(c.get("smiles", "")),
            "vina_score": float(c.get("vina_score", 0.0)),
            "bbb_probability": float(c.get("bbb_probability", 0.0)),
            "qed": float(c.get("qed", 0.0)),
            "lipinski_violations": int(c.get("lipinski_violations", 0)),
            "composite_score": float(c.get("composite_score", 0.0)),
            "on_pareto_front": bool(c.get("on_pareto_front", False)),
            "target_scores": target_scores
        })

    try:
        # THE FIX: Target correct table and use idempotent upsert
        response = supabase.table("polypharmacology_results").upsert(
            sanitized_payload, 
            on_conflict="candidate_code"
        ).execute()
        
        logging.info(f"Database Sync complete. Upserted {len(sanitized_payload)} candidates.")
    except Exception as e:
        logging.error(f"Supabase upload failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    upload_final_candidates()