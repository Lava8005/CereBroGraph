"""Pipeline Ranker: Evaluates polypharmacology candidates using Pareto skyline optimization."""

import json
import logging
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import pandas as pd
from pandas.errors import EmptyDataError

try:
    from ranking.multi_objective import compute_composite_scores
    from ranking.pareto_front import compute_pareto_front
except ImportError:
    logging.warning("Optimization modules missing. Bypassing advanced ranking.")
    def compute_composite_scores(df): return df
    def compute_pareto_front(df): return df

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_target_scores(score_val) -> dict:
    if isinstance(score_val, dict):
        return score_val
    try:
        return json.loads(str(score_val)) if pd.notna(score_val) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def extract_best_vina(score_dict: dict) -> float:
    if not score_dict:
        return 0.0
    return min(score_dict.values())


def _write_empty_json(output_json: Path) -> list:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump([], f)
    return []


def generate_final_ranking(
    input_csv: Path = Path("data/physics_engine/docking_results.csv"),
    output_json: Path = Path("data/ranking/final_candidates.json"),
    top_k: int = 20  # THE FIX: Parameterize the final payload size
):
    """Process physics output, rank, truncate to top K, and serialize for DB sync."""
    if not input_csv.exists():
        logging.error(f"Input file missing: {input_csv}")
        return _write_empty_json(output_json)
        
    try:
        df = pd.read_csv(input_csv)
    except EmptyDataError:
        logging.error(f"Corrupted cache detected: {input_csv}. Run with --force.")
        return _write_empty_json(output_json)

    if df.empty:
        return _write_empty_json(output_json)

    if "target_scores" in df.columns:
        df["target_scores"] = df["target_scores"].apply(parse_target_scores)
        df["vina_score"] = df["target_scores"].apply(extract_best_vina)
    
    df = compute_composite_scores(df)
    df = compute_pareto_front(df)
    
    # THE FIX: Deterministic Sorting & Truncation
    # 1. Prioritize Pareto optimal molecules
    # 2. Then highest composite score (GNN + QED)
    # 3. Then best (most negative) thermodynamic Vina affinity
    if "composite_score" in df.columns and "on_pareto_front" in df.columns:
        df = df.sort_values(
            by=["on_pareto_front", "composite_score", "vina_score"], 
            ascending=[False, False, True]
        )
    
    # Slice the dataframe to strictly the top 20 candidates
    df = df.head(top_k)
    
    candidates = df.to_dict(orient="records")
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=4)
        
    logging.info(f"Ranking complete. Top {len(candidates)} candidates sanitized for DB sync.")
    return candidates


if __name__ == "__main__":
    generate_final_ranking()