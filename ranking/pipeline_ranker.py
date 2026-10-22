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

from ranking.pareto_front import compute_pareto_fronts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_target_scores(score_val) -> dict:
    """Safely parse stringified dictionaries from pandas CSV."""
    if isinstance(score_val, dict):
        return score_val
    try:
        clean_str = str(score_val).replace("'", '"')
        return json.loads(clean_str) if pd.notna(score_val) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def _write_empty_json(output_json: Path) -> list:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump([], f)
    return []


def generate_final_ranking(
    input_csv: Path = Path("data/physics_engine/docking_results.csv"),
    output_json: Path = Path("data/ranking/final_candidates.json"),
    top_k: int = 20
):
    if not input_csv.exists():
        return _write_empty_json(output_json)
        
    try:
        df = pd.read_csv(input_csv)
    except EmptyDataError:
        return _write_empty_json(output_json)

    if df.empty:
        return _write_empty_json(output_json)

    # Expand multi-target scores into separate columns for true Pareto analysis
    target_cols = []
    if "target_scores" in df.columns:
        df["target_scores"] = df["target_scores"].apply(parse_target_scores)
        all_targets = set()
        for ts in df["target_scores"]:
            all_targets.update(ts.keys())
            
        for t in all_targets:
            col_name = f"vina_{t}"
            target_cols.append(col_name)
            # Penalty: If docking failed (0.0) or affinity is positive, assign +99.0 to ensure domination
            df[col_name] = df["target_scores"].apply(lambda s: float(s.get(t, 99.0)) if float(s.get(t, 0.0)) < 0 else 99.0)

    # Execute NSGA-II non-dominated sorting
    minimize_axes = target_cols if target_cols else ["vina_score"]
    df = compute_pareto_fronts(df, minimize_cols=minimize_axes, maximize_cols=["bbb_probability", "qed"])
    
    # Truncate queue strictly by Pareto rank, breaking ties by mean Vina affinity
    if target_cols:
        df["mean_vina"] = df[target_cols].apply(
            lambda row: 99.0 if 99.0 in row.values else row.mean(), axis=1
        )
        df = df.sort_values(by=["pareto_rank", "mean_vina"], ascending=[True, True])
        
    df = df.head(top_k)
    
    # Backwards compatibility / UI fallback
    df["composite_score"] = 1.0 / df["pareto_rank"]
    if "vina_score" not in df.columns and "mean_vina" in df.columns:
        df["vina_score"] = df["mean_vina"]
    
    candidates = df.to_dict(orient="records")
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=4)
        
    logging.info(f"Ranking complete. Top {len(candidates)} candidates sanitized for DB sync.")
    return candidates


if __name__ == "__main__":
    generate_final_ranking()