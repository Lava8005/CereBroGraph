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
    """Safely parse stringified dictionaries from pandas CSV."""
    if isinstance(score_val, dict):
        return score_val
    try:
        # Replace single quotes with double quotes for strict JSON compliance
        clean_str = str(score_val).replace("'", '"')
        return json.loads(clean_str) if pd.notna(score_val) else {}
    except (json.JSONDecodeError, ValueError):
        return {}


def extract_weighted_affinity(score_dict: dict, weights_dict: dict) -> float:
    """Enforce user-weighted Polypharmacology (or equal mean if no weights provided)."""
    if not score_dict:
        return 0.0
        
    total_score = 0.0
    total_weight = 0.0
    
    for disease, score in score_dict.items():
        w = float(weights_dict.get(disease, 1.0))
        total_score += float(score) * w
        total_weight += w
        
    if total_weight == 0:
        return 0.0
        
    return round(total_score / total_weight, 3)


def _get_target_weights() -> dict:
    """Extract clinical weights from the user's config."""
    config_path = Path(project_root) / "config" / "cns_targets.json"
    weights = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            targets = json.load(f)
            for t in targets:
                weights[t.get("disease")] = t.get("weight", 1.0)
    return weights


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

    target_weights = _get_target_weights()

    if "target_scores" in df.columns:
        df["target_scores"] = df["target_scores"].apply(parse_target_scores)
        df["vina_score"] = df["target_scores"].apply(lambda s: extract_weighted_affinity(s, target_weights))

    df = compute_composite_scores(df)
    df = compute_pareto_front(df)
    
    if "composite_score" in df.columns and "on_pareto_front" in df.columns:
        df = df.sort_values(
            by=["on_pareto_front", "composite_score", "vina_score"], 
            ascending=[False, False, True] 
        )

    df = df.head(top_k)
    candidates = df.to_dict(orient="records")
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(candidates, f, indent=4)
        
    logging.info(f"Ranking complete. Top {len(candidates)} candidates sanitized for DB sync.")
    return candidates


if __name__ == "__main__":
    generate_final_ranking()