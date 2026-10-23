"""Pipeline Ranker: Evaluates polypharmacology candidates using Pareto skyline optimization."""

import ast
import json
import logging
import sys
from pathlib import Path

project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from ranking.pareto_front import compute_pareto_fronts

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

MAXIMIZE_COLS = ["bbb_probability", "qed"]


def parse_target_scores(score_val) -> dict:
    """Parse a per-target score mapping from a dict, JSON string or Python-literal string."""
    if isinstance(score_val, dict):
        return score_val
    if score_val is None or (not isinstance(score_val, str) and pd.isna(score_val)):
        return {}
    text = str(score_val)
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            continue
    return {}


def _write_json(output_json: Path, payload: list) -> list:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4, allow_nan=False)
    return payload


def _minmax(series: pd.Series, higher_is_better: bool) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(1.0, index=series.index)
    scaled = (series - lo) / (hi - lo)
    return scaled if higher_is_better else 1.0 - scaled


def generate_final_ranking(
    input_csv: Path = Path("data/physics_engine/docking_results.csv"),
    output_json: Path = Path("data/ranking/final_candidates.json"),
    top_k: int = 20,
    min_affinity: float = -7.0,
):
    if not input_csv.exists():
        return _write_json(output_json, [])

    try:
        df = pd.read_csv(input_csv)
    except EmptyDataError:
        return _write_json(output_json, [])

    if df.empty:
        return _write_json(output_json, [])

    target_cols = []
    if "target_scores" in df.columns:
        df["target_scores"] = df["target_scores"].apply(parse_target_scores)
        all_targets = sorted({k for ts in df["target_scores"] for k in ts})

        for t in all_targets:
            col_name = f"vina_{t}"
            target_cols.append(col_name)
            # A docking score is only valid if it is a negative binding energy.
            df[col_name] = df["target_scores"].apply(
                lambda s, t=t: float(s[t]) if s.get(t) is not None and float(s[t]) < 0 else np.nan
            )
    elif "vina_score" in df.columns:
        target_cols = ["vina_score"]
        df["vina_score"] = df["vina_score"].where(df["vina_score"] < 0)
    else:
        raise ValueError(
            f"{input_csv} has neither a 'target_scores' nor a 'vina_score' column; cannot rank."
        )

    # Per-target failure report, so one broken receptor is visible instead of silently
    # emptying the whole ranking.
    for col in target_cols:
        failed = int(df[col].isna().sum())
        if failed:
            logging.warning(f"{col}: {failed}/{len(df)} candidates have no valid docking score.")

    # STRICT POLYPHARMACOLOGY: drop candidates that failed on any required target.
    initial_len = len(df)
    df = df.dropna(subset=target_cols).copy()
    if len(df) < initial_len:
        logging.info(f"Dropped {initial_len - len(df)} candidates due to failed/missing dockings.")
    # POLYPHARMACOLOGY FLOOR: Pareto only needs a molecule to be non-dominated, so a candidate with
    # near-zero binding on one target can still reach the front. Require a real affinity on every target.
    if min_affinity is not None and not df.empty:
        before = len(df)
        df = df[(df[target_cols] <= min_affinity).all(axis=1)].copy()
        logging.info(f"Affinity floor {min_affinity} kcal/mol on every target removed {before - len(df)} candidates.")
    if df.empty:
        logging.error("No candidate docked successfully on every target; ranking is empty.")
        return _write_json(output_json, [])

    maximize = [c for c in MAXIMIZE_COLS if c in df.columns]
    df = compute_pareto_fronts(df, minimize_cols=target_cols, maximize_cols=maximize, stop_after=top_k)

    # Mean affinity across targets (kcal/mol, more negative is better).
    df["mean_vina"] = df[target_cols].mean(axis=1)
    df["vina_score"] = df["mean_vina"]

    # Composite: mean of min-max normalized objectives over all docked candidates
    # (0..1, higher is better). Gives a meaningful ordering inside a Pareto front.
    parts = [_minmax(df[c], higher_is_better=False) for c in target_cols]
    parts += [_minmax(df[c].astype(float), higher_is_better=True) for c in maximize]
    df["composite_score"] = pd.concat(parts, axis=1).mean(axis=1).round(4)

    # Rank by Pareto tier, then spread (crowding), then best mean affinity.
    df = df.sort_values(
        by=["pareto_rank", "crowding_distance", "mean_vina"],
        ascending=[True, False, True],
        kind="stable",
    ).head(top_k)

    candidates = json.loads(df.to_json(orient="records"))
    _write_json(output_json, candidates)
    logging.info(f"Ranking complete. Top {len(candidates)} candidates written to {output_json}.")
    return candidates


if __name__ == "__main__":
    generate_final_ranking()
