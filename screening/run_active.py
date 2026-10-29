"""Run the surrogate-guided Pareto screen with real Vina and write pipeline-compatible results.

python -m screening.run_active --input data/physics_engine/filtered_step2.csv \
    --targets data/target/target_index.json --budget-frac 0.25 --workers 4

Resumable: docking scores are cached in <out>/dock_cache.csv, and the loop is seeded, so re-running
the same command continues instead of re-docking. Output: <out>/docking_results.csv has the same
format as the exhaustive docking stage, so `ranking/pipeline_ranker.py` can consume it unchanged.
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from screening.surrogate import run_screening
from screening.vina_oracle import VinaOracle

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/physics_engine/filtered_step2.csv")
    ap.add_argument("--targets", default="data/target/target_index.json")
    ap.add_argument("--out", default="results/active_screen")
    ap.add_argument("--bbb-col", default="bbb_probability")
    ap.add_argument("--mode", default="pareto", choices=["random", "greedy", "ucb", "pareto"])
    ap.add_argument("--budget-frac", type=float, default=0.25, help="fraction of the library to dock in total")
    ap.add_argument("--init-frac", type=float, default=0.08)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--exhaustiveness", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    N = len(df)
    n_init = max(5, int(a.init_frac * N))
    batch = max(1, int((a.budget_frac * N - n_init) / a.rounds))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    oracle = VinaOracle(Path(a.targets), out / "dock_cache.csv", max_workers=a.workers,
                        exhaustiveness=a.exhaustiveness, timeout=a.timeout)
    logging.info(f"Library {N}; init {n_init}, {a.rounds} rounds x {batch} -> {n_init + batch * a.rounds} dockings "
                 f"({(n_init + batch * a.rounds) / N:.0%} of library), mode={a.mode}")
    res = run_screening(df["smiles"].tolist(), df[a.bbb_col].to_numpy(float), oracle, mode=a.mode, n_init=n_init,
                        batch=batch, rounds=a.rounds, seed=a.seed, epochs=a.epochs)

    rows = df.iloc[res["order"]].copy()
    sc = res["scores"]
    rows["target_scores"] = [json.dumps({n: (0.0 if np.isnan(v) else float(v)) for n, v in zip(oracle.names, r)}) for r in sc]
    rows.to_csv(out / "docking_results.csv", index=False)
    (out / "run_summary.json").write_text(json.dumps({
        "library": N, "docked": len(res["order"]), "fraction": len(res["order"]) / N, "mode": a.mode,
        "rounds": [len(r) for r in res["rounds"]], "targets": oracle.names, "seed": a.seed,
        "exhaustiveness": a.exhaustiveness}, indent=1))
    logging.info(f"Done. Docked {len(res['order'])}/{N} molecules. Results in {out}/docking_results.csv")


if __name__ == "__main__":
    main()
