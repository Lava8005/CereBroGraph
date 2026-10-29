"""Evaluate surrogate-guided screening against random acquisition.

  --source pilot      real Vina scores from results/docking_pilot (AChE, MAO-B), small library
  --source synthetic  a descriptor-based *synthetic* scoring function on B3DB molecules, used ONLY to
                      check that the loop learns at a larger scale. It is not docking data.
Outputs results/surrogate_eval/{source}_summary.md/json
"""
import argparse, json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdMolDescriptors
from screening.surrogate import MODES, evaluate_screening, run_screening

RDLogger.DisableLog("rdApp.*")
OUT = ROOT / "results" / "surrogate_eval"


def load_pilot():
    df = pd.read_csv(ROOT / "results/docking_pilot/docking_results.csv")
    sc = df["target_scores"].apply(json.loads).apply(pd.Series)
    Y = sc[["Alzheimers_AChE", "Parkinsons_MAOB"]].to_numpy(float)
    ok = (Y < 0).all(axis=1)
    return df.loc[ok, "smiles"].tolist(), Y[ok], df.loc[ok, "bbb_prob"].to_numpy(float)


def load_csv(path, bbb_col):
    df = pd.read_csv(path)
    sc = df["target_scores"].apply(json.loads).apply(pd.Series)
    Y = sc.to_numpy(float)
    ok = (Y < 0).all(axis=1)
    return df.loc[ok, "smiles"].tolist(), Y[ok], df.loc[ok, bbb_col].to_numpy(float)


def load_synthetic(n, seed=0):
    raw = pd.read_csv(ROOT / "data/training/B3DB_classification.tsv", sep="\t")
    rng = np.random.default_rng(seed)
    smis = []
    for s in raw["SMILES"].sample(frac=1.0, random_state=seed):
        m = Chem.MolFromSmiles(str(s))
        if m is None or not 10 <= m.GetNumHeavyAtoms() <= 35 or Chem.GetFormalCharge(m) != 0:
            continue
        smis.append(Chem.MolToSmiles(m))
        if len(smis) >= n:
            break
    f = lambda m: (m.GetNumHeavyAtoms(), Descriptors.MolLogP(m), rdMolDescriptors.CalcNumAromaticRings(m),
                   rdMolDescriptors.CalcTPSA(m), rdMolDescriptors.CalcNumRotatableBonds(m))
    F = np.array([f(Chem.MolFromSmiles(s)) for s in smis])
    # two correlated "targets": size/aromaticity driven with different weights + noise (kcal/mol-like scale)
    t1 = -2.0 - 0.22 * F[:, 0] - 0.6 * F[:, 2] - 0.15 * F[:, 1] + rng.normal(0, 0.5, len(F))
    t2 = -2.0 - 0.12 * F[:, 0] - 0.9 * F[:, 2] + 0.01 * F[:, 3] + rng.normal(0, 0.5, len(F))
    bbb = 1 / (1 + np.exp(0.06 * (F[:, 3] - 75) + 0.0))
    return smis, np.column_stack([t1, t2]), bbb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["pilot", "synthetic", "csv"], default="pilot")
    ap.add_argument("--csv", default=None, help="docking_results.csv from an exhaustive run (with --source csv)")
    ap.add_argument("--bbb-col", default="bbb_probability")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--modes", nargs="+", default=list(MODES))
    ap.add_argument("--tag", default="")
    ap.add_argument("--top-weight", type=float, default=0.0)
    ap.add_argument("--n-init", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=60)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.source == "pilot":
        smis, Y, bbb = load_pilot()
    elif a.source == "csv":
        smis, Y, bbb = load_csv(a.csv, a.bbb_col)
    else:
        smis, Y, bbb = load_synthetic(a.n)
    N = len(smis)
    n_init = a.n_init or max(10, N // 10)
    batch = a.batch or max(5, N // 20)
    budgets = [n_init + batch * r for r in range(a.rounds + 1)]
    oracle = lambda sm: np.array([Y[smi_idx[s]] for s in sm])
    smi_idx = {s: i for i, s in enumerate(smis)}
    min_aff = -7.0 if a.source in ("pilot", "csv") else float(np.percentile(Y.mean(axis=1), 30))
    rows = []
    for mode in a.modes:
        for seed in range(a.seeds):
            res = run_screening(smis, bbb, oracle, mode=mode, n_init=n_init, batch=batch, rounds=a.rounds,
                                seed=seed, epochs=a.epochs, min_affinity=min_aff, top_weight=a.top_weight)
            for r in evaluate_screening(res["order"], Y, bbb, budgets, min_affinity=min_aff):
                rows.append({"mode": mode, "seed": seed, **r})
            print(mode, seed, "done", flush=True)
    df = pd.DataFrame(rows)
    agg = df.groupby(["mode", "budget"])[["budget_frac", "top_recall", "pareto_recall", "feasible_recall"]].agg(["mean", "std"])
    df.to_csv(OUT / f"{a.source}{a.tag}_runs.csv", index=False)
    lines = [f"# Surrogate screening ({a.source}); library N={N}, {a.seeds} seeds, init={n_init}, batch={batch}\n",
             f"True top set: top 10% by mean score; feasible = every target <= {min_aff:.2f} and BBB p >= 0.5.\n",
             "| mode | docked | frac | top-10% recall | Pareto recall | feasible recall |", "|---|---|---|---|---|---|"]
    for (mode, b), r in agg.iterrows():
        f = lambda c: f"{r[(c,'mean')]:.2f} ± {r[(c,'std')]:.2f}"
        lines.append(f"| {mode} | {b} | {r[('budget_frac','mean')]:.2f} | {f('top_recall')} | {f('pareto_recall')} | {f('feasible_recall')} |")
    (OUT / f"{a.source}{a.tag}_summary.md").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
