"""Aggregate results/bbb_benchmark/seed_runs.jsonl into tables and a figure."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = ROOT / "results" / "bbb_benchmark"
NAMES = {
    "rule_MW500_TPSA90_HBD3": "Rule (MW≤500, TPSA≤90, HBD≤3)",
    "logreg_6_descriptors": "Logistic regression (6 descriptors)",
    "rf_morgan": "Random forest (Morgan FP)",
    "rf_rdkit_descriptors": "Random forest (RDKit descriptors)",
    "xgboost_desc_morgan": "XGBoost (descriptors + FP)",
    "gine_gnn": "GINE graph network",
    "gine_gnn_tuned": "GINE graph network (tuned)",
}
METRICS = ["roc_auc", "pr_auc", "f1", "mcc", "balanced_accuracy"]


def load(path):
    runs = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    by_split = {}
    for r in runs:
        by_split.setdefault(r["split"], []).append(r)
    return by_split


def main(path=OUT / "seed_runs.jsonl"):
    by_split = load(path)
    lines = []
    summary = {}
    for split, runs in sorted(by_split.items()):
        runs = sorted(runs, key=lambda r: r["seed"])
        models = [m for m in NAMES if all(m in r["models"] for r in runs)]
        lines.append(f"### {split.capitalize()} split — {len(runs)} seeds "
                     f"(test n ≈ {int(np.mean([r['sizes']['test'] for r in runs]))}, mean ± SD)\n")
        lines.append("| Model | " + " | ".join(m.upper().replace("_", "-") for m in METRICS) + " |")
        lines.append("|---|" + "---|" * len(METRICS))
        summary[split] = {}
        for m in models:
            vals = {k: np.array([r["models"][m][k] for r in runs]) for k in METRICS}
            summary[split][m] = {k: (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0) for k, v in vals.items()}
            lines.append(f"| {NAMES[m]} | " + " | ".join(f"{vals[k].mean():.3f} ± {vals[k].std(ddof=1) if len(vals[k]) > 1 else 0:.3f}" for k in METRICS) + " |")
        lines.append("")
        # paired comparison of the GNN against every other model on ROC-AUC
        if "gine_gnn" in models:
            lines.append("Paired difference in ROC-AUC, GNN minus other model (same splits):\n")
            lines.append("| Other model | mean diff | 95% CI | Wilcoxon p |")
            lines.append("|---|---|---|---|")
            g = np.array([r["models"]["gine_gnn"]["roc_auc"] for r in runs])
            for m in models:
                if m == "gine_gnn":
                    continue
                o = np.array([r["models"][m]["roc_auc"] for r in runs])
                d = g - o
                se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
                tcrit = stats.t.ppf(0.975, len(d) - 1) if len(d) > 1 else float("nan")
                try:
                    p = stats.wilcoxon(d).pvalue
                except ValueError:
                    p = float("nan")
                lines.append(f"| {NAMES[m]} | {d.mean():+.3f} | [{d.mean() - tcrit * se:+.3f}, {d.mean() + tcrit * se:+.3f}] | {p:.3g} |")
            lines.append("")
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # figure: ROC-AUC per model, per split
    fig, axes = plt.subplots(1, len(summary), figsize=(6.2 * len(summary), 4.2), squeeze=False)
    for ax, (split, ms) in zip(axes[0], sorted(summary.items())):
        keys = list(ms)
        mean = [ms[k]["roc_auc"][0] for k in keys]
        sd = [ms[k]["roc_auc"][1] for k in keys]
        cols = ["#9aa5b1"] * len(keys)
        for i, k in enumerate(keys):
            if k.startswith("gine"):
                cols[i] = "#1f3a5f"
        ax.barh(range(len(keys)), mean, xerr=sd, color=cols, capsize=3)
        ax.set_yticks(range(len(keys)))
        ax.set_yticklabels([NAMES[k] for k in keys], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0.6, 1.0)
        ax.set_xlabel("Test ROC-AUC (mean ± SD over seeds)")
        ax.set_title(f"{split.capitalize()} split", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "roc_auc_comparison.png", dpi=160)
    print((OUT / "summary.md").read_text())


if __name__ == "__main__":
    main()
