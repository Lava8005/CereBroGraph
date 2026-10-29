"""Train a 5-member BBB ensemble on ONE scaffold split and evaluate calibration + conformal prediction.

Usage: python -m experiments.bbb_uncertainty [--members 5] [--split-seed 0] [--epochs 50]
Outputs results/bbb_uncertainty/{members/*.pt, summary.json, summary.md, test_predictions.csv}
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd

from vector_engine.b3db_loader import load_b3db_dataset
from vector_engine.train_eval import run_training
from vector_engine import uncertainty as U

OUT = ROOT / "results" / "bbb_uncertainty"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=7)
    a = ap.parse_args()
    (OUT / "members").mkdir(parents=True, exist_ok=True)

    datasets = load_b3db_dataset(split_mode="scaffold", seed=a.split_seed)
    train, val, test = datasets
    cks = []
    for k in range(a.members):
        ck = OUT / "members" / f"member{k}.pt"
        cks.append(ck)
        if ck.exists():
            continue
        # init seed differs per member (1000+k); the data split is shared
        run_training(epochs=a.epochs, patience=a.patience, seed=1000 + k, split_mode="scaffold",
                     checkpoint_path=ck, metrics_path=None, datasets=datasets)

    models = U.load_members(cks)
    gv, gt = list(val), list(test)
    pv, pt = U.member_probs(models, gv), U.member_probs(models, gt)
    yv = np.array([int(g.y.item()) for g in gv]); yt = np.array([int(g.y.item()) for g in gt])
    ev, et = U.ensemble_summary(pv), U.ensemble_summary(pt)

    res = {"n_val": len(yv), "n_test": len(yt), "members": a.members, "split_seed": a.split_seed,
           "test_pos_rate": float(yt.mean())}
    single = [U.safe_auc(yt, pt[k]) for k in range(a.members)]
    res["single_member_test_auc_mean"] = float(np.mean(single)); res["single_member_test_auc_sd"] = float(np.std(single))
    res["ensemble_test_auc"] = U.safe_auc(yt, et["p_mean"])
    res["ece_single_mean"] = float(np.mean([U.expected_calibration_error(pt[k], yt) for k in range(a.members)]))
    res["ece_ensemble"] = U.expected_calibration_error(et["p_mean"], yt)

    res["conformal"] = {}
    for alpha in (0.05, 0.1, 0.2):
        q = U.mondrian_thresholds(ev["p_mean"], yv, alpha)
        res["conformal"][str(alpha)] = U.evaluate_conformal(et["p_mean"], yt, q)

    # selective prediction: keep the most-confident fraction by ensemble std and by |p-0.5|
    conf = np.abs(et["p_mean"] - 0.5)
    order_std = np.argsort(et["p_std"]); order_conf = np.argsort(-conf)
    pred = (et["p_mean"] >= 0.5).astype(int)
    res["selective_accuracy"] = {}
    for frac in (1.0, 0.8, 0.6, 0.4):
        n = int(len(yt) * frac)
        res["selective_accuracy"][str(frac)] = {
            "by_low_std": float((pred[order_std[:n]] == yt[order_std[:n]]).mean()),
            "by_confidence": float((pred[order_conf[:n]] == yt[order_conf[:n]]).mean())}

    pd.DataFrame({"y": yt, "p_mean": et["p_mean"], "p_std": et["p_std"]}).to_csv(OUT / "test_predictions.csv", index=False)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    lines = ["# BBB ensemble + conformal prediction (scaffold split, seed %d)\n" % a.split_seed,
             f"Members: {a.members}; val n={len(yv)}, test n={len(yt)}.\n",
             f"Single-member test ROC-AUC: {res['single_member_test_auc_mean']:.3f} ± {res['single_member_test_auc_sd']:.3f}; ensemble: {res['ensemble_test_auc']:.3f}.",
             f"ECE: single {res['ece_single_mean']:.3f} vs ensemble {res['ece_ensemble']:.3f}.\n",
             "| alpha | coverage | cov BBB+ | cov BBB- | singleton frac | acc on singletons |", "|---|---|---|---|---|---|"]
    for al, r in res["conformal"].items():
        lines.append(f"| {al} | {r['coverage']:.3f} | {r['coverage_pos']:.3f} | {r['coverage_neg']:.3f} | {r['singleton_fraction']:.3f} | {r['accuracy_on_singletons']:.3f} |")
    lines += ["", "| kept fraction | acc (lowest ensemble std) | acc (highest confidence) |", "|---|---|---|"]
    for f, r in res["selective_accuracy"].items():
        lines.append(f"| {f} | {r['by_low_std']:.3f} | {r['by_confidence']:.3f} |")
    (OUT / "summary.md").write_text("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
