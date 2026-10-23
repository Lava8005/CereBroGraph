"""Step 1: multi-seed BBB benchmark (GNN vs baselines) on identical splits.

For each (split_mode, seed) the same train/val/test split is used for every model.
Results are appended as JSON lines so long runs can be resumed or batched.

Usage:
    python -m experiments.bbb_benchmark --split scaffold --seeds 0 1 2 3 4
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score, f1_score,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from vector_engine.b3db_loader import load_b3db_dataset
from vector_engine.train_eval import run_training

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
torch.set_num_threads(2)

OUT_DIR = ROOT / "results" / "bbb_benchmark"
FEATURE_CACHE = OUT_DIR / "features_cache.npz"
SIMPLE_DESCRIPTORS = ["MolWt", "TPSA", "NumHDonors", "NumHAcceptors", "MolLogP", "NumRotatableBonds"]


def metrics(y_true, prob, thresh=0.5):
    y_true = np.asarray(y_true)
    prob = np.asarray(prob, dtype=float)
    pred = (prob >= thresh).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
    }


def featurize(smiles_list):
    """Morgan fingerprint (2048, r=2), full RDKit 2D descriptors, simple descriptors."""
    fps, desc, simple = [], [], []
    names = [n for n, _ in Descriptors.descList]
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = np.zeros(2048, dtype=np.uint8)
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        fp[list(bv.GetOnBits())] = 1
        fps.append(fp)
        row = []
        for _, fn in Descriptors.descList:
            try:
                row.append(float(fn(mol)))
            except Exception:
                row.append(np.nan)
        desc.append(row)
        simple.append([float(getattr(Descriptors, n)(mol)) for n in SIMPLE_DESCRIPTORS])
    desc = np.nan_to_num(np.array(desc, dtype=np.float64), nan=0.0, posinf=1e6, neginf=-1e6)
    desc = np.clip(desc, -1e6, 1e6)
    return np.array(fps), desc, np.array(simple)


def feature_table(smiles_all):
    """Compute (or load from cache) features for every unique SMILES."""
    if FEATURE_CACHE.exists():
        z = np.load(FEATURE_CACHE, allow_pickle=True)
        cached = list(z["smiles"])
        if set(smiles_all) <= set(cached):
            idx = {s: i for i, s in enumerate(cached)}
            return idx, z["fp"], z["desc"], z["simple"]
    uniq = sorted(set(smiles_all))
    fp, desc, simple = featurize(uniq)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(FEATURE_CACHE, smiles=np.array(uniq, dtype=object), fp=fp, desc=desc, simple=simple)
    return {s: i for i, s in enumerate(uniq)}, fp, desc, simple


def run_baselines(train, val, test, seed, table):
    idx, FP, DESC, SIMPLE = table
    def take(ds):
        rows = [idx[g.smiles] for g in ds]
        y = np.array([int(g.y.item()) for g in ds])
        return rows, y
    tr_r, ytr = take(train)
    va_r, yva = take(val)
    te_r, yte = take(test)
    out = {}

    # 1) The pipeline's own physicochemical rule used as a classifier (no training)
    s = SIMPLE[te_r]
    rule = ((s[:, 0] <= 500) & (s[:, 1] <= 90) & (s[:, 2] <= 3)).astype(float)
    out["rule_MW500_TPSA90_HBD3"] = metrics(yte, rule)

    # 2) Logistic regression on six physicochemical descriptors
    sc = StandardScaler().fit(SIMPLE[tr_r])
    lr = LogisticRegression(max_iter=2000, random_state=seed).fit(sc.transform(SIMPLE[tr_r]), ytr)
    out["logreg_6_descriptors"] = metrics(yte, lr.predict_proba(sc.transform(SIMPLE[te_r]))[:, 1])

    # 3) Random forest on Morgan fingerprints
    rf = RandomForestClassifier(n_estimators=500, n_jobs=2, random_state=seed).fit(FP[tr_r], ytr)
    out["rf_morgan"] = metrics(yte, rf.predict_proba(FP[te_r])[:, 1])

    # 4) Random forest on all RDKit descriptors
    rf2 = RandomForestClassifier(n_estimators=500, n_jobs=2, random_state=seed).fit(DESC[tr_r], ytr)
    out["rf_rdkit_descriptors"] = metrics(yte, rf2.predict_proba(DESC[te_r])[:, 1])

    # 5) XGBoost on descriptors + fingerprints, early-stopped on the validation split
    X = lambda r: np.hstack([DESC[r], FP[r]])
    xgb = XGBClassifier(n_estimators=600, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                        n_jobs=2, random_state=seed, eval_metric="auc", early_stopping_rounds=30, verbosity=0)
    xgb.fit(X(tr_r), ytr, eval_set=[(X(va_r), yva)], verbose=False)
    out["xgboost_desc_morgan"] = metrics(yte, xgb.predict_proba(X(te_r))[:, 1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["scaffold", "random"], default="scaffold")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "seed_runs.jsonl")
    ap.add_argument("--skip-gnn", action="store_true")
    args = ap.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.out.exists():
        for line in args.out.read_text().splitlines():
            r = json.loads(line)
            done.add((r["split"], r["seed"]))

    for seed in args.seeds:
        if (args.split, seed) in done:
            print(f"skip {args.split} seed {seed} (already done)")
            continue
        t0 = time.time()
        train, val, test = load_b3db_dataset(split_mode=args.split, seed=seed)
        table = feature_table([g.smiles for ds in (train, val, test) for g in ds])
        record = {"split": args.split, "seed": seed,
                  "sizes": {"train": len(train), "val": len(val), "test": len(test)},
                  "test_pos_fraction": float(np.mean([g.y.item() for g in test])),
                  "models": run_baselines(train, val, test, seed, table)}
        if not args.skip_gnn:
            res = run_training(epochs=args.epochs, patience=args.patience, seed=seed, split_mode=args.split,
                               checkpoint_path=OUT_DIR / "checkpoints" / f"gine_{args.split}_seed{seed}.pt",
                               metrics_path=None, datasets=(train, val, test))
            record["models"]["gine_gnn"] = {k: float(v) for k, v in res["test"].items()}
            record["gnn_epochs_run"] = res["epochs_run"]
        record["seconds"] = round(time.time() - t0, 1)
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
        m = record["models"]
        print(f"[{args.split} seed {seed}] {record['seconds']}s  " +
              "  ".join(f"{k}:{v['roc_auc']:.3f}" for k, v in m.items()), flush=True)


if __name__ == "__main__":
    main()
