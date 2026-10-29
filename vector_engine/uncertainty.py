"""Uncertainty-aware BBB prediction: deep ensemble + class-conditional (Mondrian) conformal prediction.

Ensemble members share ONE data split and differ only in initialisation / batch order, so the
spread of their predictions is a clean epistemic-uncertainty signal and the shared validation
set can be used for conformal calibration without leakage into the test set.
"""
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Batch

from vector_engine.b3db_to_graph import smiles_to_graph
from vector_engine.gnn_model import BBBGraphModel


# ----------------------------------------------------------------------------- ensemble
def load_members(checkpoints: Sequence[Path]) -> List[BBBGraphModel]:
    models = []
    for ck in checkpoints:
        m = BBBGraphModel(num_node_features=27, edge_dim=7, global_dim=2, hidden_dim=64)
        m.load_state_dict(torch.load(ck, map_location="cpu", weights_only=True))
        m.eval()
        models.append(m)
    return models


@torch.no_grad()
def member_probs(models: Sequence[BBBGraphModel], graphs: list, batch_size: int = 256) -> np.ndarray:
    """Return an array (n_members, n_graphs) of BBB+ probabilities."""
    out = np.zeros((len(models), len(graphs)))
    for i in range(0, len(graphs), batch_size):
        b = Batch.from_data_list(graphs[i:i + batch_size])
        for k, m in enumerate(models):
            logit = m(b.x, b.edge_index, b.batch, edge_attr=b.edge_attr, global_features=b.global_features)
            out[k, i:i + batch_size] = torch.sigmoid(logit).view(-1).numpy()
    return out


def smiles_probs(models: Sequence[BBBGraphModel], smiles: Sequence[str]) -> np.ndarray:
    return member_probs(models, [smiles_to_graph(s, is_inference=True) for s in smiles])


def ensemble_summary(probs: np.ndarray) -> dict:
    """Mean probability and member disagreement (std)."""
    return {"p_mean": probs.mean(axis=0), "p_std": probs.std(axis=0)}


# ----------------------------------------------------------------------------- conformal
def mondrian_thresholds(p_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.1) -> dict:
    """Class-conditional split-conformal thresholds on the score s = 1 - p(class).

    Guarantees P(true class in prediction set | class = c) >= 1 - alpha under exchangeability.
    """
    q = {}
    for c in (0, 1):
        p_c = p_cal[y_cal == c] if c == 1 else 1 - p_cal[y_cal == c]
        scores = np.sort(1 - p_c)
        n = len(scores)
        if n == 0:
            q[c] = 1.0
            continue
        k = int(np.ceil((n + 1) * (1 - alpha)))
        q[c] = 1.0 if k > n else float(scores[k - 1])
    return q


def prediction_sets(p: np.ndarray, q: dict) -> np.ndarray:
    """Boolean array (n, 2): column c is True if class c is in the prediction set.

    Nonconformity score of class 1 is 1 - p; of class 0 it is p.
    """
    return np.column_stack([p <= q[0], (1 - p) <= q[1]])


def decisions(sets: np.ndarray) -> np.ndarray:
    """'pos' / 'neg' when the set is a singleton, otherwise 'uncertain' (both or empty)."""
    out = np.full(len(sets), "uncertain", dtype=object)
    out[(sets[:, 1]) & (~sets[:, 0])] = "pos"
    out[(sets[:, 0]) & (~sets[:, 1])] = "neg"
    return out


def expected_calibration_error(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def evaluate_conformal(p: np.ndarray, y: np.ndarray, q: dict) -> dict:
    sets = prediction_sets(p, q)
    dec = decisions(sets)
    covered = sets[np.arange(len(y)), y.astype(int)]
    single = dec != "uncertain"
    pred = (dec == "pos").astype(int)
    return {
        "coverage": float(covered.mean()),
        "coverage_pos": float(covered[y == 1].mean()),
        "coverage_neg": float(covered[y == 0].mean()),
        "singleton_fraction": float(single.mean()),
        "accuracy_on_singletons": float((pred[single] == y[single]).mean()) if single.any() else float("nan"),
        "uncertain_fraction": float((~single).mean()),
    }


def safe_auc(y, p) -> float:
    return float(roc_auc_score(y, p)) if len(set(y)) > 1 else float("nan")
