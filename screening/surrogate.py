"""Surrogate-guided (active-learning) multi-target docking screen with a BBB objective.

Idea: docking is the expensive step. Dock a small random sample, train a GNN ensemble to predict the
docking scores of the rest, and spend further docking calls only on molecules the surrogate ranks
as promising on the multi-objective problem (docking score on every target + BBB probability).

`oracle` is any callable  smiles_list -> array (n, n_targets)  of docking scores (more negative =
better, NaN = failed). In simulation it is a lookup into a fully docked table; in production it
wraps Vina.
"""
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from ranking.pareto_front import compute_pareto_fronts
from vector_engine.b3db_to_graph import smiles_to_graph
from vector_engine.gnn_model import BBBGraphModel

MODES = ("random", "greedy", "ucb", "pareto", "pareto_dock", "feasible")
# pareto_dock : BBB-agnostic Pareto acquisition (our re-implementation of the Fromer/Graff/Coley idea; NOT MolPAL itself)
# feasible    : rank by predicted probability that a molecule passes every gate (all targets <= min_affinity, BBB)


class SurrogateEnsemble:
    """Small deep ensemble of GINE regressors predicting one docking score per target."""

    def __init__(self, n_targets: int, n_members: int = 3, hidden_dim: int = 64, epochs: int = 80,
                 lr: float = 2e-3, batch_size: int = 32, seed: int = 0, top_weight: float = 0.0):
        self.T, self.M, self.h, self.epochs, self.lr, self.bs, self.seed = n_targets, n_members, hidden_dim, epochs, lr, batch_size, seed
        self.top_weight = top_weight  # >0 emphasises the best (most negative) scores in the loss
        self.models: List[nn.Module] = []
        self._graphs: Dict[str, object] = {}

    def _graph(self, smi: str):
        if smi not in self._graphs:
            self._graphs[smi] = smiles_to_graph(smi, is_inference=True)
        return self._graphs[smi]

    def fit(self, smiles: Sequence[str], Y: np.ndarray) -> "SurrogateEnsemble":
        Y = np.asarray(Y, dtype=float)
        self.mu, self.sd = Y.mean(axis=0), Y.std(axis=0) + 1e-6
        Z = (Y - self.mu) / self.sd
        data = []
        for s, z in zip(smiles, Z):
            g = self._graph(s).clone()
            g.y = torch.tensor(z, dtype=torch.float).view(1, -1)
            g.w = torch.tensor([float(np.exp(-self.top_weight * np.mean(z)))])  # z low = good score
            data.append(g)
        self.models = []
        for m in range(self.M):
            torch.manual_seed(self.seed * 100 + m)
            model = BBBGraphModel(num_node_features=27, edge_dim=7, global_dim=2, hidden_dim=self.h, out_dim=self.T)
            opt = torch.optim.AdamW(model.parameters(), lr=self.lr, weight_decay=1e-4)
            drop_last = len(data) % self.bs == 1
            loader = DataLoader(data, batch_size=self.bs, shuffle=True, drop_last=drop_last)
            model.train()
            for _ in range(self.epochs):
                for b in loader:
                    opt.zero_grad()
                    pred = model(b.x, b.edge_index, b.batch, edge_attr=b.edge_attr, global_features=b.global_features)
                    err = ((pred - b.y.view(-1, self.T)) ** 2).mean(dim=1)
                    w = b.w.view(-1)
                    loss = (w * err).sum() / w.sum()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
            model.eval()
            self.models.append(model)
        return self

    @torch.no_grad()
    def predict(self, smiles: Sequence[str]):
        """Return (mean, std), each (n, T), in original docking-score units."""
        graphs = [self._graph(s) for s in smiles]
        preds = np.zeros((self.M, len(graphs), self.T))
        for i in range(0, len(graphs), 256):
            b = Batch.from_data_list(graphs[i:i + 256])
            for k, model in enumerate(self.models):
                preds[k, i:i + 256] = model(b.x, b.edge_index, b.batch, edge_attr=b.edge_attr,
                                            global_features=b.global_features).numpy()
        preds = preds * self.sd + self.mu
        return preds.mean(axis=0), preds.std(axis=0)


def select_batch(mean: np.ndarray, std: np.ndarray, bbb: np.ndarray, k: int, mode: str,
                 rng: np.random.Generator, beta: float = 1.0, min_affinity: float = -7.0) -> np.ndarray:
    """Return indices (into the candidate arrays) of the k molecules to dock next."""
    n = len(mean)
    k = min(k, n)
    if mode == "random":
        return rng.choice(n, size=k, replace=False)
    if mode == "greedy":
        return np.argsort(mean.mean(axis=1))[:k]
    if mode == "ucb":  # optimistic: lower is better, so subtract beta*std
        return np.argsort((mean - beta * std).mean(axis=1))[:k]
    if mode == "feasible":
        from scipy.stats import norm
        p_aff = norm.cdf((min_affinity - mean) / (std + 1e-3)).prod(axis=1)  # P(every target <= floor)
        p = p_aff * np.asarray(bbb, dtype=float)
        return np.lexsort((mean.mean(axis=1), -p))[:k]   # highest P(feasible), ties by best mean score
    if mode in ("pareto", "pareto_dock"):
        opt = mean - beta * std
        cols = {f"t{j}": opt[:, j] for j in range(opt.shape[1])}
        if mode == "pareto":
            cols["negbbb"] = -np.asarray(bbb, dtype=float)
        df = pd.DataFrame(cols)
        fr = compute_pareto_fronts(df, list(cols), [])
        order = fr.sort_values(["pareto_rank", "crowding_distance"], ascending=[True, False]).index
        return np.asarray(order[:k])
    raise ValueError(f"mode must be one of {MODES}")


def run_screening(smiles: Sequence[str], bbb: np.ndarray, oracle: Callable, mode: str = "pareto",
                  n_init: int = 20, batch: int = 10, rounds: int = 4, seed: int = 0,
                  n_members: int = 3, epochs: int = 80, beta: float = 1.0, min_affinity: float = -7.0,
                  top_weight: float = 0.0) -> dict:
    """Run the loop. Returns the order in which molecules were docked and their scores."""
    smiles = list(smiles); bbb = np.asarray(bbb, dtype=float)
    rng = np.random.default_rng(seed)
    labeled = list(rng.choice(len(smiles), size=min(n_init, len(smiles)), replace=False))
    Y = {i: y for i, y in zip(labeled, oracle([smiles[i] for i in labeled]))}
    history = [list(labeled)]
    for _ in range(rounds):
        pool = [i for i in range(len(smiles)) if i not in Y]
        if not pool:
            break
        if mode == "random":
            pick = rng.choice(len(pool), size=min(batch, len(pool)), replace=False)
        else:
            ok = [i for i in Y if np.isfinite(Y[i]).all()]
            sur = SurrogateEnsemble(n_targets=len(Y[ok[0]]), n_members=n_members, epochs=epochs, seed=seed, top_weight=top_weight)
            sur.fit([smiles[i] for i in ok], np.array([Y[i] for i in ok]))
            mean, std = sur.predict([smiles[i] for i in pool])
            pick = select_batch(mean, std, bbb[pool], batch, mode, rng, beta, min_affinity)
        new = [pool[j] for j in pick]
        for i, y in zip(new, oracle([smiles[i] for i in new])):
            Y[i] = y
        labeled += new
        history.append(list(new))
    return {"order": labeled, "scores": np.array([Y[i] for i in labeled]), "rounds": history}


def evaluate_screening(order: Sequence[int], Y_true: np.ndarray, bbb: np.ndarray, budgets: Sequence[int],
                       top_frac: float = 0.10, min_affinity: float = -7.0, bbb_min: float = 0.5) -> List[dict]:
    """Recall metrics after docking the first `b` molecules of `order`, for each b in budgets.

    * top_recall      fraction of the true top `top_frac` (by mean docking score) found
    * pareto_recall   fraction of the true Pareto front (all targets + BBB) found
    * feasible_recall fraction of molecules that pass the pipeline's own gates found
                      (every target <= min_affinity and BBB probability >= bbb_min)
    """
    Y_true = np.asarray(Y_true, float); bbb = np.asarray(bbb, float)
    n = len(Y_true)
    mean = Y_true.mean(axis=1)
    top = set(np.argsort(mean)[: max(1, int(round(top_frac * n)))])
    df = pd.DataFrame({f"t{j}": Y_true[:, j] for j in range(Y_true.shape[1])}); df["negbbb"] = -bbb
    fr = compute_pareto_fronts(df, list(df.columns), [])
    front = set(np.where(fr["pareto_rank"].to_numpy() == 1)[0])
    feas = set(np.where((Y_true <= min_affinity).all(axis=1) & (bbb >= bbb_min))[0])
    out = []
    for b in budgets:
        got = set(order[:b])
        out.append({"budget": int(b), "budget_frac": b / n,
                    "top_recall": len(got & top) / len(top),
                    "pareto_recall": len(got & front) / max(1, len(front)),
                    "feasible_recall": len(got & feas) / max(1, len(feas)) if feas else float("nan"),
                    "n_top": len(top), "n_front": len(front), "n_feasible": len(feas)})
    return out
