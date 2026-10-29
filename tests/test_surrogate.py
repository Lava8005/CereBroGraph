"""Surrogate screening loop: shapes, determinism, and that the selection modes run."""
import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors

from screening.surrogate import SurrogateEnsemble, evaluate_screening, run_screening, select_batch

SMILES = ["CCO", "CCN", "CCC", "c1ccccc1", "CC(=O)O", "CCCC", "c1ccncc1", "CCOC", "CC(C)O", "CCCCC",
          "c1ccc(O)cc1", "CC(=O)N", "CCCN", "C1CCCCC1", "c1ccsc1", "CCCO", "CC(C)N", "CCCCO", "c1ccoc1", "CC#N",
          "CCCCN", "C1CCOC1", "CCOCC", "c1ccc(N)cc1", "CC(C)=O", "CCS", "CCCS", "CCCCC=O", "OCCO", "NCCN"]


def oracle(smis):
    return np.array([[-0.5 * Chem.MolFromSmiles(s).GetNumHeavyAtoms() - 3.0,
                      -0.2 * Descriptors.MolLogP(Chem.MolFromSmiles(s)) - 4.0] for s in smis])


def test_surrogate_predict_shapes():
    sur = SurrogateEnsemble(n_targets=2, n_members=2, epochs=3).fit(SMILES[:20], oracle(SMILES[:20]))
    m, s = sur.predict(SMILES[20:])
    assert m.shape == (10, 2) and s.shape == (10, 2) and np.isfinite(m).all()


@pytest.mark.parametrize("mode", ["random", "greedy", "ucb", "pareto", "pareto_dock", "feasible"])
def test_run_screening_modes(mode):
    res = run_screening(SMILES, np.linspace(0.2, 0.9, len(SMILES)), oracle, mode=mode,
                        n_init=8, batch=4, rounds=2, n_members=2, epochs=3)
    assert len(res["order"]) == 8 + 4 * 2 and len(set(res["order"])) == len(res["order"])
    assert res["scores"].shape == (16, 2)


def test_select_batch_pareto_prefers_dominant():
    mean = np.array([[-9., -9.], [-5., -5.], [-9., -4.], [-4., -9.]])
    pick = select_batch(mean, np.zeros_like(mean), np.array([0.9, 0.1, 0.9, 0.9]), 1, "pareto",
                        np.random.default_rng(0))
    assert pick[0] == 0


def test_evaluate_recall_full_budget_is_one():
    Y = oracle(SMILES)
    out = evaluate_screening(list(range(len(SMILES))), Y, np.full(len(SMILES), 0.9), [len(SMILES)], min_affinity=-5.0)
    assert out[0]["top_recall"] == 1.0 and out[0]["pareto_recall"] == 1.0


def test_vina_oracle_caches_and_resumes(tmp_path, monkeypatch):
    import json
    import screening.vina_oracle as vo
    calls = []
    monkeypatch.setattr(vo, "prepare_ligand_pdbqt", lambda s: "REMARK dummy\n")
    def fake_dock(exe, path, target, exh, to):
        calls.append(target["disease"])
        return target["disease"], -7.5 if target["disease"] == "A" else 0.0   # target B always fails
    monkeypatch.setattr(vo, "dock_precomputed_target", fake_dock)
    idx = tmp_path / "t.json"
    idx.write_text(json.dumps([{"disease": "A", "pdbqt": "x", "grid": "y"}, {"disease": "B", "pdbqt": "x", "grid": "y"}]))
    o = vo.VinaOracle(idx, tmp_path / "cache.csv", max_workers=2, work_dir=tmp_path / "w")
    out = o(["CCO", "CCN", "CCO"])
    assert out.shape == (3, 2) and (out[:, 0] == -7.5).all() and np.isnan(out[:, 1]).all()
    assert o.docking_calls == 2
    o2 = vo.VinaOracle(idx, tmp_path / "cache.csv", max_workers=2, work_dir=tmp_path / "w")  # resume
    n_before = len(calls)
    o2(["CCO", "CCN"])
    assert len(calls) == n_before and o2.docking_calls == 0


def test_feasible_mode_prefers_confident_feasible():
    mean = np.array([[-9., -9.], [-9., -3.], [-6., -6.], [-8., -8.]])
    std = np.full_like(mean, 0.3)
    pick = select_batch(mean, std, np.array([0.9, 0.9, 0.9, 0.2]), 2, "feasible", np.random.default_rng(0), min_affinity=-7.0)
    assert set(pick) == {0, 3} or pick[0] == 0
    assert pick[0] == 0 and 1 not in pick


def test_top_weighted_loss_runs():
    sur = SurrogateEnsemble(n_targets=2, n_members=1, epochs=2, top_weight=1.0).fit(SMILES[:20], oracle(SMILES[:20]))
    assert sur.predict(SMILES[20:])[0].shape == (10, 2)
