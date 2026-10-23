"""Unit tests for docking helpers that do not need the Vina binary."""

import pytest

from physics_engine import dock_vina

VINA_OUTPUT = """mode |   affinity | dist from best mode
     | (kcal/mol) | rmsd l.b.| rmsd u.b.
-----+------------+----------+----------
   1         -7.5      0.000      0.000
   2         -7.1      1.200      2.000
"""


def test_best_pose_regex_takes_first_mode_only():
    scores = [float(m.group(1)) for line in VINA_OUTPUT.split("\n")
              if (m := dock_vina._VINA_BEST_POSE.match(line))]
    assert scores == [-7.5]


def test_ligand_preparation_produces_pdbqt():
    pytest.importorskip("meeko")
    pdbqt = dock_vina.prepare_ligand_pdbqt("CCO")
    assert pdbqt and "ATOM" in pdbqt


def test_invalid_smiles_returns_none():
    assert dock_vina.prepare_ligand_pdbqt("not-a-smiles") is None


def test_docking_exits_when_meeko_missing(tmp_path, monkeypatch):
    inp = tmp_path / "in.csv"; inp.write_text("candidate_code,smiles\nA,CCO\n")
    cfg = tmp_path / "cfg.json"; cfg.write_text("[]")
    monkeypatch.setattr(dock_vina, "MoleculePreparation", None)
    with pytest.raises(SystemExit) as exc:
        dock_vina.batch_execute_docking(inp, tmp_path / "out.csv", cfg)
    assert exc.value.code == 1


def test_missing_receptor_file_aborts(tmp_path, monkeypatch):
    inp = tmp_path / "in.csv"; inp.write_text("candidate_code,smiles\nA,CCO\n")
    cfg = tmp_path / "cfg.json"
    cfg.write_text('[{"disease": "T", "pdbqt": "/nope.pdbqt", "grid": "/nope.json"}]')
    monkeypatch.setattr(dock_vina.subprocess, "run", lambda *a, **k: None)  # pretend vina exists
    with pytest.raises(SystemExit) as exc:
        dock_vina.batch_execute_docking(inp, tmp_path / "out.csv", cfg)
    assert exc.value.code == 1
