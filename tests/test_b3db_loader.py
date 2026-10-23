"""Biological data leakage prevention, label handling and splits for the B3DB loader."""

from pathlib import Path

import pytest

from vector_engine.b3db_loader import load_b3db_dataset, FDA_CONTROL_SMILES

SAFINAMIDE = "CC(NCc1ccc(OCc2cccc(F)c2)cc1)C(N)=O"
HEADER = "NO.\tcompound_name\tIUPAC_name\tSMILES\tBBB+/BBB-\tlogBB\n"


def _write(tmp_path: Path, rows: list) -> str:
    path = tmp_path / "mock_b3db.tsv"
    path.write_text(HEADER + "".join(f"{i}\t{n}\t\t{s}\t{l}\t0.0\n" for i, (n, s, l) in enumerate(rows, 1)),
                    encoding="utf-8")
    return str(path)


def test_loader_quarantine_and_labels(tmp_path: Path):
    """FDA controls are dropped by name and by topology; remaining labels map to y."""
    csv_path = _write(tmp_path, [
        ("Donepezil-derivative", "CCO", "BBB+"),   # dropped by name
        ("Unknown", SAFINAMIDE, "BBB+"),            # dropped by topology
        ("Valid-Linear", "CCCC", "BBB+"),
        ("Valid-Ring", "c1ccccc1", "BBB-"),
    ])
    train, val, test = load_b3db_dataset(csv_path=csv_path)

    graphs = [g for ds in (train, val, test) for g in ds]
    assert len(graphs) == 2
    assert {g.smiles: int(g.y.item()) for g in graphs} == {"CCCC": 1, "c1ccccc1": 0}


def test_safinamide_control_is_the_real_structure():
    from rdkit import Chem
    from rdkit.Chem.rdMolDescriptors import CalcMolFormula
    assert CalcMolFormula(Chem.MolFromSmiles(SAFINAMIDE)) == "C17H19FN2O2"
    assert Chem.MolToSmiles(Chem.MolFromSmiles(SAFINAMIDE)) in FDA_CONTROL_SMILES


def test_split_is_deterministic_and_scaffold_disjoint(tmp_path: Path):
    smiles = ["c1ccccc1C" * 1, "c1ccccc1CC", "c1ccccc1CCC", "C1CCCCC1C", "C1CCCCC1CC", "C1CCCCC1CCC",
              "c1ccncc1C", "c1ccncc1CC", "C1CCNCC1C", "C1CCNCC1CC", "c1ccoc1C", "c1ccoc1CC"]
    rows = [(f"m{i}", s, "BBB+" if i % 2 else "BBB-") for i, s in enumerate(smiles)]
    csv_path = _write(tmp_path, rows)

    a = load_b3db_dataset(csv_path=csv_path, seed=7)
    b = load_b3db_dataset(csv_path=csv_path, seed=7)
    assert [[g.smiles for g in ds] for ds in a] == [[g.smiles for g in ds] for ds in b]

    scaf = [{g.scaffold for g in ds} for ds in a]
    assert not (scaf[0] & scaf[1]) and not (scaf[0] & scaf[2]) and not (scaf[1] & scaf[2])


def test_invalid_split_mode(tmp_path: Path):
    with pytest.raises(ValueError):
        load_b3db_dataset(csv_path=_write(tmp_path, [("a", "CCCC", "BBB+")]), split_mode="nope")


def test_empty_dataset_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        load_b3db_dataset(csv_path=_write(tmp_path, [("Donepezil", "CCO", "BBB+")]))
