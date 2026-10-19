"""TDD Suite for Phase 2 Scientific Design: Scaffold Splitting & Leakage Prevention."""

import pytest
import torch
from pathlib import Path
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from vector_engine.b3db_to_graph import (
    extract_murcko_scaffold,
    get_b3db_dataloaders,
    FDA_CONTROL_SMILES,
)
from vector_engine.gnn_model import BBBGraphModel
from vector_engine.train_eval import compute_classification_metrics


def test_murcko_scaffold_extraction():
    """Verify RDKit extracts identical Bemis-Murcko scaffolds for congeneric analogs."""
    # Benzene and Toluene share the exact same 6-membered aromatic ring scaffold
    s1 = extract_murcko_scaffold("c1ccccc1")
    s2 = extract_murcko_scaffold("Cc1ccccc1")
    assert s1 == s2 == "c1ccccc1"


def test_zero_scaffold_leakage_between_splits(tmp_path: Path):
    """Ensure no Bemis-Murcko scaffold is shared between train and validation splits."""
    mock_tsv = tmp_path / "b3db_scaffold_test.tsv"
    rows = [
        "SMILES\tlogBB\tBBB+/BBB-",
        # Scaffold 1: Benzene derivatives (3 molecules)
        "c1ccccc1\t0.40\tBBB+",
        "Cc1ccccc1\t0.45\tBBB+",
        "CCc1ccccc1\t0.50\tBBB+",
        # Scaffold 2: Pyridine derivatives (3 molecules)
        "c1ccncc1\t-0.50\tBBB-",
        "Cc1ccncc1\t-0.40\tBBB-",
        "CCc1ccncc1\t-0.30\tBBB-",
        # Scaffold 3: Indole derivatives (2 molecules)
        "c1ccc2[nH]ccc2c1\t0.20\tBBB+",
        "Cc1cc2ccccc2[nH]1\t0.25\tBBB+",
        # Scaffold 4: Quinoline derivatives (2 molecules)
        "c1ccc2ncccc2c1\t-0.60\tBBB-",
        "Cc1ccnc2ccccc12\t-0.55\tBBB-",
    ]
    mock_tsv.write_text("\n".join(rows), encoding="utf-8")

    train_loader, val_loader = get_b3db_dataloaders(
        csv_path=mock_tsv,
        batch_size=4,
        val_split=0.2,
        random_seed=42,
        split_mode="scaffold",
    )

    train_scaffolds = {g.scaffold for g in train_loader.dataset}
    val_scaffolds = {g.scaffold for g in val_loader.dataset}

    assert len(train_scaffolds) > 0
    assert len(val_scaffolds) > 0
    # Strict mathematical proof: intersection of scaffold sets must be empty
    assert train_scaffolds.isdisjoint(val_scaffolds)


def test_fda_control_leakage_quarantine(tmp_path: Path):
    """Ensure FDA control spikes (Donepezil, Safinamide, Loperamide) are quarantined from training."""
    donepezil_smiles = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"
    loperamide_smiles = "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1"

    mock_tsv = tmp_path / "b3db_leakage.tsv"
    rows = [
        "SMILES\tlogBB\tBBB+/BBB-",
        f"{donepezil_smiles}\t0.89\tBBB+",
        f"{loperamide_smiles}\t-0.25\tBBB+",
        "c1ccccc1\t0.40\tBBB+",
        "c1ccncc1\t-0.60\tBBB-",
    ]
    mock_tsv.write_text("\n".join(rows), encoding="utf-8")

    train_loader, _ = get_b3db_dataloaders(
        csv_path=mock_tsv,
        batch_size=2,
        val_split=0.0,
        exclude_controls=True,
    )

    loaded_smiles = {g.smiles for g in train_loader.dataset}
    canonical_donepezil = Chem.MolToSmiles(Chem.MolFromSmiles(donepezil_smiles), canonical=True)
    canonical_loperamide = Chem.MolToSmiles(Chem.MolFromSmiles(loperamide_smiles), canonical=True)

    assert canonical_donepezil not in loaded_smiles
    assert canonical_loperamide not in loaded_smiles
    assert len(train_loader.dataset) == 2


def test_classification_metrics_suite():
    """Ensure ROC-AUC, Precision, Recall, and F1 are accurately calculated."""
    y_true = [1.0, 1.0, 0.0, 0.0]
    y_probs = [0.92, 0.81, 0.15, 0.22]

    metrics = compute_classification_metrics(y_true, y_probs, threshold=0.5)

    assert metrics["roc_auc"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["accuracy"] == 1.0