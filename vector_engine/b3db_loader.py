"""B3DB Dataset Loader & FDA Control Quarantine.

Parses the Blood-Brain Barrier dataset, extracts mathematically rigorous labels
and enforces a deterministic Bemis-Murcko scaffold split.
Implements a dual-layer quarantine (Topology + Lexicography) to mathematically 
guarantee FDA controls never leak into the training split.
"""

import logging
import pandas as pd
import torch
from torch_geometric.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
from rdkit import Chem

from vector_engine.b3db_to_graph import smiles_to_graph

# Scientifically Accurate FDA Controls
_RAW_FDA = [
    "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2", # Donepezil
    "CC(NCc1ccc(OCc2cccc(F)c2)cc1)C(N)=O",          # Safinamide (Corrected)
    "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1", # Loperamide
    "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2", # Risperidone
    "CC(C)COc1ccc(CNC(=O)N(Cc2ccc(F)cc2)C2CCN(C)CC2)cc1" # Pimavanserin (Corrected)
]

FDA_CONTROL_NAMES = ["DONEPEZIL", "SAFINAMIDE", "LOPERAMIDE", "RISPERIDONE", "PIMAVANSERIN"]

# Enforce absolute structural parity at runtime to prevent string-matching leakage
FDA_CONTROL_SMILES = set()
for s in _RAW_FDA:
    mol = Chem.MolFromSmiles(s)
    if mol:
        FDA_CONTROL_SMILES.add(Chem.MolToSmiles(mol, canonical=True))


class BBBDataset(Dataset):
    def __init__(self, data_list):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def load_b3db_dataset(
    csv_path: str = "data/training/B3DB_classification.tsv",
    split_mode: str = "scaffold",
    seed: int = 42,
) -> tuple:
    """Load B3DB, quarantine FDA controls, and return (train, val, test) datasets.

    split_mode="scaffold" keeps molecules sharing a Bemis-Murcko scaffold in the same
    split (80/10/10); split_mode="random" shuffles molecules individually.
    """
    if split_mode not in ("scaffold", "random"):
        raise ValueError(f"split_mode must be 'scaffold' or 'random', got {split_mode!r}")

    df = pd.read_csv(csv_path, sep="\t")

    graphs = []
    scaffolds = []
    quarantine_count = 0
    failed_count = 0

    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(str(row["SMILES"]))
        if mol is None:
            failed_count += 1
            continue

        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)

        # 1. Quarantine Check (Topology)
        if canonical_smiles in FDA_CONTROL_SMILES:
            quarantine_count += 1
            continue

        # 2. Quarantine Check (Lexicography)
        compound_name = str(row.get("compound_name", "")).upper()
        if any(control in compound_name for control in FDA_CONTROL_NAMES):
            quarantine_count += 1
            continue

        label = 1 if row["BBB+/BBB-"] == "BBB+" else 0

        try:
            graph_data = smiles_to_graph(canonical_smiles, label=label)
        except Exception as e:
            logging.warning(f"Featurization failed for {canonical_smiles}: {e}")
            failed_count += 1
            continue
        if graph_data is None:
            failed_count += 1
            continue

        graphs.append(graph_data)
        scaffolds.append(getattr(graph_data, "scaffold", "") or "ACYCLIC_CORE")

    logging.info(
        f"Loaded {len(graphs)} molecules. Quarantined {quarantine_count} FDA controls. "
        f"Skipped {failed_count} unparseable rows."
    )

    if not graphs:
        raise ValueError("Dataset is empty after parsing and quarantine.")

    n = len(graphs)
    groups = scaffolds if split_mode == "scaffold" else list(range(n))
    idx_df = pd.DataFrame({"group": groups})

    # 80% train / 20% temp, then temp split 50/50 -> 10% val / 10% test
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(gss.split(idx_df, groups=idx_df["group"]))

    temp_groups = idx_df.iloc[temp_idx]["group"].tolist()
    if len(set(temp_groups)) > 1:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        val_rel, test_rel = next(gss2.split(temp_idx, groups=temp_groups))
        val_idx, test_idx = temp_idx[val_rel], temp_idx[test_rel]
    else:
        val_idx, test_idx = temp_idx, []

    return (
        BBBDataset([graphs[i] for i in train_idx]),
        BBBDataset([graphs[i] for i in val_idx]),
        BBBDataset([graphs[i] for i in test_idx]),
    )
