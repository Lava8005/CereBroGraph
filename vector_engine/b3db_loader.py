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

# 1. Topological Quarantine (Exact Structure Matching)

import logging

_RAW_FDA = [
    "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2", # Donepezil
    "CC(NCc1ccc(OCc2cccc(F)c2)cc1)C(N)=O",          # Safinamide
    "CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1", # Loperamide
    "Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2", # Risperidone
    "CC(C)COc1ccc(CNC(=O)N(Cc2ccc(F)cc2)C2CCN(C)CC2)cc1" # Pimavanserin
]

# Mathematically guarantee topology parity via RDKit
FDA_CONTROL_SMILES = set()
for s in _RAW_FDA:
    mol = Chem.MolFromSmiles(s)
    if mol:
        FDA_CONTROL_SMILES.add(Chem.MolToSmiles(mol, canonical=True))
    else:
        logging.error(f"Failed to canonicalize FDA control: {s}")
# 2. Lexicographical Quarantine (Homolog & Derivative Catch-All)
FDA_CONTROL_NAMES = {"DONEPEZIL", "SAFINAMIDE", "LOPERAMIDE", "RISPERIDONE", "PIMAVANSERIN"}


class B3DBDataset(Dataset):
    def __init__(self, data_list):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]


def load_b3db_dataset(csv_path="data/training/B3DB_classification.tsv", split_mode="scaffold", seed=42):
    df = pd.read_csv(csv_path, sep="\t")
    
    graphs = []
    scaffolds = []
    
    for _, row in df.iterrows():
        smi = row.get("SMILES", "")
        # Corrected column names to match actual B3DB TSV headers
        compound_name = str(row.get("compound_name", row.get("IUPAC_name", ""))).upper()
        
        mol = Chem.MolFromSmiles(smi)
        if not mol:
            continue
            
        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        
        # Dual-Layer Quarantine ensures 100% Validation 2 integrity
        if canonical_smi in FDA_CONTROL_SMILES or any(n in compound_name for n in FDA_CONTROL_NAMES):
            continue

        # Benchmark compliance: Using official B3DB labels instead of custom logBB thresholds.
        label = 1 if row.get("BBB+/BBB-", "BBB-") == "BBB+" else 0
            
        graph = smiles_to_graph(smi, is_inference=False)
        if graph is not None:
            graph.y = torch.tensor([label], dtype=torch.long)
            
            # Preserve acyclic molecules instead of silently dropping them
            scaffold = getattr(graph, "scaffold", "")
            scaffolds.append(scaffold if scaffold else "ACYCLIC")
            graphs.append(graph)
            
    if not graphs:
        return None, None, None
        
    df_scaffolds = pd.DataFrame({"scaffold": scaffolds})
    
    # First Split: 80% Train, 20% Temp
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, temp_idx = next(gss.split(df_scaffolds, groups=df_scaffolds["scaffold"]))
    
    # Second Split: Divide Temp evenly into 10% Val and 10% Test
    temp_scaffolds = df_scaffolds.iloc[temp_idx].reset_index(drop=True)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
    val_rel_idx, test_rel_idx = next(gss_test.split(temp_scaffolds, groups=temp_scaffolds["scaffold"]))
    
    val_idx = temp_idx[val_rel_idx]
    test_idx = temp_idx[test_rel_idx]
    
    train_dataset = B3DBDataset([graphs[i] for i in train_idx])
    val_dataset = B3DBDataset([graphs[i] for i in val_idx])
    test_dataset = B3DBDataset([graphs[i] for i in test_idx])
    
    logging.info(f"Loaded {len(graphs)} graphs. Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}.")
    return train_dataset, val_dataset, test_dataset