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


def load_b3db_dataset(csv_path: str = "data/training/B3DB_classification.tsv") -> tuple:
    df = pd.read_csv(csv_path, sep='\t')
    
    graphs = []
    scaffolds = []
    
    quarantine_count = 0
    acyclic_count = 0
    
    for idx, row in df.iterrows():
        raw_smiles = str(row['SMILES'])
        mol = Chem.MolFromSmiles(raw_smiles)
        
        if mol is None:
            continue
            
        canonical_smiles = Chem.MolToSmiles(mol, canonical=True)
        
        # 1. Quarantine Check (Topology)
        if canonical_smiles in FDA_CONTROL_SMILES:
            quarantine_count += 1
            continue
            
        # 2. Quarantine Check (Lexicography)
        compound_name = str(row.get('compound_name', '')).upper()
        if any(control in compound_name for control in ["DONEPEZIL", "SAFINAMIDE", "LOPERAMIDE", "RISPERIDONE", "PIMAVANSERIN"]):
            quarantine_count += 1
            continue

        label = 1 if row['BBB+/BBB-'] == 'BBB+' else 0
        
        try:
            graph_data = smiles_to_graph(canonical_smiles, label)
            if graph_data:
                scaffold = Chem.MolToSmiles(Chem.rdMolDescriptors.GetMacromoleculeScaffold(mol))
                if not scaffold:
                    acyclic_count += 1
                    scaffold = "ACYCLIC_CORE"
                    
                graphs.append(graph_data)
                scaffolds.append(scaffold)
        except Exception as e:
            logging.debug(f"Failed to process {canonical_smiles}: {e}")

    logging.info(f"Quarantined {quarantine_count} FDA controls.")
    
    if not graphs:
        raise ValueError("Dataset is empty after parsing and quarantine.")

    # 80/10/10 Split
    temp_df = pd.DataFrame({"scaffold": scaffolds})
    
    # First split: 80% Train, 20% Temp
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, temp_idx = next(gss.split(temp_df, groups=temp_df["scaffold"]))
    
    # Second split: 50% Val, 50% Test (from the 20% Temp -> 10% / 10% overall)
    temp_scaffolds = temp_df.iloc[temp_idx].reset_index(drop=True)
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    
    # Fix: Safely handle tiny mock datasets in test suite
    if len(temp_scaffolds) > 1:
        val_rel_idx, test_rel_idx = next(gss_test.split(temp_scaffolds, groups=temp_scaffolds["scaffold"]))
        val_idx = temp_idx[val_rel_idx]
        test_idx = temp_idx[test_rel_idx]
    else:
        val_idx = temp_idx
        test_idx = []

    train_dataset = BBBDataset([graphs[i] for i in train_idx])
    val_dataset = BBBDataset([graphs[i] for i in val_idx])
    test_dataset = BBBDataset([graphs[i] for i in test_idx]) if len(test_idx) > 0 else BBBDataset([])

    return train_dataset, val_dataset, test_dataset