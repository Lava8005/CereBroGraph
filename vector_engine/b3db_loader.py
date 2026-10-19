"""B3DB Dataset Loader & FDA Control Quarantine."""

import pandas as pd
import torch
import logging
from torch_geometric.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
from vector_engine.b3db_to_graph import smiles_to_graph
from rdkit import Chem
FDA_CONTROL_SMILES = {
    Chem.MolToSmiles(Chem.MolFromSmiles("COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2"), canonical=True), # Donepezil
    Chem.MolToSmiles(Chem.MolFromSmiles("CC(Nc1ccc(OCc2ccccc2F)cc1)C(N)=O"), canonical=True), # Safinamide
    Chem.MolToSmiles(Chem.MolFromSmiles("CN(C)C(=O)C(CCN1CCC(O)(c2ccc(Cl)cc2)CC1)(c1ccccc1)c1ccccc1"), canonical=True), # Loperamide (THE FIX)
    Chem.MolToSmiles(Chem.MolFromSmiles("Cc1nc2n(c(=O)c1CCN1CCC(c3noc4cc(F)ccc34)CC1)CCCC2"), canonical=True), # Risperidone
    Chem.MolToSmiles(Chem.MolFromSmiles("CC(C)C1CC(C)CC(C1)N(Cc1ccc(F)cc1)C(=O)N1CCN(Cc2ccc(OCC(F)(F)F)cc2)CC1"), canonical=True) # Pimavanserin
}

class B3DBDataset(Dataset):
    def __init__(self, data_list):
        super().__init__(root=None, transform=None, pre_transform=None)
        self.data_list = data_list
    def len(self): return len(self.data_list)
    def get(self, idx): return self.data_list[idx]

def load_b3db_dataset(csv_path="data/training/B3DB_classification.tsv", split_mode="scaffold", seed=42):
    df = pd.read_csv(csv_path, sep="\t")
    graphs, scaffolds = [], []
    
    for _, row in df.iterrows():
        smi = row.get("SMILES", "")
        mol = Chem.MolFromSmiles(smi)
        if not mol: continue
        canonical_smi = Chem.MolToSmiles(mol, canonical=True)
        
        if canonical_smi in FDA_CONTROL_SMILES:
            continue

        # logBB Rule: Use continuous logBB if available, otherwise fallback to class
        if "logBB" in row and pd.notna(row["logBB"]):
            label = 1 if float(row["logBB"]) >= -0.1 else 0
        else:
            label = 1 if row.get("BBB+/BBB-", "BBB-") == "BBB+" else 0
            
        graph = smiles_to_graph(smi, is_inference=False)
        if graph is not None:
            graph.y = torch.tensor([label], dtype=torch.long)
            # Handle acyclic molecules
            scaffold = getattr(graph, "scaffold", "")
            scaffolds.append(scaffold if scaffold else "ACYCLIC")
            graphs.append(graph)
            
    if not graphs: return None, None
        
    df_scaffolds = pd.DataFrame({"scaffold": scaffolds})
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(df_scaffolds, groups=df_scaffolds["scaffold"]))
    
    train_dataset = B3DBDataset([graphs[i] for i in train_idx])
    val_dataset = B3DBDataset([graphs[i] for i in val_idx])
    logging.info(f"Loaded {len(graphs)} graphs. Train: {len(train_dataset)}, Val: {len(val_dataset)}.")
    return train_dataset, val_dataset