"""B3DB Dataset Loader & FDA Control Quarantine."""

import pandas as pd
import torch
from torch_geometric.data import Dataset
from sklearn.model_selection import GroupShuffleSplit
from vector_engine.b3db_to_graph import smiles_to_graph
import logging

# Strictly quarantine these from training to prevent Validation 2 data leakage
FDA_CONTROLS = {"DONEPEZIL", "SAFINAMIDE", "LOPERAMIDE", "RISPERIDONE", "PIMAVANSERIN"}

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
        # Apply standard logBB threshold for BBB+ vs BBB-
        label = 1 if row.get("BBB+/BBB-", "BBB-") == "BBB+" else 0
        
        # Enforce FDA Quarantine
        if any(control in smi.upper() for control in FDA_CONTROLS):
            continue
            
        graph = smiles_to_graph(smi, is_inference=False)
        if graph is not None and hasattr(graph, "scaffold") and graph.scaffold:
            graph.y = torch.tensor([label], dtype=torch.long)
            graphs.append(graph)
            scaffolds.append(graph.scaffold)
            
    if not graphs:
        return None, None
        
    df_scaffolds = pd.DataFrame({"scaffold": scaffolds})
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
    train_idx, val_idx = next(gss.split(df_scaffolds, groups=df_scaffolds["scaffold"]))
    
    train_dataset = B3DBDataset([graphs[i] for i in train_idx])
    val_dataset = B3DBDataset([graphs[i] for i in val_idx])
    
    logging.info(f"Loaded {len(graphs)} graphs. Train: {len(train_dataset)}, Val: {len(val_dataset)}.")
    return train_dataset, val_dataset