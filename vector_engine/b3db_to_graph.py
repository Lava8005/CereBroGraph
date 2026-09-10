"""SMILES to PyTorch Geometric Graph Converter & Dataset Loader.

Translates 1D chemical string representations into 2D graph tensors
and orchestrates batched DataLoaders for the Blood-Brain Barrier dataset.
"""

import csv
import logging
from pathlib import Path
from typing import Optional, Tuple

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split

try:
    from rdkit import Chem
except ImportError:
    raise ImportError("RDKit is required. Install via: pip install rdkit")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def smiles_to_graph(smiles: str) -> Optional[Data]:
    """
    Convert a SMILES string into a PyTorch Geometric Data object.
    Extracts exactly 5 node features using modern RDKit APIs.
    """
    if not smiles or not isinstance(smiles, str):
        return None

    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # 1. Node Features Extraction (Shape: [Num_Atoms, 5])
        node_features = []
        for atom in mol.GetAtoms():
            # Modern API usage prevents terminal I/O bottlenecks
            implicit_valence = atom.GetValence(Chem.ValenceType.IMPLICIT)
            
            features = [
                float(atom.GetAtomicNum()),
                float(atom.GetDegree()),
                float(atom.GetTotalNumHs()),
                float(implicit_valence),
                float(atom.GetIsAromatic())
            ]
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float)

        # 2. Edge Index Extraction (Adjacency Matrix in COO format)
        edge_indices = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            
            # PyG requires directed edges, so we add both directions for undirected chemical bonds
            edge_indices.append([i, j])
            edge_indices.append([j, i])

        if edge_indices:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        else:
            # Handle edge-case single-atom molecules (e.g., isolated ions)
            edge_index = torch.empty((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    except Exception as e:
        logging.debug(f"Failed to process SMILES '{smiles}': {e}")
        return None


def get_b3db_dataloaders(
    csv_path: Path = Path("data/training/B3DB_classification.tsv"),
    batch_size: int = 128,
    val_split: float = 0.2,
    random_seed: int = 42
) -> Tuple[DataLoader, DataLoader]:
    """
    Parse the B3DB dataset, execute RDKit graph conversions, and yield pinned DataLoaders.
    Dynamically supports both CSV and TSV formats.
    """
    if not isinstance(csv_path, Path):
        csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training dataset missing at {csv_path}. "
            "Please ensure the B3DB classification dataset is placed in data/training/."
        )

    logging.info(f"Loading and processing dataset from {csv_path}...")
    
    dataset_graphs = []
    dropped_count = 0

    # Auto-detect delimiter based on file extension
    delimiter = '\t' if csv_path.suffix.lower() == '.tsv' else ','

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        
        # Dynamically map the target label column based on known B3DB schemas
        label_col = next((col for col in reader.fieldnames if col in ["BBB+/BBB-", "p_np", "label", "BBB"]), None)
        if not label_col:
            raise ValueError(f"Could not identify a target label column in {csv_path}. Found: {reader.fieldnames}")

        for row in reader:
            # Handle various standard SMILES column names
            raw_smiles = row.get("SMILES", row.get("smiles", row.get("compound_smiles")))
            raw_label = row.get(label_col)
            
            if not raw_smiles or not raw_label:
                continue
                
            # Parse label into binary 1.0 or 0.0
            if isinstance(raw_label, str):
                y_val = 1.0 if "BBB+" in raw_label or raw_label == "1" else 0.0
            else:
                y_val = float(raw_label)

            graph = smiles_to_graph(raw_smiles)
            
            if graph is not None:
                graph.y = torch.tensor([y_val], dtype=torch.float)
                dataset_graphs.append(graph)
            else:
                dropped_count += 1

    total_valid = len(dataset_graphs)
    if total_valid == 0:
        raise RuntimeError("Dataset processing failed: 0 valid graphs were generated.")

    logging.info(f"Successfully generated {total_valid} graphs. Dropped {dropped_count} invalid SMILES.")

    # Stratified split to strictly maintain BBB+ / BBB- class balance
    labels = [g.y.item() for g in dataset_graphs]
    train_graphs, val_graphs = train_test_split(
        dataset_graphs, 
        test_size=val_split, 
        random_state=random_seed, 
        stratify=labels
    )

    logging.info(f"Dataset split: {len(train_graphs)} Train | {len(val_graphs)} Validation")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader