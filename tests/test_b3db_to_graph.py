import pytest
import torch
from pathlib import Path
from vector_engine.b3db_to_graph import smiles_to_graph, get_b3db_dataloaders

def test_smiles_to_graph_topology():
    """Ensure RDKit safely translates topology into PyG tensors of the correct shape."""
    graph = smiles_to_graph("CCO")  # Ethanol: 3 heavy atoms
    assert graph is not None
    assert graph.x.shape == (3, 5)  # 3 nodes, 5 topological features
    assert graph.edge_index.shape == (2, 4)  # 2 undirected bonds = 4 directed edges

def test_tsv_classification_loader(tmp_path: Path):
    """Ensure the dataloader correctly parses TSV files and stratifies batches."""
    mock_tsv = tmp_path / "B3DB_classification.tsv"
    mock_tsv.write_text("SMILES\tBBB+/BBB-\nCCO\tBBB+\nCCN\tBBB-\n")
    
    train_loader, val_loader = get_b3db_dataloaders(
        csv_path=mock_tsv, 
        batch_size=2, 
        val_split=0.0
    )
    
    assert len(train_loader.dataset) == 2
    for batch in train_loader:
        assert isinstance(batch.x, torch.Tensor)
        assert set(batch.y.view(-1).tolist()).issubset({0.0, 1.0})
        break