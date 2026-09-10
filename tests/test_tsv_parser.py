import pytest
import torch
from pathlib import Path
from vector_engine.b3db_to_graph import get_b3db_dataloaders

def test_tsv_classification_parsing(tmp_path: Path):
    """Ensure the loader properly processes tab-separated classification files."""
    mock_tsv = tmp_path / "B3DB_classification.tsv"
    # Mocking standard B3DB TSV format with tabs
    mock_tsv.write_text("SMILES\tBBB+/BBB-\nCCO\tBBB+\nCCN\tBBB-\n")
    
    train_loader, val_loader = get_b3db_dataloaders(
        csv_path=mock_tsv, 
        batch_size=2, 
        val_split=0.0
    )
    
    assert len(train_loader.dataset) == 2
    for batch in train_loader:
        assert isinstance(batch.y, torch.Tensor)
        # Ensure labels are binary probabilities
        assert set(batch.y.view(-1).tolist()).issubset({0.0, 1.0})
        break