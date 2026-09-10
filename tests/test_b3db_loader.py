import pytest
import torch
from pathlib import Path
from vector_engine.b3db_to_graph import get_b3db_dataloaders

def test_dataloader_batching_and_filtering(tmp_path: Path):
    """Ensure the loader handles valid/invalid SMILES, extracts labels, and splits batches."""
    mock_csv = tmp_path / "b3db_mock.csv"
    mock_csv.write_text("SMILES,BBB+/BBB-\nCCO,BBB+\nINVALID_SMILES,BBB-\nCCN,BBB-\n")
    
    # Execute
    train_loader, val_loader = get_b3db_dataloaders(csv_path=mock_csv, batch_size=2, val_split=0.0)
    
    # Assertions
    assert len(train_loader.dataset) == 2  # The invalid SMILES must be dropped
    
    for batch in train_loader:
        assert isinstance(batch.x, torch.Tensor)
        assert isinstance(batch.y, torch.Tensor)
        assert batch.y.shape == (2,) # Batch size of 2
        break