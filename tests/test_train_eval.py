import pytest
import torch
from pathlib import Path
from vector_engine.gnn_model import BBBGraphModel

def test_model_checkpoint_serialization(tmp_path: Path):
    """Ensure the model's state_dict can be successfully written to disk."""
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "best_bbb_gnn.pt"
    
    # Initialize dummy model
    model = BBBGraphModel(num_node_features=5, hidden_dim=64)
    
    # Simulate saving best weights
    torch.save(model.state_dict(), checkpoint_path)
    
    assert checkpoint_path.exists()
    assert checkpoint_path.stat().st_size > 1000  # Verify it's not an empty file