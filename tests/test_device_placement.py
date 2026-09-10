import pytest
import torch
from unittest.mock import MagicMock
from vector_engine.train_eval import train_epoch

def test_epoch_device_transfer(monkeypatch):
    """Ensure the epoch loops explicitly cast CPU dataloader batches to the model's device."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Mock model and criterion
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.8], [0.2]], device=device)
    mock_optimizer = MagicMock()
    mock_criterion = MagicMock()
    mock_criterion.return_value = torch.tensor(0.5, device=device)
    
    # Mock a batch that originates on the CPU
    class MockBatch:
        def __init__(self):
            self.x = torch.ones((2, 5))
            self.edge_index = torch.ones((2, 2))
            self.batch = torch.zeros(2)
            self.y = torch.tensor([1.0, 0.0])
            self.num_graphs = 2
            
        def to(self, dev):
            self.x = self.x.to(dev)
            self.y = self.y.to(dev)
            return self

    mock_loader = [MockBatch()]
    mock_loader.dataset = [1, 2] # Dummy len
    
    # Execute
    loss = train_epoch(mock_model, mock_loader, mock_optimizer, mock_criterion, device)
    
    # Verify the model was called with tensors on the correct device
    args, _ = mock_model.call_args
    assert args[0].device == device