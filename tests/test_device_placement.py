"""TDD Suite for Device Placement & GPU Memory Pinning."""

import pytest
import torch
from unittest.mock import MagicMock
from vector_engine.train_eval import train_epoch

def test_epoch_device_transfer():
    """Ensure the epoch loops explicitly cast CPU dataloader batches to the model's device."""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Mock model
    mock_model = MagicMock()
    mock_model.return_value = torch.tensor([[0.8], [0.2]], device=device)
    mock_optimizer = MagicMock()
    
    # THE FIX: Decouple the loss from the PyTorch autograd engine using MagicMock.
    # This safely intercepts loss.backward() and loss.item() without throwing grad_fn errors.
    mock_loss = MagicMock()
    mock_loss.item.return_value = 0.5
    
    mock_criterion = MagicMock()
    mock_criterion.return_value = mock_loss
    
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
            
    # Wrap the batches in a mock DataLoader class
    class MockDataLoader:
        def __init__(self, batches):
            self.batches = batches
            self.dataset = [1, 2]  # Dummy dataset length to satisfy len(loader.dataset) denominator
            
        def __iter__(self):
            return iter(self.batches)
            
        def __len__(self):
            return len(self.batches)
            
    mock_loader = MockDataLoader([MockBatch()])
    
    # Execute
    loss = train_epoch(mock_model, mock_loader, mock_optimizer, mock_criterion, device)
    
    # Verify the model was called with tensors on the correct device
    args, _ = mock_model.call_args
    assert args[0].device == device
    
    # Verify autograd backward and optimizer steps were successfully bypassed/called
    mock_loss.backward.assert_called_once()
    mock_optimizer.step.assert_called_once()