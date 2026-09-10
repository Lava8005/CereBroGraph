import pytest
import torch
from torch_geometric.data import Data, Batch
from vector_engine.gnn_model import BBBGraphModel

def test_gnn_model_initialization():
    """Ensure model initializes with correct layer dimensions."""
    model = BBBGraphModel(num_node_features=5, hidden_dim=64)
    assert model is not None
    assert model.conv1.in_channels == 5
    assert model.classifier[0].in_features == 64

def test_gnn_model_forward_pass():
    """Verify the model processes variable-sized graphs and outputs a valid 0-1 probability."""
    model = BBBGraphModel(num_node_features=5, hidden_dim=32)
    model.eval()

    # Mock Graph 1 (e.g., Ethanol: 3 atoms, 2 bonds)
    x1 = torch.rand((3, 5))
    edge_index1 = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    data1 = Data(x=x1, edge_index=edge_index1)

    # Mock Graph 2 (e.g., Benzene: 6 atoms, 6 bonds)
    x2 = torch.rand((6, 5))
    edge_index2 = torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,0], [1,0,2,1,3,2,4,3,5,4,0,5]], dtype=torch.long)
    data2 = Data(x=x2, edge_index=edge_index2)

    # Batch them together as PyG DataLoader would
    batch = Batch.from_data_list([data1, data2])

    with torch.no_grad():
        out = model(batch.x, batch.edge_index, batch.batch)

    assert out.shape == (2, 1), "Output should be shape [batch_size, 1]"
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0), "Output must be a probability between 0 and 1"