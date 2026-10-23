import torch
from torch_geometric.data import Data, Batch
from vector_engine.gnn_model import BBBGraphModel


def test_gnn_model_initialization():
    """Ensure model initializes with correct layer dimensions."""
    model = BBBGraphModel(num_node_features=5, edge_dim=7, global_dim=2, hidden_dim=64)
    assert model.node_emb.in_features == 5
    assert model.node_emb.out_features == 64
    assert model.mlp[0].in_features == 64 * 2 + 2  # add-pool + max-pool + globals
    assert model.mlp[-1].out_features == 1


def test_gnn_model_forward_pass():
    """Variable-sized graphs batch together and yield one logit each; sigmoid gives a probability."""
    model = BBBGraphModel(num_node_features=5, edge_dim=7, global_dim=2, hidden_dim=32)
    model.eval()

    def make_graph(n_atoms, edges):
        ei = torch.tensor(edges, dtype=torch.long)
        return Data(
            x=torch.rand((n_atoms, 5)),
            edge_index=ei,
            edge_attr=torch.rand((ei.shape[1], 7)),
            global_features=torch.rand((1, 2)),
        )

    g1 = make_graph(3, [[0, 1, 1, 2], [1, 0, 2, 1]])  # ethanol-like
    g2 = make_graph(6, [[0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 0], [1, 0, 2, 1, 3, 2, 4, 3, 5, 4, 0, 5]])  # ring
    batch = Batch.from_data_list([g1, g2])

    with torch.no_grad():
        logits = model(batch.x, batch.edge_index, batch.batch,
                       edge_attr=batch.edge_attr, global_features=batch.global_features)
        probs = torch.sigmoid(logits)

    assert logits.shape == (2, 1)
    assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)
