import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class BBBGraphModel(nn.Module):
    """
    Graph Convolutional Network for Blood-Brain Barrier permeability prediction.
    Aggregates atomic features into a molecular embedding, outputting a 0-1 probability.
    """
    def __init__(self, num_node_features: int = 5, hidden_dim: int = 64):
        super(BBBGraphModel, self).__init__()
        
        # Graph Convolutional Layers
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)

        # Multi-Layer Perceptron (MLP) Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the GNN.
        x: Node feature matrix [num_atoms, num_features]
        edge_index: Graph connectivity matrix [2, num_edges]
        batch: Indicator vector mapping atoms to their respective graphs in a batch
        """
        # Message Passing Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Message Passing Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)

        # Message Passing Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)

        # Global Pooling: Collapse [num_atoms, hidden_dim] -> [batch_size, hidden_dim]
        # This handles molecules of different sizes dynamically.
        x = global_mean_pool(x, batch)

        # Final Classification
        logits = self.classifier(x)
        probability = torch.sigmoid(logits)
        
        return probability