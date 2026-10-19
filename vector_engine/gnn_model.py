"""Blood-Brain Barrier (BBB) Graph Neural Network.

Utilizes Graph Isomorphism Networks with Edges (GINEConv) to process rich
chemical graphs. Concatenates global topological descriptors (TPSA, MW) 
directly into the final multi-layer perceptron.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool
from torch.nn import Sequential, Linear, BatchNorm1d, ReLU, Dropout


class BBBGraphModel(nn.Module):
    def __init__(self, num_node_features=27, edge_dim=7, global_dim=2, hidden_dim=64):
        super(BBBGraphModel, self).__init__()
        
        # 1. Initial Encoders
        self.node_emb = Linear(num_node_features, hidden_dim)
        self.edge_emb = Linear(edge_dim, hidden_dim)
        
        # 2. GINE Convolutions (Requires an internal MLP per layer)
        nn1 = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv1 = GINEConv(nn1, edge_dim=hidden_dim)
        self.bn1 = BatchNorm1d(hidden_dim)
        
        nn2 = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv2 = GINEConv(nn2, edge_dim=hidden_dim)
        self.bn2 = BatchNorm1d(hidden_dim)
        
        nn3 = Sequential(Linear(hidden_dim, hidden_dim), BatchNorm1d(hidden_dim), ReLU(), Linear(hidden_dim, hidden_dim))
        self.conv3 = GINEConv(nn3, edge_dim=hidden_dim)
        self.bn3 = BatchNorm1d(hidden_dim)
        
        # 3. Final Prediction Head
        # Input = AddPool(hidden) + MaxPool(hidden) + Globals
        mlp_in_dim = (hidden_dim * 2) + global_dim
        
        self.mlp = Sequential(
            Linear(mlp_in_dim, hidden_dim),
            BatchNorm1d(hidden_dim),
            ReLU(),
            Dropout(0.3),
            Linear(hidden_dim, 1)
        )

    def forward(self, x, edge_index, batch, edge_attr=None, global_features=None):
        # Embed nodes and edges
        x = self.node_emb(x)
        if edge_attr is not None:
            edge_attr = self.edge_emb(edge_attr)
            
        # Message Passing
        x = F.relu(self.bn1(self.conv1(x, edge_index, edge_attr)))
        x = F.relu(self.bn2(self.conv2(x, edge_index, edge_attr)))
        x = F.relu(self.bn3(self.conv3(x, edge_index, edge_attr)))
        
        # Dual-Pooling Aggregation
        x_add = global_add_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_pool = torch.cat([x_add, x_max], dim=1)
        
        # Inject global BBB descriptors (TPSA, MW) directly into the classifier
        if global_features is not None:
            x_pool = torch.cat([x_pool, global_features], dim=1)
            
        # Return raw logits (FP32 Sigmoid is applied during inference script)
        return self.mlp(x_pool)