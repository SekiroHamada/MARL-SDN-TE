import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv, global_mean_pool

class GNNEncoder(nn.Module):
    def __init__(self, node_in_dim=4, edge_in_dim=3, hidden_dim=64, num_layers=3):
        super(GNNEncoder, self).__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.convs = nn.ModuleList()
        self.layer_norms = nn.ModuleList()
        
        # Initial node embedding
        self.node_emb = nn.Linear(node_in_dim, hidden_dim)
        
        for _ in range(num_layers):
            nn_func = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
            self.convs.append(GINEConv(nn_func, edge_dim=hidden_dim))
            self.layer_norms.append(nn.LayerNorm(hidden_dim))
            
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_emb(x)
        edge_attr_emb = self.edge_mlp(edge_attr)
        
        for conv, norm in zip(self.convs, self.layer_norms):
            residual = x
            x = conv(x, edge_index, edge_attr_emb)
            x = norm(x)
            x = torch.relu(x)
            x = x + residual # Guard against oversmoothing
            
        graph_emb = global_mean_pool(x, batch)
        return x, graph_emb # Returns node_embeddings, pooled_graph_embedding