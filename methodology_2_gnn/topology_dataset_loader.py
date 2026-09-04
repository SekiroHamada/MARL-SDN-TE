import torch
import networkx as nx
import json
from torch_geometric.data import Data
import os

class TopologyLoader:
    def __init__(self, synthetic_dir="data/topologies/synthetic/"):
        self.synthetic_dir = synthetic_dir
        self.train_pool = [os.path.join(synthetic_dir, f) for f in os.listdir(synthetic_dir) if f.endswith('.json')]
        # GEANT and Abilene are strictly held out for zero-shot testing
        self.zero_shot_pool = ["data/topologies/geant.json", "data/topologies/abilene.json"]

    def load_graph(self, path):
        with open(path, 'r') as f:
            data = json.load(f)
        return nx.node_link_graph(data)

    def to_pyg_data(self, graph, node_states, edge_states):
        """
        Converts NetworkX graph + live states to PyG Data.
        Node features: [queue_occupancy, ingress_rate, neighbor_utilization, normalized_degree]
        Edge features: [capacity, delay, utilization]
        """
        num_nodes = len(graph.nodes)
        
        # Build node features (x)
        x = torch.zeros((num_nodes, 4), dtype=torch.float)
        for i in range(num_nodes):
            x[i] = torch.tensor(node_states[i], dtype=torch.float)
            
        # Build edge index and attributes
        edge_index = []
        edge_attr = []
        
        edges = list(graph.edges(data=True))
        # PyG requires undirected graphs to have bi-directional edges explicitly
        for u, v, data in edges:
            edge_idx = (u, v)
            util = edge_states.get(edge_idx, 0.0)
            
            # Forward edge
            edge_index.append([u, v])
            edge_attr.append([data['capacity'], data['delay'], util])
            
            # Reverse edge
            edge_index.append([v, u])
            edge_attr.append([data['capacity'], data['delay'], util])
            
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)