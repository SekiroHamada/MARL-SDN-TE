import sys
import os
import argparse
import torch
import torch.nn as nn
from torch.optim import Adam
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from environments.shared_env_contract import SDNInterfaceContract
from methodology_2_gnn.topology_dataset_loader import TopologyLoader

class MLPBaseline(nn.Module):
    def __init__(self, num_nodes, node_features=4, num_actions=4):
        super(MLPBaseline, self).__init__()
        input_dim = num_nodes * node_features
        self.actor = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_nodes * num_actions)
        )
        self.num_nodes = num_nodes
        self.num_actions = num_actions

    def forward(self, flat_node_states):
        raw_actions = self.actor(flat_node_states)
        actions = raw_actions.view(self.num_nodes, self.num_actions)
        return torch.softmax(actions, dim=-1)

def train_mlp_baseline(topology_file):
    loader = TopologyLoader()
    graph = loader.load_graph(topology_file)
    num_nodes = len(graph.nodes)
    
    env = SDNInterfaceContract(num_next_hops=4)
    model = MLPBaseline(num_nodes=num_nodes)
    optimizer = Adam(model.parameters(), lr=3e-4)
    
    print(f"==================================================")
    print(f"Training Fixed-Structure MLP on {os.path.basename(topology_file)}")
    print(f"Fixed Input Dimension: {num_nodes * 4} (Will crash if topology size changes)")
    print(f"==================================================")
    
    node_states, edge_states = env.reset(topology=graph)
    
    for update in range(101):
        flat_obs = torch.tensor(node_states, dtype=torch.float32).flatten()
        actions = model(flat_obs)
        node_states, edge_states, reward, done, _ = env.step(actions.detach().numpy())
        
        loss = -torch.tensor(reward, requires_grad=True) 
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if update % 20 == 0:
            print(f"Update {update:03d} | Baseline Reward: {reward:.2f}")

    print("==================================================")
    print(f"MLP Baseline established for {os.path.basename(topology_file)}.")
    print("==================================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, required=True, help="Topology name (e.g., nsfnet, geant)")
    args = parser.parse_args()
    
    # Map the short argument to the actual file path
    topo_name = args.topology.lower().replace('.json', '')
    file_path = f"data/topologies/{topo_name}.json"
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find {file_path}. Make sure the JSON file exists in data/topologies/")
        sys.exit(1)
        
    train_mlp_baseline(file_path)