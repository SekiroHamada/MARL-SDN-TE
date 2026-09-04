import torch
import torch.nn as nn
from torch.distributions import Normal
from gnn_policy_encoder import GNNEncoder

class GNNActorCritic(nn.Module):
    def __init__(self, config):
        super(GNNActorCritic, self).__init__()
        self.encoder = GNNEncoder(
            hidden_dim=config['gnn']['hidden_dim'],
            num_layers=config['gnn']['num_layers']
        )
        
        # Actor: Outputs flow-split weights per node
        self.actor_head = nn.Sequential(
            nn.Linear(config['gnn']['hidden_dim'], 32),
            nn.ReLU(),
            nn.Linear(32, 1) # Outputs raw logits for outgoing edges
        )
        self.actor_std = nn.Parameter(torch.zeros(1))
        
        # Critic: Centralized value function for PPO advantage
        self.critic_head = nn.Sequential(
            nn.Linear(config['gnn']['hidden_dim'], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, data):
        node_emb, graph_emb = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        
        # Value estimate
        value = self.critic_head(graph_emb)
        
        # Action logits
        action_mean = self.actor_head(node_emb).squeeze(-1)
        action_std = self.actor_std.exp().expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        action = dist.sample()
        
        # In deployment, apply node-wise softmax using scatter to ensure split weights sum to 1.0
        # (Simplified continuous distribution for PPO logic)
        log_prob = dist.log_prob(action).sum()
        
        return action, log_prob, value

    def evaluate_actions(self, data, action):
        node_emb, graph_emb = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        value = self.critic_head(graph_emb)
        
        action_mean = self.actor_head(node_emb).squeeze(-1)
        action_std = self.actor_std.exp().expand_as(action_mean)
        
        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum()
        entropy = dist.entropy().mean()
        
        return log_prob, entropy, value