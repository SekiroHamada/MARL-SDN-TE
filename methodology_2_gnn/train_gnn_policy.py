import yaml
import torch
import torch.nn as nn
from torch.optim import Adam
from torch_geometric.data import Batch
import numpy as np
import random
import sys
import os

# forces Python to look in the parent directory (root folder) for modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Un-commented and linked to your shared team environment modules
from environments.shared_env_contract import SDNInterfaceContract
from methodology_2_gnn.evaluation.metrics_logger import GNNTrainingLogger
from topology_dataset_loader import TopologyLoader
from gnn_actor_critic import GNNActorCritic

def compute_gae(rewards, values, next_value, gamma, lam):
    advantages = []
    gae = 0
    for r, v in zip(reversed(rewards), reversed(values)):
        delta = r + gamma * next_value - v
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
        next_value = v
    return advantages

def train():
    # Load configuration
    with open("methodology_2_gnn/configs/gnn_config.yaml", 'r') as f:
        config = yaml.safe_load(f)
        
    loader = TopologyLoader()
    model = GNNActorCritic(config)
    optimizer = Adam(model.parameters(), lr=float(config['ppo']['lr']))
    
    # Initialize the shared simulation contract and logger
    env = SDNInterfaceContract(num_next_hops=4)
    logger = GNNTrainingLogger()
    clip_eps = float(config['ppo']['clip_eps'])
    
    print("Starting Domain-Randomized PPO Training...")
    
    for update in range(config['training']['total_updates']):
        # ---------------------------------------------------------
        # 1. Collect Rollouts (Domain Randomization)
        # ---------------------------------------------------------
        obs_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []

        # Sample a fresh random graph from the synthetic training pool
        topo_path = random.choice(loader.train_pool)
        graph = loader.load_graph(topo_path)
        
        # Reset environment with the new topology
        node_states, edge_states = env.reset(topology=graph)
        
        for step in range(config['training']['episode_length']):
            # Convert raw network states to PyG Data format
            pyg_data = loader.to_pyg_data(graph, node_states, edge_states)
            obs_list.append(pyg_data)
            
            # Forward pass to get actions and state value
            with torch.no_grad():
                action, log_prob, value = model(pyg_data)
            
            actions_list.append(action)
            log_probs_list.append(log_prob)
            values_list.append(value)
            
            # Step the environment (passing the raw tensor as numpy)
            node_states, edge_states, reward, done, _ = env.step(action.numpy())
            rewards_list.append(reward)
            
            if done:
                break
                
        # Get the final next_value for the GAE calculation
        with torch.no_grad():
            final_data = loader.to_pyg_data(graph, node_states, edge_states)
            _, _, next_value = model(final_data)

        # Batch the collected variable-sized graphs into a single PyG Batch for parallel processing
        batch = Batch.from_data_list(obs_list)
        old_actions = torch.cat(actions_list)
        old_log_probs = torch.stack(log_probs_list)
        values = torch.stack(values_list).squeeze()

        # ---------------------------------------------------------
        # 2. Compute Advantages (GAE)
        # ---------------------------------------------------------
        advantages_list = compute_gae(
            rewards_list, 
            values.tolist(), 
            next_value.item(), 
            config['ppo']['gamma'], 
            config['ppo']['gae_lambda']
        )
        advantages = torch.tensor(advantages_list, dtype=torch.float32)
        returns = advantages + values
        
        # Normalize advantages for training stability
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # ---------------------------------------------------------
        # 3. PPO Update Epochs
        # ---------------------------------------------------------
        for epoch in range(config['ppo']['epochs_per_update']):
            # Evaluate the old actions using the updated policy
            log_prob, entropy, value = model.evaluate_actions(batch, old_actions)
            value = value.squeeze()
            
            # Calculate ratios and clipped surrogate objective
            ratio = torch.exp(log_prob - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            # Critic loss (MSE between estimated value and actual returns)
            critic_loss = nn.MSELoss()(value, returns)
            
            # Combined PPO loss function
            loss = actor_loss + float(config['ppo']['value_coef']) * critic_loss - float(config['ppo']['entropy_coef']) * entropy
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['ppo']['max_grad_norm'])
            optimizer.step()
            
        # Log the episode metrics using the shared team logger
        ep_reward = sum(rewards_list)
        logger.log_metrics(
            episode=update, 
            reward=ep_reward, 
            actor_loss=actor_loss.item(), 
            critic_loss=critic_loss.item()
        )
            
        if update % 10 == 0:
            print(f"Update {update} complete | Ep Reward: {ep_reward:.2f} | Checkpointing...")
            torch.save(model.state_dict(), "methodology_2_gnn/results/checkpoints/gnn_policy_latest.pt")
            
if __name__ == "__main__":
    train()