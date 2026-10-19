# environments/shared_env_contract.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random
import sys
import os
import networkx as nx

# Dynamically import the configuration weights
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from configs.shared_env_config import REWARD_WEIGHTS
except ImportError:
    # Fallback weights in case the config file is missing during execution
    REWARD_WEIGHTS = {"alpha": 1.0, "beta": 0.5, "gamma": 2.0}

class SDNInterfaceContract(gym.Env):
    def __init__(self, num_next_hops=4):
        """
        num_next_hops: The number of available outbound edges for a given node.
        This dictates the size of the flow-splitting action vector.
        """
        super(SDNInterfaceContract, self).__init__()
        self.num_next_hops = num_next_hops
        
        # Observation Space: S_i(t) = [queue_occupancy, ingress_rate, neighbor_link_utilization, normalized_degree]
        # Bounded between 0.0 and 1.0 for neural network stability
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0, 
            shape=(4,), 
            dtype=np.float32
        )
        
        # Action Space: A(t) = [flow_split_weights]
        # A continuous vector outputting a ratio (0.0 to 1.0) for each available path
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_next_hops,),
            dtype=np.float32
        )
        
        # Internal state tracking
        self.current_topology = None
        self.step_count = 0
        self.max_steps = 100
        self.node_states = None
        self.edge_states = None

    def reset(self, topology=None, seed=None):
        """
        Resets the environment state using a provided NetworkX graph.
        """
        super().reset(seed=seed)
        self.step_count = 0
        
        if topology is not None:
            self.current_topology = topology
            
        if self.current_topology is None:
            raise ValueError("A NetworkX topology must be provided to reset the environment.")
            
        num_nodes = len(self.current_topology.nodes)
        
        # Initialize node states: [queue, ingress, neighbor_util, degree]
        self.node_states = np.random.uniform(0.1, 0.4, (num_nodes, 4))
        
        # Initialize bi-directional edge states (utilization percentages)
        self.edge_states = {}
        for u, v in self.current_topology.edges():
            self.edge_states[(u, v)] = random.uniform(0.1, 0.4)
            self.edge_states[(v, u)] = random.uniform(0.1, 0.4)
            
        return self.node_states.tolist(), self.edge_states

    def step(self, actions):
        """
        Advances the network state by one tick based on the routing actions.
        """
        self.step_count += 1
        num_nodes = len(self.current_topology.nodes)
        
        # Simulate network physics: apply a slight random walk to queue depths and link utilization
        # (Person 1 will replace this block with Pareto/Poisson traffic distributions later)
        self.node_states = np.clip(self.node_states + np.random.normal(0.0, 0.05, (num_nodes, 4)), 0.0, 1.0)
        
        for u, v in self.current_topology.edges():
            self.edge_states[(u, v)] = np.clip(self.edge_states[(u, v)] + random.normalvariate(0.0, 0.05), 0.0, 1.0)
            self.edge_states[(v, u)] = np.clip(self.edge_states[(v, u)] + random.normalvariate(0.0, 0.05), 0.0, 1.0)
            
        # Generate mock telemetry matching real network bounds
        throughput = random.uniform(60, 95)  # Gbps
        delay = random.uniform(2, 15)        # ms
        packet_loss = random.uniform(0, 2)   # Percentage
        
        reward = self.compute_reward(throughput, delay, packet_loss)
        done = self.step_count >= self.max_steps
        info = {'delay': delay, 'throughput': throughput, 'packet_loss': packet_loss}
        
        return self.node_states.tolist(), self.edge_states, reward, done, info

    @staticmethod
    def compute_reward(throughput, delay, packet_loss):
        """
        Calculates R(t) = alpha * Throughput - beta * Delay - gamma * PacketLoss
        """
        alpha = REWARD_WEIGHTS.get("alpha", 1.0)
        beta = REWARD_WEIGHTS.get("beta", 0.5)
        gamma = REWARD_WEIGHTS.get("gamma", 2.0)
        
        return (alpha * throughput) - (beta * delay) - (gamma * packet_loss)