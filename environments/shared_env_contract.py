# environments/shared_env_contract.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os

# Dynamically import the configuration weights
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs.shared_env_config import REWARD_WEIGHTS

class SDNInterfaceContract:
    def __init__(self, num_next_hops):
        """
        num_next_hops: The number of available outbound edges for a given node.
        This dictates the size of the flow-splitting action vector.
        """
        
        # Observation Space: S_i(t) = [queue_occupancy, ingress_rate, neighbor_link_utilization]
        # Assuming values are normalized between 0.0 (0%) and 1.0 (100%) for neural network stability
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0, 
            shape=(3,), 
            dtype=np.float32
        )
        
        # Action Space: A(t) = [flow_split_weights][cite: 2]
        # A continuous vector outputting a ratio (0.0 to 1.0) for each available path
        self.action_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(num_next_hops,),
            dtype=np.float32
        )

    @staticmethod
    def compute_reward(throughput, delay, packet_loss):
        """
        Calculates R(t) = alpha * Throughput - beta * Delay - gamma * PacketLoss[cite: 2]
        """
        alpha = REWARD_WEIGHTS["alpha"]
        beta = REWARD_WEIGHTS["beta"]
        gamma = REWARD_WEIGHTS["gamma"]
        
        reward = (alpha * throughput) - (beta * delay) - (gamma * packet_loss)
        return reward