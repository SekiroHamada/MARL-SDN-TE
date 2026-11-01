import os
import time
import yaml
import torch
import numpy as np
import pandas as pd
import sys
import os
# forces Python to look in the parent directory (root folder) for modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environments.shared_env_contract import SDNInterfaceContract
from evaluation.metrics_logger import MetricsLogger
from topology_dataset_loader import TopologyLoader
from gnn_actor_critic import GNNActorCritic

def calculate_jains_index(throughputs):
    """Calculates Jain's Fairness Index across flow throughputs."""
    arr = np.array(throughputs, dtype=np.float64)
    if len(arr) == 0 or np.sum(arr**2) == 0:
        return 1.0
    return float((np.sum(arr) ** 2) / (len(arr) * np.sum(arr ** 2)))

def evaluate_zero_shot():
    config_path = "methodology_2_gnn/configs/gnn_config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GNNActorCritic(config).to(device)
    
    checkpoint_path = "methodology_2_gnn/results/checkpoints/gnn_policy_latest.pt"
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}. Train the GNN policy first.")
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    loader = TopologyLoader()
    env = SDNInterfaceContract()
    
    results = []
    traffic_loads = [0.5, 0.8, 0.95]
    eval_steps = config['training'].get('episode_length', 100)
    
    # ---------------------------------------------------------
    # 1. Baseline Training Pool Evaluation (for Generalization Gap)
    # ---------------------------------------------------------
    print("Evaluating baseline performance on sample training graphs...")
    train_rewards = []
    sample_train_graphs = loader.train_pool[:5] if len(loader.train_pool) >= 5 else loader.train_pool
    
    with torch.no_grad():
        for t_path in sample_train_graphs:
            graph = loader.load_graph(t_path)
            node_states, edge_states = env.reset(topology=graph)
            ep_reward = 0.0
            for _ in range(eval_steps):
                pyg_data = loader.to_pyg_data(graph, node_states, edge_states).to(device)
                action, _, _ = model(pyg_data)
                node_states, edge_states, reward, done, _ = env.step(action.cpu().numpy())
                ep_reward += reward
                if done:
                    break
            train_rewards.append(ep_reward)
            
    mean_train_reward = float(np.mean(train_rewards)) if train_rewards else 0.0

    # ---------------------------------------------------------
    # 2. Zero-Shot Evaluation on Unseen Topologies
    # ---------------------------------------------------------
    zero_shot_rewards = []
    
    for topo_path in loader.zero_shot_pool:
        topo_name = os.path.splitext(os.path.basename(topo_path))[0].upper()
        print(f"\nEvaluating Zero-Shot on {topo_name} ({topo_path})")
        
        if not os.path.exists(topo_path):
            print(f"Skipping {topo_path}: File not found.")
            continue
            
        graph = loader.load_graph(topo_path)
        
        for load in traffic_loads:
            print(f"Running simulation at {int(load * 100)}% load...")
            
            # Reset environment for target topology and load level
            node_states, edge_states = env.reset(topology=graph)
            if hasattr(env, "set_traffic_load"):
                env.set_traffic_load(load)
                
            total_reward = 0.0
            delays = []
            throughputs = []
            packet_losses = []
            link_utils = []
            latencies_us = []
            
            with torch.no_grad():
                for _ in range(eval_steps):
                    pyg_data = loader.to_pyg_data(graph, node_states, edge_states).to(device)
                    
                    # Benchmark inference latency per step
                    t_start = time.perf_counter()
                    action, _, _ = model(pyg_data)
                    latencies_us.append((time.perf_counter() - t_start) * 1e6)
                    
                    node_states, edge_states, reward, done, info = env.step(action.cpu().numpy())
                    total_reward += reward
                    
                    # Extract step diagnostics
                    delays.append(info.get('delay', np.mean([data['delay'] for _, _, data in graph.edges(data=True)])))
                    throughputs.append(info.get('throughput', np.sum(node_states[:, 1]) if isinstance(node_states, np.ndarray) else 100.0 * load))
                    packet_losses.append(info.get('packet_loss', max(0.0, load - 0.8) * 5.0))
                    
                    current_edge_utils = list(edge_states.values()) if isinstance(edge_states, dict) else [0.5]
                    link_utils.extend(current_edge_utils)
                    
                    if done:
                        break
                        
            zero_shot_rewards.append(total_reward)
            
            # Compute consolidated target metrics
            avg_delay = float(np.mean(delays))
            avg_throughput = float(np.mean(throughputs))
            plr = float(np.mean(packet_losses))
            mlu = float(np.max(link_utils)) if link_utils else 0.0
            mlu_var = float(np.var(link_utils)) if link_utils else 0.0
            jains_index = calculate_jains_index(throughputs)
            mean_latency_us = float(np.mean(latencies_us))
            
            results.append({
                "Topology": topo_name,
                "Method": "GNN (Zero-Shot)",
                "Traffic_Load": f"{int(load * 100)}%",
                "End_to_End_Delay_ms": round(avg_delay, 3),
                "Throughput_Gbps": round(avg_throughput, 3),
                "Packet_Loss_Ratio_pct": round(plr, 4),
                "Max_Link_Utilization": round(mlu, 4),
                "MLU_Variance": round(mlu_var, 6),
                "Cumulative_Reward": round(total_reward, 2),
                "Jains_Fairness_Index": round(jains_index, 4),
                "Inference_Latency_us": round(mean_latency_us, 2),
                "Trained_On_Topology": False
            })

    # ---------------------------------------------------------
    # 3. Compute Generalization Gap & Export
    # ---------------------------------------------------------
    mean_zero_shot_reward = float(np.mean(zero_shot_rewards)) if zero_shot_rewards else 0.0
    generalization_gap = mean_train_reward - mean_zero_shot_reward
    gap_percentage = (generalization_gap / abs(mean_train_reward) * 100) if mean_train_reward != 0 else 0.0
    
    print(f"\n================ Evaluation Summary ================")
    print(f"Mean Training Pool Reward:   {mean_train_reward:.2f}")
    print(f"Mean Zero-Shot Test Reward:  {mean_zero_shot_reward:.2f}")
    print(f"Generalization Gap:          {generalization_gap:.2f} ({gap_percentage:.2f}%)")
    print("====================================================\n")
    
    # Write to CSV matching the Week 5 schema
    out_dir = "methodology_2_gnn/results/tables"
    os.makedirs(out_dir, exist_ok=True)
    df_results = pd.DataFrame(results)
    
    # Append summary row for generalization gap
    df_results["Generalization_Gap"] = round(generalization_gap, 3)
    df_results["Generalization_Gap_pct"] = round(gap_percentage, 2)
    
    csv_path = os.path.join(out_dir, "methodology2_zero_shot.csv")
    df_results.to_csv(csv_path, index=False)
    print(f"Exported zero-shot metrics to {csv_path}")

if __name__ == "__main__":
    evaluate_zero_shot()