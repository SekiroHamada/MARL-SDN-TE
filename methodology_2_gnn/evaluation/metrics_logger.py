import os
import time
import pandas as pd

class GNNTrainingLogger:
    """
    Logs PPO training progression, convergence, and wall-clock time 
    specifically for Methodology 2.
    """
    def __init__(self):
        self.records = []
        self.start_time = time.time()

    def log_metrics(self, episode, reward, actor_loss, critic_loss):
        elapsed_hours = (time.time() - self.start_time) / 3600.0
        self.records.append({
            "Episode": episode,
            "Reward": round(reward, 3),
            "Actor_Loss": round(actor_loss, 4),
            "Critic_Loss": round(critic_loss, 4),
            "Wall_Clock_Hours": round(elapsed_hours, 4)
        })

    def export_to_csv(self, filepath="methodology-2-gnn/results/logs/gnn_training_curve.csv"):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        df = pd.DataFrame(self.records)
        df.to_csv(filepath, index=False)