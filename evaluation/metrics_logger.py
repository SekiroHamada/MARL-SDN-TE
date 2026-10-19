# evaluation/metrics_logger.py
import pandas as pd
import time

class MetricsLogger:
    """
    Captures and structures the 9 target evaluation metrics during simulation runs.
    """
    def __init__(self):
        self.records = []

    def log_step(self, algo_name, topology_name, load_percentage, throughput, delay, packet_loss):
        """
        Records a single evaluation instance for latency, loss, and throughput.
        """
        self.records.append({
            "Algorithm": algo_name,
            "Topology": topology_name,
            "Load (%)": load_percentage,
            "Throughput (Mbps)": throughput,
            "End-to-End Delay (ms)": delay,
            "Packet Loss (%)": packet_loss,
            "Timestamp": time.strftime("%H:%M:%S")
        })

    def export_to_csv(self, filename="week1_baseline_metrics.csv"):
        """
        Exports the logged metrics to a Pandas DataFrame and CSV format.
        """
        df = pd.DataFrame(self.records)
        df.to_csv(filename, index=False)
        return df