# baselines/baseline_ospf_ecmp.py

import networkx as nx
import json
import os

class BaselineRouting:
    def __init__(self, topology_path):
        """
        Loads the standardized NetworkX graph from the JSON blueprint.
        """
        if not os.path.exists(topology_path):
            raise FileNotFoundError(f"Topology file not found at {topology_path}")
            
        with open(topology_path, 'r') as f:
            data = json.load(f)
        self.graph = nx.node_link_graph(data)

    def compute_ospf_paths(self, source, target):
        """
        OSPF Implementation: Dijkstra Shortest Path.
        Computes the single primary routing path based on minimum edge delay.
        """
        try:
            # Dijkstra calculation minimizing the 'delay' weight
            path = nx.shortest_path(self.graph, source=source, target=target, weight='delay')
            return path
        except nx.NetworkXNoPath:
            return None

    def compute_ecmp_paths(self, source, target):
        """
        ECMP Implementation: Equal-Cost Multi-Path.
        Finds all paths between source and target that share the exact same minimum delay.
        """
        try:
            # Extracts all shortest paths with identical weights
            paths = list(nx.all_shortest_paths(self.graph, source=source, target=target, weight='delay'))
            return paths
        except nx.NetworkXNoPath:
            return []

    def distribute_ecmp_flow(self, source, target, traffic_volume):
        """
        Statically distributes a given traffic volume equally across all valid ECMP paths.
        """
        paths = self.compute_ecmp_paths(source, target)
        if not paths:
            return {}
        
        split_ratio = traffic_volume / len(paths)
        flow_distribution = {
            f"path_{i}": {"nodes": paths[i], "allocated_traffic": split_ratio} 
            for i in range(len(paths))
        }
        
        return flow_distribution

# Example Execution
if __name__ == "__main__":
    # Point to the shared Sprint 0 topology
    routing = BaselineRouting("../data/topologies/nsfnet.json")
    
    # Calculate between node 0 and node 9
    print("OSPF Primary Path:", routing.compute_ospf_paths(0, 9))
    
    flow_split = routing.distribute_ecmp_flow(0, 9, traffic_volume=100)
    print("ECMP Flow Distribution:", json.dumps(flow_split, indent=2))