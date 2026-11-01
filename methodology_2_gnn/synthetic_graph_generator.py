import networkx as nx
import json
import random
import os

def generate_synthetic_graph(n_nodes, graph_type="waxman"):
    """Generates a graph with identical link attributes to the test set."""
    if graph_type == "waxman":
        # Increased alpha and beta to ensure higher edge density
        G = nx.waxman_graph(n_nodes, beta=0.8, alpha=0.8)
    else:
        G = nx.barabasi_albert_graph(n_nodes, m=2)
    
    # Ensure connectivity with a strict retry limit
    retries = 0
    while not nx.is_connected(G) and retries < 50:
        if graph_type == "waxman":
            G = nx.waxman_graph(n_nodes, beta=0.8, alpha=0.8)
        else:
            G = nx.barabasi_albert_graph(n_nodes, m=2)
        retries += 1
        
    # Fallback to Barabasi-Albert (which mathematically guarantees connectivity for m>=1) 
    # if Waxman stubbornly fails after 50 attempts
    if not nx.is_connected(G):
        G = nx.barabasi_albert_graph(n_nodes, m=2)

    # Assign physical constraints matching NSFNet/GEANT
    for u, v in G.edges():
        G[u][v]['capacity'] = 100
        G[u][v]['delay'] = random.randint(2, 10)
        G[u][v]['queue'] = 100
    
    return G

def build_pool(n_graphs=200, out_dir="data/topologies/synthetic/"):
    os.makedirs(out_dir, exist_ok=True)
    for i in range(n_graphs):
        n_nodes = random.randint(10, 40)
        g_type = random.choice(["waxman", "ba"])
        G = generate_synthetic_graph(n_nodes, g_type)
        
        # Export to JSON schema compatible with existing network_generator.py
        data = nx.node_link_data(G)
        with open(os.path.join(out_dir, f"{g_type}_{i:03d}.json"), 'w') as f:
            json.dump(data, f, indent=4)

if __name__ == "__main__":
    build_pool()