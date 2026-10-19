import networkx as nx
import random
import json
from networkx.readwrite import json_graph
import os

def assign_edge_attributes(G):
    """
    Applies the shared project constraints to every edge in the graph.
    These act as the weights and capacities for your OSPF shortest-path calculations.
    """
    for u, v in G.edges():
        G[u][v]['capacity'] = 100  # Uniform 100 Mbps capacity[cite: 2]
        G[u][v]['delay'] = random.randint(2, 10)  # Random delay between 2-10 ms[cite: 2]
        G[u][v]['queue_length'] = 100  # Finite FIFO queue buffer[cite: 2]
    return G

def generate_shared_topologies():
    # Ensure the target directory exists
    os.makedirs('../data/topologies', exist_ok=True)
    
    # 1. Construct 14-Node NSFNet[cite: 2]
    nsfnet = nx.Graph(name="NSFNet")
    nsfnet_edges = [
        (0,1), (0,2), (0,3), (1,2), (1,7), (2,5), (3,4), (3,8), 
        (4,5), (4,6), (5,12), (6,7), (6,9), (7,10), (8,9), 
        (8,11), (9,10), (9,12), (10,11), (10,13), (11,13)
    ]
    nsfnet.add_edges_from(nsfnet_edges)
    nsfnet = assign_edge_attributes(nsfnet)
    
    # 2. Construct 23-Node GEANT[cite: 2]
    geant = nx.Graph(name="GEANT")
    # Provide the standard 23-node adjacency list here
    # geant.add_edges_from([...]) 
    #
    geant_edges = [
        (0, 1), (0, 2), (1, 3), (1, 8), (2, 4), (3, 5), (4, 5), 
        (4, 6), (5, 7), (6, 8), (6, 11), (7, 9), (8, 10), (9, 10), 
        (9, 12), (10, 13), (11, 14), (12, 15), (13, 14), (13, 16), 
        (14, 17), (15, 16), (15, 18), (16, 19), (17, 20), (18, 19), 
        (18, 21), (19, 22), (20, 21), (21, 22), (0, 12), (2, 17), 
        (3, 19), (7, 21), (11, 22), (1, 15), (8, 20)
    ]
    geant.add_edges_from(geant_edges)
    #
    geant = assign_edge_attributes(geant)
    
    # 3. Construct Fat-Tree (K=4)[cite: 2]
    fattree = nx.Graph(name="FatTree_K4")
    # A K=4 Fat-Tree requires linking 16 hosts, 8 edge switches, 
    # 8 aggregation switches, and 4 core switches.
    # fattree.add_edges_from([...])
    #
    k = 4
    core_nodes = list(range(0, 4))
    agg_nodes = list(range(4, 12))
    edge_nodes = list(range(12, 20))
    host_nodes = list(range(20, 36))
    
    fattree_edges = []
    
    # 1. Connect Core to Aggregation (Each core connects to one agg per pod)
    for i in range(4): # 4 pods
        for j in range(2): # 2 core switches per agg block
            core_index = j * 2 + (i % 2)
            agg_index = 4 + (i * 2) + j
            fattree_edges.append((core_nodes[core_index], agg_nodes[i * 2 + j]))
            
    # Standard Core to Agg mapping for K=4
    fattree_edges = [
        # Core to Agg
        (0, 4), (0, 6), (0, 8), (0, 10),
        (1, 4), (1, 6), (1, 8), (1, 10),
        (2, 5), (2, 7), (2, 9), (2, 11),
        (3, 5), (3, 7), (3, 9), (3, 11),
        # Agg to Edge (Full bipartite within each of the 4 pods)
        (4, 12), (4, 13), (5, 12), (5, 13),  # Pod 0
        (6, 14), (6, 15), (7, 14), (7, 15),  # Pod 1
        (8, 16), (8, 17), (9, 16), (9, 17),  # Pod 2
        (10, 18), (10, 19), (11, 18), (11, 19), # Pod 3
        # Edge to Host (2 hosts per edge switch)
        (12, 20), (12, 21), (13, 22), (13, 23),
        (14, 24), (14, 25), (15, 26), (15, 27),
        (16, 28), (16, 29), (17, 30), (17, 31),
        (18, 32), (18, 33), (19, 34), (19, 35)
    ]
    fattree.add_edges_from(fattree_edges)
    #
    fattree = assign_edge_attributes(fattree)

    # 4. Serialize and Export
    # Exporting as JSON allows seamless integration for Person 2's PyG data loaders
    graphs = [nsfnet, geant, fattree]
    filenames = ["nsfnet", "geant", "fattree_k4"]
    
    for graph, name in zip(graphs, filenames):
        data = json_graph.node_link_data(graph)
        filepath = f"../data/topologies/{name}.json"
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Exported {name} to {filepath}")

if __name__ == "__main__":
    generate_shared_topologies()