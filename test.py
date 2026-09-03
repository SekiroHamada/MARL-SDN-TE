import json
import os
from mininet.topo import Topo
from mininet.net import Mininet
from mininet.cli import CLI
from mininet.link import TCLink
from mininet.node import OVSController

class JSONTopology(Topo):
    def __init__(self, json_file):
        Topo.__init__(self)

        # 1. Load the static blueprint
        if not os.path.exists(json_file):
            raise FileNotFoundError(f"Cannot find {json_file}")
            
        with open(json_file, 'r') as f:
            data = json.load(f)

        # 2. Build the Switches and Hosts
        for node in data['nodes']:
            node_id = str(node['id'])
            
            # Create a switch (e.g., 's0', 's1')
            #switch_name = self.addSwitch(f's{node_id}')
            switch_name = self.addSwitch(f's{node_id}', stp=True)
            
            
            # Create a dummy host for each switch to generate/receive traffic (e.g., 'h0', 'h1')
            host_name = self.addHost(f'h{node_id}')
            
            # Connect the host to its respective switch
            self.addLink(host_name, switch_name)

        # 3. Build the Network Edges
        for edge in data['edges']:
            source_switch = f"s{edge['source']}"
            target_switch = f"s{edge['target']}"
            
            # Extract your Step 1 attributes
            capacity = edge['capacity']        # in Mbps
            delay_ms = f"{edge['delay']}ms"    # formatted for Mininet
            q_len = edge['queue_length']       # packet limit
            
            # Wire the switches together with hardware constraints
            self.addLink(
                source_switch, 
                target_switch, 
                bw=capacity, 
                delay=delay_ms, 
                max_queue_size=q_len
            )

if __name__ == '__main__':
    # Point this to your generated Step 1 topology
    topo_path = "data/topologies/nsfnet.json" 
    
    print(f"Building network from {topo_path}...")
    custom_topo = JSONTopology(topo_path)
    
    # Must specify link=TCLink to enforce bandwidth/delay constraints
    net = Mininet(topo=custom_topo, controller=OVSController, link=TCLink)
    
    net.start()
    print("Network active! Dropping to CLI...")
    CLI(net)
    net.stop()