"""
Topology loading for Methodology 2 (v2).

* Real WANs come from the Internet Topology Zoo (graphml, lat/long per node).
  Link propagation delay = great-circle distance / (2e5 km/s), i.e. 5 us/km.
* Synthetic training graphs are generated with *WAN-like* sparsity
  (mean degree ~2.2-4.5) so the training distribution covers the test set.
* Legacy team JSON files (data/topologies/*.json) are also supported.

Every graph returned is: undirected, connected, nodes relabelled 0..N-1,
edge attrs: capacity (Mbps), delay (ms), queue (packets).
"""
import json, math, os, random
import networkx as nx
import numpy as np

DEFAULT_CAPACITY = 100.0   # Mbps (team Sprint-0 constraint)
DEFAULT_QUEUE = 100        # packets (team Sprint-0 constraint)


def _finalize(G, name):
    G = nx.Graph(G)                       # drop multi-edges / direction
    G.remove_edges_from(nx.selfloop_edges(G))
    G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    G = nx.convert_node_labels_to_integers(G, ordering="sorted")
    for u, v, d in G.edges(data=True):
        d["capacity"] = float(d.get("capacity", DEFAULT_CAPACITY))
        d["delay"] = float(d.get("delay", 5.0))
        d["queue"] = int(d.get("queue", d.get("queue_length", DEFAULT_QUEUE)))
    G.graph["name"] = name
    return G


def _haversine_km(lat1, lon1, lat2, lon2):
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def load_topology_zoo(path, mean_delay_ms=5.0):
    """Load a Topology Zoo yEd .graphml file (real link structure).

    These files carry the yEd drawing coordinates (which trace the network's
    map) rather than lat/long, so link delay is set proportional to drawn
    distance and rescaled so the mean link delay is `mean_delay_ms`
    (team constraint: 2-10 ms). Delays are clipped to [0.5, 4*mean].
    """
    import xml.etree.ElementTree as ET
    ns = {"g": "http://graphml.graphdrawing.org/xmlns", "y": "http://www.yworks.com/xml/graphml"}
    root = ET.parse(path).getroot()
    graph = root.find("g:graph", ns)
    pos, G = {}, nx.Graph()
    for node in graph.findall("g:node", ns):
        geo = node.find(".//y:Geometry", ns)
        if geo is not None:
            pos[node.get("id")] = (float(geo.get("x")), float(geo.get("y")))
        G.add_node(node.get("id"))
    for e in graph.findall("g:edge", ns):
        u, v = e.get("source"), e.get("target")
        if u != v:
            G.add_edge(u, v)
    raw = {}
    for u, v in G.edges():
        raw[(u, v)] = math.dist(pos[u], pos[v]) if (u in pos and v in pos) else None
    known = [d for d in raw.values() if d]
    scale = mean_delay_ms / (float(np.mean(known)) if known else 1.0)
    for (u, v), d in raw.items():
        dl = d * scale if d else mean_delay_ms
        G[u][v]["delay"] = float(np.clip(dl, 0.5, 4 * mean_delay_ms))
    name = os.path.splitext(os.path.basename(path))[0]
    return _finalize(G, name)


def load_team_json(path):
    """Team Sprint-0 JSON (node-link format; edge list under 'edges' or 'links')."""
    import inspect
    with open(path) as f:
        data = json.load(f)
    key = "edges" if "edges" in data else "links"
    if "edges" in inspect.signature(nx.node_link_graph).parameters:
        G = nx.node_link_graph(data, edges=key)
    else:                                   # networkx < 3.4 always uses 'links'
        data["links"] = data.pop(key)
        G = nx.node_link_graph(data)
    return _finalize(G, os.path.splitext(os.path.basename(path))[0])


def load_any(path):
    if path.endswith(".graphml"):
        return load_topology_zoo(path)
    return load_team_json(path)


# ----------------------------------------------------------------------------
# Synthetic, WAN-like training graphs (domain randomisation)
# ----------------------------------------------------------------------------
def _geo_delays(G, pos, rng, span_ms):
    """Delay proportional to Euclidean distance in the unit square."""
    for u, v, d in G.edges(data=True):
        dist = math.dist(pos[u], pos[v])
        d["delay"] = max(0.5, dist * span_ms * rng.uniform(0.9, 1.1))


def synthetic_graph(n, rng):
    """One random connected WAN-like graph with n nodes."""
    kind = rng.choice(["waxman", "ba", "geo_tree_plus", "ring_chords"])
    pos = {i: (rng.random(), rng.random()) for i in range(n)}
    if kind == "waxman":
        for _ in range(200):
            G = nx.waxman_graph(n, beta=rng.uniform(0.15, 0.4), alpha=rng.uniform(0.15, 0.35),
                                domain=(0, 0, 1, 1), seed=rng.randrange(1 << 30))
            if nx.is_connected(G) and 2 * G.number_of_edges() / n <= 5.0:
                break
        else:
            kind = "geo_tree_plus"
        if kind == "waxman":
            pos = nx.get_node_attributes(G, "pos")
    if kind == "ba":
        G = nx.barabasi_albert_graph(n, rng.choice([1, 2, 2]), seed=rng.randrange(1 << 30))
        if min(dict(G.degree()).values()) == 1 and rng.random() < 0.7:
            _add_random_edges(G, pos, rng, frac=0.25)
    if kind == "geo_tree_plus":
        # Euclidean MST + short extra links: looks like a real backbone
        K = nx.Graph()
        for i in range(n):
            for j in range(i + 1, n):
                K.add_edge(i, j, weight=math.dist(pos[i], pos[j]))
        G = nx.minimum_spanning_tree(K)
        _add_random_edges(G, pos, rng, frac=rng.uniform(0.2, 0.8))
    if kind == "ring_chords":
        order = sorted(range(n), key=lambda i: math.atan2(pos[i][1] - .5, pos[i][0] - .5))
        G = nx.cycle_graph(order)
        _add_random_edges(G, pos, rng, frac=rng.uniform(0.1, 0.6))
    _geo_delays(G, pos, rng, span_ms=rng.uniform(10, 40))
    for u, v, d in G.edges(data=True):
        d["capacity"] = DEFAULT_CAPACITY
        d["queue"] = DEFAULT_QUEUE
    return _finalize(G, f"syn_{kind}_{n}")


def _add_random_edges(G, pos, rng, frac):
    """Add ~frac*n extra edges, preferring geographically short ones."""
    n = G.number_of_nodes()
    k = max(1, int(frac * n))
    nodes = list(G.nodes())
    for _ in range(k):
        u = rng.choice(nodes)
        cands = sorted((math.dist(pos[u], pos[v]), v) for v in nodes if v != u and not G.has_edge(u, v))
        if not cands:
            continue
        v = cands[min(len(cands) - 1, int(rng.expovariate(1 / 2.0)))][1]
        G.add_edge(u, v)


def synthetic_pool(size, n_range=(8, 45), seed=0):
    rng = random.Random(seed)
    return [synthetic_graph(rng.randint(*n_range), rng) for _ in range(size)]


# ----------------------------------------------------------------------------
# Held-out real test sets
# ----------------------------------------------------------------------------
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZOO_DIR = os.path.join(REPO_ROOT, "data", "topologies", "topologyzoo")

# Primary zero-shot targets named in the project plan
PRIMARY_TEST = ["Abilene", "Nsfnet", "Geant2012"]


def exists_exact(path):
    """Case-SENSITIVE os.path.exists for a file inside a run folder: checks that the file
    AND its parent folder exist with exactly this capitalisation. Needed on Windows
    drives (WSL /mnt/c), where 'specialist_mlp_nsfnet' would otherwise match the
    Topology-Zoo 'specialist_mlp_Nsfnet' (a different, 13-node network)."""
    parent, name = os.path.split(os.path.abspath(path))
    grand, pname = os.path.split(parent)
    return _exact(grand, pname) and _exact(parent, name)


def _exact(folder, filename):
    """Case-SENSITIVE existence check. os.path.exists is case-insensitive on Windows
    drives (e.g. WSL /mnt/c), where 'nsfnet' would wrongly match Zoo 'Nsfnet.graphml'
    (13 nodes) instead of the team's 'nsfnet.json' (14 nodes)."""
    try:
        return filename in os.listdir(folder)
    except FileNotFoundError:
        return False


def load_named(name):
    """Topology Zoo graph if `<name>.graphml` exists with exactly this case, else the
    team's `data/topologies/<name>.json`. E.g. 'Nsfnet' = Zoo (13 nodes), 'nsfnet' = team (14)."""
    if _exact(ZOO_DIR, f"{name}.graphml"):
        return load_topology_zoo(os.path.join(ZOO_DIR, f"{name}.graphml"))
    team = os.path.join(REPO_ROOT, "data", "topologies")
    if _exact(team, f"{name}.json"):
        return load_team_json(os.path.join(team, f"{name}.json"))
    raise FileNotFoundError(f"topology '{name}' not found (names are case-sensitive): "
                            f"looked for {ZOO_DIR}/{name}.graphml and {team}/{name}.json")
