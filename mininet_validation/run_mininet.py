#!/usr/bin/env python3
"""
Mininet runner = the data-plane half of the validation (run as root with the
SYSTEM python3 that has Mininet; standard library + Mininet only).

It builds the topology (OVS switches, one host per switch, TCLink links with the
same capacity / delay / 100-packet buffers as the simulator, optionally scaled
down in bandwidth), installs OpenFlow 1.3 rules (one select group per
destination per switch), replays the SAME bursty traffic schedule for every
method, and each control interval:
    reads real link counters (tx bytes, tc backlog, tc drops)
    -> sends them to the policy server -> installs the returned split weights.
Throughput, loss and one-way delay are measured end-to-end by the receivers.

    # terminal 1 (venv):   python mininet_validation/policy_server.py --topology nsfnet
    # terminal 2:
    sudo python3 mininet_validation/run_mininet.py --check            # build + OSPF + pingall
    sudo python3 mininet_validation/run_mininet.py                    # full comparison

Results -> results_mininet/<topology>_runs.csv  (one row per method x load x seed)
"""
import argparse, csv, json, os, re, socket, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
AGENT = os.path.join(HERE, "traffic_agent.py")
DEFAULT_METHODS = ["OSPF", "ECMP-hop", "M2 GNN zero-shot (seed0)", "M1 IPPO (seed0)", "M3 MAPPO (seed0)"]


# ----------------------------------------------------------------------------- helpers
class Server:
    def __init__(self, addr):
        h, p = addr.split(":")
        self.sock = socket.create_connection((h, int(p)))
        self.f = self.sock.makefile("rwb")

    def call(self, **req):
        self.f.write((json.dumps(req) + "\n").encode()); self.f.flush()
        rep = json.loads(self.f.readline())
        if "error" in rep:
            raise RuntimeError("policy server: " + rep["error"])
        return rep


def ip(i):
    return f"10.0.{(i + 1) // 250}.{(i + 1) % 250 or 250}"


def mac(i):
    return "00:00:00:00:%02x:%02x" % (((i + 1) >> 8) & 255, (i + 1) & 255)


def ofctl(*args, stdin=None):
    r = subprocess.run(["ovs-ofctl", "-O", "OpenFlow13", *args], input=stdin, text=True,
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"ovs-ofctl {' '.join(args)}: {r.stderr.strip()}")


def group_spec(t, buckets):
    return f"group_id={t + 1},type=select," + ",".join(f"bucket=weight:{w},output:{p}" for p, w in buckets)


QDISC_RE = re.compile(r"^qdisc \S+ \S+ dev (\S+) ", re.M)


def read_tc():
    """{dev: (dropped_pkts, backlog_pkts)}. Mininet shapes links with htb + netem; depending on
    the kernel, backlog/drops show on the root or the child qdisc, so take the max per device
    (never the sum, which would double-count)."""
    out = subprocess.run(["tc", "-s", "qdisc", "show"], capture_output=True, text=True).stdout
    res = {}
    for b in re.split(r"(?=^qdisc )", out, flags=re.M):
        m = QDISC_RE.match(b)
        if not m:
            continue
        d = re.search(r"dropped (\d+)", b)
        bl = re.search(r"backlog \S+ (\d+)p", b)
        cur = res.get(m.group(1), (0, 0))
        res[m.group(1)] = (max(cur[0], int(d.group(1)) if d else 0), max(cur[1], int(bl.group(1)) if bl else 0))
    return res


def read_counter(dev, name):
    with open(f"/sys/class/net/{dev}/statistics/{name}") as f:
        return int(f.read())


# ----------------------------------------------------------------------------- network
def build(topo, scale):
    from mininet.net import Mininet
    from mininet.node import OVSSwitch
    from mininet.link import TCLink, Link
    n = topo["nodes"]
    net = Mininet(controller=None, build=False, autoSetMacs=False, autoStaticArp=False)
    hosts = [net.addHost(f"h{i + 1}", ip=ip(i) + "/8", mac=mac(i)) for i in range(n)]
    sws = [net.addSwitch(f"s{i + 1}", cls=OVSSwitch, protocols="OpenFlow13", failMode="secure",
                         dpid="%016x" % (i + 1)) for i in range(n)]
    hlinks = [net.addLink(hosts[i], sws[i], cls=Link) for i in range(n)]
    port, dev, cap = {}, {}, {}
    for u, v, c, d, q in topo["links"]:
        l = net.addLink(sws[u], sws[v], cls=TCLink, bw=c * scale, delay=f"{d:.3f}ms",
                        max_queue_size=q, use_htb=True)
        for a, b, intf in ((u, v, l.intf1), (v, u, l.intf2)):
            port[(a, b)] = sws[a].ports[intf]
            dev[(a, b)] = intf.name
            cap[(a, b)] = c * scale * 1e6
    net.build(); net.start()
    hport = [sws[i].ports[hlinks[i].intf2] for i in range(n)]
    for i, h in enumerate(hosts):                     # static ARP: no flooding, no loops
        h.cmd("sysctl -qw net.ipv6.conf.all.disable_ipv6=1")
        for j in range(n):
            if j != i:
                h.setARP(ip(j), mac(j))
    return net, hosts, sws, port, dev, cap, hport


class Installer:
    """Keeps OVS groups in sync with the policy's split weights (only changed ones are sent)."""

    def __init__(self, sws, port, hport, n):
        self.sws, self.port, self.hport, self.n = sws, port, hport, n
        self.cur = {}
        self.pool = ThreadPoolExecutor(16)

    def buckets(self, v, nh):
        tot = sum(w for _, w in nh) or 1.0
        return tuple(sorted((self.port[(v, u)], max(1, int(round(1000 * w / tot)))) for u, w in nh))

    def apply(self, routes, first=False):
        jobs = []
        for v, t, nh in routes:
            b = self.buckets(v, nh)
            old = self.cur.get((v, t))
            if old is not None and len(old) == len(b) and all(
                    p1 == p2 and abs(w1 - w2) <= 20 for (p1, w1), (p2, w2) in zip(old, b)):
                continue                                     # change < 2%: keep
            self.cur[(v, t)] = b
            jobs.append((("add-group" if old is None else "mod-group"), self.sws[v].name, group_spec(t, b)))
        t0 = time.time()
        list(self.pool.map(lambda j: ofctl(*j), jobs))
        if first:                                            # flows pointing at the groups
            for v in range(self.n):
                lines = [f"priority=100,ip,nw_dst={ip(v)},actions=output:{self.hport[v]}"]
                lines += [f"priority=100,ip,nw_dst={ip(t)},actions=group:{t + 1}" for t in range(self.n) if t != v]
                ofctl("add-flows", self.sws[v].name, "-", stdin="\n".join(lines) + "\n")
        return len(jobs), (time.time() - t0) * 1e3

    def reset(self, sws):
        for s in sws:
            ofctl("del-flows", s.name)
            subprocess.run(["ovs-ofctl", "-O", "OpenFlow13", "del-groups", s.name], capture_output=True)
        self.cur = {}


# ----------------------------------------------------------------------------- one run
def run_once(a, srv, topo, net, hosts, sws, port, dev, cap, hport, method, load, seed):
    n, D = topo["nodes"], [tuple(x) for x in topo["directed"]]
    K, T = a.intervals, a.interval_s
    st = srv.call(cmd="start", method=method, load=load, seed=seed, intervals=K, scale=a.scale)
    rates = st["rates_mbps"]                                   # [k][src][dst]
    inst = Installer(sws, port, hport, n)
    inst.reset(sws)
    zero = [0.0] * len(D)
    rep = srv.call(cmd="step", k=0, util=zero, queue=zero, drop=zero)
    inst.apply(rep["routes"], first=True)
    os.makedirs(a.workdir, exist_ok=True)
    for f in os.listdir(a.workdir):
        os.remove(os.path.join(a.workdir, f))
    start = time.time() + 3.0
    for i in range(n):
        json.dump(dict(interval_s=T, start_time=start, rates_mbps={str(i): [rates[k][i] for k in range(K)]},
                       dest_ip=[ip(j) for j in range(n)], payload_bytes=a.payload, flows_per_pair=a.flows),
                  open(f"{a.workdir}/sched_{i}.json", "w"))
    until = start + K * T + 2.0
    recv = [h.popen([a.host_python, AGENT, "recv", "--me", str(i), "--out", f"{a.workdir}/recv_{i}.json",
                     "--until", str(until)]) for i, h in enumerate(hosts)]
    send = [h.popen([a.host_python, AGENT, "send", "--me", str(i), "--schedule", f"{a.workdir}/sched_{i}.json",
                     "--out", f"{a.workdir}/sent_{i}.json"]) for i, h in enumerate(hosts)]
    tx_prev = {e: read_counter(dev[e], "tx_bytes") for e in D}
    tc_prev = read_tc()
    txp_prev = {e: read_counter(dev[e], "tx_packets") for e in D}
    mlu, ustd, dec_ms, inst_ms, n_mod = [], [], [], [], []
    for k in range(1, K):
        while time.time() < start + k * T + 0.02:
            time.sleep(0.005)
        tx = {e: read_counter(dev[e], "tx_bytes") for e in D}
        txp = {e: read_counter(dev[e], "tx_packets") for e in D}
        tc = read_tc()
        util = [8 * (tx[e] - tx_prev[e]) / (cap[e] * T) for e in D]
        drops = [tc.get(dev[e], (0, 0))[0] - tc_prev.get(dev[e], (0, 0))[0] for e in D]
        drop = [d / max(1, d + txp[e] - txp_prev[e]) for d, e in zip(drops, D)]
        queue = [min(1.0, tc.get(dev[e], (0, 0))[1] / 100.0) for e in D]
        tx_prev, txp_prev, tc_prev = tx, txp, tc
        if k > a.warmup:
            mlu.append(max(util)); m = sum(util) / len(util)
            ustd.append((sum((x - m) ** 2 for x in util) / len(util)) ** 0.5)
        rep = srv.call(cmd="step", k=k, util=util, queue=queue, drop=drop)
        dec_ms.append(rep["decision_ms"])
        cnt, ms = inst.apply(rep["routes"])
        inst_ms.append(ms); n_mod.append(cnt)
    for p in send:
        p.wait()
    time.sleep(max(0.0, until - time.time()))
    for p in recv:
        p.wait(timeout=10)
    sent = [json.load(open(f"{a.workdir}/sent_{i}.json")) for i in range(n)]
    got = [json.load(open(f"{a.workdir}/recv_{i}.json")) for i in range(n)]
    S = R = DS = 0.0
    ratios = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            s = sum(c for k, c in sent[i].get(str(j), {}).items() if int(k) >= a.warmup)
            r = [v for k, v in got[j].get(str(i), {}).items() if int(k) >= a.warmup]
            rc = sum(v[0] for v in r); ds = sum(v[1] for v in r)
            if s > 0:
                ratios.append(min(1.0, rc / s))
            S += s; R += min(rc, s); DS += ds
    span = (K - a.warmup) * T
    wire = (a.payload + 42) * 8
    loss = 1 - R / S if S else 0.0
    delay = DS / R if R else float("nan")
    jain = (sum(ratios) ** 2) / (len(ratios) * sum(x * x for x in ratios)) if ratios else 1.0
    thr = R * wire / span / 1e6
    reward = (1 - loss) - 0.5 * delay / st["ref_delay_ms"] - 2 * loss
    return dict(method=method, load=load, seed=seed, reward=round(reward, 5), loss_pct=round(100 * loss, 4),
                delay_ms=round(delay, 3), throughput_mbps_emulated=round(thr, 3),
                throughput_mbps_model=round(thr / a.scale, 2), mlu=round(sum(mlu) / max(1, len(mlu)), 4),
                util_std=round(sum(ustd) / max(1, len(ustd)), 4), jain=round(jain, 5),
                decision_ms=round(sum(dec_ms) / len(dec_ms), 3), install_ms=round(sum(inst_ms) / len(inst_ms), 1),
                groups_modified_per_interval=round(sum(n_mod) / len(n_mod), 1),
                packets_sent=int(S), packets_received=int(R))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="127.0.0.1:6655")
    ap.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    ap.add_argument("--loads", type=float, nargs="+", default=[0.8, 0.95])
    ap.add_argument("--seeds", type=int, nargs="+", default=[101])
    ap.add_argument("--intervals", type=int, default=60, help="control intervals per run")
    ap.add_argument("--interval-s", type=float, default=1.0, help="seconds per control interval")
    ap.add_argument("--warmup", type=int, default=5, help="intervals excluded from metrics")
    ap.add_argument("--scale", type=float, default=0.1, help="link bandwidth scale (100 Mbps -> 10 Mbps)")
    ap.add_argument("--flows", type=int, default=8, help="UDP flows per host pair")
    ap.add_argument("--payload", type=int, default=1000)
    ap.add_argument("--host-python", default=sys.executable)
    ap.add_argument("--workdir", default="/tmp/mnval")
    ap.add_argument("--out", default=os.path.join(ROOT, "results_mininet"))
    ap.add_argument("--check", action="store_true", help="build, install OSPF, pingall, exit")
    ap.add_argument("--cli", action="store_true", help="open the Mininet CLI after installing OSPF")
    a = ap.parse_args()
    if os.geteuid() != 0:
        sys.exit("run with sudo (Mininet needs root)")
    from mininet.log import setLogLevel
    from mininet.clean import cleanup
    setLogLevel("warning")
    cleanup()
    srv = Server(a.server)
    topo = srv.call(cmd="hello")
    print(f"topology: {topo['nodes']} nodes, {len(topo['links'])} links | server methods: {topo['methods']}")
    net, hosts, sws, port, dev, cap, hport = build(topo, a.scale)
    try:
        if a.check or a.cli:
            srv.call(cmd="start", method="OSPF", load=0.5, seed=1, intervals=2, scale=a.scale)
            z = [0.0] * len(topo["directed"])
            inst = Installer(sws, port, hport, topo["nodes"])
            inst.reset(sws)
            inst.apply(srv.call(cmd="step", k=0, util=z, queue=z, drop=z)["routes"], first=True)
            loss = net.pingAll(timeout="1")
            print(f"pingAll over OSPF groups: {loss:.1f}% dropped")
            if a.cli:
                from mininet.cli import CLI
                CLI(net)
            return
        os.makedirs(a.out, exist_ok=True)
        path = os.path.join(a.out, f"{topo.get('name', 'topology')}_runs.csv")
        rows = []
        for load in a.loads:
            for seed in a.seeds:
                for m in a.methods:
                    t0 = time.time()
                    r = run_once(a, srv, topo, net, hosts, sws, port, dev, cap, hport, m, load, seed)
                    rows.append(r)
                    print(f"[{time.time() - t0:5.0f}s] load {load} seed {seed} {m:28s} reward {r['reward']:+.4f} "
                          f"loss {r['loss_pct']:.2f}% delay {r['delay_ms']:.1f} ms MLU {r['mlu']:.2f}", flush=True)
                    with open(path, "w", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print("results ->", path)
    finally:
        net.stop()


if __name__ == "__main__":
    main()
