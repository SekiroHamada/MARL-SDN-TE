#!/usr/bin/env python3
"""
Traffic agent that runs INSIDE a Mininet host (standard library only).

  send : follows a per-interval rate schedule (Mbps per destination), spreading each
         destination's traffic over K UDP source ports (= K flows, so OpenFlow
         select groups can hash them onto different next hops). Every packet
         carries (src, dst, flow, interval, seq, send_time_ns) so the receiver can
         measure loss and one-way delay - Mininet hosts share the kernel clock.
  recv : counts packets and delay per (source, interval) and dumps JSON stats.

Usage (the runner starts these for you):
  traffic_agent.py send --me 0 --schedule sched.json --out sent_0.json
  traffic_agent.py recv --me 0 --out recv_0.json --until <epoch_s>
"""
import argparse, heapq, json, os, signal, socket, struct, sys, time

HDR = struct.Struct("!HHHIIQ")          # src, dst, flow, interval, seq, send_time_ns
PORT = 5001


def atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def send(a):
    sc = json.load(open(a.schedule))
    me, T, start = a.me, sc["interval_s"], sc["start_time"]
    rates = sc["rates_mbps"][str(me)] if isinstance(sc["rates_mbps"], dict) else sc["rates_mbps"][me]
    dest_ip = sc["dest_ip"]                       # list: ip of host t
    size, K = sc.get("payload_bytes", 1000), sc.get("flows_per_pair", 8)
    wire_bits = (size + 42) * 8                   # UDP+IP+Ethernet headers
    pad = b"\0" * (size - HDR.size)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 << 20)
    socks = []                                    # one socket per flow -> distinct source ports
    for k in range(K):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("0.0.0.0", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 1 << 20)
        socks.append(s)
    sent = {}                                     # "dst" -> {interval: packets}
    seq = 0
    while time.time() < start:
        time.sleep(min(0.05, max(0.0, start - time.time())))
    for k, row in enumerate(rates):
        t0 = start + k * T
        heap = []
        for dst, r in enumerate(row):
            if dst == me or r <= 0:
                continue
            n = int(round(r * 1e6 * T / wire_bits))
            for j in range(n):
                heapq.heappush(heap, (t0 + (j + 0.5) * T / n, dst, j % K))
        cnt = {}
        while heap:
            when, dst, flow = heapq.heappop(heap)
            d = when - time.time()
            if d > 0.0005:
                time.sleep(d)
            if time.time() > t0 + T + 0.5 * T:        # hopelessly behind: skip rest of interval
                break
            seq += 1
            try:
                socks[flow].sendto(HDR.pack(me, dst, flow, k, seq & 0xFFFFFFFF, time.time_ns()) + pad,
                                   (dest_ip[dst], PORT))
                cnt[dst] = cnt.get(dst, 0) + 1
            except OSError:
                pass
        for dst, c in cnt.items():
            sent.setdefault(str(dst), {})[str(k)] = c
        while time.time() < t0 + T:
            time.sleep(0.001)
    atomic_dump(sent, a.out)


def recv(a):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
    sock.bind(("0.0.0.0", PORT))
    sock.settimeout(0.2)
    stats = {}                                    # "src" -> {interval: [packets, delay_sum_ms]}
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    last_dump = time.time()
    while not stop["flag"] and time.time() < a.until:
        try:
            data, _ = sock.recvfrom(65535)
        except socket.timeout:
            data = None
        now = time.time_ns()
        if data and len(data) >= HDR.size:
            src, dst, flow, k, seq, ts = HDR.unpack_from(data)
            e = stats.setdefault(str(src), {}).setdefault(str(k), [0, 0.0])
            e[0] += 1
            e[1] += (now - ts) / 1e6
        if time.time() - last_dump > 0.5:
            atomic_dump(stats, a.out); last_dump = time.time()
    atomic_dump(stats, a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["send", "recv"])
    ap.add_argument("--me", type=int, required=True)
    ap.add_argument("--schedule")
    ap.add_argument("--out", required=True)
    ap.add_argument("--until", type=float, default=1e18)
    a = ap.parse_args()
    send(a) if a.mode == "send" else recv(a)
