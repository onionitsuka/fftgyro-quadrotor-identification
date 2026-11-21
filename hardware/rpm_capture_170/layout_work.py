"""Breadboard topology and routed geometry; coordinates in 2.54 mm pitches."""

import heapq
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COLS = dict(zip("ABCDEFGHIJ", [0, 1, 2, 3, 4, 7, 8, 9, 10, 11], strict=False))


def xy(h):
    return COLS[h[0]], int(h[1:]) - 1


def strip(h):
    return ("L" if h[0] < "F" else "R") + h[1:]


IC = [(i, f"E{5 + i}") for i in range(1, 8)] + [(i, f"F{20 - i}") for i in range(8, 15)]
PARTS = []


def part(ref, val, a, b):
    PARTS.append(dict(ref=ref, value=val, a=a, b=b))


for ch, pu, rin, cf, ro in [
    (1, ("A1", "A3"), ("D3", "D6"), ("C6", "C5"), ("B7", "B4")),
    (2, ("A17", "A14"), ("D14", "D10"), ("C10", "C12"), ("B11", "B15")),
    (3, ("J1", "J3"), ("G3", "G7"), ("H7", "H5"), ("I8", "J4")),
    (4, ("J17", "J14"), ("G14", "G11"), ("H11", "H13"), ("I12", "I15")),
]:
    for typ, val, ends in [
        ("RP", "4.7k", pu),
        ("RI", "1k", rin),
        ("CF", "1n", cf),
        ("RO", "100", ro),
    ]:
        part(f"{typ}{ch}", val, *ends)
part("F1", "MF-R050", "C2", "E1")
part("C1", "100n", "J6", "J9")
part("C2", "100n", "I1", "I2")
part("C3", "10u 25V MLCC", "F1", "F2")
NETS = {s: s for s in [f"{lr}{r}" for lr in "LR" for r in range(1, 18)]}
for net, strips in {
    "GND": ["L5", "L8", "L12", "L13", "L16", "R2", "R5", "R9", "R13", "R16"],
    "SENSOR_5V": ["L1", "R1", "L17", "R17"],
    "USB_5V": ["L2", "R6"],
}.items():
    for s in strips:
        NETS[s] = net
for ch, ins, outs, raw, post in [
    (1, "L6", "L7", "L3", "L4"),
    (2, "L10", "L11", "L14", "L15"),
    (3, "R7", "R8", "R3", "R4"),
    (4, "R11", "R12", "R14", "R15"),
]:
    for s, n in [(ins, f"IN{ch}"), (outs, f"OUT{ch}"), (raw, f"RAW{ch}"), (post, f"D{ch + 1}")]:
        NETS[s] = n


# C1's second terminal is the tied unused input, pin 11.
def net(h):
    return NETS[strip(h)]


HOLES = [f"{c}{r}" for c in COLS for r in range(1, 18)]
OCC = {h for _, h in IC} | {p[k] for p in PARTS for k in ["a", "b"]}
assert len(OCC) == 54
SEG = [(xy(p["a"]), xy(p["b"])) for p in PARTS]


def distance(p, a, b):
    dx, dy = b[0] - a[0], b[1] - a[1]
    t = max(0, min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - a[0] - t * dx, p[1] - a[1] - t * dy)


# Block bare leads and the IC body. Routing grid is half a pitch.
GRID = [(x, y) for x in range(-5, 28) for y in range(-5, 38)]
BASE = set()
for q in GRID:
    p = (q[0] / 2, q[1] / 2)
    if (4.15 <= p[0] <= 6.85 and 4.5 <= p[1] <= 11.5) or any(
        distance(p, a, b) < 0.34 for a, b in SEG
    ):
        BASE.add(q)


def point(h):
    return tuple(round(v * 2) for v in xy(h))


FREE = [h for h in HOLES if h not in OCC and point(h) not in BASE]
# External leads terminate at edge holes. Their off-board routes are not optimized.
EXTERNAL = [
    ("Nano 5V", "B2"),
    ("Nano GND", "A8"),
    ("Nano D2", "A4"),
    ("Nano D3", "C15"),
    ("Nano D4", "H4"),
    ("Nano D5", "H15"),
    ("S1 +", "B1"),
    ("S1 OUT", "B3"),
    ("S1 GND signal", "A5"),
    ("S1 GND power", "E16"),
    ("S2 +", "B17"),
    ("S2 OUT", "C14"),
    ("S2 GND signal", "A13"),
    ("S2 GND power", "C13"),
    ("S3 +", "H1"),
    ("S3 OUT", "I3"),
    ("S3 GND signal", "H2"),
    ("S3 GND power", "G16"),
    ("S4 +", "I17"),
    ("S4 OUT", "H14"),
    ("S4 GND signal", "I16"),
    ("S4 GND power", "H16"),
]
assert len(set(h for _, h in EXTERNAL)) == len(EXTERNAL)
assert not OCC & {h for _, h in EXTERNAL}
# A terminal next to a bare lead is allowed, but not on its projected line.
assert all(point(h) not in BASE for _, h in EXTERNAL), [h for _, h in EXTERNAL if point(h) in BASE]
OCC.update(h for _, h in EXTERNAL)
FREE = [h for h in FREE if h not in OCC]
BASE.update(point(h) for h in OCC)


def route(sources, targets, blocked, wire_grid=frozenset()):
    goals = {point(h): h for h in targets}
    if not goals:
        return None
    queue = []
    cost = {}
    prev = {}
    origin = {}
    for h in sources:
        q = point(h)
        if q in blocked:
            continue
        cost[q] = 0
        origin[q] = h
        heapq.heappush(queue, (0, q))
    while queue:
        d, q = heapq.heappop(queue)
        if d != cost[q]:
            continue
        if q in goals:
            path = [q]
            while q in prev:
                q = prev[q]
                path.append(q)
            path.reverse()
            return origin[q], goals[path[-1]], path, d
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            v = (q[0] + dx, q[1] + dy)
            if v[0] < -5 or v[0] > 27 or v[1] < -5 or v[1] > 37 or v in blocked:
                continue
            # Favor staying on board and few turns.
            turn = 0
            if q in prev:
                old = prev[q]
                turn = 0.2 if (q[0] - old[0], q[1] - old[1]) != (dx, dy) else 0
            nd = (
                d
                + 1
                + turn
                + (80 if v in wire_grid else 0)
                + (0.15 if not (0 <= v[0] <= 22 and 0 <= v[1] <= 32) else 0)
            )
            if nd < cost.get(v, 1e9):
                cost[v] = nd
                prev[v] = q
                origin[v] = origin[q]
                heapq.heappush(queue, (nd, v))
    return None


def attempt(seed):
    rng = random.Random(seed)
    free = set(FREE)
    blocked = set(BASE)
    wires = []
    wire_grid = set()
    groups = [
        ("GND", [s for s, n in NETS.items() if n == "GND"]),
        ("SENSOR_5V", [s for s, n in NETS.items() if n == "SENSOR_5V"]),
        ("USB_5V", [s for s, n in NETS.items() if n == "USB_5V"]),
    ]
    rng.shuffle(groups)
    # Short common-ground bridge is intentionally above the insulated IC package.
    bridge = [
        (3, 7),
        (3.5, 7),
        (3.5, 7.5),
        (4, 7.5),
        (4.5, 7.5),
        (5, 7.5),
        (5.5, 7.5),
        (6, 7.5),
        (6.5, 7.5),
        (7, 7.5),
        (7.5, 7.5),
        (7.5, 8),
        (8, 8),
    ]
    wires.append(dict(net="GND", a="D8", b="G9", path=bridge, cost=12, over_ic=True))
    free.difference_update(["D8", "G9"])
    for x, y in bridge:
        for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
            wire_grid.add((round(x * 2) + dx, round(y * 2) + dy))
    for name, ss in groups:
        rng.shuffle(ss)
        if name == "GND":
            joined = {"L8", "R9"}
            ss = [s for s in ss if s not in joined]
        else:
            joined = {ss.pop()}
        while ss:
            sources = [h for h in free if strip(h) in joined]
            targets = [h for h in free if strip(h) in ss]
            if len(sources) <= 1 and len(ss) > 1:
                targets = [h for h in targets if sum(strip(k) == strip(h) for k in targets) > 1]
            route_blocked = set(blocked)
            if name == "GND":
                # One insulated bridge corridor above the IC body, between pin rows 8 and 9.
                route_blocked.difference_update((x, 15) for x in range(9, 14))
            r = route(sorted(sources), sorted(targets), route_blocked, wire_grid)
            if r is None:
                return None
            a, b, path, d = r
            free.remove(a)
            free.remove(b)
            joined.add(strip(b))
            ss.remove(strip(b))
            wires.append(dict(net=name, a=a, b=b, path=[[x / 2, y / 2] for x, y in path], cost=d))
            # Full pitch between adjacent routed jumper centerlines except terminals.
            for x, y in path:
                for dx, dy in [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]:
                    wire_grid.add((x + dx, y + dy))
    return wires


if __name__ == "__main__":
    best = None
    valid = 0
    for seed in range(200):
        wires = attempt(seed)
        if wires is None:
            continue
        valid += 1
        score = sum(w["cost"] for w in wires)
        if best is None or score < best[0]:
            best = (score, seed, wires)
            print("best", seed, round(score, 1), flush=True)
    assert best is not None, "no non-crossing routing"
    out = dict(
        ic=IC,
        parts=PARTS,
        external=EXTERNAL,
        nets=NETS,
        wires=best[2],
        seed=best[1],
        valid_candidates=valid,
        candidates=200,
    )
    (ROOT / "layout.json").write_text(json.dumps(out, indent=2))
    print("VALID", valid, "wires", len(best[2]))
