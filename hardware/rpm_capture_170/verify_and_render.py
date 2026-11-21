# ruff: noqa: E501
# Long strings below are generated CAD/HTML text, not control logic.
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import layout_work as m
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

root = Path(__file__).resolve().parent

d = json.loads((root / "layout.json").read_text())
# Verify physical strip unions independently against KiCad's exported electrical netlist.
parent = {f"{s}{r}": f"{s}{r}" for s in "LR" for r in range(1, 18)}


def rt(x):
    while parent[x] != x:
        x = parent[x]
    return x


def join(a, b):
    parent[rt(m.strip(a))] = rt(m.strip(b))


for w in d["wires"]:
    join(w["a"], w["b"])
term = {}
for p in d["parts"]:
    for n, k in [("1", "a"), ("2", "b")]:
        term[p["ref"], n] = p[k]
for n, h in d["ic"]:
    term["U1", str(n)] = h
ext = dict(d["external"])
for n, key in enumerate(["Nano 5V", "Nano GND", "Nano D2", "Nano D3", "Nano D4", "Nano D5"], 1):
    term["J5", str(n)] = ext[key]
for ch in range(1, 5):
    for n, suffix in enumerate(["+", "OUT", "GND signal", "GND power"], 1):
        term[f"J{ch}", str(n)] = ext[f"S{ch} {suffix}"]
phys = {k: rt(m.strip(h)) for k, h in term.items()}
x = ET.parse(root / "rpm_capture.net.xml")
elec = {}
for net in x.findall("./nets/net"):
    for node in net.findall("node"):
        elec[node.attrib["ref"], node.attrib["pin"]] = net.attrib["name"]
assert set(phys) == set(elec), (set(phys) - set(elec), set(elec) - set(phys))
for a in phys:
    for b in phys:
        assert (phys[a] == phys[b]) == (elec[a] == elec[b]), (a, b)
allholes = (
    [h for _, h in d["ic"]]
    + [p[k] for p in d["parts"] for k in ["a", "b"]]
    + [w[k] for w in d["wires"] for k in ["a", "b"]]
    + [h for _, h in d["external"]]
)
assert len(set(allholes)) == len(allholes)


# Count projected intersections, including touching; ignore segments of the same wire.
def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def hit(a, b, c, e):
    if (
        max(a[0], b[0]) + 1e-8 < min(c[0], e[0])
        or max(c[0], e[0]) + 1e-8 < min(a[0], b[0])
        or max(a[1], b[1]) + 1e-8 < min(c[1], e[1])
        or max(c[1], e[1]) + 1e-8 < min(a[1], b[1])
    ):
        return False
    return cross(a, b, c) * cross(a, b, e) <= 1e-10 and cross(c, e, a) * cross(c, e, b) <= 1e-10


parts = [(m.xy(p["a"]), m.xy(p["b"])) for p in d["parts"]]
wires = [list(zip(w["path"], w["path"][1:], strict=False)) for w in d["wires"]]
bare = sum(hit(*a, *b) for i, a in enumerate(parts) for b in parts[i + 1 :])
wp = sum(any(hit(*a, *b) for a in w) for w in wires for b in parts)
ww = sum(
    any(hit(*a, *b) for a in w for b in v) for i, w in enumerate(wires) for v in wires[i + 1 :]
)
assert bare == wp == ww == 0, (bare, wp, ww)
# Measure the old straight-line placement fairly: old jumper bends were unspecified.
old = json.loads((root / "previous_layout.json").read_text())
oldparts = [(m.xy(a), m.xy(b)) for _, _, a, b in old["parts"]]
oldw = [(m.xy(a), m.xy(b)) for _, a, b in old["wires"]]
oldbare = sum(hit(*a, *b) for i, a in enumerate(oldparts) for b in oldparts[i + 1 :])
oldwp = sum(hit(*a, *b) for a in oldw for b in oldparts)
oldww = sum(hit(*a, *b) for i, a in enumerate(oldw) for b in oldw[i + 1 :])
# Drawing in pitch coordinates. Lead centerlines are exact; body envelopes are illustrative.
colors = {"USB_5V": "#d48b00", "GND": "#516274", "SENSOR_5V": "#da4646"}


def simple(path):
    out = [path[0]]
    for a, b, c in zip(path, path[1:], path[2:], strict=False):
        if (b[0] - a[0], b[1] - a[1]) != (c[0] - b[0], c[1] - b[1]):
            out.append(b)
    return out + [path[-1]]


for w in d["wires"]:
    w["bends"] = simple(w["path"])
fig, ax = plt.subplots(figsize=(8, 10))
fig.patch.set_facecolor("#f5f7fa")
ax.set_facecolor("#f5f7fa")
ax.add_patch(Rectangle((-0.65, -0.65), 12.3, 17.3, facecolor="white", edgecolor="#aab3c2", lw=2))
ax.add_patch(Rectangle((4.6, -0.65), 1.8, 17.3, facecolor="#ecf0f4", edgecolor="none"))
for r in range(1, 18):
    for x1, x2 in [(0, 4), (7, 11)]:
        ax.plot([x1, x2], [r - 1] * 2, color="#e1e5eb", lw=4, zorder=1)
    for _c, xv in m.COLS.items():
        ax.scatter(xv, r - 1, s=18, color="#b9c0cc", zorder=2)
    ax.text(-0.9, r - 1, str(r), ha="right", va="center", fontsize=9)
for c, xv in m.COLS.items():
    ax.text(xv, -1, c, ha="center", fontsize=10, weight="bold")
ax.add_patch(Rectangle((4.25, 4.6), 2.5, 6.8, facecolor="#253347", edgecolor="black", zorder=8))
ax.add_patch(Circle((5.5, 4.6), 0.36, facecolor="#f5f7fa", zorder=9))
ax.text(
    5.5,
    8,
    "TC74HC14AP",
    color="white",
    ha="center",
    va="center",
    rotation=90,
    zorder=10,
    fontsize=9,
)
for n, h in d["ic"]:
    xx, yy = m.xy(h)
    ax.plot([xx, 4.25 if h[0] == "E" else 6.75], [yy, yy], color="#526175", lw=3, zorder=7)
    ax.text(
        xx + (0.13 if h[0] == "E" else -0.13),
        yy - 0.12,
        str(n),
        fontsize=6,
        ha="left" if h[0] == "E" else "right",
        zorder=10,
        color="#f6ba52",
    )
for _i, w in enumerate(d["wires"], 1):
    xs, ys = zip(*w["path"], strict=False)
    ax.plot(xs, ys, color=colors[w["net"]], lw=1.9, zorder=12 if w.get("over_ic") else 4)
    for h in [w["a"], w["b"]]:
        ax.scatter(*m.xy(h), s=20, facecolor="white", edgecolor=colors[w["net"]], lw=1.3, zorder=6)
for p in d["parts"]:
    a, b = m.xy(p["a"]), m.xy(p["b"])
    ax.plot([a[0], b[0]], [a[1], b[1]], color="#394553", lw=1.8, zorder=5)
    cx, cy = (a[0] + b[0]) / 2, (a[1] + b[1]) / 2
    # Stagger resistor bodies relative to neighboring capacitors.
    if p["ref"] == "RO1":
        cx, cy = a[0], a[1] - 0.4
    if p["ref"].startswith("R"):
        ax.add_patch(
            Rectangle(
                (cx - 0.27, cy - 0.55),
                0.54,
                1.1,
                facecolor="#e3bc84",
                edgecolor="#876748",
                zorder=7,
            )
        )
    else:
        ax.add_patch(
            Circle(
                (cx, cy),
                0.29,
                facecolor="#42a8c0" if p["ref"] != "F1" else "#66a75b",
                edgecolor="#294c5a",
                zorder=7,
            )
        )
    ax.text(cx + 0.32, cy, p["ref"], fontsize=6.8, va="center", color="#172c43", zorder=11)
for _label, h in d["external"]:
    ax.scatter(*m.xy(h), s=30, marker="s", facecolor="none", edgecolor="#7d50aa", zorder=9)
ax.set(xlim=(-2, 14), ylim=(18, -2))
ax.set_aspect("equal")
ax.axis("off")
fig.suptitle("170 holes / 4ch RPM / Nano Every external", fontsize=15, weight="bold", y=0.98)
fig.text(
    0.5,
    0.035,
    "Red: sensor 5V   Gold: USB 5V   Gray: GND\nSquares: external terminals. Follow the interactive drawing for each wire.",
    ha="center",
    fontsize=10,
)
fig.tight_layout(rect=(0, 0.06, 1, 0.96))
fig.savefig(root / "placement.png", dpi=180)
fig.savefig(root / "placement.svg")
plt.close(fig)
# KiCad native board is a placement/strip-connectivity model, not a fabricated PCB.
netnames = sorted(set(d["nets"].values()))
ids = {n: i + 1 for i, n in enumerate(netnames)}


def xy_mm(p):
    return (50 + p[0] * 2.54, 40 + p[1] * 2.54)


def hole_mm(h):
    return xy_mm(m.xy(h))


board = [
    '(kicad_pcb (version 20241229) (generator "pcbnew") (general (thickness 1.6)) (paper "A4") (layers (0 "F.Cu" signal) (31 "B.Cu" signal) (36 "B.SilkS" user "b.silkscreen") (37 "F.SilkS" user "f.silkscreen") (40 "Dwgs.User" user "user.drawings") (44 "Edge.Cuts" user)) (setup (pad_to_mask_clearance 0))'
]
for n, i in ids.items():
    board.append(f'(net {i} "{n}")')
for h in m.HOLES:
    xx, yy = hole_mm(h)
    n = d["nets"][m.strip(h)]
    board.append(
        f'(footprint "Hole:{h}" (layer "F.Cu") (at {xx} {yy}) (attr through_hole) (fp_text reference "{h}" (at 0 0) (layer "F.SilkS") hide (effects (font (size 0.5 0.5) (thickness 0.1)))) (pad "1" thru_hole circle (at 0 0) (size 1.5 1.5) (drill 0.9) (layers "*.Cu" "*.Mask") (net {ids[n]} "{n}")))'
    )
for r in range(1, 18):
    for letters in ["ABCDE", "FGHIJ"]:
        for a, b in zip(letters, letters[1:], strict=False):
            x1, y1 = hole_mm(f"{a}{r}")
            x2, y2 = hole_mm(f"{b}{r}")
            n = d["nets"][m.strip(f"{a}{r}")]
            board.append(
                f'(segment (start {x1} {y1}) (end {x2} {y2}) (width 0.4) (layer "B.Cu") (net {ids[n]}))'
            )
for w in d["wires"]:
    for a, b in zip(w["bends"], w["bends"][1:], strict=False):
        x1, y1 = xy_mm(a)
        x2, y2 = xy_mm(b)
        board.append(
            f'(segment (start {x1} {y1}) (end {x2} {y2}) (width 0.3) (layer "F.Cu") (net {ids[w["net"]]}))'
        )
for p in d["parts"]:
    x1, y1 = hole_mm(p["a"])
    x2, y2 = hole_mm(p["b"])
    board.append(
        f'(gr_line (start {x1} {y1}) (end {x2} {y2}) (stroke (width 0.35) (type default)) (layer "Dwgs.User"))'
    )
    board.append(
        f'(gr_text "{p["ref"]} {p["value"]}" (at {(x1 + x2) / 2 + 1} {(y1 + y2) / 2} 90) (layer "Dwgs.User") (effects (font (size 0.65 0.65) (thickness 0.1))))'
    )
for a, b in [
    ((-0.65, -0.65), (11.65, -0.65)),
    ((11.65, -0.65), (11.65, 16.65)),
    ((11.65, 16.65), (-0.65, 16.65)),
    ((-0.65, 16.65), (-0.65, -0.65)),
]:
    x1, y1 = xy_mm(a)
    x2, y2 = xy_mm(b)
    board.append(
        f'(gr_line (start {x1} {y1}) (end {x2} {y2}) (stroke (width 0.1) (type default)) (layer "Edge.Cuts"))'
    )
# Draw the IC package and hole coordinates for native KiCad review.
x1, y1 = xy_mm((4.25, 4.6))
x2, y2 = xy_mm((6.75, 11.4))
board.append(
    f'(gr_rect (start {x1} {y1}) (end {x2} {y2}) (stroke (width 0.2) (type default)) (fill none) (layer "Dwgs.User"))'
)
for number, hole in d["ic"]:
    x, y = hole_mm(hole)
    board.append(
        f'(gr_text "{number}" (at {x} {y - 0.8}) (layer "Dwgs.User") (effects (font (size 0.7 0.7) (thickness 0.1))))'
    )
for letter, x in m.COLS.items():
    x, y = xy_mm((x, -1.4))
    board.append(
        f'(gr_text "{letter}" (at {x} {y}) (layer "Dwgs.User") (effects (font (size 0.9 0.9) (thickness 0.15))))'
    )
for row in range(1, 18):
    x, y = xy_mm((-3, row - 1))
    board.append(
        f'(gr_text "{row}" (at {x} {y}) (layer "Dwgs.User") (effects (font (size 0.9 0.9) (thickness 0.15))))'
    )
board.append(
    '(gr_text "BREADBOARD PLACEMENT ONLY - NOT FOR FABRICATION" (at 65 32) (layer "Dwgs.User") (effects (font (size 1 1) (thickness 0.15))))'
)
board.append(")")
(root / "rpm_capture.kicad_pcb").write_text("\n".join(board))
# Analyze ngspice waveforms (synthetic test data, not experiment data).
w = np.loadtxt(str(root / "simulation_wave.tmp"), skiprows=1)
t = w[:, 0]
counts = [int(np.count_nonzero((w[:-1, i] < 2.5) & (w[1:, i] >= 2.5))) for i in range(4, 8)]
assert counts == [10] * 4, counts
spike = (t >= 0.004600) & (t <= 0.004601)
report = dict(
    kicad_version="10.0.5",
    erc_errors=0,
    erc_warnings=0,
    occupied_holes=len(allholes),
    jumper_count=len(d["wires"]),
    candidates=d["candidates"],
    valid_candidates=d["valid_candidates"],
    selected_seed=d["seed"],
    bare_crossings=bare,
    jumper_part_crossings=wp,
    jumper_jumper_crossings=ww,
    old_straight_line_reference=dict(
        bare_crossings=oldbare, jumper_part_crossings=oldwp, jumper_jumper_crossings=oldww
    ),
    kicad_netlist_matches_physical_strips=True,
    routing_refinements=d.get("routing_refinements", 0),
    refinement_seed=d.get("refinement_seed"),
    simulation=dict(
        model="Illustrative ideal sensor switch + RC + behavioral Schmitt; not vendor model",
        pulse_frequency_hz=2000,
        duration_ms=5,
        additional_glitch_ns=100,
        output_rising_edges=counts,
        rc1_min_during_glitch_v=float(w[spike, 3].min()),
    ),
)
(root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
fig, ax = plt.subplots(2, 1, figsize=(9, 6))
for i, label in [(2, "Sensor OUT"), (3, "After RC"), (4, "Nano input")]:
    ax[0].plot(t * 1e6, w[:, i], label=label, lw=1.2)
ax[0].set(xlim=(100, 150), ylabel="V", title="Illustrative transient: RC + ideal Schmitt")
ax[0].legend()
for i, label in [(2, "Sensor OUT"), (3, "After RC"), (4, "Nano input")]:
    ax[1].plot(t * 1e6, w[:, i], label=label, lw=1.2)
ax[1].set(
    xlim=(4599, 4605),
    ylabel="V",
    xlabel="Time [us]",
    title="Injected 100 ns sensor pull-down glitch",
)
ax[1].legend()
fig.tight_layout()
fig.savefig(root / "simulation.png", dpi=150)
plt.close(fig)
(root / "layout.json").write_text(json.dumps(d, indent=2) + "\n")
print(json.dumps(report, indent=2))
