# ruff: noqa: E501
# Long strings below are generated CAD/HTML text, not control logic.
import json
import uuid
from pathlib import Path

root = Path(__file__).resolve().parent
d = json.loads((root / "layout.json").read_text())


def uid():
    return str(uuid.uuid4())


def q(s):
    return json.dumps(str(s), ensure_ascii=False)


def effect(size=1.27):
    return f"(effects (font (size {size} {size})))"


def pin(num, name, typ, x, y, ang):
    return f"(pin {typ} line (at {x} {y} {ang}) (length 2.54) (name {q(name)} {effect()}) (number {q(num)} {effect()}))"


lib = []


def libsym(name, pins, w=5.08, h=2.54):
    s = f'(symbol {q(name)} (pin_names (offset 0.5)) (in_bom yes) (on_board yes) (property "Reference" "X" (at 0 4 0) {effect()}) (property "Value" {q(name)} (at 0 -4 0) {effect()})'
    s += f"(symbol {q(name + '_0_1')} (rectangle (start {-w} {h}) (end {w} {-h}) (stroke (width 0.254) (type default)) (fill (type background))))"
    s += f"(symbol {q(name + '_1_1')}" + "".join(pin(*p) for p in pins) + "))"
    lib.append(s)


libsym("Passive", [(1, "", "passive", -7.62, 0, 0), (2, "", "passive", 7.62, 0, 180)])
upins = []
for i, (p, o) in enumerate([(1, 2), (3, 4), (5, 6), (9, 8), (11, 10), (13, 12)]):
    y = 12.7 - i * 5.08
    upins.extend([(p, f"{p}A", "input", -12.7, y, 0), (o, f"{p}Y", "output", 12.7, y, 180)])
upins.extend([(14, "VCC", "power_in", 0, 20.32, 270), (7, "GND", "power_in", 0, -20.32, 90)])
libsym("HC14", upins, 10.16, 17.78)
npins = [
    (i, n, "power_out" if i <= 2 else "input", -7.62, 12.7 - (i - 1) * 5.08, 0)
    for i, n in enumerate(["5V", "GND", "D2", "D3", "D4", "D5"], 1)
]
libsym("NanoInterface", npins, 5.08, 15.24)
spins = [
    (i, n, "passive", -7.62, 7.62 - (i - 1) * 5.08, 0)
    for i, n in enumerate(["+", "OUT", "GND signal", "GND power"], 1)
]
libsym("Sensor", spins, 5.08, 10.16)
sheet = uid()
body = []


def text(s, x, y, size=1.5):
    body.append(f"(text {q(s)} (at {x} {y} 0) {effect(size)} (uuid {q(uid())}))")


def inst(name, ref, val, x, y, pins, nets):
    x = round(x / 1.27) * 1.27
    y = round(y / 1.27) * 1.27
    ident = uid()
    s = f"(symbol (lib_id {q('RPM:' + name)}) (at {x} {y} 0) (unit 1) (in_bom yes) (on_board yes) (dnp no) (uuid {q(ident)})"
    s += f'(property "Reference" {q(ref)} (at {x} {y - 5 if name == "Passive" else y - 23} 0) {effect()}) (property "Value" {q(val)} (at {x} {y + 5 if name == "Passive" else y + 24} 0) {effect(1)})'
    s += "".join(f"(pin {q(p[0])} (uuid {q(uid())}))" for p in pins)
    s += f'(instances (project "rpm_capture" (path {q("/" + sheet)} (reference {q(ref)}) (unit 1)))))'
    body.append(s)
    for p in pins:
        no, _, _, px, py, ang = p
        tx, ty = x + px, y - py
        n = nets[str(no)]
        if n is None:
            body.append(f"(no_connect (at {tx} {ty}) (uuid {q(uid())}))")
            continue
        dx, dy = {0: (-2.54, 0), 180: (2.54, 0), 270: (0, -2.54), 90: (0, 2.54)}[ang]
        ex, ey = tx + dx, ty + dy
        body.append(
            f"(wire (pts (xy {tx} {ty}) (xy {ex} {ey})) (stroke (width 0) (type default)) (uuid {q(uid())}))"
        )
        angle = 180 if ang == 0 else 0
        body.append(
            f"(label {q(n)} (at {ex} {ey} {angle}) (effects (font (size 1 1)) (justify left bottom)) (uuid {q(uid())}))"
        )


passive = [(1, "", "passive", -7.62, 0, 0), (2, "", "passive", 7.62, 0, 180)]
for ch in range(1, 5):
    y = 40 + (ch - 1) * 35
    for col, typ in enumerate(["RP", "RI", "CF", "RO"]):
        p = next(p for p in d["parts"] if p["ref"] == f"{typ}{ch}")
        inst(
            "Passive",
            p["ref"],
            p["value"],
            55 + col * 48,
            y,
            passive,
            {
                str(i + 1): d["nets"][("L" if p[k][0] < "F" else "R") + p[k][1:]]
                for i, k in enumerate(["a", "b"])
            },
        )
    text(f"RPM{ch}", 18, y, 2)
for i, ref in enumerate(["F1", "C1", "C2", "C3"]):
    p = next(p for p in d["parts"] if p["ref"] == ref)
    inst(
        "Passive",
        ref,
        p["value"],
        55 + 48 * i,
        190,
        passive,
        {
            str(j + 1): d["nets"][("L" if p[k][0] < "F" else "R") + p[k][1:]]
            for j, k in enumerate(["a", "b"])
        },
    )
icnet = {str(i): d["nets"][("L" if h[0] < "F" else "R") + h[1:]] for i, h in d["ic"]}
icnet["4"] = None
icnet["10"] = None
inst("HC14", "U1", "TC74HC14AP", 272, 74, upins, icnet)
inst(
    "NanoInterface",
    "J5",
    "Nano Every (external)",
    345,
    74,
    npins,
    dict(zip(map(str, range(1, 7)), ["USB_5V", "GND", "D2", "D3", "D4", "D5"], strict=False)),
)
for ch in range(1, 5):
    inst(
        "Sensor",
        f"J{ch}",
        f"EE-SPY401 #{ch}",
        265 + (ch - 1) % 2 * 80,
        145 + (ch - 1) // 2 * 55,
        spins,
        {"1": "SENSOR_5V", "2": f"RAW{ch}", "3": "GND", "4": "GND"},
    )
text("4ch RPM / 170-hole breadboard / Nano Every external", 160, 15, 3)
text("Mapping: 1->2 / 5->6 / 13->12 / 9->8. Unused inputs 3,11 to GND.", 158, 225)
text("Each sensor: local 100nF + 1uF across supply and GND (off this board).", 158, 236)
text("C3: 10uF 25V non-polar MLCC. All repeated labels are connected.", 158, 247)
(root / "rpm_capture.kicad_sch").write_text(
    f'(kicad_sch (version 20250114) (generator "eeschema") (uuid {q(sheet)}) (paper "A3") (lib_symbols '
    + "".join(
        s.replace('(symbol "' + name + '"', '(symbol "RPM:' + name + '"', 1)
        for s, name in zip(lib, ["Passive", "HC14", "NanoInterface", "Sensor"], strict=False)
    )
    + ")"
    + "".join(body)
    + ")"
)
print(root / "rpm_capture.kicad_sch")

(root / "RPM.kicad_sym").write_text(
    '(kicad_symbol_lib (version 20241209) (generator "kicad_symbol_editor") ' + "".join(lib) + ")"
)
(root / "sym-lib-table").write_text(
    '(sym_lib_table (lib (name "RPM") (type "KiCad") (uri "${KIPRJMOD}/RPM.kicad_sym") (options "") (descr "Project-local symbols")))'
)
