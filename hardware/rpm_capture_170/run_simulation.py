# ruff: noqa: E501
# Long strings below are generated CAD/HTML text, not control logic.
import argparse
import ctypes as c
from pathlib import Path

root = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True, help="KiCad libngspice shared library path")
libpath = parser.parse_args().library
lib = c.CDLL(libpath)
logs = []


@c.CFUNCTYPE(c.c_int, c.c_char_p, c.c_int, c.c_void_p)
def msg(s, i, u):
    logs.append(s.decode(errors="replace"))
    return 0


@c.CFUNCTYPE(c.c_int, c.c_int, c.c_bool, c.c_bool, c.c_int, c.c_void_p)
def ex(a, b, d, i, u):
    logs.append(f"EXIT {a}")
    return 0


lib.ngSpice_Init(msg, msg, ex, None, None, None, None)
lines = [
    "4-channel RPM receiver - illustrative switch models, not vendor validation",
    "Vcc vcc 0 5",
    ".model SENSOR SW(Ron=10 Roff=1e12 Vt=2.5 Vh=0)",
    ".model HCBEHAV SW(Ron=1 Roff=1e12 Vt=2.5 Vh=0.5)",
]
for ch in range(1, 5):
    lines += [
        f"Vdrive{ch} drive{ch} 0 PULSE(0 5 {100 + ch * 20}u 10n 10n 200u 500u)",
        f"Vglitch{ch} glitch{ch} 0 PULSE(0 5 4600u 1n 1n 100n 10m)",
        f"Bctrl{ch} ctrl{ch} 0 V=V(drive{ch})+V(glitch{ch})",
        f"Ssens{ch} raw{ch} 0 ctrl{ch} 0 SENSOR",
        f"RP{ch} vcc raw{ch} 4.7k",
        f"RI{ch} raw{ch} rc{ch} 1k",
        f"CF{ch} rc{ch} 0 1n",
        f"Shc{ch} out{ch} 0 rc{ch} 0 HCBEHAV",
        f"Rlogic{ch} vcc out{ch} 1k",
        f"RO{ch} out{ch} nano{ch} 100",
        f"Cload{ch} nano{ch} 0 10p",
    ]
lines += [".tran 0.05u 5m 0 0.05u", ".end"]
(root / "receiver_demo.cir").write_text("\n".join(lines) + "\n")
a = (c.c_char_p * (len(lines) + 1))(*(line.encode() for line in lines), None)
assert lib.ngSpice_Circ(a) == 0
for cmd in [
    "set noaskquit",
    "run",
    "set wr_singlescale",
    "set wr_vecnames",
    f"wrdata {root / 'simulation_wave.tmp'} v(ctrl1) v(raw1) v(rc1) v(nano1) v(nano2) v(nano3) v(nano4)",
]:
    assert lib.ngSpice_Command(cmd.encode()) == 0, cmd
(root / "ngspice.log").write_text("\n".join(logs) + "\n")
print("\n".join(logs[-12:]))
