"""Causal four-timestamp affine mapping, client monotonic -> server monotonic.

Software-clock bounds only; USB/sensor sample latency is not measured here.
All exchanges constrain the fit. No asymmetry or outlier rejection assumption.
"""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path

from .smooth_clock import SmoothIntervalClock, fit_smooth_interval_clock


@dataclass(frozen=True)
class PiMap:
    client_origin_ns: int
    server_origin_ns: int
    vertices: tuple[tuple[float, float], ...]
    first_ns: int
    last_ns: int

    def map(self, stamp: int) -> tuple[int, int, int]:
        if not self.first_ns <= stamp <= self.last_ns:
            raise ValueError("timestamp outside clock exchange coverage")
        x = (stamp - self.client_origin_ns) * 1e-9
        values = [a * x + b for a, b in self.vertices]
        return (
            self.server_origin_ns + round(sum(values) / len(values) * 1e9),
            self.server_origin_ns + math.floor(min(values) * 1e9),
            self.server_origin_ns + math.ceil(max(values) * 1e9),
        )


@dataclass(frozen=True)
class PiecewisePiMap:
    """Piecewise midpoint mapping for explicitly reviewed non-affine clocks."""

    client_ns: tuple[int, ...]
    offsets_ns: tuple[int, ...]
    lower_offsets_ns: tuple[int, ...]
    upper_offsets_ns: tuple[int, ...]
    first_ns: int
    last_ns: int

    def map(self, stamp: int) -> tuple[int, int, int]:
        if not self.first_ns <= stamp <= self.last_ns:
            raise ValueError("timestamp outside clock exchange coverage")
        right = bisect.bisect_left(self.client_ns, stamp)
        if right == 0:
            left = right = 0
            weight = 0.0
        elif right == len(self.client_ns):
            left = right = len(self.client_ns) - 1
            weight = 0.0
        else:
            left = right - 1
            weight = (stamp - self.client_ns[left]) / (self.client_ns[right] - self.client_ns[left])

        def blend(values: tuple[int, ...]) -> int:
            return values[left] + round(weight * (values[right] - values[left]))

        estimate = stamp + blend(self.offsets_ns)
        lower = stamp + blend(self.lower_offsets_ns)
        upper = stamp + blend(self.upper_offsets_ns)
        return estimate, min(lower, estimate), max(upper, estimate)


def fit_piecewise_midpoint(rows: list[tuple[int, int, int, int]]) -> PiecewisePiMap:
    """Interpolate NTP-style midpoint offsets without an affine-clock claim."""
    if len(rows) < 2:
        raise ValueError("need at least two clock exchanges")
    client = tuple((t1 + t4) // 2 for t1, _, _, t4 in rows)
    if any(b <= a for a, b in zip(client, client[1:], strict=False)):
        raise ValueError("piecewise clock anchors must strictly increase")
    server = tuple((t2 + t3) // 2 for _, t2, t3, _ in rows)
    offsets = tuple(s - c for s, c in zip(server, client, strict=True))
    lower = tuple(t2 - t4 for t1, t2, t3, t4 in rows)
    upper = tuple(t3 - t1 for t1, t2, t3, t4 in rows)
    return PiecewisePiMap(client, offsets, lower, upper, rows[0][0], rows[-1][3])


def fit_smooth_interval(rows: list[tuple[int, int, int, int]]) -> SmoothIntervalClock:
    """Fit a smooth monotone client-to-server mapping inside NTP bounds."""
    if len(rows) < 3:
        raise ValueError("need at least three clock exchanges")
    client = [(t1 + t4) // 2 for t1, _, _, t4 in rows]
    lower = [center + row[1] - row[3] for center, row in zip(client, rows, strict=True)]
    upper = [center + row[2] - row[0] for center, row in zip(client, rows, strict=True)]
    return fit_smooth_interval_clock(
        client, lower, upper, first_stamp=rows[0][0], last_stamp=rows[-1][3]
    )


def fit(rows: list[tuple[int, int, int, int]], slack_ns: int = 0) -> PiMap:
    if len(rows) < 3 or slack_ns < 0:
        raise ValueError("need three exchanges and nonnegative slack")
    for i, (t1, t2, t3, t4) in enumerate(rows):
        if min(t1, t2, t3, t4) < 0 or t4 <= t1 or t3 < t2:
            raise ValueError("invalid exchange timestamp ordering")
        if i and (t1 <= rows[i - 1][3] or t2 <= rows[i - 1][2]):
            raise ValueError("overlapping/reordered exchanges or clock reset")
    c0, s0 = rows[0][:2]
    points = [
        (
            (t1 - c0) * 1e-9,
            (t2 - s0 + slack_ns) * 1e-9,
            (t3 - s0 - slack_ns) * 1e-9,
            (t4 - c0) * 1e-9,
        )
        for t1, t2, t3, t4 in rows
    ]
    f, z = points[0], points[-1]
    amin = (z[2] - f[1]) / (z[3] - f[0])
    amax = (z[1] - f[2]) / (z[0] - f[3])
    if not 0 < amin <= amax:
        raise ValueError("no bounded positive affine clock")
    vertices = [(amin, f[2] - amax * f[3]), (amax, f[2] - amax * f[3]), (amax, f[1]), (amin, f[1])]
    for x1, y2, y3, x4 in points:
        # a*x1+b <= y2; a*x4+b >= y3.
        for x, y, sign in ((x1, y2, -1), (x4, y3, 1)):
            clipped = []
            if not vertices:
                raise ValueError("no affine clock satisfies every exchange")
            prev = vertices[-1]
            pd = sign * (prev[0] * x + prev[1] - y)
            for cur in vertices:
                cd = sign * (cur[0] * x + cur[1] - y)
                if (pd >= 0) != (cd >= 0):
                    fraction = pd / (pd - cd)
                    clipped.append(
                        (
                            prev[0] + fraction * (cur[0] - prev[0]),
                            prev[1] + fraction * (cur[1] - prev[1]),
                        )
                    )
                if cd >= 0:
                    clipped.append(cur)
                prev, pd = cur, cd
            vertices = clipped
    if not vertices:
        raise ValueError("no affine clock satisfies every exchange")
    return PiMap(c0, s0, tuple(vertices), rows[0][0], rows[-1][3])


def read_exchanges(path: Path) -> list[tuple[int, int, int, int]]:
    with path.open(newline="", encoding="utf-8") as source:
        return [
            tuple(
                int(row[k])
                for k in (
                    "client_send_ns",
                    "server_receive_ns",
                    "server_send_ns",
                    "client_receive_ns",
                )
            )
            for row in csv.DictReader(source)
        ]
