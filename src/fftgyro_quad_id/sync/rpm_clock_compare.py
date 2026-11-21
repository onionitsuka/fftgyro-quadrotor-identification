"""Compare global affine clock with overlapping, RTT-weighted local clocks.

No automatic promotion of an estimator. Local intervals are conditional on
local affine validity, not directly comparable accuracy certificates.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path

from .rpm_pid import ClockModel, Exchange, fit_clock, read_capture
from .smooth_clock import SmoothIntervalClock, fit_smooth_interval_clock


@dataclass(frozen=True)
class LocalClock:
    centers: tuple[int, ...]
    models: tuple[ClockModel, ...]

    def map(self, nano_us: int) -> tuple[int, int, int]:
        if not self.centers[0] <= nano_us <= self.centers[-1]:
            raise ValueError("local clock does not extrapolate outside Q/T anchors")
        right = bisect.bisect_right(self.centers, nano_us)
        if right == len(self.centers):
            return self.models[-1].map(nano_us)
        left = max(0, right - 1)
        weight = (nano_us - self.centers[left]) / (self.centers[right] - self.centers[left])
        a, b = self.models[left].map(nano_us), self.models[right].map(nano_us)
        # Blend overlapping fits, not independent hard-switching segments.
        return (
            a[0] + round(weight * (b[0] - a[0])),
            a[1] + math.floor(weight * (b[1] - a[1])),
            a[2] + math.ceil(weight * (b[2] - a[2])),
        )


@dataclass(frozen=True)
class PiecewiseMidpointClock:
    """Conservative fallback for operator-approved non-affine Nano clocks.

    Each Q/T exchange brackets the Nano timestamp between request send and
    reply receive.  The estimate interpolates exchange midpoints and the
    bounds interpolate their causal intervals.  It never extrapolates.
    """

    stamps: tuple[int, ...]
    estimates: tuple[int, ...]
    lowers: tuple[int, ...]
    uppers: tuple[int, ...]

    def map(self, nano_us: int) -> tuple[int, int, int]:
        if not self.stamps[0] <= nano_us <= self.stamps[-1]:
            raise ValueError("piecewise clock does not extrapolate outside Q/T anchors")
        right = bisect.bisect_left(self.stamps, nano_us)
        if right == 0:
            return self.estimates[0], self.lowers[0], self.uppers[0]
        if right == len(self.stamps):
            return self.estimates[-1], self.lowers[-1], self.uppers[-1]
        left = right - 1
        span = self.stamps[right] - self.stamps[left]
        weight = (nano_us - self.stamps[left]) / span

        def blend(values: tuple[int, ...]) -> int:
            return values[left] + round(weight * (values[right] - values[left]))

        estimate, lower, upper = (blend(self.estimates), blend(self.lowers), blend(self.uppers))
        return estimate, min(lower, estimate), max(upper, estimate)


def fit_piecewise_midpoint(
    exchanges: list[Exchange], *, slack_ns: int = 10_000
) -> PiecewiseMidpointClock:
    """Fit a bracket-preserving piecewise clock for explicitly reviewed runs."""
    if slack_ns < 0 or len(exchanges) < 2:
        raise ValueError("invalid slack or insufficient exchanges")
    stamps = tuple(exchange.nano_us for exchange in exchanges)
    if any(b <= a for a, b in zip(stamps, stamps[1:], strict=False)):
        raise ValueError("anchors must strictly increase")
    lowers = tuple(exchange.sent_ns - slack_ns for exchange in exchanges)
    uppers = tuple(exchange.received_ns + slack_ns for exchange in exchanges)
    estimates = tuple((low + high) // 2 for low, high in zip(lowers, uppers, strict=True))
    if any(b <= a for a, b in zip(estimates, estimates[1:], strict=False)):
        raise ValueError("piecewise midpoint clock would reverse time")
    return PiecewiseMidpointClock(stamps, estimates, lowers, uppers)


def fit_smooth_interval(
    exchanges: list[Exchange], *, slack_ns: int = 10_000
) -> SmoothIntervalClock:
    """Fit the preferred offline fallback without following midpoint jitter."""
    if slack_ns < 0:
        raise ValueError("invalid slack")
    return fit_smooth_interval_clock(
        [exchange.nano_us for exchange in exchanges],
        [exchange.sent_ns - slack_ns for exchange in exchanges],
        [exchange.received_ns + slack_ns for exchange in exchanges],
    )


def fit_local(
    exchanges: list[Exchange], *, window_s: float = 6, step_s: float = 2, slack_ns: int = 10_000
) -> LocalClock:
    if not (
        math.isfinite(window_s)
        and math.isfinite(step_s)
        and window_s > 0
        and 0 < step_s <= window_s / 2
    ):
        raise ValueError("require 0 < step <= window/2")
    if len(exchanges) < 4:
        raise ValueError("at least four Q/T pairs required for local estimation")
    stamps = [e.nano_us for e in exchanges]
    if any(b <= a for a, b in zip(stamps, stamps[1:], strict=False)):
        raise ValueError("anchors must strictly increase")
    centers = [stamps[0]]
    while centers[-1] + step_s * 1e6 < stamps[-1]:
        centers.append(round(centers[-1] + step_s * 1e6))
    centers.append(stamps[-1])
    models = []
    for center in centers:
        # Both neighboring local models cover every point in their blend.
        selected = [e for e in exchanges if abs(e.nano_us - center) <= window_s * 5e5]
        if len(selected) < 4:
            selected = sorted(
                sorted(exchanges, key=lambda e: abs(e.nano_us - center))[:4],
                key=lambda e: e.nano_us,
            )
        models.append(fit_clock(selected, slack_ns, weight_by_rtt=True))
    clock = LocalClock(tuple(centers), tuple(models))
    # Blending two affine functions gives a quadratic. Its derivative is
    # affine, so endpoint checks establish monotonicity throughout each span.
    for i in range(len(centers) - 1):
        span_s = (centers[i + 1] - centers[i]) * 1e-6
        for stamp, slope in ((centers[i], models[i].slope), (centers[i + 1], models[i + 1].slope)):
            delta_s = (models[i + 1].map(stamp)[0] - models[i].map(stamp)[0]) * 1e-9
            if slope + delta_s / span_s <= 0:
                raise ValueError("local blend would reverse time")
    return clock


def compare(
    exchanges: list[Exchange],
    stamps: list[int],
    *,
    window_s: float = 6,
    step_s: float = 2,
    slack_ns: int = 10_000,
) -> tuple[dict, list[tuple]]:
    if not stamps:
        raise ValueError("empty comparison timeline")
    # Withhold whole one-second host-time blocks, not neighboring burst samples.
    origin = exchanges[0].sent_ns
    held_out = [e for e in exchanges if ((e.sent_ns - origin) // 10**9) % 5 == 2]
    training = [e for e in exchanges if ((e.sent_ns - origin) // 10**9) % 5 != 2]
    summary = {
        "window_s": window_s,
        "step_s": step_s,
        "timestamp_slack_ns": slack_ns,
        "exchange_count": len(exchanges),
        "held_out_count": len(held_out),
        "status": "COMPARISON_ONLY_NOT_QUALIFIED",
        "models": {},
        "limitations": [
            "Held-out interval consistency is not ground-truth accuracy.",
            "Local bounds assume each overlapping local affine model is valid.",
            "Extra exchanges cannot identify fixed USB path asymmetry.",
            "No RPM/control waveform correlation or latency removal.",
        ],
    }
    predictions = {}
    for name in ("global", "local"):

        def fit_selected(pairs, model_name=name):
            return (
                fit_clock(pairs, slack_ns)
                if model_name == "global"
                else fit_local(pairs, window_s=window_s, step_s=step_s, slack_ns=slack_ns)
            )

        try:
            model = fit_selected(exchanges)
            mapped = [model.map(stamp) for stamp in stamps]
            if any(b[0] <= a[0] for a, b in zip(mapped, mapped[1:], strict=False)):
                raise ValueError("nonmonotonic mapped timeline")
            predictions[name] = mapped
            item = {
                "fit": "OK",
                "max_interval_width_ms": max((hi - lo) / 1e6 for _, lo, hi in mapped),
            }
        except ValueError as error:
            item = {"fit": "REJECTED", "reason": str(error)}
        try:
            if not held_out:
                raise ValueError("no held-out blocks")
            validation_model = fit_selected(training)
            excesses = []
            for e in held_out:
                estimate, _, _ = validation_model.map(e.nano_us)
                excesses.append(
                    max(e.sent_ns - slack_ns - estimate, estimate - e.received_ns - slack_ns, 0)
                    / 1e6
                )
            item["held_out"] = {
                "count": len(excesses),
                "violations": sum(e > 0 for e in excesses),
                "max_excess_ms": max(excesses),
            }
        except ValueError as error:
            item["held_out"] = {"unavailable": str(error)}
        summary["models"][name] = item
    rows = []
    for i, stamp in enumerate(stamps):
        row = [stamp]
        for name in ("global", "local"):
            row.extend(predictions[name][i] if name in predictions else ("", "", ""))
        rows.append(tuple(row))
    if "global" in predictions and "local" in predictions:
        shifts = [
            abs(a[0] - b[0]) / 1e6
            for a, b in zip(predictions["global"], predictions["local"], strict=True)
        ]
        summary["absolute_model_difference_ms"] = {
            "median": statistics.median(shifts),
            "maximum": max(shifts),
        }
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rpm_csv", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--window-s", type=float, default=6)
    parser.add_argument("--step-s", type=float, default=2)
    args = parser.parse_args()
    events, exchanges = read_capture(args.rpm_csv)
    # Compare only inside anchors; no speculative local extrapolation.
    stamps = sorted(
        {
            int(e.unwrapped_us)
            for e in events
            if e.kind == "E"
            and exchanges[0].nano_us <= int(e.unwrapped_us) <= exchanges[-1].nano_us
        }
    )
    if not stamps:
        stamps = [e.nano_us for e in exchanges]  # Stationary capture still tests clocks.
    summary, rows = compare(exchanges, stamps, window_s=args.window_s, step_s=args.step_s)
    summary["source"] = str(args.rpm_csv.resolve())
    args.output.mkdir(exist_ok=False)
    with (args.output / "comparison.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            (
                "nano_unwrapped_us",
                "global_ns",
                "global_lower_ns",
                "global_upper_ns",
                "local_ns",
                "local_lower_ns",
                "local_upper_ns",
            )
        )
        writer.writerows(rows)
    (args.output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
