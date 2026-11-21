"""速度ノードごとの荷重を線形補間する試作LPVモデル。

L_k = physical + B_k + [C_k(z)-C_k(reference)]
B_kは有効な固定試験T窓だけでfit。C_kは保存済み動的補正式を速度kへ局所化。
RPM依存は既存のphysicalと局所Cで扱い、基準Bに追加RPM傾きを推定しない。
独立な速度別動的再同定ではない。速度の一定成分は差分Cから除きBで扱う。
固定窓は動いている可能性があり、独立な絶対校正/自由飛行検証ではない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from simulate_graybox_forward_flight import OUTPUTS, Aircraft, rotation


def interpolation_weights(speed: float, nodes: np.ndarray) -> np.ndarray:
    """連続・非負・和1。範囲外を端値で黙ってclipしない。"""
    if not np.isfinite(speed) or speed < nodes[0] - 1e-10 or speed > nodes[-1] + 1e-10:
        raise ValueError("Speed outside scheduled model support; extrapolation disabled")
    weights = np.zeros(len(nodes))
    if speed >= nodes[-1]:
        weights[-1] = 1
        return weights
    hi = int(np.searchsorted(nodes, speed, side="right"))
    hi = max(1, hi)
    lo = hi - 1
    weight = (speed - nodes[lo]) / (nodes[hi] - nodes[lo])
    weights[lo], weights[hi] = 1 - weight, weight
    return weights


def fixed_air(row: dict) -> np.ndarray:
    """Same nominal Rz(heading) Ry(pitch) Rz(yaw) as the current reconstruction."""
    theta, psi, heading = np.deg2rad([row["pitch_deg"], row["yaw_deg"], row["heading_deg"]])
    ry = rotation(np.array([np.cos(theta / 2), 0, np.sin(theta / 2), 0]))
    rz = rotation(np.array([np.cos(psi / 2), 0, 0, np.sin(psi / 2)]))
    flow = np.array([np.cos(heading), -np.sin(heading), 0])
    return (ry @ rz).T @ flow * row["wind_m_s"]


class ScheduledAircraft(Aircraft):
    def __init__(self, artifact: dict, route: str, schedule: dict, *, anchored: bool = True):
        self.local = schedule["routes"][route]
        self.nodes = np.array(schedule["speeds_m_s"], float)
        if len(self.nodes) < 2 or np.any(np.diff(self.nodes) <= 0):
            raise ValueError("Need strictly increasing scheduled speeds")
        self.raw_model = Aircraft(artifact, route, anchored=False)
        self.physics = Aircraft(artifact, "physical", anchored=False)
        super().__init__(artifact, route, anchored=anchored)

    def correction(self, air: np.ndarray, omega: np.ndarray, rotor: np.ndarray) -> np.ndarray:
        zero = np.zeros(4)
        return self.raw_model.loads(air, omega, rotor, zero) - self.physics.loads(
            air, omega, rotor, zero
        )

    def node_load(
        self, index: int, direction: np.ndarray, omega: np.ndarray, rotor: np.ndarray
    ) -> np.ndarray:
        node = self.local[index]
        speed = self.nodes[index]
        bar2 = float(np.mean(rotor**2))
        balanced = np.full(4, np.sqrt(bar2))
        reference_air = np.array(node["reference_air_m_s"])
        delta = self.correction(speed * direction, omega, rotor) - self.correction(
            reference_air, np.zeros(3), balanced
        )
        basis = np.array([1, bar2 / node["reference_rotor_squared_rad_s2"] - 1])
        return basis @ np.array(node["baseline_coefficients"]) + delta

    def loads(
        self, air: np.ndarray, omega: np.ndarray, rotor: np.ndarray, rotor_dot: np.ndarray
    ) -> np.ndarray:
        speed = float(np.linalg.norm(air))
        direction = air / speed if speed > 1e-12 else np.array([1.0, 0, 0])
        weights = interpolation_weights(speed, self.nodes)
        residual = sum(
            weights[k] * self.node_load(k, direction, omega, rotor) for k in np.flatnonzero(weights)
        )
        return self.physics.loads(air, omega, rotor, rotor_dot) + residual + self.anchor


def fit_schedule(artifact: dict, rows: list[dict]) -> dict:
    # Extraction records the status policy. 2026-10-05 includes 0x44 per user instruction.
    eligible = [
        r
        for r in rows
        if r["role"] == "T"
        and abs(r["heading_deg"]) <= 1
        and abs(r["yaw_deg"]) <= 2
        and r["rate_max_deg_s"] <= 5
    ]
    speeds = sorted({r["wind_m_s"] for r in eligible})
    speeds = [v for v in speeds if len({r["run"] for r in eligible if r["wind_m_s"] == v}) >= 2]
    # Enumerate every observed speed; never silently use a hand-picked node list.
    coverage = []
    for speed in sorted({r["wind_m_s"] for r in rows}):
        observed = [r for r in rows if r["wind_m_s"] == speed]
        accepted = [r for r in eligible if r["wind_m_s"] == speed]
        used = speed in speeds
        coverage.append(dict(
            speed_m_s=speed,
            observed_runs=sorted({r["run"] for r in observed}),
            observed_roles=sorted({r["role"] for r in observed}),
            eligible_T_runs=sorted({r["run"] for r in accepted}),
            eligible_T_windows=len(accepted),
            adopted=used,
            reason="ALL_ELIGIBLE_T_RUNS_USED" if used else
                   "NO_T_WINDOWS" if not any(r["role"] == "T" for r in observed) else
                   "FEWER_THAN_TWO_ELIGIBLE_T_RUNS",
        ))
    if not speeds or speeds[0] != 0 or speeds[-1] < 10:
        raise ValueError("Insufficient fixed T coverage for a 0-to-10 scheduled prototype")
    result = dict(
        speeds_m_s=speeds,
        speed_node_policy="All observed speeds with >=2 eligible fixed T runs; no H/D fitting",
        speed_node_coverage=coverage,
        routes={},
        status="EXPLORATORY_FIXED_BASELINE_LPV",
        baseline_training_role="T only; no per-run fitted offsets",
        leptrino_status_policies=sorted({r.get("leptrino_status_policy", "historical_strict_or_unspecified") for r in rows}),
        baseline_basis="Constant residual at each speed; extra RPM slope fixed to zero",
        local_dynamic_model="Saved common dynamic coefficients evaluated at each node speed",
        strict_static_gate="rate<=5 deg/s AND accel<=20 deg/s2",
        actual_baseline_gate="rate<=5 deg/s; corrected fixed-window means, NOT certified static",
        limitations=[
            "Absolute zero/configuration epochs across fixed runs are unverified",
            "High-speed fixed windows fail the strict acceleration gate",
            "Different speeds are not the same verified battery/mounting epoch",
            "RPM and pose extrapolation may remain even inside speed support",
            "No independent local dynamic identification; no free-flight validation",
            "Reported speed provenance retained; no planning-speed replacement",
            "Extra RPM baseline slope not identified: narrow zero-wind RPM span; frozen to zero",
            "Fixed baseline uses measured RPM for both coefficient routes; "
            "not command-only validation",
        ],
    )
    for route in ("measured", "command_route"):
        raw = Aircraft(artifact, route, anchored=False)
        physics = Aircraft(artifact, "physical", anchored=False)
        nodes = []
        for speed in speeds:
            part = [r for r in eligible if r["wind_m_s"] == speed]
            pose = dict(
                wind_m_s=speed,
                heading_deg=0,
                pitch_deg=float(np.median([r["pitch_deg"] for r in part])),
                yaw_deg=float(np.median([r["yaw_deg"] for r in part])),
            )
            reference_air = fixed_air(pose)
            rotors = [
                np.array([r[f"rpm_M{j}"] for j in range(1, 5)]) * 2 * np.pi / 60 for r in part
            ]
            bar2 = np.array([np.mean(w * w) for w in rotors])
            reference_bar2 = float(np.median(bar2))
            counts = {
                name: sum(r["run"] == name for r in part) for name in {r["run"] for r in part}
            }
            weight = np.array([1 / np.sqrt(counts[r["run"]]) for r in part])
            x = np.ones((len(part), 1))
            y = []
            for row, w in zip(part, rotors, strict=True):
                air = fixed_air(row)
                balanced = np.full(4, np.sqrt(np.mean(w * w)))
                actual_p = physics.loads(air, np.zeros(3), w, np.zeros(4))
                actual_c = raw.loads(air, np.zeros(3), w, np.zeros(4)) - actual_p
                reference_c = raw.loads(
                    reference_air, np.zeros(3), balanced, np.zeros(4)
                ) - physics.loads(reference_air, np.zeros(3), balanced, np.zeros(4))
                truth = np.array([row[f"target_{out}"] for out in OUTPUTS])
                y.append(truth - actual_p - actual_c + reference_c)
            coef, _, rank, singular = np.linalg.lstsq(
                x * weight[:, None], np.array(y) * weight[:, None], rcond=1e-10
            )
            if rank != 1:
                raise ValueError(f"Fixed baseline rank deficient at speed {speed}")
            residual = x @ coef - y
            node = dict(
                speed_m_s=speed,
                reference_air_m_s=reference_air.tolist(),
                reference_pose=pose,
                reference_rotor_squared_rad_s2=reference_bar2,
                baseline_coefficients=np.vstack([coef, np.zeros((1, 6))]).tolist(),
                T_runs=sorted(counts),
                windows=len(part),
                adopted_0x44_samples=sum(r.get("status_0x44_samples", 0) for r in part),
                rank=int(rank),
                singular_values=singular.tolist(),
                pose_pitch_range_deg=[
                    min(r["pitch_deg"] for r in part),
                    max(r["pitch_deg"] for r in part),
                ],
                rotor_rms_range_RPM=(np.sqrt([bar2.min(), bar2.max()]) * 60 / (2 * np.pi)).tolist(),
                strict_static_windows=sum(r["accel_max_deg_s2"] <= 20 for r in part),
                wind_basis=sorted({r["wind_basis"] for r in part}),
                training_rmse=np.sqrt(np.mean(residual**2, axis=0)).tolist(),
            )
            nodes.append(node)
        result["routes"][route] = nodes
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--fixed", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--extraction-audit", type=Path,
                        help="Preserve rejected/zero-window acquisition speeds and reasons")
    args = parser.parse_args()
    artifact, rows = json.loads(args.models.read_text()), json.loads(args.fixed.read_text())
    fitted = fit_schedule(artifact, rows)
    if args.extraction_audit:
        audit = json.loads(args.extraction_audit.read_text())
        fitted["extraction_audit_sha256"] = hashlib.sha256(args.extraction_audit.read_bytes()).hexdigest()
        fitted["unusable_fixed_acquisitions"] = [
            {key: row.get(key) for key in ("run", "wind_m_s", "wind_basis", "status",
                                          "windows", "reason", "strict_status_valid_rows", "status_histogram")}
            for row in audit if row.get("status") == "REJECTED" or row.get("windows", 0) == 0
        ]
    fitted["model_sha256"] = hashlib.sha256(args.models.read_bytes()).hexdigest()
    fitted["fixed_sha256"] = hashlib.sha256(args.fixed.read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(fitted, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                "speeds_m_s": fitted["speeds_m_s"],
                "T_windows": [n["windows"] for n in fitted["routes"]["measured"]],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
