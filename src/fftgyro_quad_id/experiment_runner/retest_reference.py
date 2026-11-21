"""Retest evidence links only: no hardware, tare, or automatic model acceptance."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .empty_sensor_decay import (
    POSES as EMPTY_POSES,
)
from .empty_sensor_decay import (
    STAGE as EMPTY_DECAY_STAGE,
)
from .empty_sensor_decay import (
    STAGES as EMPTY_STAGES,
)
from .empty_sensor_decay import (
    STATIC_STAGE as EMPTY_STATIC_STAGE,
)

SCHEMA = "fftgyro-retest-context-v1"
STAGES = ("Z-pre", "F-before", "D-slow", "D-fast", "F-after", "Z-post")
IDENTITY = ("configuration_id", "battery_id", "zero_epoch_id", "calibration_block_id", "split")
PAIR_IDENTITY = (*IDENTITY, "pair_id", "axis", "location_id", "wind_m_s")


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def validate_context(context: dict) -> None:
    if context.get("schema") != SCHEMA:
        raise ValueError("invalid retest context schema")
    for field in (*IDENTITY, "location_id"):
        if not isinstance(context.get(field), str) or not context[field].strip():
            raise ValueError(f"missing retest field: {field}")
    if context["split"] not in ("T", "D", "H"):
        raise ValueError("split must be T/D/H; do not split a battery block")
    if context.get("stage") not in (*STAGES, "Z", "S", "M", "HT", *EMPTY_STAGES):
        raise ValueError("invalid retest stage")
    wind = context.get("wind_m_s")
    if (
        isinstance(wind, bool)
        or not isinstance(wind, (int, float))
        or not math.isfinite(wind)
        or wind < 0
    ):
        raise ValueError("wind_m_s must be finite and nonnegative")
    if context["stage"] in ("F-before", "D-slow", "D-fast", "F-after"):
        if not isinstance(context.get("pair_id"), str) or not context["pair_id"].strip():
            raise ValueError("A requires pair_id")
        if context.get("axis") not in ("PITCH", "YAW"):
            raise ValueError("A requires PITCH/YAW")
    if context["stage"] in ("Z", "Z-pre", "Z-post"):
        if wind != 0 or context.get("outside_wind_tunnel") is not True:
            raise ValueError("Z requires wind-free measurement outside the wind tunnel")
        if context.get("motor_power_disconnected") is not True:
            raise ValueError("Z requires motor main power disconnected")
    if context["stage"] in EMPTY_STAGES:
        if wind != 0 or context.get("axis") not in ("PITCH", "YAW"):
            raise ValueError("empty-sensor decay requires no wind and PITCH/YAW")
        for key in (
            "outside_wind_tunnel",
            "motor_power_disconnected",
            "aircraft_removed",
            "sensor_measurement_side_empty",
            "sensor_support_side_fixed",
            "other_axes_locked",
        ):
            if context.get(key) is not True:
                raise ValueError(f"empty-sensor decay requires {key}")
        static = context["stage"] == EMPTY_STATIC_STAGE
        if context.get("selected_axis_free") is not (not static):
            raise ValueError("empty-sensor selected_axis_free mismatch")
        if context.get("all_axes_locked") is not static:
            raise ValueError("empty-sensor all_axes_locked mismatch")
        pose_ids = tuple(pose[0] for pose in EMPTY_POSES)
        allowed = ("pitch-free",) if not static and context["axis"] == "PITCH" else pose_ids
        if context.get("pose_id") not in allowed:
            raise ValueError("empty-sensor pose_id mismatch")


def read_run(folder: Path) -> tuple[dict, dict]:
    manifest = load_json(folder / "run.json")
    if manifest.get("state") != "CAPTURED":
        raise ValueError(f"reference capture not complete: {folder}")
    qc = load_json(folder / "immediate-qc.json")
    if qc.get("decision") != "PASS":
        raise ValueError(f"reference QC not PASS: {folder}")
    context = manifest.get("retest_context", {})
    validate_context(context)
    return manifest, context


def match(left: dict, right: dict, fields: tuple[str, ...]) -> None:
    for field in fields:
        if left.get(field) != right.get(field):
            raise ValueError(f"retest reference mismatch: {field}")


def check_zero(folder: Path, current: dict, stage: str) -> dict:
    manifest, reference = read_run(folder)
    match(current, reference, IDENTITY)
    if reference["stage"] != stage or not manifest.get("idle"):
        raise ValueError(f"require idle {stage} reference")
    if not manifest.get("qc_stationary_required"):
        raise ValueError("zero reference must require stationary QC")
    if (
        reference["location_id"] != current["location_id"]
        and current.get("calibration_transfer_confirmed") is not True
    ):
        raise ValueError("calibration transfer requires explicit unchanged-mounting confirmation")
    return reference


def prepare_context(
    path: Path, *, idle: bool, fixed: bool, stationary: bool, px4_log: bool, session: dict | None
) -> dict:
    """Reject incomplete evidence before the coordinator makes any connections."""
    context = load_json(path)
    validate_context(context)
    stage = context["stage"]
    if stage in ("Z", "Z-pre", "Z-post"):
        if not (idle and fixed and stationary and px4_log):
            raise ValueError("Z requires --idle --fixed --idle-px4-log --qc-stationary-required")
    elif stage == "S":
        if not (idle and px4_log):
            raise ValueError("S requires idle and PX4 logging")
    elif stage == EMPTY_DECAY_STAGE:
        if not idle or fixed or stationary or session is not None:
            raise ValueError(
                "empty-sensor decay requires idle manual motion, not fixed/zero/powered capture"
            )
    elif stage == EMPTY_STATIC_STAGE:
        if not (idle and fixed and stationary) or session is not None:
            raise ValueError("empty-sensor static requires idle, fixed and stationary QC")
    else:
        if idle or session is None:
            raise ValueError("powered retest requires a separately qualified session")
        if not isinstance(context.get("pre_zero_run"), str) or not context["pre_zero_run"]:
            raise ValueError("powered retest requires pre_zero_run")

        def resolve(name: str) -> Path:
            return (path.parent / context[name]).resolve()

        check_zero(resolve("pre_zero_run"), context, "Z-pre")
        if stage in ("F-before", "F-after", "M", "HT") and not fixed:
            raise ValueError("fixed retest stage requires --fixed")
        if stage in ("D-slow", "D-fast"):
            if fixed or session.get("pid", {}).get("axis", "").upper() != context["axis"]:
                raise ValueError("dynamic retest axis/session mismatch")
        if stage in ("D-slow", "D-fast", "F-after"):
            if not isinstance(context.get("previous_run"), str) or not context["previous_run"]:
                raise ValueError("A sequence requires previous_run")
            _, previous = read_run(resolve("previous_run"))
            match(context, previous, PAIR_IDENTITY)
            if previous["stage"] != STAGES[STAGES.index(stage) - 1]:
                raise ValueError("A must follow F-before -> D-slow -> D-fast -> F-after")
        # Preserve resolved links and exact evidence hashes in run.json.
        for field in ("pre_zero_run", "previous_run"):
            if context.get(field):
                folder = resolve(field)
                context[field] = str(folder)
                context[field + "_sha256"] = hashlib.sha256(
                    (folder / "run.json").read_bytes()
                ).hexdigest()
    return context


def audit_block(folders: list[Path]) -> dict:
    """Cross-run metadata audit; never call complete metadata absolute validation."""
    issues: list[str] = []
    records: list[dict] = []
    manifests: list[dict] = []
    valid_folders: list[Path] = []
    for folder in folders:
        try:
            manifest, context = read_run(folder)
            if context["stage"] in ("Z", "Z-pre", "Z-post") and (
                not manifest.get("idle") or not manifest.get("qc_stationary_required")
            ):
                raise ValueError("Z must be idle with stationary QC")
            records.append(context)
            manifests.append(manifest)
            valid_folders.append(folder.resolve())
        except (OSError, ValueError) as error:
            issues.append(f"{folder.name}: {error}")
    if records:
        if len(set(valid_folders)) != len(valid_folders):
            issues.append("duplicate run folders are not independent records")
        for before, after in zip(manifests, manifests[1:], strict=False):
            if not (
                isinstance(before.get("capture_closed_utc_ns"), int)
                and isinstance(after.get("capture_setup_utc_ns"), int)
                and before["capture_closed_utc_ns"] <= after["capture_setup_utc_ns"]
            ):
                issues.append("missing or inconsistent acquisition chronology")
        for record in records[1:]:
            try:
                match(records[0], record, IDENTITY)
            except ValueError as error:
                issues.append(str(error))
        for stage in ("Z-pre", "Z-post"):
            if sum(r["stage"] == stage for r in records) != 1:
                issues.append(f"exactly one {stage} required")
        if records[0]["stage"] != "Z-pre" or records[-1]["stage"] != "Z-post":
            issues.append("block must be bracketed by Z-pre / Z-post")
        zero_folders = [
            folder
            for folder, record in zip(valid_folders, records, strict=True)
            if record["stage"] == "Z-pre"
        ]
        for folder, record in zip(valid_folders, records, strict=True):
            if record["stage"] in ("Z", "Z-pre", "Z-post", "S"):
                continue
            try:
                pre = Path(record.get("pre_zero_run", ""))
                if not pre.is_absolute() or pre.resolve() not in zero_folders:
                    raise ValueError("powered run not linked to this block's Z-pre")
                check_zero(pre, record, "Z-pre")
                for field in ("pre_zero_run", "previous_run"):
                    if record.get(field):
                        source = Path(record[field]) / "run.json"
                        if hashlib.sha256(source.read_bytes()).hexdigest() != record.get(
                            field + "_sha256"
                        ):
                            raise ValueError(f"changed evidence: {field}")
            except (OSError, ValueError) as error:
                issues.append(f"{folder.name}: {error}")
        for pair in sorted({r.get("pair_id") for r in records if r["stage"] in STAGES[1:5]}):
            rows = [r for r in records if r.get("pair_id") == pair and r["stage"] in STAGES[1:5]]
            if [r["stage"] for r in rows] != list(STAGES[1:5]):
                issues.append(f"{pair}: incomplete or unordered A sequence")
            for row in rows[1:]:
                try:
                    match(rows[0], row, PAIR_IDENTITY)
                except ValueError as error:
                    issues.append(f"{pair}: {error}")
    else:
        issues.append("no complete records")
    return dict(
        schema="fftgyro-retest-reference-audit-v1",
        issues=issues,
        decision="REVIEW" if issues else "METADATA_COMPLETE",
        absolute_values_validated=False,
        remaining=[
            "前後の生6軸・実姿勢の再現性",
            "零点/CGのrankと不確かさ",
            "実角度・4RPM・電圧の重なり",
            "同期/慣性/ローター項",
            "新規H評価",
        ],
    )
