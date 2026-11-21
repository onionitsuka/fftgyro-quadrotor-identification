"""Versioned acquisition selection without renumbering or deleting old trials."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .timing_policy import apply_timing_policy


LEGACY_POLICY = "auxiliary-fixed-repeat2-v1"
T_SINGLE_POLICY = "all-t-single-balanced-v1"
RECOMMENDED_319_POLICY = "recommended-windy-plus-calm-t1-d2-h2-v1"
RECOMMENDED_WINDY_ONLY_POLICY = "recommended-windy-only-t1-d2-h2-v1"
WINDY_ONE_RUN_H30_POLICY = "windy-t1-d1-h1-with-main-h30-v1"
CALM_WINDY_ONE_RUN_H30_POLICY = "calm-windy-t1-d1-h1-with-main-h30-v1"

MAIN_H30_IDS = {
    "H-N-005", "H-N-009", "H-N-014", "H-N-018",
    "H-F-003", "H-F-007", "H-F-010", "H-F-014",
}


RECOMMENDED_D_IDS = {
    *(f"D-FIX-{number:03d}" for number in (*range(1, 5), *range(9, 13), 17, 18)),
    "D-N-002", "D-N-004", "D-N-007", "D-N-009",
    "D-F-002", "D-F-004", "D-F-007", "D-F-009",
    "D-S-001", "D-S-003", "D-W-001", "D-W-002",
}
RECOMMENDED_H_IDS = {
    *(f"H-FIX-{number:03d}" for number in range(11, 25)),
    "H-N-002", "H-N-004", "H-N-007", "H-N-008",
    "H-N-011", "H-N-013", "H-N-016", "H-N-017",
    "H-F-002", "H-F-006", "H-F-009", "H-F-013",
    "H-S-001", "H-S-003", "H-W-002",
}


def _recommended_t_condition(row: dict) -> bool:
    """The documented 160-condition windy T subset from 2026-09-06."""
    group = row["group"]
    wind = row["wind_mps"]
    pitch = row["pitch_deg"]
    heading = row["heading_deg"]
    axis = row["axis"]
    if group == "B":
        return (
            wind in (5, 10) and pitch in (0, -7.5, -15, -30, -45)
            and heading in (0, -30, 30)
        ) or (
            wind in (2.5, 7.5) and pitch in (0, -15, -30, -45) and heading == 0
        ) or (
            wind in (2.5, 5) and pitch in (7.5, 15) and heading == 0
        )
    if group == "G":
        return wind in (5, 10) and (
            heading == 0 and pitch in (0, -15, -30, -45)
            or pitch == -15 and heading in (-30, 30)
        )
    if group in {"A", "M"}:
        return wind in (5, 10) and pitch in (-15, -30) and heading == 0
    if group in {"X", "FX"}:
        return wind in (5, 10) and heading == 0 and (
            row["mode"] in ("C", "P") and pitch == -15
            or row["mode"] in ("R", "Y") and pitch == 0
        )
    if group == "N":
        environment = wind in (5, 10) and heading == 0 and (
            axis == "pitch" and pitch in (-7.5, -15, -30)
            or axis == "yaw" and pitch in (0, -15, -30)
        )
        waveform = row["amplitude_deg"] == 2 or (
            row["amplitude_deg"] == 5 and row["frequency_hz"] == 0.5
        )
        return environment and waveform
    if group == "SIDE":
        return True
    if group == "RPM":
        return wind in (5, 10) and pitch == -15 and row["rpm_ratio"] in (0.8, 1.2)
    if group in {"F1", "F2", "F3"}:
        return wind in (5, 10) and pitch == -15
    if group == "S":
        return (wind, pitch) in {(5, 0), (5, -15), (10, -15), (10, -30)}
    if group == "W":
        return wind in (5, 10) and pitch == -15
    return False


def recommended_319_condition(row: dict) -> bool:
    """Keep all calm conditions plus the reviewed 211-condition windy subset."""
    if row["wind_mps"] == 0:
        return True
    if row["role"] == "T":
        return _recommended_t_condition(row)
    if row["role"] == "D":
        return row["id"] in RECOMMENDED_D_IDS
    if row["role"] == "H":
        return row["id"] in RECOMMENDED_H_IDS
    return False


def legacy_omission_reason(row: dict) -> str | None:
    """The 2026-09-14 auxiliary fixed-condition reduction."""
    if row["role"] != "T" or row["axis"] != "fixed" or row["repetition"] != 2:
        return None
    if row["wind_mps"] in (2.5, 7.5) and row["group"] == "B":
        return "中間風速の固定B：第2反復を省略（第1反復は実施）"
    if row["wind_mps"] in (5, 10) and row["group"] in ("G", "A", "M"):
        return "補助固定G/A/M：第2反復を省略（第1反復は実施）"
    return None


def expected_omissions(rows: list[dict], policy: str) -> dict[int, str]:
    """Return the complete outcome-independent omission set for a known policy."""
    if policy == LEGACY_POLICY:
        return {
            row["number"]: reason
            for row in rows
            if (reason := legacy_omission_reason(row))
        }
    if policy == RECOMMENDED_WINDY_ONLY_POLICY:
        omitted = expected_omissions(rows, RECOMMENDED_319_POLICY)
        omitted.update({
            row["number"]: "0 m/s main trial omitted from future acquisition"
            for row in rows if row["wind_mps"] == 0
        })
        return omitted
    if policy == WINDY_ONE_RUN_H30_POLICY:
        omitted = expected_omissions(rows, RECOMMENDED_WINDY_ONLY_POLICY)
        for row in rows:
            if row["id"] in MAIN_H30_IDS and row["repetition"] == 1:
                omitted.pop(row["number"], None)
            elif row["role"] in {"D", "H"} and row["repetition"] >= 2:
                omitted[row["number"]] = "D/H consecutive repetition omitted; retain source repetition 1"
        return omitted
    if policy == CALM_WINDY_ONE_RUN_H30_POLICY:
        omitted = expected_omissions(rows, RECOMMENDED_319_POLICY)
        for row in rows:
            if row["id"] in MAIN_H30_IDS and row["repetition"] == 1:
                omitted.pop(row["number"], None)
            elif row["role"] in {"D", "H"} and row["repetition"] >= 2:
                omitted[row["number"]] = "D/H consecutive repetition omitted; retain source repetition 1"
        return omitted
    if policy == RECOMMENDED_319_POLICY:
        candidate_rows = [row for row in rows if recommended_319_condition(row)]
        t_omissions = expected_omissions(candidate_rows, T_SINGLE_POLICY)
        omitted = {}
        for row in rows:
            if not recommended_319_condition(row):
                omitted[row["number"]] = "Condition outside calm + documented recommended windy subset"
            elif row["role"] == "H" and row["repetition"] == 3:
                omitted[row["number"]] = "H condition reduced from three to two independent repetitions"
            elif row["number"] in t_omissions:
                omitted[row["number"]] = t_omissions[row["number"]]
        return omitted
    if policy != T_SINGLE_POLICY:
        raise ValueError("Unsupported run-selection policy")

    # A fixed-input run already contains both command signs. Keep repetition 2
    # so the 0 m/s fixed runs acquired before this reduction remain in-plan.
    # Dynamic repetition reverses motion direction, so alternate the retained
    # repetition independently for Pitch and Yaw in immutable source order.
    dynamic_index = Counter()
    retained_by_condition: dict[str, int] = {}
    omitted: dict[int, str] = {}
    for row in rows:
        if row["role"] != "T":
            continue
        if row["axis"] == "fixed":
            retained = 2
            reason = "T fixed-input condition reduced to repetition 2"
        else:
            if row["id"] not in retained_by_condition:
                dynamic_index[row["axis"]] += 1
                retained_by_condition[row["id"]] = (
                    1 if dynamic_index[row["axis"]] % 2 else 2
                )
            retained = retained_by_condition[row["id"]]
            reason = "T dynamic condition reduced to one axis-balanced direction"
        if row["repetition"] != retained:
            omitted[row["number"]] = reason
    return omitted


@dataclass(frozen=True)
class RunSelection:
    rows: list[dict]
    omitted: dict[int, str]
    source_count: int
    document_bytes: bytes | None = None
    sha256: str | None = None
    selection_id: str = "full-original-plan"
    timing_policy: str | None = None


def select_runs(rows: list[dict], binding: dict | None, plan_hash: str, repo: Path) -> RunSelection:
    """Fail closed on stale selection/identity/counts; retained rows keep source fields."""
    if binding is None:
        return RunSelection(list(rows), {}, len(rows))
    path = Path(binding["path"])
    if not path.is_absolute():
        path = repo / path
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != binding["sha256"]:
        raise ValueError("実施選択表のhashが変更されています。削減対象を再確認してください。")
    document = json.loads(raw.decode("utf-8-sig"))
    if document.get("schema") != "fftgyro-run-selection-v1":
        raise ValueError("未対応の実施選択表です。")
    if document.get("source_plan_sha256") != plan_hash:
        raise ValueError("実施選択表と元条件表のhashが一致しません。")
    numbers = document.get("omitted_numbers")
    if (not isinstance(numbers, list) or any(type(n) is not int for n in numbers)
            or len(set(numbers)) != len(numbers)):
        raise ValueError("除外番号に重複または不正な値があります。")
    expected = expected_omissions(rows, document.get("policy"))
    if set(numbers) != set(expected):
        raise ValueError("除外番号が採用済みの補助固定反復削減規則と一致しません。")
    retained = [r for r in rows if r["number"] not in expected]
    if (len(rows) != document.get("expected_original_runs")
            or len(retained) != document.get("expected_active_runs")
            or len({r["id"] for r in retained}) != document.get("expected_conditions")):
        raise ValueError("実施選択表の本数・条件数が一致しません。")
    counts = Counter(r["id"] for r in retained)
    # UI-only metadata: never rewrite the original source's repeats or waveform.
    timing_policy = document.get("timing_policy")
    retained = [
        dict(apply_timing_policy(r, timing_policy), planned_repeats=counts[r["id"]])
        for r in retained
    ]
    return RunSelection(
        retained, expected, len(rows), raw, digest, document["selection_id"], timing_policy
    )
