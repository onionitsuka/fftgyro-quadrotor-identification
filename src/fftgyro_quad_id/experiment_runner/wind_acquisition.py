"""Separate source-plan wind, operator-declared wind and acquisition progress.

No measured wind is inferred from a setpoint or from a historical run number.
Old nominal-5/10 receipts remain untouched and are only labelled operator-reported
5.6/11.5, with their individual time intervals still awaiting verification.
"""
from __future__ import annotations

import json
from pathlib import Path

SCHEMA = "fftgyro-wind-acquisition-v1"


def policy(settings: dict) -> dict | None:
    value = settings.get("wind_acquisition")
    if value is None:
        return None
    if (value.get("schema") != SCHEMA or value.get("source_wind_mps") != 10
            or value.get("legacy_reported_wind_mps") != 11.5
            or value.get("additional_wind_mps") != 10
            or value.get("fresh5_wind_mps") != 5
            or value.get("legacy_reported_five_mps") != 5.6
            or value.get("plan_sha256") != settings["plan_sha256"]):
        raise ValueError("風速区分設定が元計画と一致しません")
    numbers = value.get("additional_numbers", [])
    if (not numbers or any(type(n) is not int or n <= 0 for n in numbers)
            or len(set(numbers)) != len(numbers)):
        raise ValueError("追加10 m/sの番号が不正です")
    fresh5 = value.get("fresh5_numbers", [])
    if (not fresh5 or any(type(n) is not int or n <= 0 for n in fresh5)
            or len(set(fresh5)) != len(fresh5) or set(fresh5) & set(numbers)):
        raise ValueError("再測定5 m/sの番号が不正です")
    return value


def condition(row: dict, settings: dict) -> dict | None:
    """Select a wind variant without altering the hash-pinned source session."""
    selected = policy(settings)
    if selected is None:
        return None
    nominal = row["wind_mps"]
    requested_scope = settings.get("_wind_scope_override")
    if nominal == selected["fresh5_wind_mps"] and row["number"] in selected["fresh5_numbers"]:
        requested_wind = settings.get("_wind_mps_override", nominal)
        if requested_wind != nominal:
            raise ValueError("再測定5.0 m/s番号の取得風速は5.0 m/sです")
        scope = requested_scope or "additional_fresh5"
        if scope not in ("additional_fresh5", "main"):
            raise ValueError("この5 m/s番号で指定できる取得枠はadditional_fresh5またはmainです")
        target = requested_wind
    elif nominal == selected["source_wind_mps"]:
        default = (selected["additional_wind_mps"]
                   if row["number"] in selected["additional_numbers"]
                   else selected["legacy_reported_wind_mps"])
        target = settings.get("_wind_mps_override", default)
        if target not in (10, 11.5):
            raise ValueError("旧10 m/s番号の取得風速は10または11.5を指定してください")
        default_scope = "additional_true10" if target == 10 else "main"
        scope = requested_scope or default_scope
        if scope not in ("additional_true10", "main"):
            raise ValueError("この旧10 m/s番号で指定できる取得枠はadditional_true10またはmainです")
        if (scope == "additional_true10") != (target == 10):
            raise ValueError("追加10 m/s枠は10 m/s、既存計画枠は11.5 m/sを指定してください")
    else:
        target = settings.get("_wind_mps_override", nominal)
        if target != nominal:
            raise ValueError("この番号は10/11.5 m/s区分の対象ではありません")
        scope = requested_scope or "main"
        if scope != "main":
            raise ValueError("この番号には追加取得枠がありません")
    return dict(schema=SCHEMA, policy_id=selected["id"], progress_scope=scope,
                source_plan_wind_mps=nominal, acquisition_wind_mps=target,
                measured_wind_mps=None, instrument_record_verified=False,
                confirmation_basis="operator_setpoint_confirmation",
                wind_log_reference=settings.get("_wind_log_reference"))


def label(record: dict) -> str:
    labels = {"additional_true10": "追加10 m/s枠",
              "additional_fresh5": "再測定5 m/s枠", "main": "既存計画枠"}
    scope = labels[record["progress_scope"]]
    return (f"今回の設定風速 {record['acquisition_wind_mps']:g} m/s ／ {scope}"
            f"（元計画 {record['source_plan_wind_mps']:g} m/s・風速の自動実測なし）")


def is_main_receipt(receipt: dict) -> bool:
    record = receipt.get("wind_condition")
    # Untagged historical receipts retain acquisition status, not verified wind.
    return record is None or (isinstance(record, dict) and record.get("schema") == SCHEMA
                              and record.get("progress_scope") in ("main", "additional_fresh5"))


def additional_completed(root: Path, plan_hash: str, selected: dict | None,
                         scope: str = "additional_true10") -> set[int]:
    """An old receipt or number-only adjudication cannot complete true-10 runs."""
    done: set[int] = set()
    if selected is None:
        return done
    if scope == "additional_true10":
        numbers, target = selected["additional_numbers"], selected["additional_wind_mps"]
    elif scope == "additional_fresh5":
        numbers, target = selected["fresh5_numbers"], selected["fresh5_wind_mps"]
    else:
        raise ValueError("未対応の追加取得枠です")
    for path in root.glob("*/operator-result.json"):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8-sig"))
            wind = receipt.get("wind_condition") or {}
            receipt_scope = wind.get("progress_scope")
            scope_matches = (receipt_scope == scope or
                             (scope == "additional_fresh5" and receipt_scope == "main"))
            if (receipt.get("plan_sha256") == plan_hash
                    and receipt.get("capture_complete") is True
                    and receipt.get("number") in numbers
                    and wind.get("schema") == SCHEMA
                    and wind.get("policy_id") == selected["id"]
                    and scope_matches
                    and wind.get("source_plan_wind_mps") == target
                    and wind.get("acquisition_wind_mps") == target
                    and wind.get("operator_confirmed_at")):
                done.add(receipt["number"])
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    if scope == "additional_true10":
        done.update(_retrospective_assignments(root, plan_hash, selected))
    return done


def _retrospective_assignments(root: Path, plan_hash: str, selected: dict) -> set[int]:
    """Validate explicit run-level assignments without rewriting capture records."""
    return set(_validated_retrospective_runs(root, plan_hash, selected).values())


def _validated_retrospective_runs(root: Path, plan_hash: str,
                                  selected: dict) -> dict[str, int]:
    assigned: dict[str, int] = {}
    for path in (root / "adjudications").glob("wind-acquisition-*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8-sig"))
            if (record.get("schema") != "fftgyro-wind-acquisition-adjudication-v1"
                    or record.get("plan_sha256") != plan_hash
                    or record.get("policy_id") != selected["id"]
                    or record.get("decision") != "ASSIGN_TO_ADDITIONAL_TRUE10"
                    or record.get("acquisition_wind_mps") != 10
                    or record.get("measurement_verified") is not False
                    or record.get("evidence_basis") != "operator_retrospective_confirmation"):
                continue
            for entry in record.get("runs", []):
                folder = root / entry["run"]
                receipt = json.loads((folder / "operator-result.json").read_text(encoding="utf-8-sig"))
                session = json.loads((folder / "operator-session.json").read_text(encoding="utf-8-sig"))
                manifest = json.loads((folder / "run.json").read_text(encoding="utf-8-sig"))
                if (receipt.get("plan_sha256") == plan_hash
                        and receipt.get("number") == entry.get("number")
                        and receipt.get("condition_id") == entry.get("condition_id")
                        and receipt.get("session_sha256") == entry.get("session_sha256")
                        and receipt.get("operator_confirmed_at") == entry.get("operator_confirmed_at")
                        and receipt.get("number") in selected["additional_numbers"]
                        and receipt.get("capture_complete") is True
                        and receipt.get("wind_condition") is None
                        and session.get("wind_m_s") == selected["source_wind_mps"]
                        and manifest.get("state") == "CAPTURED"
                        and manifest.get("remote_artifacts_verified") is True):
                    assigned[entry["run"]] = receipt["number"]
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return assigned


def inventory(root: Path, plan_hash: str, rows: list[dict], selected: dict) -> list[dict]:
    """Export wind provenance by attempt; not a scientific acceptance decision."""
    by_number = {row["number"]: row for row in rows}
    entries = []
    retrospective = _validated_retrospective_runs(root, plan_hash, selected)
    for path in sorted(root.glob("*/operator-result.json")):
        try:
            receipt = json.loads(path.read_text(encoding="utf-8-sig"))
            row = by_number.get(receipt.get("number"))
            if receipt.get("plan_sha256") != plan_hash or row is None:
                continue
            wind = receipt.get("wind_condition")
            if wind is None:
                nominal = row["wind_mps"]
                wind = dict(source_plan_wind_mps=nominal, acquisition_wind_mps=None,
                            measured_wind_mps=None, instrument_record_verified=False,
                            progress_scope="main", confirmation_basis="legacy_no_per_run_wind_record")
                if nominal == selected["source_wind_mps"]:
                    wind.update(reported_wind_mps=selected["legacy_reported_wind_mps"],
                                confirmation_basis="operator_report_scope_unverified")
                    if path.parent.name in retrospective:
                        wind.update(acquisition_wind_mps=selected["additional_wind_mps"],
                                    progress_scope="additional_true10",
                                    confirmation_basis="operator_retrospective_confirmation")
                elif nominal == selected["fresh5_wind_mps"]:
                    wind.update(reported_wind_mps=selected["legacy_reported_five_mps"],
                                confirmation_basis="operator_report_scope_unverified")
            entries.append(dict(run=path.parent.name, number=receipt["number"],
                                capture_complete=receipt.get("capture_complete", False),
                                wind_condition=wind))
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return entries
