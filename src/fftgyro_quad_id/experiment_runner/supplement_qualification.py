"""Evidence checks for promoting separately numbered dynamic supplements."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dynamic_evidence(path: Path, axis: str, tested_profile: Path) -> dict:
    evidence = _read(path)
    if (evidence.get("schema") != "fftgyro-dynamic-qualification-v1"
            or evidence.get("status") != "QUALIFIED"
            or evidence.get("axis") != axis.upper()):
        raise ValueError(f"{axis} dynamic qualification evidence is invalid")
    if evidence.get("profile_sha256") != _sha(tested_profile):
        raise ValueError(f"{axis} tested profile hash differs from qualification evidence")
    runs = evidence.get("runs", [])
    if not runs or any(run.get("state") != "CAPTURED"
                       or len(run.get("run_json_sha256", "")) != 64 for run in runs):
        raise ValueError(f"{axis} dynamic qualification run evidence is incomplete")
    return evidence


def measured_collective(path: Path) -> dict:
    evidence = _read(path)
    if (evidence.get("schema") != "fftgyro-fixed-input-qualification-v1"
            or evidence.get("status") != "QUALIFIED"):
        raise ValueError("measured collective qualification evidence is invalid")
    mapping = evidence.get("collective_by_rpm_ratio")
    required = {"0.8", "0.9", "0.95", "1", "1.05", "1.1", "1.2"}
    runs = evidence.get("runs", [])
    if (not isinstance(mapping, dict) or set(mapping) != required
            or any(not 0 < float(command) <= .8 for command in mapping.values())
            or len(runs) < 5
            or any(run.get("state") != "CAPTURED"
                   or len(run.get("run_json_sha256", "")) != 64 for run in runs)):
        raise ValueError("measured collective mapping for RPM ratio 1 is missing")
    return mapping
