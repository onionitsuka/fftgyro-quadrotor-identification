from __future__ import annotations

import csv
import json
from pathlib import Path

from .models import ExperimentSegment, PlanSummary


def export_plan(
    output_dir: Path,
    segments: list[ExperimentSegment],
    summary: PlanSummary,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.csv"
    summary_path = output_dir / "summary.json"

    rows = [segment.to_dict() for segment in segments]
    if not rows:
        raise ValueError("cannot export an empty plan")

    with manifest_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary.to_dict(), stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    return manifest_path, summary_path

