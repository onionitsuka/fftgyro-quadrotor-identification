"""Plot/export boundary: private logs and fitting are unnecessary."""

import ast
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a
import analysis_output as out


@pytest.fixture
def saved(tmp_path):
    source = tmp_path / "analysis"
    source.mkdir()
    rows, arrays = [], {}
    for mode in ("all_data_in_sample", "grouped_OOF"):
        for ref in ("increment", "independent_absolute", "assisted_direct"):
            for route in ("measured", "command_route"):
                truth = np.linspace(-0.02, 0.02, 4)
                prediction, physical = truth + 0.001, truth - 0.003
                row = dict(
                    mode=mode,
                    output="Mz",
                    route=route,
                    reference=ref,
                    wind=11.5,
                    run="test",
                    condition_id="condition1",
                    physical=a.error_metrics(truth, physical),
                    identified=a.error_metrics(truth, prediction),
                )
                rows.append(row)
                prefix = f"{mode}/Mz/{route}/test/{ref}/"
                arrays.update(
                    {
                        prefix + k: v
                        for k, v in dict(
                            time=np.arange(4.0),
                            truth=truth,
                            prediction=prediction,
                            physical=physical,
                        ).items()
                    }
                )
    summary = dict(
        profile="fftgyro-final-20261013-v1",
        per_run=rows,
        metrics=a.summarize(rows),
        code_sha256="fixture",
        limitations=["Assisted is not independent validation"],
    )
    out.write_json(source / "summary.json", summary)
    np.savez_compressed(source / "predictions.npz", **arrays)
    return source, summary, arrays


def imports(module):
    for node in ast.walk(ast.parse(Path(module.__file__).read_text())):
        if isinstance(node, ast.Import):
            yield from (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            yield node.module or ""


def test_computation_and_presentation_have_no_cross_imports():
    assert not any(s.startswith(("analysis_output", "matplotlib")) for s in imports(a))
    assert not any(s.startswith(("analysis", "scipy", "scripts", "fftgyro")) for s in imports(out))
    assert not hasattr(a, "plot_results")


@pytest.mark.parametrize("schema", [None, "fftgyro-analysis-result-v1"])
def test_export_legacy_and_new_results_without_raw_data_or_models(saved, tmp_path, schema):
    source, summary, _ = saved
    if schema:
        summary["result_schema"] = schema
        out.write_json(source / "summary.json", summary)
    hashes = {p.name: out.sha256(p) for p in source.iterdir()}
    target = tmp_path / "report"
    assert out.export_results(source, target, plots=False) == summary
    assert json.loads((target / "summary.json").read_text()) == summary
    assert hashes == {p.name: out.sha256(p) for p in source.iterdir()}
    with (target / "metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        table = list(csv.DictReader(f))
    assert len(table) == 2 * len(summary["metrics"])
    assert {r["reference"] for r in table} == {
        "increment",
        "independent_absolute",
        "assisted_absolute",
    }
    assert {r["mode"] for r in table} == {"all_data_in_sample", "grouped_OOF"}
    assert all(r["unit"] == "N m" for r in table)
    for row, stored in zip(table[::2], summary["metrics"], strict=True):
        assert float(row["run_equal_rmse"]) == stored["physical"]["run_equal_rmse"]
    manifest = json.loads((target / "report_manifest.json").read_text())
    assert manifest["source_sha256"] == hashes
    assert manifest["refit_performed"] is False
    assert not list(target.glob("*.png"))


def test_output_cannot_overwrite_input_or_existing_report(saved, tmp_path):
    source, _, _ = saved
    with pytest.raises(ValueError):
        out.export_results(source, source, plots=False)
    with pytest.raises(ValueError):
        out.export_results(source, source / "figures", plots=False)
    report = tmp_path / "report"
    report.mkdir()
    with pytest.raises(FileExistsError):
        out.export_results(source, report, plots=False)


@pytest.mark.parametrize("bad", ["nan", "length", "count", "duplicate", "missing", "schema"])
def test_malformed_results_fail_without_silent_point_removal(saved, tmp_path, bad):
    source, summary, arrays = saved
    key = next(k for k in arrays if k.endswith("/truth"))
    if bad == "nan":
        arrays[key] = np.full(4, np.nan)
    elif bad == "length":
        arrays[key] = arrays[key][:-1]
    elif bad == "count":
        summary["per_run"][0]["physical"]["samples"] = 99
    elif bad == "duplicate":
        summary["per_run"].append(summary["per_run"][0])
    elif bad == "missing":
        del arrays[key]
    else:
        summary["result_schema"] = "unknown"
    out.write_json(source / "summary.json", summary)
    np.savez_compressed(source / "predictions.npz", **arrays)
    target = tmp_path / "report"
    with pytest.raises((ValueError, KeyError)):
        out.export_results(source, target, plots=False)
    assert not target.exists()


def test_figures_can_be_generated_using_only_saved_results(saved, tmp_path):
    from PIL import Image

    source, _, _ = saved
    target = tmp_path / "report"
    out.export_results(source, target)
    plots = sorted(target.glob("*.png"))
    assert len(plots) == 3
    for path in plots:
        with Image.open(path) as im:
            assert im.size == (2700, 900)
    manifest = json.loads((target / "figure_manifest.json").read_text())
    assert manifest["all_points_shown"]
    assert manifest["reference_classes_separate"]


def test_refit_comparison_requires_same_truth_and_excludes_assisted(saved, tmp_path):
    source, summary, arrays = saved
    report = tmp_path/'comparison'
    report.mkdir()
    result = out.compare_previous(source, summary, arrays, report)
    assert result['paired_arrays_verified'] == 24
    assert not any(r['reference'].startswith('assisted_') for r in result['groups'])
    assert all(r['before_rmse'] == r['after_rmse'] for r in result['groups'])
    changed = copy.deepcopy(arrays)
    key = next(k for k in changed if '/independent_absolute/' in k and k.endswith('/truth'))
    changed[key] += .01
    with pytest.raises(AssertionError):
        out.compare_previous(source, summary, changed, report)
