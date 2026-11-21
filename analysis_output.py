#!/usr/bin/env python3
"""FFTGYRO 解析結果の図・表出力。fitも荷重再構成も実行しない。

python analysis_output.py --input ANALYSIS_OUTPUT --output NEW_REPORT

読む順番: 1 保存結果の読込 → 2 表 → 3 図 → 4 実行。
入力は analysis.py の summary.json と predictions.npz の2ファイルだけ。
旧 final-analysis-20261013-v2 の保存結果もそのまま使用できる。
図は全学習の1:1比較、表は全学習／条件外の両方。独立絶対基準と
運動補正した停止基準・モデル補助基準は混ぜない。数値・採用点・零点・モデルは変更しない。
CSVの単位はN/N m、図だけモーメントをmN mに換算する。
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
from pathlib import Path

import numpy as np

CHANNELS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


# 1. 保存結果の読込（欠損・破損を黙って除外しない）
def write_json(path, value):
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_results(source):
    summary = json.loads((source / "summary.json").read_text(encoding="utf-8-sig"))
    if summary.get("profile") != "fftgyro-final-20261013-v1":
        raise ValueError("Unknown analysis profile")
    if summary.get("result_schema", "fftgyro-analysis-result-v1") != "fftgyro-analysis-result-v1":
        raise ValueError("Unknown result schema")
    with np.load(source / "predictions.npz", allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    seen = set()
    for row in summary["per_run"]:
        key = (row[k] for k in ("mode", "output", "route", "run", "reference"))
        prefix = "/".join(map(str, key)) + "/"
        if prefix in seen:
            raise ValueError(f"Duplicate result: {prefix}")
        seen.add(prefix)
        values = [arrays[prefix + k] for k in ("time", "truth", "prediction", "physical")]
        if any(v.ndim != 1 or not v.size or not np.isfinite(v).all() for v in values):
            raise ValueError(f"Empty/nonfinite/non-vector data: {prefix}")
        if any(v.shape != values[0].shape for v in values):
            raise ValueError(f"Array length mismatch: {prefix}")
        if any(row[variant]["samples"] != len(values[0]) for variant in ("physical", "identified")):
            raise ValueError(f"Metric/sample count mismatch: {prefix}")
    if not seen or not summary["metrics"]:
        raise ValueError("No analysis results")
    return summary, arrays


# 2. 計算済み誤差の表。再集計／再fitはせず、保存値をそのまま出す
def write_table(path, rows, identifiers, metric_names):
    fields = [*identifiers, "unit", "variant", *metric_names]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for variant in ("physical", "identified"):
                writer.writerow(
                    {
                        **{k: row[k] for k in identifiers},
                        "unit": "N" if row["output"].startswith("F") else "N m",
                        "variant": variant,
                        **{k: row[variant][k] for k in metric_names},
                    }
                )


def compare_previous(source, summary, arrays, output):
    """Compare refits on identical truth, time and measured-RPM baseline.

    Model-assisted references can themselves change when refitting. They are
    deliberately excluded from this paired comparison, not treated as truth.
    """
    previous, old_arrays = load_results(source)
    identifiers = ('mode', 'output', 'route', 'run', 'reference')
    old = {tuple(r[k] for k in identifiers):r for r in previous['per_run']}
    paired = []
    checked = 0
    for row in summary['per_run']:
        if row['reference'] not in ('independent_absolute', 'motion_corrected_absolute', 'increment'):
            continue
        key = tuple(row[k] for k in identifiers)
        if key not in old:
            raise ValueError(f'No same-reference pre-fit evaluation for {key}')
        p = '/'.join(map(str, key))+'/'
        for field in ('time', 'truth', 'physical'):
            np.testing.assert_allclose(arrays[p+field], old_arrays[p+field], atol=1e-12, rtol=0)
            checked += 1
        paired.append(dict(**{k:row[k] for k in (*identifiers, 'wind', 'condition_id')},
            samples=row['identified']['samples'], physical_rmse=row['physical']['rmse'],
            before_rmse=old[key]['identified']['rmse'], after_rmse=row['identified']['rmse'],
            before_bias=old[key]['identified']['bias'], after_bias=row['identified']['bias']))
    groups = []
    for mode, channel, route in sorted({tuple(r[k] for k in ('mode','output','route')) for r in paired}):
        base = [r for r in paired if (r['mode'],r['output'],r['route']) == (mode,channel,route)]
        for reference in ('independent_absolute','motion_corrected_absolute','all_stop_absolute','increment'):
            selected = [r for r in base if (r['reference'] in ('independent_absolute','motion_corrected_absolute')
                        if reference=='all_stop_absolute' else r['reference']==reference)]
            if not selected:
                continue
            for wind in ['all']+sorted({r['wind'] for r in selected}):
                rr = [r for r in selected if wind=='all' or r['wind']==wind]
                groups.append(dict(mode=mode, output=channel, route=route, reference=reference,
                    wind=wind, runs=len(rr), samples=sum(r['samples'] for r in rr),
                    improved_runs=sum(r['after_rmse']<r['before_rmse']-1e-12 for r in rr),
                    **{k:float(np.sqrt(np.mean([r[k]**2 for r in rr])))
                       for k in ('physical_rmse','before_rmse','after_rmse')},
                    **{k:float(np.mean([r[k] for r in rr])) for k in ('before_bias','after_bias')}))
    for name, data in [('refit_comparison.csv',groups), ('refit_per_run.csv',paired)]:
        with (output/name).open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    result=dict(previous=str(source), paired_arrays_verified=checked,
        aggregation='sqrt(mean(run RMSE squared)); same truth/time/physical, both routes',
        before_summary_sha256=sha256(source/'summary.json'), groups=groups,
        limitation='New motor-off references remain conditional. All-stop totals combine old/new classes only for coverage; assisted references excluded.')
    write_json(output/'refit_comparison.json',result)
    return result


# 3. 1:1比較。元の描画処理を移設（同じ点・軸・集計式）
def plot_results(output, rows, arrays, mode="all_data_in_sample"):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 全風速を描く。絶対値の4風速だけに色を限定しない。
    winds = sorted({r["wind"] for r in rows})
    palette = (
        "#332288",
        "#117733",
        "#882255",
        "#AA4499",
        "#004488",
        "#994455",
        "#000000",
        "#6699CC",
        "#666666",
    )
    shapes = ("o", "s", "D", "^", "v", "P", "X", "<", ">")
    colors = {w: palette[i % len(palette)] for i, w in enumerate(winds)}
    markers = {w: shapes[i % len(shapes)] for i, w in enumerate(winds)}
    with plt.rc_context({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}):
        for ref in ("increment", "independent_absolute", "motion_corrected_absolute", "assisted_absolute"):
            for channel in CHANNELS:
                rr = [
                    r
                    for r in rows
                    if r["mode"] == mode
                    and r["output"] == channel
                    and (
                        r["reference"].startswith("assisted_")
                        if ref == "assisted_absolute"
                        else r["reference"] == ref
                    )
                ]
                if not rr:
                    continue
                scale = 1.0 if channel.startswith("F") else 1000.0
                unit = "N" if channel.startswith("F") else "mN m"
                fig, axes = plt.subplots(1, 3, figsize=(15, 5), layout="constrained")
                all_values = []
                for ax, (route, field, title) in zip(
                    axes,
                    [
                        ("measured", "physical", "Physical / measured RPM"),
                        ("measured", "prediction", "Identified / measured RPM"),
                        ("command_route", "prediction", "Identified / command + voltage"),
                    ],
                    strict=True,
                ):
                    subset = [r for r in rr if r["route"] == route]
                    for wind in sorted({r["wind"] for r in subset}):
                        group = [r for r in subset if r["wind"] == wind]
                        xx = []
                        yy = []
                        for r in group:
                            p = f"{mode}/{channel}/{route}/{r['run']}/{r['reference']}/"
                            xx.extend(arrays[p + "truth"] * scale)
                            yy.extend(arrays[p + field] * scale)
                        ax.scatter(
                            xx,
                            yy,
                            s=5,
                            c=colors[wind],
                            marker=markers[wind],
                            label=f"{wind:g} m/s",
                            rasterized=True,
                        )
                        all_values.extend(xx)
                        all_values.extend(yy)
                    metric = "physical" if field == "physical" else "identified"
                    rmse = np.sqrt(np.mean([r[metric]["rmse"] ** 2 for r in subset])) * scale
                    ax.set(
                        title=f"{title}\nrun-equal RMSE {rmse:.3g} {unit}",
                        xlabel=f"Reconstructed {channel} [{unit}]",
                        ylabel=f"Predicted {channel} [{unit}]",
                    )
                lo, hi = min(all_values), max(all_values)
                pad = max((hi - lo) * 0.05, 1e-8)
                for ax in axes:
                    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
                    ax.set(xlim=(lo - pad, hi + pad), ylim=(lo - pad, hi + pad), aspect="equal")
                    ax.grid(alpha=0.2)
                handles, labels = axes[0].get_legend_handles_labels()
                fig.legend(handles, labels, loc="outside lower center", ncol=5, fontsize=10)
                label = (
                    "all-data fit (NOT held out)"
                    if mode == "all_data_in_sample"
                    else "grouped out-of-fold (development data)"
                )
                fig.suptitle(f"{channel}: {ref} | {label}")
                suffix = "" if mode == "all_data_in_sample" else "-" + mode
                fig.savefig(output / f"{channel}-{ref}{suffix}.png", dpi=180, facecolor="white")
                plt.close(fig)
    write_json(
        output
        / (
            "figure_manifest.json"
            if mode == "all_data_in_sample"
            else f"figure_manifest-{mode}.json"
        ),
        dict(
            plotting_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            all_points_shown=True,
            common_limits=True,
            reference_classes_separate=True,
            metric="run-equal RMSE",
            fit_status=mode,
            pixels=[2700, 900],
            dpi=180,
            color_and_marker_by_wind=True,
        ),
    )


def plot_six_axis_overview(output, rows, arrays, mode, reference):
    """成分ごとに物理/同定を同じ点・軸で比較。外れ点はクリップしない。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False}):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10), layout="constrained")
        for channel, ax in zip(CHANNELS, axes.flat, strict=True):
            rr = [
                r
                for r in rows
                if r["mode"] == mode
                and r["reference"] == reference
                and r["output"] == channel
                and r["route"] == "measured"
            ]
            if not rr:
                ax.text(0.5, 0.5, "Unavailable", transform=ax.transAxes, ha="center")
                continue
            scale, unit = (1, "N") if channel.startswith("F") else (1000, "mN m")
            values = {}
            for field in ("truth", "physical", "prediction"):
                values[field] = scale * np.concatenate(
                    [
                        arrays[f"{mode}/{channel}/measured/{r['run']}/{reference}/{field}"]
                        for r in rr
                    ]
                )
            for field, color, marker, label in (
                ("physical", "#777777", "x", "Physical"),
                ("prediction", "#0072B2", ".", "Identified"),
            ):
                ax.scatter(
                    values["truth"],
                    values[field],
                    s=7,
                    color=color,
                    marker=marker,
                    label=label,
                    alpha=0.65,
                    rasterized=True,
                )
            lo, hi = min(v.min() for v in values.values()), max(v.max() for v in values.values())
            pad = max((hi - lo) * 0.05, 1e-8)
            pm, gm = [
                scale * np.sqrt(np.mean([r[v]["rmse"] ** 2 for r in rr]))
                for v in ("physical", "identified")
            ]
            ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=1)
            ax.set(
                xlim=(lo - pad, hi + pad),
                ylim=(lo - pad, hi + pad),
                aspect="equal",
                title=f"{channel} | {len(rr)} runs\nRMSE {pm:.3g} -> {gm:.3g} {unit}",
                xlabel=f"Reconstructed [{unit}]",
                ylabel=f"Predicted [{unit}]",
            )
            ax.grid(alpha=0.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2)
        fig.suptitle(
            f"Six-axis | {reference} | {mode}\nMeasured RPM; run-equal RMSE; different channel scopes; NOT free-flight validation",
            fontsize=15,
        )
        fig.savefig(output / f"six-axis-{reference}-{mode}.png", dpi=160, facecolor="white")
        plt.close(fig)


def write_six_axis_index(output, summary):
    """図とCSVへの入口。基準のない絶対値を検証済みとは表示しない。"""
    content = [
        '<!doctype html><html lang="ja"><meta charset="utf-8"><title>6軸解析</title>',
        "<style>body{font:17px/1.65 sans-serif;max-width:1200px;margin:32px auto;padding:0 20px}img{width:100%;height:auto}td,th{padding:6px 14px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}table{border-collapse:collapse}</style>",
        "<h1>6軸の荷重モデル比較</h1>",
        "<p>既存のMy・Mz・Fzは再現性を確認し、Fx・Fy・Mxを追加同定。"
        "全学習への一致と条件別検証を分けます。物理比較は両経路とも実測RPMです。</p>",
        "<p>追加3軸：独立停止基準付きrunの絶対基礎＋全runの基準からの変化。"
        "基準荷重もモデルで予測し、実測初期荷重へ予測を合わせません。"
        "既存のFz採用点を共通に利用。外側5条件foldの内側3条件foldで辞書・正則化を選択し、"
        "Sun型F/PSEで項を選び直します。追加軸の損失は条件等重みで、校正共分散による重み付けはしていません。</p>",
        "<p><strong>注意：</strong>Fx・Fyの物理基準はゼロ（抗力モデル比較ではありません）。"
        "Mxは固定・Pitch・Yaw中の横軸荷重で、独立したRoll運動の検証ではありません。"
        "絶対値は校正に条件付けた再構成値との誤差で、未知の零点誤差は残ります。"
        "項選択まで内側で行ったのは追加3軸のみ。My・Mz・Fzは既存構造を維持しています。"
        "全成分とも校正・RPM写像を凍結した開発データ上の評価で、実飛行の独立検証ではありません。</p>",
        "<p>集計は √mean(RMSE_run²)。改善率は100×(1−同定RMSE/物理RMSE)。"
        "成分ごとに試験範囲が異なり、改善率は荷重の相対誤差ではありません。</p>",
    ]
    for mode, mode_label in (
        ("grouped_OOF", "条件別検証"),
        ("all_data_in_sample", "全学習：学習データへの一致"),
    ):
        content.append(f"<h2>{mode_label}</h2>")
        for reference, ref_label in (
            ("increment", "基準からの変化量"),
            ("independent_absolute", "独立停止基準による絶対値"),
        ):
            content.append(
                f"<h3>{ref_label}</h3><table><tr><th>成分</th><th>run</th><th>物理</th><th>実測RPM同定</th><th>推算RPM同定</th><th>実測経路の改善率</th></tr>"
            )
            for ch in CHANNELS:
                rr = [
                    r
                    for r in summary["metrics"]
                    if r["mode"] == mode
                    and r["reference"] == reference
                    and r["output"] == ch
                    and r["wind"] == "all"
                ]
                by_route = {r["route"]: r for r in rr}
                if "measured" not in by_route:
                    continue
                m, c = by_route["measured"], by_route.get("command_route")
                scale, unit = (1.0, "N") if ch.startswith("F") else (1000.0, "mN·m")
                pm, gm = [m[v]["run_equal_rmse"] * scale for v in ("physical", "identified")]
                cm = f"{c['identified']['run_equal_rmse'] * scale:.4g}" if c else "—"
                gain = f"{100 * (1 - gm / pm):+.1f}%" if pm > 0 else "—"
                content.append(
                    f"<tr><td>{ch} [{unit}]</td><td>{m['runs']}</td><td>{pm:.4g}</td><td>{gm:.4g}</td><td>{cm}</td><td>{gain}</td></tr>"
                )
            content.append("</table>")
            name = f"six-axis-{reference}-{mode}.png"
            if (output / name).exists():
                content.append(
                    f'<a href="{name}"><img src="{name}" alt="{mode_label}、{ref_label}の6成分。黒破線は完全一致、灰色×は物理、青点は同定。数値は上表参照。"></a>'
                )
    content.append(
        "<h2>成分別の図</h2><p>全点を表示。強風Yawのモデル補助基準は独立絶対基準に合算しません。</p><ul>"
    )
    for path in sorted(output.glob("*.png")):
        content.append(f'<li><a href="{html.escape(path.name)}">{html.escape(path.name)}</a></li>')
    content.append(
        '</ul><h2>数値・来歴</h2><p><a href="metrics.csv">条件別集計</a> / <a href="per_run.csv">全run誤差</a> / <a href="summary.json">集計・制約・hash</a></p>'
    )
    content.append(
        '<p>図作成支援：scientific-visualization。Kassis, T., Agarwal, V., He, Y., Patel, D., &amp; Brueckner, A. M. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents. <a href="https://doi.org/10.48550/arXiv.2609.00065">DOI</a>。空力モデルの出典ではありません。</p></html>'
    )
    (output / "index.html").write_text("\n".join(content), encoding="utf-8")


# 4. 実行。既存の解析結果・出力ディレクトリを上書きしない
def export_results(source: Path, output: Path, *, plots: bool = True, include_oof: bool = False,
                   previous: Path | None = None):
    source, output = source.resolve(), output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Report directory must be outside the analysis input directory")
    if output.exists():
        raise FileExistsError(output)
    summary, arrays = load_results(source)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "summary.json", summary)
    write_table(
        output / "metrics.csv",
        summary["metrics"],
        ("mode", "output", "route", "reference", "wind", "runs"),
        ("run_equal_rmse", "sample_rmse", "run_equal_bias", "samples"),
    )
    write_table(
        output / "per_run.csv",
        summary["per_run"],
        ("mode", "output", "route", "reference", "wind", "run", "condition_id"),
        ("rmse", "bias", "samples"),
    )
    if previous is not None:
        compare_previous(previous, summary, arrays, output)
    if plots:
        plot_results(output, summary["per_run"], arrays)
        if include_oof:
            plot_results(output, summary["per_run"], arrays, "grouped_OOF")
        if summary.get("extra_channels_refit"):
            for mode in ("all_data_in_sample", "grouped_OOF"):
                for reference in ("increment", "independent_absolute"):
                    plot_six_axis_overview(output, summary["per_run"], arrays, mode, reference)
    if summary.get("extra_channels_refit"):
        write_six_axis_index(output, summary)
    write_json(
        output / "report_manifest.json",
        {
            "source_directory": str(source),
            "source_sha256": {
                name: sha256(source / name) for name in ("summary.json", "predictions.npz")
            },
            "output_code_sha256": sha256(Path(__file__)),
            "analysis_code_sha256": summary.get("code_sha256"),
            "refit_performed": False,
            "numerical_values_changed": False,
            "plots_created": plots,
            "figure_mode": "all_data_in_sample" if plots else None,
            "table_modes": sorted({r["mode"] for r in summary["per_run"]}),
            "csv_units": "N / N m; figures use N / mN m",
            "limitations": summary.get("limitations", []),
        },
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="analysis.pyの保存先")
    parser.add_argument("--output", type=Path, required=True, help="図・表の新規保存先")
    parser.add_argument("--no-plots", action="store_true", help="表だけを出力")
    parser.add_argument("--include-oof", action="store_true", help="条件別検証の図も出力")
    parser.add_argument("--previous", type=Path, help="同じ基準で再構成した再fit前の結果との比較")
    args = parser.parse_args()
    summary = export_results(
        args.input, args.output, plots=not args.no_plots, include_oof=args.include_oof,
        previous=args.previous
    )
    print("DONE", args.output, flush=True)
    for row in summary["metrics"]:
        if (
            row["mode"] == "all_data_in_sample"
            and row["wind"] == "all"
            and row["route"] == "measured"
        ):
            print(row, flush=True)


if __name__ == "__main__":
    main()
