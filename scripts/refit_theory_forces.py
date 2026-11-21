"""Fx/Fy physical candidates, selected on the deployed absolute + delta prediction.

No mandatory drag term or shared coefficients. Sun P3 is an alternative, not
concatenated with algebraically identical dimensional bases. All preparation,
outer folds, training runs and the measured-RPM comparison remain unchanged.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import hashlib
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a
import analysis_search as s
import analysis_output as report
from scripts import compare_candidate_dictionary as comparison

VERSION = "physical-force-composite-v1"
RIDGES = (0., .0001, .001, .01, .1, 1., 10.)
BASES = ("theory__body_drag__1", "theory__rotor_plane_linear",
         "theory__rotor_plane_quadratic")
MOTIONS = ("pbar", "qbar", "rbar", "up", "uq", "ur")
STAGES = dict(core=("sun", "physical_core", "physical_direction"),
              interactions=("sun", "physical_core", "physical_direction", "physical_coupled"),
              per_rotor=("sun", "physical_core", "physical_direction", "physical_coupled", "per_rotor"))


def feature_bank(data, cfg, output):
    """Dimensional force columns; reference averaging must happen afterwards."""
    theory = a.theory_feature_bank(data, cfg, output)
    bank = a.extra_feature_bank(data, output, "sun_inplane", cfg)
    physical = {k: v for k, v in theory.items()
                if k.startswith("theory__body_drag__") or k in BASES}
    # First-order local axial-inflow modulation of local rotor in-plane force.
    # Ω R * λ = Vz: evaluate without division by RPM, including stopped rotors.
    local = data["rotor_flow"]["velocity_frd_m_s"]
    j = 0 if output == "Fx" else 1
    physical["local_axial_inplane"] = (-cfg.rho*np.pi*cfg.rotor_radius_m**2
                                       * (local[:, :, j]*local[:, :, 2]).sum(axis=1))
    coupled = {f"coupled__{base}__{motion}": physical[base]*data["state"][motion]
               for base in BASES for motion in MOTIONS}
    bank.update(physical)
    bank.update(coupled)
    area = np.pi*cfg.rotor_radius_m**2
    speeds = data["rotor_flow"]["omega_abs_rad_s"]
    horizontal = np.linalg.norm(local[:, :, :2], axis=2)
    individual = {}
    for i in range(4):
        for label, factor in (("linear", cfg.rotor_radius_m*speeds[:, i]),
                              ("quadratic", horizontal[:, i]), ("axial", local[:, i, 2])):
            individual[f"individual__{i+1}__{label}"] = -cfg.rho*area*factor*local[:, i, j]
    bank.update(individual)
    # '1' means QΩ, NOT a dimensional intercept. Retain this existing option
    # for Ω² cross-axis force correction; never interpret it as a sensor zero.
    common = {"intercept", "1"}
    pools = dict(sun=common | set(a.extra_feature_names(output, "sun_inplane")),
                 physical_core=common | set(BASES),
                 physical_direction=common | set(physical),
                 physical_coupled=common | set(physical) | set(coupled),
                 # Do not include each sum alongside its four summands.
                 per_rotor=common | {k for k in physical if "body_drag" in k} | set(individual))
    return bank, {k: sorted(v) for k, v in pools.items()}


def attach_banks(runs, cfg, output, route):
    for r in runs:
        source = r["search_data"][route]
        bank, pools = feature_bank(source, cfg, output)
        ix, ref = source["indices"]["selected"], source["indices"]["reference"]
        d = r["channels"][output][route]
        d["features"]["systematic"] = {k: v[ix] for k, v in bank.items()}
        d["reference_features"]["systematic"] = {
            k: float(v[ref].mean()) for k, v in bank.items()} if len(ref) else {}
    return sorted({"intercept", *bank}), pools


def fit_candidates(cases, stats, names, pools, ridges, noise=1e-12):
    """Only this training subset supplies selection, coefficients and scales."""
    conditions = {c["condition"] for c in cases}
    fitted = []
    for family in ("absolute", "delta"):
        stat = s.add_statistics([v for k, v in stats[family].items() if k in conditions])
        models = {}
        for spec, pool in pools.items():
            choices = [k for k in pool if family == "absolute" or k != "intercept"]
            selected, audit = s.gram_select(s.subset_statistics(
                stat, [names.index(k) for k in choices]), choices)
            ids = [c["run"] for c in cases if family == "delta" or c["anchored"]]
            for ridge in ridges:
                model = s.fit_stats(stat, names, selected, ridge, ids)
                model.update(selection=audit, candidate_pool=spec, dictionary_version=VERSION)
                models[spec, ridge] = model
        fitted.append(models)
    return fitted, s.scales_from_training(cases, noise)


def score_pairs(cases, foundations, increments, names, norm):
    """Fast exact scores for A(reference) + B(t)-B(reference), condition equal."""
    ka, kd = list(foundations), list(increments)
    ca = np.array([[m["coefficient_by_name"].get(k, 0.) for m in foundations.values()] for k in names])
    cd = np.array([[m["coefficient_by_name"].get(k, 0.) for m in increments.values()] for k in names])
    absolute, delta = defaultdict(list), defaultdict(list)
    for c in cases:
        if not c["validation"]:
            continue
        ed = c["dx"]@cd - c["dy"][:, None]
        delta[c["condition"]].append(np.mean(ed**2, axis=0))
        if c["anchored"]:
            # y = dy + reference residual; avoid a samples × A × B allocation.
            offset = c["xr"]@ca - float((c["y"]-c["dy"])[0])
            absolute[c["condition"]].append(offset[:, None]**2
                + 2*offset[:, None]*ed.mean(axis=0)[None, :]
                + np.mean(ed**2, axis=0)[None, :])
    if not absolute or not delta:
        raise ValueError("Inner split has no independently anchored validation")
    av = np.maximum(0., np.mean([np.mean(v, axis=0) for v in absolute.values()], axis=0))
    dv = np.mean([np.mean(v, axis=0) for v in delta.values()], axis=0)
    return {(aa, dd): dict(absolute_mse=float(av[i, j]), increment_mse=float(dv[j]),
                          joint=float(.5*(av[i, j]/norm[0]+dv[j]/norm[1])))
            for i, aa in enumerate(ka) for j, dd in enumerate(kd)}


def search_channel(cases, names, pools, output, route, *, ridges=RIDGES, stages=STAGES,
                   folds=None, noise=1e-12):
    condition_folds = defaultdict(set)
    for case in cases:
        condition_folds[case["condition"]].add(case["fold"])
    if any(len(value) != 1 for value in condition_folds.values()):
        raise ValueError("A condition spans outer folds; grouped sufficient statistics would leak")
    stats = {f: s.grouped_stats(cases, f) for f in ("absolute", "delta")}
    fitted, audits = {stage: {} for stage in stages}, []
    folds = folds if folds is not None else ("final", *sorted({c["fold"] for c in cases}))
    for fold in folds:
        train = [c for c in cases if fold == "final" or c["fold"] != fold]
        # Keep the same established inner condition partition as analysis_search.
        conditions = sorted({c["condition"] for c in train}, key=lambda k:
                            hashlib.sha256((s.VERSION+str(k)).encode()).hexdigest())
        inner = {k: i % 3 for i, k in enumerate(conditions)}
        scores = defaultdict(list)
        for split in range(3):
            tt = [c for c in train if inner[c["condition"]] != split]
            vv = [c for c in train if inner[c["condition"]] == split]
            models, norm = fit_candidates(tt, stats, names, pools, ridges, noise)
            for key, value in score_pairs(vv, *models, names, norm).items():
                scores[key].append(value)
        full, _ = fit_candidates(train, stats, names, pools, ridges, noise)
        mean_scores = {k: {m: float(np.mean([v[m] for v in values])) for m in values[0]}
                       for k, values in scores.items()}
        for stage, allowed in stages.items():
            eligible = [k for k in scores if k[0][0] in allowed and k[1][0] in allowed]
            winner = min(eligible, key=lambda k: (mean_scores[k]["joint"], k))
            ma, md = (deepcopy(full[i][key]) for i, key in enumerate(winner))
            audit = dict(stage=stage, fold=fold, output=output, route=route,
                chosen=dict(foundation_pool=winner[0][0], foundation_ridge=winner[0][1],
                            increment_pool=winner[1][0], increment_ridge=winner[1][1]),
                inner_conditions=inner, chosen_score=mean_scores[winner],
                objective="equal absolute/delta normalized condition-MSE of deployed composition",
                inner_scores=[dict(foundation_pool=k[0][0], foundation_ridge=k[0][1],
                    increment_pool=k[1][0], increment_ridge=k[1][1], **mean_scores[k]) for k in eligible])
            for model in (ma, md):
                model.update(search=audit, output=output, zero_updated=False)
            fitted[stage][str(fold)] = ma, md
            audits.append(audit)
            print("FIT", stage, output, route, fold, winner,
                  "terms", len(ma["feature_names"]), len(md["feature_names"]), flush=True)
    return fitted, audits


def main(args):
    import __main__
    __main__.Settings = a.Settings
    if args.output.exists():
        raise ValueError("Use a new output directory")
    args.output.mkdir(parents=True)
    with args.trusted_prepared_checkpoint.open("rb") as stream:
        _, profile, cfg, runs, _, _ = pickle.load(stream)
    previous = a.read_json(args.previous/"models.json")
    joint = None
    if args.coefficient_fit == "joint":
        from scripts import joint_force_search as joint
    stages = {key: STAGES[key] for key in args.stages}
    models = {stage: deepcopy(previous) for stage in stages}
    audits, dictionaries = [], {}
    for output in args.outputs:
        for route in a.ROUTES:
            names, pools = attach_banks(runs, cfg, output, route)
            allowed = {pool for values in stages.values() for pool in values}
            pools = {k: v for k, v in pools.items() if k in allowed}
            dictionaries[output] = pools
            cases = s.channel_cases(runs, [], route, output, names)
            # Preserve the historical 325 training runs: do not newly add the
            # reference-less centered case that generic systematic search allows.
            eligible = set(previous["extra"][f"final/{route}/{output}/increment"]["training_runs"])
            cases = [c for c in cases if c["run"] in eligible]
            searcher = joint.search_channel if joint else search_channel
            result, search_audit = searcher(cases, names, pools, output, route, stages=stages)
            audits.extend(search_audit)
            for stage, folds in result.items():
                for fold, pair in folds.items():
                    if joint:
                        train = [c for c in cases if fold == "final" or str(c["fold"]) != fold]
                        pair = joint.verify_pair(pair, train, names)
                    for family, model in zip(("foundation", "increment"), pair):
                        key = f"{fold}/{route}/{output}/{family}"
                        old = previous["extra"][key]
                        family_ids = model.get("family_supervision_runs", model["training_runs"])
                        assert set(family_ids) == set(old["training_runs"]), key
                        if joint:
                            models[stage]["extra"][key] = model
                            continue
                        matrix = comparison.matrix_for(runs, [], profile, cfg, output, route,
                            family, old, model["feature_names"] or ["1"], model["training_runs"])
                        # Final coefficients use the established QR/SVD solver,
                        # with an explicit check against fast search predictions.
                        verified = comparison.fit_selected(matrix, model["feature_names"], model, output, family)
                        verified["candidate_ablation"] = VERSION
                        xx = matrix.x if model["feature_names"] else matrix.x[:, :0]
                        np.testing.assert_allclose(xx@np.asarray(model["coefficients"]),
                            xx@np.asarray(verified["coefficients"]), atol=1e-8, rtol=1e-7)
                        models[stage]["extra"][key] = verified
            a.write_json(args.output/"progress.json", dict(completed=output+"/"+route, search=audits))
    for stage, bundle in models.items():
        directory = args.output/stage
        directory.mkdir()
        rows, arrays = a.evaluate_extra_channels(runs, bundle["extra"], outputs=args.outputs)
        summary = dict(result_schema="fftgyro-analysis-result-v1", profile=profile["schema"],
            version=joint.VERSION if joint else VERSION, coefficient_fit=args.coefficient_fit,
            stage=stage, outputs=args.outputs, per_run=rows, metrics=a.summarize(rows),
            source_models=str(args.previous/"models.json"), preparation_checkpoint=str(args.trusted_prepared_checkpoint),
            limitations=["Repeated-development condition OOF, not a fresh blind test.",
                "Independent foundation/increment coefficients; neither physical term is mandatory.",
                "Sun F/PSE selection heuristic, not a significance test on independent time samples.",
                "Body/rotor drag can remain observationally confounded; coefficients are not unique mechanism estimates.",
                "No passivity constraint; powered-regime interactions require extrapolation validation.",
                "Zero, masks, folds, training run IDs and measured-RPM comparison fixed. SITL not updated."])
        np.savez_compressed(directory/"predictions.npz", **arrays)
        a.write_json(directory/"models.json", bundle)
        a.write_json(directory/"summary.json", summary)
        report.compare_previous(args.previous, summary, arrays, directory)
        print("SAVED", directory, flush=True)
    sources = [Path(a.__file__), Path(s.__file__), Path(comparison.__file__), Path(__file__)]
    if joint:
        sources.extend((Path(joint.__file__), Path(joint.j.__file__)))
    a.write_json(args.output/"search.json", dict(version=VERSION, coefficient_fit=args.coefficient_fit,
                                             dictionaries=dictionaries, search=audits))
    a.write_json(args.output/"manifest.json", dict(version=VERSION, coefficient_fit=args.coefficient_fit, source_sha256={
        str(p.relative_to(Path(__file__).resolve().parents[1])): report.sha256(p) for p in
        sources}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-prepared-checkpoint", type=Path, required=True)
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--outputs", nargs="+", choices=("Fx", "Fy"), default=["Fx", "Fy"])
    parser.add_argument("--coefficient-fit", choices=("separate", "joint"), default="separate")
    parser.add_argument("--stages", nargs="+", choices=tuple(STAGES), default=["core", "interactions"])
    main(parser.parse_args())
