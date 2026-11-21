"""Candidate-only ablation using the saved legacy losses and per-fold ridge.

Three arms: exact old-structure refit; reselection within old selected terms;
the same selection plus restored/cross dictionaries. Both searched arms retain
the old structure as an inner-CV candidate. Coefficients are always fit only on
the current training subset. This is development CV, not a fresh blind test.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
import gc
import hashlib
from pathlib import Path
import pickle
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analysis as a
import analysis_search as s

VERSION = "legacy-loss-candidate-only-v1"
CHOICES = ("old_structure", "old_pool", "restored", "cross2", "cross3", "cross4")


def correlation_diagnostics(blocks):
    """Post-fit audit only, NEVER a new term-selection constraint.

    Correlations/VIF use the actual weighted design. Project out its intercept
    column if present; contrasts without an intercept use uncentered design
    cosines/VIF. These are not raw, unweighted time-series Pearson statistics.
    """
    x = blocks.x.copy()
    names = list(blocks.names)
    if "intercept" in names:
        j = names.index("intercept")
        q = x[:, j]
        if q@q > 0:
            x -= np.outer(q, q@x/(q@q))
        x = np.delete(x, j, axis=1)
        names.pop(j)
    norms = np.linalg.norm(x, axis=0)
    keep = norms > 1e-20
    removed = [k for k,take in zip(names,keep) if not take]
    names = [k for k,take in zip(names,keep) if take]
    x = x[:,keep]/norms[keep]
    corr = x.T@x
    eigen = np.linalg.eigvalsh(corr) if names else np.array([])
    full_rank = bool(len(eigen) and eigen[0] > max(1e-12*eigen[-1],1e-14))
    vif = np.diag(np.linalg.inv(corr)) if full_rank else None
    pairs = sorted((dict(first=names[i],second=names[j],correlation=float(corr[i,j]))
                    for i in range(len(names)) for j in range(i)),
                   key=lambda d:abs(d["correlation"]),reverse=True)
    return dict(basis="weighted, intercept-residualized" if "intercept" in blocks.names
                else "weighted, uncentered contrast design",postfit_only=True,
        zero_information_columns=removed,correlation_names=names,correlation_matrix=corr.tolist(),
        strongest_pairs=pairs[:10],vif_by_name=dict(zip(names,vif.tolist())) if vif is not None else None,
        vif_status="finite" if full_rank else "rank-deficient or empty; not replaced with pseudoinverse")


def model_key(output, route, fold, family):
    if output in a.EXTRA_CHANNELS:
        return "extra", f"{fold}/{route}/{output}/{family}"
    if output == "Mz" and family == "increment":
        return "yaw", f"{fold}/{route}"
    return family, f"{fold}/{route}/{output}"


def matrix_for(runs, yaw_runs, profile, cfg, output, route, family, old, names, ids):
    """Reuse the ORIGINAL weighting routines; never normalize absolute/delta anew."""
    ids = set(ids)
    template = dict(old, spec="systematic", feature_names=list(names), training_runs=sorted(ids))
    if output == "Mz" and family == "increment":
        rr = [r for r in yaw_runs if r["run"] in ids]
        x, y = a.training_matrix(rr, route, "augmented", names,
            extra_ratio=old["extra_ratio"], balanced=False, noise=profile["yaw_noise"])
        return a.RegressionBlocks(x, y, tuple(names), dict(training_runs=sorted(ids)))
    if output in a.EXTRA_CHANNELS:
        # extra_matrix has an explicit dictionary; reproduce its scalar weights
        # without changing its public interface or the ordinary analysis path.
        rr = [r for r in runs if r["run"] in ids]
        counts = a.Counter(r["condition_id"] for r in rr)
        xx, yy = [], []
        for r in rr:
            ch = r["channels"][output]
            d, take = ch[route], ch["take"]
            n = int(take.sum())
            x = np.column_stack([np.ones(n) if k == "intercept" else
                                 d["features"]["systematic"][k][take] for k in names])
            y = (ch["truth"]-d["physical"])[take]
            if family == "increment":
                x -= [d["reference_features"]["systematic"][k] for k in names]
                y -= ch["reference"]-d["reference_physical"]
            weight = 1/np.sqrt(n*counts[r["condition_id"]])
            xx.append(x*weight); yy.append(y*weight)
        return a.RegressionBlocks(np.vstack(xx), np.concatenate(yy), tuple(names),
                                  dict(training_runs=sorted(ids)))
    if family == "foundation" and output != "Fz":
        x, y, names, audit = a.assemble_foundation(runs, profile["calibration"], route, output, template)
        return a.RegressionBlocks(x, y, names, dict(training_runs=sorted(ids), contributions=audit))
    return a.frozen_fit(runs, profile["calibration"], route, output, template, cfg, matrix_only=True)


def fit_selected(matrix, names, old, output, family):
    """Exact QR/SVD solver, bounds and scaled ridge from the legacy model."""
    if names:
        small = a.subset_blocks(matrix, names)
        compressed = a.compress_regression_blocks(small)
        if output == "Mz" and family == "increment":
            fit = a.fit_yaw_matrix(small.x, small.y, names, old["ridge"], "augmented",
                                  local_penalty=old["local_penalty"])
        else:
            bounds = {k: tuple(-np.inf if v is None and i == 0 else np.inf if v is None else v
                       for i, v in enumerate(b)) for k, b in old.get("bounds", {}).items() if k in names}
            fit = a.fit_regularized(compressed, ridge=old["ridge"],
                                   bounds=bounds, ridge_exempt=old.get("ridge_exempt", ("intercept",)))
        fit["correlation_diagnostics"] = correlation_diagnostics(compressed)
    else:
        fit = dict(feature_names=[], coefficients=[], coefficient_by_name={}, column_scales=[],
                   scaled_coefficients=[], ridge=old["ridge"], numerical_rank=0)
    result = {**old, **fit, "spec": "systematic", "training_runs":matrix.audit["training_runs"],
              "zero_updated":False, "candidate_ablation":VERSION}
    if output == "Mz" and family == "increment":
        result["bank"] = "augmented"
    return result


def fitted_choices(matrix, pools, old, output, family):
    stats = s.statistics(matrix.x, matrix.y)
    fitted = {}
    for choice in ("old_structure", *pools):
        names = old["feature_names"] if choice == "old_structure" else pools[choice]
        if choice != "old_structure":
            ids = [matrix.names.index(k) for k in names]
            names, selection = s.gram_select(s.subset_statistics(stats, ids), names)
        else:
            selection = dict(method="unchanged_saved_structure_refit", candidate_names=names)
        fit = fit_selected(matrix, names, old, output, family)
        fit["selection"] = selection
        fit["frozen_structure"] = choice == "old_structure"
        coefficients = np.array([fit["coefficient_by_name"].get(k, 0.) for k in matrix.names])
        error = matrix.x @ coefficients-matrix.y
        fit["ablation_training_mse"] = float(error@error/len(error))
        fitted[choice] = fit
    return fitted


def score_models(matrix, models):
    """Same legacy weighted residual loss for ALL dictionaries, without ridge."""
    return {key: float(np.mean((matrix.x @ np.array([
        model["coefficient_by_name"].get(k, 0.) for k in matrix.names])-matrix.y)**2))
        for key, model in models.items()}


def attach_dictionary(runs, yaw_runs, cfg, output, route, legacy):
    yi = {r["run"]:r for r in yaw_runs}
    restored = None
    for r in runs:
        if output not in r["channels"]:
            continue
        source = r["search_data"][route]
        # Cross/local names are generated by expanded_bank AFTER restored_bank.
        # Do not pass them into its plain-monomial fallback when reselecting an
        # already expanded model.
        plain = [k for k in legacy if not k.startswith(("cross__", "localflow__"))]
        bank, restored = s.expanded_bank(source, cfg, output, plain)
        d = r["channels"][output][route]
        ix, ref = source["indices"]["selected"], source["indices"]["reference"]
        d["features"]["systematic"] = {k:v[ix] for k,v in bank.items()}
        d["reference_features"]["systematic"] = {k:float(v[ref].mean()) for k,v in bank.items()} if len(ref) else {}
        if output == "Mz" and r["run"] in yi:
            ii, ri = source["indices"]["yaw"], source["indices"]["yaw_reference"]
            yi[r["run"]]["route"][route]["augmented"].update(
                {k:a.reference_contrast(v, ii, ri) for k,v in bank.items()})
    return set(bank), restored


def attach_latest_dictionary(runs, yaw_runs, cfg, output, route, legacy):
    """Add the newest dimensional candidates without changing targets or weights."""
    from scripts import six_axis_theory_bank as theory
    available, restored = attach_dictionary(runs, yaw_runs, cfg, output, route, legacy)
    yi = {r["run"]: r for r in yaw_runs}
    expected = None
    for r in runs:
        if output not in r["channels"]:
            continue
        source = r["search_data"][route]
        bank, pools = theory.feature_bank(source, cfg, output)
        if expected is not None and pools != expected:
            raise ValueError("Candidate eligibility changed across runs")
        expected = pools
        d = r["channels"][output][route]
        ix, ref = source["indices"]["selected"], source["indices"]["reference"]
        for k, value in bank.items():
            if k in d["features"]["systematic"]:
                np.testing.assert_allclose(d["features"]["systematic"][k], value[ix],
                                           atol=1e-12, rtol=1e-12)
            d["features"]["systematic"][k] = value[ix]
            if len(ref):
                d["reference_features"]["systematic"][k] = float(value[ref].mean())
        if output == "Mz" and r["run"] in yi:
            ii, ri = source["indices"]["yaw"], source["indices"]["yaw_reference"]
            yi[r["run"]]["route"][route]["augmented"].update(
                {k: a.reference_contrast(v, ii, ri) for k, v in bank.items()})
        available.update(bank)
    return available, restored, expected


def compare_family(runs, yaw_runs, profile, cfg, output, route, fold, family, old, available, restored,
                   latest_pools=None):
    ids = set(old["training_runs"])
    conditions = {r["run"]:r["condition_id"] for r in runs}
    ordered = sorted({conditions[k] for k in ids}, key=lambda k:hashlib.sha256(str(k).encode()).hexdigest())
    inner = {k:i%3 for i,k in enumerate(ordered)}
    # Same intercept eligibility as before. No new intercept in a contrast-only model.
    intercept = {"intercept"} if "intercept" in old["feature_names"] else set()
    pools = dict(old_pool=list(old["feature_names"]), restored=sorted(restored|intercept))
    for degree in (2,3,4):
        pools[f"cross{degree}"] = sorted(restored|intercept|set(s.uniform_names(degree))|set(s.local_names()))
    for key, values in (latest_pools or {}).items():
        pools["latest_"+key] = sorted((set(values)-{"intercept"})|intercept)
    choices = ("old_structure", *pools)
    names = sorted(available|intercept)
    scores = defaultdict(list)
    for split in range(3):
        train = {k for k in ids if inner[conditions[k]] != split}
        valid = ids-train
        matrix = matrix_for(runs,yaw_runs,profile,cfg,output,route,family,old,names,train)
        fitted = fitted_choices(matrix,pools,old,output,family)
        del matrix
        used = sorted({k for model in fitted.values() for k in model["feature_names"]})
        validation = matrix_for(runs,yaw_runs,profile,cfg,output,route,family,old,used,valid)
        for key, value in score_models(validation,fitted).items():
            scores[key].append(value)
        del validation
    means = {k:float(np.mean(v)) for k,v in scores.items()}
    control = min(CHOICES[:2], key=lambda k:(means[k],CHOICES.index(k)))
    expanded = min(choices, key=lambda k:(means[k],choices.index(k)))
    matrix = matrix_for(runs,yaw_runs,profile,cfg,output,route,family,old,names,ids)
    fitted = fitted_choices(matrix,pools,old,output,family)
    audit = dict(output=output,route=route,fold=str(fold),family=family,
        inner_conditions=inner,inner_scores=means,control=control,expanded=expanded,
        candidate_counts={k:len(v) for k,v in pools.items()},
        ridge=old["ridge"],loss=old.get("loss"),predictive_candidate=old.get("predictive_candidate"),
        local_penalty=old.get("local_penalty"),model_discrepancy_sd=old.get("model_discrepancy_sd",0.),
        scoring="mean of 3 inner-fold legacy-weighted validation MSEs; not a new absolute/delta mixture",
        training_mse={k:v["ablation_training_mse"] for k,v in fitted.items()})
    for model in fitted.values():
        model["ablation_search"] = audit
    print("MATCHED",output,route,fold,family,control,expanded,
          "old/train",fitted["old_structure"]["ablation_training_mse"],
          "new/train",fitted[expanded]["ablation_training_mse"],flush=True)
    return dict(replay=fitted["old_structure"],control=fitted[control],expanded=fitted[expanded]),audit


def preserve_transferred_references(previous_summary, previous_arrays, rows, arrays):
    """Keep saved transferred zeros AND donor-model offsets fixed in this ablation.

    A new dictionary must not manufacture new observations by refitting the
    donor transfer. The saved constant prediction offset is part of that frozen
    legacy transfer. This is assisted validation, not an independent anchor.
    """
    for old in previous_summary["per_run"]:
        if not old["reference"].startswith("assisted_"):
            continue
        p = f'{old["mode"]}/{old["output"]}/{old["route"]}/{old["run"]}/'
        if p+"absolute_prediction" not in arrays:
            continue
        q = p+old["reference"]+"/"
        correction = previous_arrays[q+"prediction"]-previous_arrays[p+"absolute_prediction"]
        np.testing.assert_allclose(correction, correction[0], atol=1e-10, rtol=0)
        prediction = arrays[p+"absolute_prediction"]+correction[0]
        for field in ("truth", "time", "physical"):
            arrays[q+field] = previous_arrays[q+field].copy()
        arrays[q+"prediction"] = prediction
        row = dict(old, identified=a.error_metrics(arrays[q+"truth"], prediction))
        rows.append(row)


def main(args):
    import __main__
    __main__.Settings = a.Settings
    if args.output.exists():
        raise ValueError("Use a new output directory; preserve completed results")
    args.output.mkdir(parents=True)
    # SECURITY: only our own saved preparation checkpoint is accepted by the operator.
    with args.trusted_prepared_checkpoint.open("rb") as stream:
        original_args,profile,cfg,runs,yaw_runs,transfer = pickle.load(stream)
    old_models = a.read_json(args.previous/"models.json")
    outputs = tuple(args.outputs)
    results = {key:deepcopy(old_models) for key in ("replay","control","expanded")}
    if args.models_from:
        results = {key:a.read_json(args.models_from/key/"models.json") for key in results}
    audits = []
    for output in outputs:
        for route in a.ROUTES:
            families = ("foundation",) if output == "Fz" else ("foundation","increment")
            folds = ("final",*sorted({r["fold"] for r in runs}))
            templates = {(str(fold),family): old_models[model_key(output,route,fold,family)[0]][
                model_key(output,route,fold,family)[1]] for fold in folds for family in families}
            legacy = {k for old in templates.values() for k in old["feature_names"]}
            latest_pools = None
            if args.latest_dictionary:
                from scripts import six_axis_theory_bank as theory
                available,restored,latest_pools = attach_latest_dictionary(
                    runs,yaw_runs,cfg,output,route,legacy)
            else:
                available,restored = attach_dictionary(runs,yaw_runs,cfg,output,route,legacy)
            if args.models_from:
                continue
            for fold in folds:
                for family in families:
                    old = templates[str(fold),family]
                    if args.latest_dictionary:
                        old = dict(old, bounds={**old.get("bounds", {}),
                            **theory.coefficient_bounds(cfg, output, available)})
                    if args.verify_only:
                        matrix = matrix_for(runs,yaw_runs,profile,cfg,output,route,family,old,
                                            old["feature_names"],old["training_runs"])
                        model = fit_selected(matrix,old["feature_names"],old,output,family)
                        fitted = {arm:model for arm in results}
                        audit = dict(output=output,route=route,fold=str(fold),family=family,verify_only=True)
                        print("REPLAY",output,route,fold,family,flush=True)
                    else:
                        fitted,audit = compare_family(runs,yaw_runs,profile,cfg,output,route,fold,family,
                                                     old,available,restored,latest_pools)
                    audits.append(audit)
                    group,key = model_key(output,route,fold,family)
                    for arm,model in fitted.items():
                        results[arm][group][key] = model
                        if output == "Fz":
                            results[arm]["increment"][key] = deepcopy(model)
                    a.write_json(args.output/"progress.json",dict(completed=key,search=audits))
            # Retain only bases used by saved models for prediction, not all 1,000 columns.
            keep = {k for arm in results.values() for fold in folds for family in families
                    for k in arm[model_key(output,route,fold,family)[0]][
                        model_key(output,route,fold,family)[1]]["feature_names"]}
            for r in runs:
                if output in r["channels"]:
                    d = r["channels"][output][route]
                    for field in ("features","reference_features"):
                        d[field]["systematic"] = {k:v for k,v in d[field]["systematic"].items() if k in keep}
            if output == "Mz":
                for r in yaw_runs:
                    d = r["route"][route]
                    d["augmented"] = {k:v for k,v in d["augmented"].items() if k in keep}
            gc.collect()
    # Output is numerical only; figure generation remains a separate workflow.
    import analysis_output as report
    old_summary,old_arrays = report.load_results(args.previous)
    for arm,models in results.items():
        if args.verify_only and arm != "replay":
            continue
        directory = args.output/arm
        directory.mkdir()
        if set(outputs) <= set(a.EXTRA_CHANNELS):
            rows,arrays = a.evaluate_extra_channels(runs,models["extra"],outputs=outputs)
        else:
            rows,arrays = a.evaluate(runs,yaw_runs,models)
            preserve_transferred_references(old_summary,old_arrays,rows,arrays)
        summary = dict(result_schema="fftgyro-analysis-result-v1",profile=profile["schema"],
            per_run=rows,metrics=a.summarize(rows),
            arm=arm,version=VERSION,outputs=outputs,preparation_checkpoint=str(args.trusted_prepared_checkpoint),
            source_models=str(args.previous/"models.json"),latest_dictionary=args.latest_dictionary,limitations=[
                "Legacy per-outer-fold ridge/loss/structure are frozen development choices, not new blind CV.",
                "Old-pool control only reselects previously selected terms; expanded includes restored dictionaries.",
                "Absolute and increment have separate LEGACY losses, not a newly normalized joint score.",
                "No new zero correction, model-assisted reference, sample exclusion or default-model promotion."])
        np.savez_compressed(directory/"predictions.npz",**arrays)
        a.write_json(directory/"models.json",models)
        a.write_json(directory/"summary.json",summary)
        report.compare_previous(args.previous,summary,arrays,directory)
        if arm == "replay":
            for key,value in arrays.items():
                if key in old_arrays:
                    np.testing.assert_allclose(value,old_arrays[key],atol=2e-8,rtol=0,
                                               err_msg="Legacy replay mismatch: "+key)
        del arrays
    a.write_json(args.output/"search.json",dict(version=VERSION,search=audits))
    print("DONE",args.output,flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trusted-prepared-checkpoint",type=Path,required=True)
    parser.add_argument("--previous",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--outputs",nargs="+",choices=a.OUTPUTS,default=a.CHANNELS)
    parser.add_argument("--verify-only",action="store_true",help="Verify old coefficients/predictions before expensive search")
    parser.add_argument("--latest-dictionary",action="store_true",
                        help="Add latest dimensional physical/per-rotor pools; keep legacy loss")
    parser.add_argument("--models-from", type=Path,
                        help="Reevaluate completed arm models without repeating term selection")
    main(parser.parse_args())
