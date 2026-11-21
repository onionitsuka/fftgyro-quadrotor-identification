"""Opt-in dictionary audit/search; reconstruction and ordinary replay stay in analysis.py.

1. Restore the complete CURRENT dictionaries (not just previously selected terms).
2. Add identical flow/rate/control polynomial rules to every output.
3. Select terms on training data, tune degree/ridge by inner condition CV.
4. Score the deployed absolute composition AND its reference increment.

This is repeated-development nested CV, not a fresh blind experiment. No load,
run identifier, zero correction, or measured RPM from the other route is a feature.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
import hashlib
import numpy as np
from scipy.optimize import lsq_linear

import analysis as a

VERSION = "restored-uniform-cross-v1"
FLOW = ("mu_x", "mu_y", "mu_z")
MOTION = ("pbar", "qbar", "rbar", "up", "uq", "ur")
LINEAR_MOTION = (*MOTION, *("abs_"+k for k in MOTION))
SPECS = ("selected_control", "restored", "cross2", "cross3", "cross4")
RIDGES = (0., .001, .01, .1, 1.)


def retain_state(data, selected, reference, yaw, yaw_reference):
    """Keep only requested samples; reference polynomials are averaged AFTER evaluation."""
    reference = np.flatnonzero(reference) if np.asarray(reference).dtype == bool else reference
    indices = np.unique(np.concatenate([selected, reference,
        yaw if yaw is not None else [], yaw_reference if yaw_reference is not None else []]).astype(int))
    compact = {k: np.asarray(data[k])[indices] for k in (
        "physical", "force_scale", "moment_scale", "rpm", "air_frd_m_s",
        "omega_body_rad_s", "rms_speed_ratio")}
    compact["state"] = {k: v[indices] for k, v in data["state"].items()}
    compact["rotor_flow"] = {k: v if k == "positions_frd_m" else v[indices]
                             for k, v in data["rotor_flow"].items()}
    compact["indices"] = dict(selected=np.searchsorted(indices, selected),
                              reference=np.searchsorted(indices, reference))
    if yaw is not None:
        compact["indices"].update(yaw=np.searchsorted(indices, yaw),
                                  yaw_reference=np.searchsorted(indices, yaw_reference))
    return compact


def term_name(term):
    return "*".join(k + (f"^{p}" if p != 1 else "") for k, p in term) or "1"


def uniform_names(degree):
    """P_d(flow)*(1 + six signed motions + their magnitudes), plus
    P_(d-2)(flow)*all quadratic signed-motion products.

    All components/permutations have identical eligibility. d=2,3,4; highest
    total degree is d+1. This is a finite block-polynomial dictionary, NOT all
    monomials of arbitrary degree and NOT Sun's exact output-specific dictionary.
    """
    flow = a.monomials(FLOW, degree)
    motion = [t for t in a.monomials(MOTION, 2) if sum(p for _, p in t) == 2]
    terms = list(flow)
    terms += [f + ((m, 1),) for f in flow for m in LINEAR_MOTION]
    terms += [f + m for f in a.monomials(FLOW, degree - 2) for m in motion]
    return ["cross__" + term_name(t) for t in terms]


def local_names():
    # Same individual-rotor normalization and all six motions for every rotor/axis.
    return [f"localflow__{i}__{k}__{m}" for i in range(4)
            for k in range(3) for m in ("1", *LINEAR_MOTION)]


def restored_bank(data, cfg, output, old_names=()):
    """All available dictionaries, including formerly rejected and old selected terms."""
    scale = data["force_scale"] if output.startswith("F") else data["moment_scale"]
    if output in a.EXTRA_CHANNELS:
        bank = a.extra_feature_bank(data, output, "expanded", cfg)
    else:
        bank = {}
        for spec in ("compact", "expanded"):
            bank.update(a.evaluate_features(data["state"], data["physical"][:, a.OUTPUTS.index(output)],
                                            scale, output, spec, cfg))
        if output in ("My", "Mz"):
            bank.update(a.speed_features(data, output))
        if output == "Mz":
            bank.update(a.continuous_bank(data))
            bank.update(a.physics_bank(data, cfg, a.SIGNS))
            historical = ["1*"+region for region in a.SUN_REGIONS]
            historical += [term_name(t+(("rbar", 1), (region, 1)))
                for region in a.SUN_REGIONS for t in a.monomials(FLOW, 3)]
            historical += ["rbar_absrbar*"+region for region in a.SUN_REGIONS]
            bank.update(a.evaluate_names(data["state"], scale, historical))
        if output == "My":
            # Complete previous sun_pitch/dynamic_update pools, not their selected subset.
            historical = [term_name(f+side+motion)
                for f in a.monomials(("mu_x", "mu_z"), 5)
                for side in a.monomials(("abs_mu_y",), 2)
                for motion in ((), (("qbar", 1),), (("uq", 1),))]
            bank.update(a.evaluate_names(data["state"], scale, historical))
    bank.update(a.theory_feature_bank(data, cfg, output))
    # Preserve historical chosen columns not enumerated by today's small dictionary.
    missing = [k for k in old_names if k not in bank and k != "intercept"]
    if missing:
        bank.update(a.evaluate_names(data["state"], scale, missing))
    # Fz's previous generator explicitly suppressed mu_z^2. Restore it too.
    if output == "Fz":
        bank["mu_z^2"] = scale * data["state"]["mu_z"]**2
    return bank


def expanded_bank(data, cfg, output, old_names=()):
    bank = restored_bank(data, cfg, output, old_names)
    restored = set(bank)
    scale = data["force_scale"] if output.startswith("F") else data["moment_scale"]
    terms = uniform_names(4)
    values = a.evaluate_names(data["state"], scale, [k.removeprefix("cross__") for k in terms])
    bank.update({k: values[k.removeprefix("cross__")] for k in terms})
    for name in local_names():
        _, i, k, m = name.split("__")
        value = (scale * data["rotor_flow"]["omega_squared_weight"][:, int(i)] / 4
                 * data["rotor_flow"]["mu"][:, int(i), int(k)])
        bank[name] = value if m == "1" else value * data["state"][m]
    return bank, restored


def statistics(x, y):
    """Additive sufficient statistics; N and sums retain the original F/PSE definition."""
    return dict(g=x.T @ x, h=x.T @ y, yy=float(y @ y), sx=x.sum(axis=0),
                sy=float(y.sum()), n=len(y))


def add_statistics(parts):
    if not parts:
        raise ValueError("No training conditions")
    return {k: sum(p[k] for p in parts) for k in parts[0]}


def subset_statistics(s, ids):
    return dict(s, g=s["g"][np.ix_(ids, ids)], h=s["h"][ids], sx=s["sx"][ids])


def gram_select(s, names):
    """Same forward/backward F/PSE rules as analysis.select_extra_terms.

    Sufficient statistics avoid repeated tall QR for hundreds of correlated
    candidates. Rank is checked on normalized Gram matrices (squared condition
    number); near-zero residual columns are rejected, never silently sampled out.
    """
    n = s["n"]
    scales = np.sqrt(np.maximum(np.diag(s["g"]), 0))
    safe = np.maximum(scales, 1e-30)
    g, h, sx = s["g"] / np.outer(safe, safe), s["h"] / safe, s["sx"] / safe
    yy, sy = s["yy"], s["sy"]
    variance = max(0., yy/n - (sy/n)**2)
    active = [i for i, k in enumerate(names) if k == "intercept"]
    required = set(active)
    pool = {int(i) for i in np.flatnonzero(scales > 1e-25)} - required
    history, reason = [], "candidate_pool_exhausted"

    def solve(ids):
        if not ids:
            return np.empty(0), yy, yy/n
        mat = g[np.ix_(ids, ids)]
        eigen = np.linalg.eigvalsh(mat)
        if eigen[0] <= max(1e-12*eigen[-1], 1e-14):
            raise ValueError("Numerically dependent columns")
        coef = np.linalg.solve(mat, h[ids])
        sse = max(0., yy - 2*coef@h[ids] + coef@mat@coef)
        return coef, sse, sse/n + variance*len(ids)/n

    coef, sse, previous = solve(active)
    initial = previous
    while pool and n > len(active)+1:
        ids = sorted(pool)
        cross = g[np.ix_(active, ids)]
        projection = np.linalg.solve(g[np.ix_(active, active)], cross) if active else cross
        residual_h = h[ids] - (cross.T @ coef if active else 0.)
        residual_sum = sy - (sx[active]@coef if active else 0.)
        column_sum = sx[ids] - (sx[active]@projection if active else 0.)
        norm2 = np.diag(g)[ids] - (np.sum(cross*projection, axis=0) if active else 0.)
        centered_norm2 = norm2 - column_sum**2/n
        target_norm2 = max(0., sse-residual_sum**2/n)
        if target_norm2 <= max(1e-26, yy*1e-14):
            reason = "zero_residual_variance"
            break
        corr = np.where(centered_norm2 > 1e-12,
            np.abs(residual_h-column_sum*residual_sum/n) /
            np.sqrt(np.maximum(centered_norm2*target_norm2, 1e-300)), -np.inf)
        if not np.isfinite(corr).any():
            reason = "no_independent_correlated_candidates"
            break
        j = ids[int(np.argmax(corr))]
        before = active.copy()
        try:
            _, newsse, _ = solve(active+[j])
        except ValueError:
            pool.remove(j)
            history.append(dict(action="numerical_rejection", column=j))
            continue
        df = n-len(active)-1
        denominator = max(newsse/df, yy*1e-15/df, np.finfo(float).tiny)
        fin = max(0., sse-newsse)/denominator
        if fin <= 5:
            reason = "forward_F_not_above_5"
            break
        pool.remove(j)
        active.append(j)
        backward = [np.inf if k in required else
            max(0., solve([v for v in active if v != k])[1]-newsse) /
            denominator for k in active]
        out = active[int(np.argmin(backward))] if min(backward) < 4 else None
        if out is not None:
            active.remove(out)
        coef, sse, pse = solve(active)
        accepted = pse < .99*previous
        history.append(dict(added=j, removed=out, F_in=float(fin), pse=float(pse), accepted=bool(accepted)))
        if not accepted:
            active = before
            reason = "PSE_not_1_percent_better_rollback"
            break
        previous = pse
        if pse < .001*initial:
            reason = "PSE_below_0_001_initial"
            break
    return [names[i] for i in active], dict(candidate_names=list(names), history=history,
        stop_reason=reason, training_rows=n, method="Sun_F_PSE_sufficient_statistics",
        F_in=5, F_out=4, PSE_ratio=.99, term_cap=None,
        numerical_policy="normalized Gram min/max eigenvalue > 1e-12; near dependent columns rejected")


def fit_stats(s, names, selected, ridge, train_ids, bounds=None):
    ids = [names.index(k) for k in selected]
    if not ids:
        return dict(feature_names=[], coefficients=[], coefficient_by_name={}, column_scales=[],
                    scaled_coefficients=[], ridge=ridge, training_runs=sorted(train_ids), spec="systematic")
    ss = subset_statistics(s, ids)
    scale = np.sqrt(np.diag(ss["g"]))
    g = ss["g"]/np.outer(scale, scale)
    penalty = np.array([0. if k == "intercept" else 1. for k in selected])
    bounds = {k: v for k, v in (bounds or {}).items() if k in selected}
    if bounds:
        chol = np.linalg.cholesky(g)
        design = np.vstack([chol.T, np.sqrt(ridge)*np.diag(penalty)])
        target = np.r_[np.linalg.solve(chol, ss["h"]/scale), np.zeros(len(selected))]
        lower = [bounds.get(k, (-np.inf, np.inf))[0]*v for k, v in zip(selected, scale)]
        upper = [bounds.get(k, (-np.inf, np.inf))[1]*v for k, v in zip(selected, scale)]
        fit = lsq_linear(design, target, bounds=(lower, upper), tol=1e-10, max_iter=1000)
        if not fit.success:
            raise RuntimeError(fit.message)
        coef = fit.x
    else:
        coef = np.linalg.solve(g + ridge*np.diag(penalty), ss["h"]/scale)
    eig = np.linalg.eigvalsh(g)
    return dict(feature_names=selected, coefficients=(coef/scale).tolist(),
                coefficient_by_name=dict(zip(selected, (coef/scale).tolist())),
                column_scales=scale.tolist(), scaled_coefficients=coef.tolist(), ridge=ridge,
                numerical_rank=int(np.sum(eig > eig[-1]*1e-12)),
                normalized_condition_number=float(np.sqrt(eig[-1]/eig[0])),
                training_runs=sorted(train_ids), spec="systematic", zero_updated=False,
                bounds={k:list(v) for k, v in bounds.items()},
                solver=dict(method="normalized_sufficient_statistics", success=True),
                frozen_structure=False, dictionary_version=VERSION)


def channel_cases(runs, yaw_runs, route, output, names):
    """Construct original absolute/contrast rows; unknown zeros ONLY enter contrasts."""
    yi = {r["run"]: r for r in yaw_runs}
    cases = []
    for r in runs:
        if output not in r["channels"]:
            continue
        ch = r["channels"][output]
        if not np.any(ch["take"]):
            continue
        d, take = ch[route], ch["take"]
        xx = np.column_stack([np.ones(int(take.sum())) if k == "intercept" else
                              d["features"]["systematic"][k][take] for k in names])
        y = ch["truth"][take]-d["physical"][take]
        if ch["reference_available"]:
            xr = np.array([1. if k == "intercept" else d["reference_features"]["systematic"][k] for k in names])
            yr = ch["reference"]-d["reference_physical"]
        else:
            # Retain the original centered training-only run (e0343). No new
            # initial reference is claimed, and it remains outside scored rows.
            xr, yr = xx.mean(axis=0), float(y.mean())
        dx, dy = xx-xr, y-yr
        train_dx, train_dy, weights = dx, dy, np.full(len(dy), 1/len(dy))
        if output == "Mz":
            yy = yi[r["run"]]
            yd = yy["route"][route]
            train_dx = np.column_stack([np.zeros(len(yy["y"])) if k == "intercept" else
                                        yd["augmented"][k] for k in names])
            train_dy = yy["y"]-yd["physical"]
            n = yy["original_count"]
            # Existing preparation-stage weight 0.25 is retained, duration does not dominate.
            weights = np.r_[np.full(n, 1/n), np.full(len(train_dy)-n, .25/max(1, len(train_dy)-n))]
        if not all(np.isfinite(v).all() for v in (xx, xr, y, dx, dy, train_dx, train_dy)):
            raise ValueError("Nonfinite candidate on frozen valid samples: "+r["run"])
        cases.append(dict(run=r["run"], condition=r["condition_id"], fold=r["fold"], anchored=r["anchored"],
            validation=ch["reference_available"],
            x=xx, xr=xr, y=y, dx=dx, dy=dy, train_dx=train_dx, train_dy=train_dy, weights=weights))
    return cases


def grouped_stats(cases, family):
    groups = defaultdict(list)
    for c in cases:
        if family == "delta" or c["anchored"]:
            groups[c["condition"]].append(c)
    result = {}
    for condition, group in groups.items():
        xx, yy = [], []
        for c in group:
            if family == "absolute":
                x, y, weights = c["x"], c["y"], np.full(len(c["y"]), 1/len(c["y"]))
            else:
                x, y, weights = c["train_dx"], c["train_dy"], c["weights"]
            w = np.sqrt(weights/len(group))
            xx.append(x*w[:, None]); yy.append(y*w)
        result[condition] = statistics(np.vstack(xx), np.concatenate(yy))
    return result


def scales_from_training(cases, noise):
    """Own-route physical residuals set training-only comparison units; no route leakage."""
    result = []
    for key in ("y", "dy"):
        groups = defaultdict(list)
        for c in cases:
            if key == "dy" or c["anchored"]:
                groups[c["condition"]].append(float(np.mean(c[key]**2)))
        result.append(max(noise**2, float(np.mean([np.mean(v) for v in groups.values()]))))
    return result


def prediction(c, model, names, kind):
    ids = [names.index(k) for k in model["feature_names"]]
    return c[kind][..., ids] @ np.asarray(model["coefficients"])


def pair_scores(cases, models_a, models_d, names, norm, same=False):
    scores = {}
    for ra, ma in models_a.items():
        for rd, md in models_d.items():
            if same and ra != rd:
                continue
            absolute, delta = defaultdict(list), defaultdict(list)
            for c in cases:
                if not c.get("validation", True):
                    continue
                pd = prediction(c, md, names, "dx")
                pa = prediction(c, ma, names, "xr") + pd
                delta[c["condition"]].append(float(np.mean((pd-c["dy"])**2)))
                if c["anchored"]:
                    absolute[c["condition"]].append(float(np.mean((pa-c["y"])**2)))
            if not absolute or not delta:
                raise ValueError("Inner split lacks absolute or increment validation")
            av = float(np.mean([np.mean(v) for v in absolute.values()]))
            dv = float(np.mean([np.mean(v) for v in delta.values()]))
            scores[ra, rd] = dict(absolute_mse=av, increment_mse=dv,
                                  joint=.5*(av/norm[0]+dv/norm[1]))
    return scores


def fit_channel(cases, names, candidate_sets, old_selected, noise, output, route):
    stats_a, stats_d = grouped_stats(cases, "absolute"), grouped_stats(cases, "delta")
    models, audits = {}, []
    for fold in ("final", *sorted({c["fold"] for c in cases})):
        train = [c for c in cases if fold == "final" or c["fold"] != fold]
        conditions = sorted({c["condition"] for c in train},
            key=lambda k: hashlib.sha256((VERSION+str(k)).encode()).hexdigest())
        inner = {k: i % 3 for i, k in enumerate(conditions)}
        scores = defaultdict(list)

        def candidates(tt, spec):
            conditions_t = {c["condition"] for c in tt}
            sa = add_statistics([v for k, v in stats_a.items() if k in conditions_t])
            sd = add_statistics([v for k, v in stats_d.items() if k in conditions_t])
            norm = scales_from_training(tt, noise)
            if output == "Fz":
                # One Fz function; normalized absolute + contrast sums, equal
                # weights per condition within each available target type.
                combined = {}
                for k in sa:
                    if k == "n": combined[k] = sa[k]+sd[k]
                    elif k in ("sx", "sy"): combined[k] = sa[k]/np.sqrt(2*norm[0])+sd[k]/np.sqrt(2*norm[1])
                    else: combined[k] = sa[k]/(2*norm[0])+sd[k]/(2*norm[1])
                sa = sd = combined
            fitted, selections = [], []
            for family, stat in (("foundation", sa), ("increment", sd)):
                if output == "Fz" and fitted:
                    fitted.append(fitted[0]); selections.append(selections[0]); break
                choices = old_selected[str(fold)][family] if spec == "selected_control" else candidate_sets[spec]
                choices = sorted(set(choices) & set(names))
                if family == "increment" and output != "Fz":
                    choices = [k for k in choices if k != "intercept"]
                # selected_control restricts the candidate POOL, not the final
                # structure: it too is reselected inside the training split.
                selected, audit = gram_select(subset_statistics(stat, [names.index(k) for k in choices]), choices)
                ids = [c["run"] for c in tt if family == "increment" or c["anchored"] or output == "Fz"]
                fitted.append({ridge: fit_stats(stat, names, selected, ridge, ids,
                    bounds={"prop_gain":(-.5, .5)} if output == "Fz" else None) for ridge in RIDGES})
                selections.append(audit)
            return fitted, selections, norm

        for split in range(3):
            tt = [c for c in train if inner[c["condition"]] != split]
            vv = [c for c in train if inner[c["condition"]] == split]
            for spec in SPECS:
                fitted, _, norm = candidates(tt, spec)
                for (ra, rd), metric in pair_scores(vv, *fitted, names, norm, output == "Fz").items():
                    scores[spec, ra, rd].append(metric)
        winner = min(scores, key=lambda k: (np.mean([s["joint"] for s in scores[k]]), SPECS.index(k[0]), k[1:]))
        spec, ra, rd = winner
        fitted, selection, norm = candidates(train, spec)
        ma, md = fitted[0][ra], fitted[1][rd]
        audit = dict(output=output, route=route, fold=fold, chosen=dict(spec=spec, absolute_ridge=ra, delta_ridge=rd),
            inner_conditions=inner, normalization="training own-route physics residual MSE; floor=noise^2",
            objective="0.5*absolute_condition_MSE/train_absolute_scale + 0.5*delta_condition_MSE/train_delta_scale",
            inner_scores=[dict(spec=s, absolute_ridge=lr, delta_ridge=ld,
                **{k: float(np.mean([m[k] for m in v])) for k in v[0]}) for (s, lr, ld), v in scores.items()])
        for model, sel in zip((ma, md), selection):
            model.update(selection=sel, search=audit, output=output)
        models[str(fold)] = (ma, md)
        audits.append(audit)
        print("SYSTEMATIC", output, route, fold, winner, len(ma["feature_names"]), len(md["feature_names"]), flush=True)
    return models, audits


def fit_systematic(profile, runs, yaw_runs, cal, cfg, outputs, directory):
    """One component/route at a time bounds memory. Keep only selected bases afterward."""
    result = deepcopy(profile["models"])
    result.setdefault("extra", {})
    yaw_index = {r["run"]: r for r in yaw_runs}
    all_audits, dictionary_audit = [], {}
    for output in outputs:
        for route in a.ROUTES:
            legacy = set()
            old_selected = {}
            for fold in ("final", *sorted({r["fold"] for r in runs})):
                key = f"{fold}/{route}/{output}"
                if output in a.EXTRA_CHANNELS:
                    aa = ["intercept", *a.extra_feature_names(output, "expanded")]
                    dd = a.extra_feature_names(output, "expanded")
                else:
                    aa = profile["models"]["foundation"][key]["feature_names"]
                    dd = (profile["models"]["yaw"][f"{fold}/{route}"]["feature_names"] if output == "Mz" else
                          profile["models"]["increment"][key]["feature_names"] if output == "My" else aa)
                old_selected[str(fold)] = dict(foundation=aa, increment=dd)
                legacy.update(aa); legacy.update(dd)
            for r in runs:
                if output not in r["channels"]:
                    continue
                source = r["search_data"][route]
                bank, restored = expanded_bank(source, cfg, output, legacy)
                ix, ref = source["indices"]["selected"], source["indices"]["reference"]
                d = r["channels"][output][route]
                d["features"]["systematic"] = {k: v[ix] for k, v in bank.items()}
                d["reference_features"]["systematic"] = {k: float(v[ref].mean()) for k, v in bank.items()} if len(ref) else {}
                if output == "Mz" and r["run"] in yaw_index:
                    yi, ri = source["indices"]["yaw"], source["indices"]["yaw_reference"]
                    yaw_index[r["run"]]["route"][route]["augmented"].update(
                        {k: a.reference_contrast(v, yi, ri) for k, v in bank.items()})
            names = sorted({"intercept", *bank})
            candidate_sets = dict(restored=sorted({"intercept", *restored}))
            for degree in (2, 3, 4):
                candidate_sets[f"cross{degree}"] = sorted({"intercept", *restored, *uniform_names(degree), *local_names()})
            dictionary_audit[output+"/"+route] = candidate_sets
            cases = channel_cases(runs, yaw_runs, route, output, names)
            # Extra-axis bundle has no independently estimated noise floor.
            # Use only a numerical floor there; do not borrow another axis' noise.
            noise_floor = cal["noise"].get(output, 1e-12)
            fitted, audits = fit_channel(cases, names, candidate_sets, old_selected, noise_floor, output, route)
            all_audits.extend(audits)
            keep = set()
            for fold, (ma, md) in fitted.items():
                key = f"{fold}/{route}/{output}"
                keep.update(ma["feature_names"]); keep.update(md["feature_names"])
                if output in a.EXTRA_CHANNELS:
                    result["extra"][key+"/foundation"] = ma
                    result["extra"][key+"/increment"] = md
                else:
                    result["foundation"][key] = ma
                    if output == "Mz":
                        md.update(bank="augmented", local_penalty=1., extra_ratio=.25)
                        result["yaw"][f"{fold}/{route}"] = md
                    elif output == "My":
                        result["increment"][key] = md
            for r in runs:
                if output not in r["channels"]:
                    continue
                d = r["channels"][output][route]
                for group in ("features", "reference_features"):
                    d[group]["systematic"] = {k: v for k, v in d[group]["systematic"].items() if k in keep}
                if output == "Mz" and r["run"] in yaw_index:
                    yd = yaw_index[r["run"]]["route"][route]
                    yd["augmented"] = {k: v for k, v in yd["augmented"].items() if k in keep}
            del cases
            a.write_json(directory/"systematic-checkpoint.json", result)
            a.write_json(directory/"systematic-progress.json", dict(completed=output+"/"+route,
                dictionaries=dictionary_audit, search=all_audits))
    a.write_json(directory/"systematic-search.json", dict(version=VERSION, dictionaries=dictionary_audit,
        search=all_audits, limitations=["Restored CURRENT dictionary, not every discarded script in repository history.",
        "Condition-equal fitting replaces earlier GLS weighting in this opt-in experiment.",
        "No coefficient-sign or passivity constraint; cross-axis effects may lack excitation.",
        "Fz uses one function with normalized, condition-equal absolute and contrast sums; other channels use foundation + increment.",
        "No model-assisted absolute references enter fitting or independent scoring."]))
    return result


def resume_checkpoint(path):
    """Resume OUR OWN local preparation checkpoint after a code-only fit failure.

    Pickle executes Python: never pass a downloaded or third-party file here.
    Existing completed results are never overwritten.
    """
    import pickle
    import __main__
    from pathlib import Path
    __main__.Settings = a.Settings  # Created when analysis.py was the entry point.
    with Path(path).open("rb") as stream:
        args, profile, cfg, runs, yaw_runs, transfer = pickle.load(stream)
    if (args.output/"summary.json").exists():
        raise ValueError("Completed result already exists")
    outputs = args.search_outputs or (a.OUTPUTS if args.six_axis else a.CHANNELS)
    models = fit_systematic(profile, runs, yaw_runs, profile["calibration"], cfg, outputs, args.output)
    rows, arrays = a.evaluate(runs, yaw_runs, models)
    if args.six_axis:
        extra_rows, extra_arrays = a.evaluate_extra_channels(runs, models["extra"])
        rows.extend(extra_rows); arrays.update(extra_arrays)
    np.savez_compressed(args.output/"predictions.npz", **arrays)
    a.write_json(args.output/"models.json", models)
    a.write_json(args.output/"summary.json", dict(result_schema="fftgyro-analysis-result-v1",
        profile=profile["schema"], signal_runs=len(runs), yaw_training_runs=len(yaw_runs),
        coefficient_refit=True, systematic_search=True, systematic_search_outputs=list(outputs),
        term_selection_repeated=True, per_run=rows, metrics=a.summarize(rows),
        source_profile_sha256=hashlib.sha256((args.bundle/"profile.json").read_bytes()).hexdigest(),
        code_sha256=hashlib.sha256(Path(a.__file__).read_bytes()).hexdigest(),
        systematic_search_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        preparation_checkpoint=str(path),
        zero_recovery=(dict(source=str(args.zero_recovery),
            drift_prior=str(args.drift_prior), calibration_file=str(args.output/"calibration.json"))
            if args.zero_recovery else None),
        limitations=["Own local preparation checkpoint resumed; reconstruction unchanged.",
            "Fx/Fy/Mx use a numerical normalization floor, not a measured noise estimate.",
            "Model-assisted absolute transfer is not recomputed or scored in this search.",
            "Repeated-development grouped OOF, not a new blind experiment."]))
    print("DONE", args.output, flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Resume a trusted, self-created search checkpoint")
    parser.add_argument("--trusted-prepared-checkpoint", required=True)
    resume_checkpoint(parser.parse_args().trusted_prepared_checkpoint)
