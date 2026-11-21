"""Select independent A/B terms by fitting their actual deployed composition.

Absolute targets exist only for independently anchored runs. All eligible runs
can constrain B through differences. No run offset is estimated here.
"""
from collections import defaultdict
from copy import deepcopy
import hashlib

import numpy as np

import analysis as a
import analysis_search as s
from scripts import joint_force_fit as j
from scripts import compare_candidate_dictionary as comparison
from scripts.refit_theory_forces import RIDGES, STAGES

VERSION = "physical-force-joint-coefficients-v1"


def models_for(train, names, pools, ridges, noise=1e-12):
    labels = j.label_names(names)
    norm = s.scales_from_training(train, noise)
    stats = j.joint_statistics(train, names, norm)
    models = {}
    for pool_a, aa in pools.items():
        for pool_d, dd in pools.items():
            choices = ["intercept" if k == "intercept" else "A::"+k for k in aa]
            choices += ["B::"+k for k in dd if k != "intercept"]
            small = j.subset_stats(stats, labels, choices)
            selected, selection = s.gram_select(small, choices)
            for ra in ridges:
                for rd in ridges:
                    pair = j.fit_joint(stats, labels, selected, ra, rd,
                        [c["run"] for c in train], [c["run"] for c in train if c["anchored"]])
                    for model, pool in zip(pair, (pool_a, pool_d)):
                        model.update(candidate_pool=pool, selection=selection,
                                     dictionary_version=VERSION, coefficient_fit="joint_composition")
                    models[pool_a, ra, pool_d, rd] = pair
    return models, norm


def joint_coefficients(pair, names):
    aa, dd = pair
    return np.array([aa["coefficient_by_name"].get(k, 0.) for k in names]
                    + [dd["coefficient_by_name"].get(k, 0.) for k in names if k != "intercept"])


def score_models(validation, models, names, norm):
    # Score actual validation rows, not any future augmented training rows.
    validation = [dict(c, train_dx=c["dx"], train_dy=c["dy"],
                       weights=np.full(len(c["dy"]), 1/len(c["dy"])))
                  for c in validation if c["validation"]]
    sa, sd = j.component_statistics(validation, names)
    coefficients = np.column_stack([joint_coefficients(pair, names) for pair in models.values()])

    def error(stat):
        value = np.sum(coefficients*(stat["g"]@coefficients), axis=0)
        value += stat["yy"]-2*stat["h"]@coefficients
        if value.min() < -1e-8:
            raise ValueError("Negative validation SSE beyond roundoff")
        return np.maximum(value, 0.)

    absolute, delta = error(sa), error(sd)
    return {key: dict(absolute_mse=float(av), increment_mse=float(dv),
                     joint=float(.5*(av/norm[0]+dv/norm[1])))
            for key, av, dv in zip(models, absolute, delta)}


def search_channel(cases, names, pools, output, route, *, ridges=RIDGES, stages=STAGES,
                   folds=None, noise=1e-12):
    condition_folds = defaultdict(set)
    for c in cases:
        condition_folds[c["condition"]].add(c["fold"])
    if any(len(v) != 1 for v in condition_folds.values()):
        raise ValueError("A condition spans outer folds")
    fitted, audits = {stage: {} for stage in stages}, []
    folds = folds if folds is not None else ("final", *sorted({c["fold"] for c in cases}))
    for fold in folds:
        train = [c for c in cases if fold == "final" or c["fold"] != fold]
        conditions = sorted({c["condition"] for c in train}, key=lambda k:
                            hashlib.sha256((s.VERSION+str(k)).encode()).hexdigest())
        inner = {k: i % 3 for i, k in enumerate(conditions)}
        scores = defaultdict(list)
        for split in range(3):
            tt = [c for c in train if inner[c["condition"]] != split]
            vv = [c for c in train if inner[c["condition"]] == split]
            models, norm = models_for(tt, names, pools, ridges, noise)
            for key, value in score_models(vv, models, names, norm).items():
                scores[key].append(value)
        full, _ = models_for(train, names, pools, ridges, noise)
        means = {k: {m: float(np.mean([v[m] for v in values])) for m in values[0]}
                 for k, values in scores.items()}
        for stage, allowed in stages.items():
            eligible = [k for k in scores if k[0] in allowed and k[2] in allowed]
            winner = min(eligible, key=lambda k: (means[k]["joint"], k))
            pair = deepcopy(full[winner])
            audit = dict(stage=stage, fold=fold, output=output, route=route,
                coefficient_fit="joint_composition", inner_conditions=inner,
                chosen=dict(foundation_pool=winner[0], foundation_ridge=winner[1],
                            increment_pool=winner[2], increment_ridge=winner[3]),
                chosen_score=means[winner],
                objective="equal absolute/delta normalized condition-MSE of deployed composition",
                inner_scores=[dict(foundation_pool=k[0], foundation_ridge=k[1],
                    increment_pool=k[2], increment_ridge=k[3], **means[k]) for k in eligible])
            for model in pair:
                model.update(search=audit, output=output, zero_updated=False)
            fitted[stage][str(fold)] = pair
            audits.append(audit)
            print("JOINT", stage, output, route, fold, winner,
                  "terms", *(len(m["feature_names"]) for m in pair), flush=True)
    return fitted, audits


def verify_pair(pair, cases, names, noise=1e-12):
    """Independent tall-design QR/SVD check of the sufficient-statistics fit."""
    norm = s.scales_from_training(cases, noise)
    blocks = j.joint_matrix(cases, names, pair[0]["feature_names"], pair[1]["feature_names"], norm)
    if not blocks.names:
        return pair
    compressed = a.compress_regression_blocks(blocks)
    scale = np.linalg.norm(compressed.x, axis=0)
    penalty = [0. if name == "intercept" else pair[int(name.startswith("B::"))]["ridge"]
               for name in compressed.names]
    design = np.vstack([compressed.x/scale, np.diag(np.sqrt(penalty))])
    target = np.r_[compressed.y, np.zeros(len(scale))]
    coef = np.linalg.lstsq(design, target, rcond=None)[0]/scale
    expected = np.array(pair[0]["coefficients"]+pair[1]["coefficients"])
    np.testing.assert_allclose(compressed.x@coef, compressed.x@expected, atol=1e-8, rtol=1e-7)
    diagnostics = comparison.correlation_diagnostics(compressed)
    offset = 0
    for model in pair:
        count = len(model["feature_names"])
        model["coefficients"] = coef[offset:offset+count].tolist()
        model["coefficient_by_name"] = dict(zip(model["feature_names"], model["coefficients"]))
        model["scaled_coefficients"] = (coef[offset:offset+count]*scale[offset:offset+count]).tolist()
        model["joint_correlation_diagnostics"] = diagnostics
        model["solver"] = dict(method="verified_tall_joint_design_QR_SVD", success=True)
        offset += count
    return pair
