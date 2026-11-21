"""Joint force fitting: A(reference) + B(t)-B(reference), without shared coefficients.

Only the supplied training cases enter statistics. Absolute supervision uses
anchored cases; contrasts use every supplied case. Each target type is weighted
equally across conditions, then equally across repeated runs of a condition.
"""
from collections import Counter

import numpy as np

import analysis as a
import analysis_search as s


def label_names(names):
    names = tuple(names)
    if len(names) != len(set(names)):
        raise ValueError("Feature names must be unique")
    return tuple("intercept" if k == "intercept" else "A::" + k for k in names) + tuple(
        "B::" + k for k in names if k != "intercept")


def subset_stats(stat, labels, selected):
    """Keep selected labelled columns in the requested order."""
    if len(selected) != len(set(selected)):
        raise ValueError("Selected labels must be unique")
    return s.subset_statistics(stat, [list(labels).index(k) for k in selected])


def _setup(cases, names):
    if not cases:
        raise ValueError("No training cases")
    label_names(names)
    absolute = Counter(c["condition"] for c in cases if c["anchored"])
    delta = Counter(c["condition"] for c in cases)
    if not absolute:
        raise ValueError("Joint absolute fitting requires anchored training cases")
    return absolute, delta, [i for i, k in enumerate(names) if k != "intercept"]


def _normalization(norm):
    norm = np.asarray(norm, float)
    if norm.shape != (2,) or not np.isfinite(norm).all() or (norm <= 0).any():
        raise ValueError("Two finite positive training normalization values required")
    return norm


def _delta_arrays(c, size, ib):
    x, y, weights = (np.asarray(c[k], float) for k in ("train_dx", "train_dy", "weights"))
    if x.ndim != 2 or x.shape[1] != size or y.shape != (len(x),) or weights.shape != y.shape:
        raise ValueError("Contrast array shape mismatch")
    if not len(y) or not all(np.isfinite(v).all() for v in (x, y, weights)):
        raise ValueError("Contrast arrays must be nonempty and finite")
    if (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Contrast weights must be nonnegative with positive total")
    return x[:, ib], y, weights


def _absolute_arrays(c, size, ib):
    xr, dx, y = (np.asarray(c[k], float) for k in ("xr", "dx", "y"))
    if xr.shape != (size,) or dx.ndim != 2 or dx.shape[1] != size or y.shape != (len(dx),):
        raise ValueError("Absolute array shape mismatch")
    if not len(y) or not all(np.isfinite(v).all() for v in (xr, dx, y)):
        raise ValueError("Absolute arrays must be nonempty and finite")
    return xr, dx[:, ib], y


def joint_matrix(cases, names, pool_a, pool_d, norm):
    """Explicit weighted design, used for independent QR/SVD verification."""
    norm = _normalization(norm)
    counts_a, counts_d, ib = _setup(cases, names)
    labels = label_names(names)
    selected = tuple("intercept" if k == "intercept" else "A::" + k for k in pool_a)
    selected += tuple("B::" + k for k in pool_d if k != "intercept")
    if len(selected) != len(set(selected)):
        raise ValueError("Selected labels must be unique")
    columns = [labels.index(k) for k in selected]
    pa, xx, yy = len(names), [], []
    for c in cases:
        if c["anchored"]:
            xr, dx, y = _absolute_arrays(c, pa, ib)
            w = np.sqrt(.5 / (norm[0] * len(counts_a) * counts_a[c["condition"]] * len(y)))
            x = np.column_stack([np.broadcast_to(xr, (len(y), pa)), dx])
            xx.append(x[:, columns] * w)
            yy.append(y * w)
        dx, y, weights = _delta_arrays(c, pa, ib)
        w = np.sqrt(.5 * weights / (norm[1] * len(counts_d) * counts_d[c["condition"]]))
        x = np.column_stack([np.zeros((len(y), pa)), dx])
        xx.append(x[:, columns] * w[:, None])
        yy.append(y * w)
    return a.RegressionBlocks(np.vstack(xx), np.concatenate(yy), selected, dict(
        training_runs=sorted({c["run"] for c in cases}),
        absolute_training_runs=sorted({c["run"] for c in cases if c["anchored"]}),
        loss="joint_deployed_absolute_and_contrast_condition_equal", normalization=norm.tolist()))


def component_statistics(cases, names):
    """Return absolute/contrast condition-mean losses as separate full block Grams.

    For a labelled coefficient vector c, c'Gc-2c'h+yy equals that component's
    condition-mean squared error. No absolute/delta normalization is applied.
    """
    counts_a, counts_d, ib = _setup(cases, names)
    pa, p = len(names), len(label_names(names))

    def empty():
        return dict(g=np.zeros((p, p)), h=np.zeros(p), yy=0., sx=np.zeros(p), sy=0., n=0)

    sa, sd = empty(), empty()
    for c in cases:
        if c["anchored"]:
            xr, dx, y = _absolute_arrays(c, pa, ib)
            w2 = 1. / (len(counts_a) * counts_a[c["condition"]] * len(y))
            w, n = np.sqrt(w2), len(y)
            sumx, sumy = dx.sum(axis=0), float(y.sum())
            sa["g"][:pa, :pa] += w2*n*np.outer(xr, xr)
            cross = w2*np.outer(xr, sumx)
            sa["g"][:pa, pa:] += cross
            sa["g"][pa:, :pa] += cross.T
            sa["g"][pa:, pa:] += w2*(dx.T @ dx)
            sa["h"][:pa] += w2*xr*sumy
            sa["h"][pa:] += w2*(dx.T @ y)
            sa["yy"] += w2*float(y @ y)
            sa["sx"][:pa] += w*n*xr
            sa["sx"][pa:] += w*sumx
            sa["sy"] += w*sumy
            sa["n"] += n
        dx, y, weights = _delta_arrays(c, pa, ib)
        w = np.sqrt(weights / (len(counts_d)*counts_d[c["condition"]]))
        xw, yw = dx*w[:, None], y*w
        sd["g"][pa:, pa:] += xw.T @ xw
        sd["h"][pa:] += xw.T @ yw
        sd["yy"] += float(yw @ yw)
        sd["sx"][pa:] += xw.sum(axis=0)
        sd["sy"] += float(yw.sum())
        sd["n"] += len(y)
    return sa, sd


def joint_statistics(cases, names, norm):
    """Loss-exact sufficient statistics without a tall repeated-reference design."""
    norm = _normalization(norm)
    sa, sd = component_statistics(cases, names)
    factors = .5 / norm
    result = {k: factors[0]*sa[k] + factors[1]*sd[k] for k in ("g", "h", "yy")}
    result.update({k: np.sqrt(factors[0])*sa[k] + np.sqrt(factors[1])*sd[k]
                   for k in ("sx", "sy")})
    result["n"] = sa["n"] + sd["n"]
    return result


def fit_joint(stat, labels, selected, ridge_a, ridge_d, train_ids, absolute_ids):
    """Fit independent A/B coefficients with separate normalized ridge strengths.

    Both outputs were jointly estimated from all training cases. Their distinct
    family_supervision_runs describe the anchored/contrast observations, not
    independent coefficient fits. The A intercept is exempt from ridge.
    """
    if not np.isfinite([ridge_a, ridge_d]).all() or min(ridge_a, ridge_d) < 0:
        raise ValueError("Ridge strengths must be finite and nonnegative")
    train_ids, absolute_ids = sorted(set(train_ids)), sorted(set(absolute_ids))
    if not set(absolute_ids) <= set(train_ids):
        raise ValueError("Absolute supervision must belong to joint training cases")
    selected = list(selected)
    ss = subset_stats(stat, labels, selected)
    scales = np.sqrt(np.maximum(np.diag(ss["g"]), 0.))
    keep = scales > 1e-25
    dropped = [k for k, ok in zip(selected, keep) if not ok]
    selected = [k for k, ok in zip(selected, keep) if ok]
    ss = s.subset_statistics(ss, np.flatnonzero(keep))
    scales = scales[keep]
    if selected:
        g = ss["g"] / np.outer(scales, scales)
        penalty = np.array([0. if k == "intercept" else ridge_d if k.startswith("B::") else ridge_a
                            for k in selected])
        normalized = np.linalg.solve(g + np.diag(penalty), ss["h"] / scales)
        coefficients = normalized / scales
        eigen = np.linalg.eigvalsh(g)
        rank = int(np.sum(eigen > max(1e-12*eigen[-1], 1e-14)))
        condition = float(np.sqrt(eigen[-1]/eigen[0])) if eigen[0] > 0 else None
    else:
        normalized = coefficients = np.empty(0)
        rank, condition = 0, None
    result = []
    for family, prefix, ridge in (("foundation", "A::", ridge_a), ("increment", "B::", ridge_d)):
        ix = [i for i, k in enumerate(selected) if k.startswith(prefix) or
              (family == "foundation" and k == "intercept")]
        names = [selected[i].removeprefix(prefix) for i in ix]
        coef = coefficients[ix]
        result.append(dict(feature_names=names, coefficients=coef.tolist(),
            coefficient_by_name=dict(zip(names, coef.tolist())),
            column_scales=scales[ix].tolist(), scaled_coefficients=normalized[ix].tolist(),
            ridge=float(ridge), ridge_exempt=["intercept"], spec="systematic", zero_updated=False,
            training_runs=train_ids, family_supervision_runs=absolute_ids if family == "foundation" else train_ids,
            joint_fit=True, joint_feature_names=selected, joint_dropped_columns=dropped,
            joint_numerical_rank=rank, joint_normalized_condition_number=condition,
            solver=dict(method="joint_normalized_sufficient_statistics", success=True)))
    return tuple(result)
