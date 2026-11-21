"""保存済み6軸モデルの特徴量評価。元の解析コードから純粋関数だけを抽出。

生ログ、零点、回帰、通信、ファイル読込を含まない。
model.jsonで採用済みの項だけを評価する。
"""
import math
from itertools import product
import numpy as np
from scipy.interpolate import CubicSpline

OUTPUTS = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
EXTRA_CHANNELS = ("Fx", "Fy", "Mx")
SIGNS = np.array([1., 1., -1., -1.])
SPIN = SIGNS
ROLL = np.array([-1., 1., 1., -1.])
PITCH = np.array([1., -1., 1., -1.])
FLU_TO_FRD = np.array([1., -1., -1.])

def inflow(mu: np.ndarray, ct: float) -> float:
    """Same positive unique momentum root as graybox_pipeline.induced_inflow."""
    h2, z = float(mu[0] ** 2 + mu[1] ** 2), float(mu[2])

    def flux(nu: float) -> float:
        return 2 * nu * math.sqrt(h2 + (nu - z) ** 2)

    disc = z * z - 8 * h2
    if z > 0 and disc > 0:
        nmax, nmin = (3 * z - math.sqrt(disc)) / 4, (3 * z + math.sqrt(disc)) / 4
        if flux(nmin) - 1e-12 <= ct <= flux(nmax) + 1e-12:
            raise ValueError("Ambiguous quasi-steady inflow; no arbitrary root selected")
    lo, hi = 0.0, max(z, 0) + math.sqrt(ct / 2) + 1
    for _ in range(48):
        mid = (lo + hi) / 2
        if flux(mid) < ct:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2 - z


def monomials(variables, degree):
    return [
        tuple((name, power) for name, power in zip(variables, powers, strict=True) if power)
        for powers in product(range(degree + 1), repeat=len(variables))
        if sum(powers) <= degree
    ]


def smooth_switch(value, boundary, halfwidth):
    """Quintic transition: weights and first/second derivatives join continuously."""
    t = np.clip((value - boundary + halfwidth) / (2 * halfwidth), 0.0, 1.0)
    return t**3 * (10 - 15 * t + 6 * t**2)


def sun_yaw_regions(mu, rbar, ur, yaw_rate, cfg=None):
    """Sun Eq.2/Table12--13: beta=asin(v/sqrt(u²+v²)), not aircraft yaw.

    atan2(v,abs(u)) is equivalent and handles reverse flow. mu_x still retains
    the forward/backward distinction. At zero horizontal flow beta is undefined;
    use zero only as a partition convention, and expose that limitation.
    """
    advance = np.linalg.norm(mu, axis=1)
    beta = np.arctan2(mu[:, 1], np.abs(mu[:, 0]))
    ab = np.abs(beta)
    low = advance <= 0.05
    result = dict(
        sun_low=low.astype(float),
        sun_beta_0_30=((~low) & (ab < np.pi / 6)).astype(float),
        sun_beta_30_60=((~low) & (ab >= np.pi / 6) & (ab < np.pi / 3)).astype(float),
        sun_beta_60_90=((~low) & (ab >= np.pi / 3)).astype(float),
        sun_beta=beta,
        sun_mu=advance,
        yaw_rate=yaw_rate,
        ur_absur=ur * np.abs(ur),
        rbar_absrbar=rbar * np.abs(rbar),
        muy_absmuy=mu[:, 1] * np.abs(mu[:, 1]),
    )
    if cfg is not None and cfg.yaw_motion_model == "sun_blended":
        high = smooth_switch(advance, 0.05, cfg.sun_mu_halfwidth)
        a = smooth_switch(np.rad2deg(ab), 30.0, cfg.sun_beta_halfwidth_deg)
        b = smooth_switch(np.rad2deg(ab), 60.0, cfg.sun_beta_halfwidth_deg)
        result.update(
            sun_low=1 - high,
            sun_beta_0_30=high * (1 - a),
            sun_beta_30_60=high * (a - b),
            sun_beta_60_90=high * b,
        )
    return result


def rotor_local_flow(raw, rotor_speed, cfg):
    """Each rotor's local freestream, normalized by its own |Omega_i| R.

    mu_h,i = hypot(V_x,i, V_y,i)/(|Omega_i| R); lambda_i = V_z,i/(|Omega_i| R).
    lambda is axial freestream, NOT total inflow including the induced wake.
    Zero/missing speed or nonfinite ratios stay unavailable; no RPM floor or
    measured-RPM substitution is used. Geometry is the existing nominal plane.
    """
    speed = np.abs(np.asarray(rotor_speed, dtype=float))
    if speed.ndim != 2 or speed.shape[1] != 4:
        raise ValueError("Expected four rotor angular speeds per sample")
    if cfg.rotor_radius_m <= 0 or cfg.arm_xy_m <= 0:
        raise ValueError("Rotor radius and arm length must be positive")
    shift = (np.asarray(cfg.cg_flu_m) - cfg.propulsion_reference_cg_flu_m) * FLU_TO_FRD
    positions = cfg.arm_xy_m * np.array([[1, 1, 0], [-1, -1, 0], [1, -1, 0], [-1, 1, 0]]) - shift
    velocity = raw["air"][:, None, :] + np.cross(raw["omega"][:, None, :], positions)
    tip_speed = speed * cfg.rotor_radius_m
    available = np.isfinite(tip_speed) & (tip_speed > 0) & np.isfinite(velocity).all(axis=2)
    mu = np.full_like(velocity, np.nan)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        np.divide(velocity, tip_speed[:, :, None], out=mu, where=available[:, :, None])
        available &= np.isfinite(mu).all(axis=2)
        mu[~available] = np.nan
        # S_total = sum_i rho*A*(Omega_i*R)^2.  Consequently averaging
        # (Omega_i^2 / mean(Omega^2))*f(mu_i) preserves each rotor's load scale.
        mean_squared = np.mean(speed**2, axis=1)
        weight = np.full_like(speed, np.nan)
        np.divide(
            speed**2,
            mean_squared[:, None],
            out=weight,
            where=np.isfinite(mean_squared[:, None]) & (mean_squared[:, None] > 0),
        )
    return dict(
        positions_frd_m=positions,
        velocity_frd_m_s=velocity,
        omega_abs_rad_s=speed,
        mu=mu,
        mu_h=np.hypot(mu[:, :, 0], mu[:, :, 1]),
        lambda_axial=mu[:, :, 2],
        omega_squared_weight=weight,
        available=available,
    )


def rotor_local_yaw(raw, rotor_speed, cfg, *, flow=None):
    """Rotor-plane drag and empirical axial-flow torque; no induced-wake model.

    v_i = v_air,CG + omega_body cross r_i (body relative to air, FRD).
    Nominal rotor plane passes through the reference CG; its height is not measured.
    Positive linear/quadratic coefficients give in-plane forces opposing local flow.
    Each rotor uses its own |Omega_i| R. Its contribution is weighted by
    Omega_i^2/mean(Omega^2) before summing with the existing moment scale.
    For these linear/quadratic features this is algebraically equivalent to
    the previous shared-RMS implementation; old fitted coefficients stay valid.
    """
    flow = rotor_local_flow(raw, rotor_speed, cfg) if flow is None else flow
    positions = flow["positions_frd_m"]
    plane = flow["mu"].copy()
    plane[:, :, 2] = 0
    weighted_plane = plane * flow["omega_squared_weight"][:, :, None]
    linear = -np.cross(positions / cfg.arm_xy_m, weighted_plane)[:, :, 2].mean(axis=1)
    quadratic = -np.cross(
        positions / cfg.arm_xy_m, np.linalg.norm(plane, axis=2)[:, :, None] * weighted_plane
    )[:, :, 2].mean(axis=1)
    axial = np.mean(SIGNS * flow["omega_squared_weight"] * flow["lambda_axial"], axis=1)
    return dict(local_yaw_linear=linear, local_yaw_quadratic=quadratic, local_yaw_axial=axial)


def evaluate_names(state, scale, names):
    """Evaluate every monomial BEFORE taking any time/reference average."""
    features = {}
    for name in names:
        factors = []
        for factor in name.split("*"):
            if factor == "1":
                continue
            key, power = factor.split("^") if "^" in factor else (factor, "1")
            factors.append(np.asarray(state[key], float) ** int(power))
        features[name] = (
            np.asarray(scale, float) * np.prod(factors, axis=0)
            if factors
            else np.asarray(scale, float).copy()
        )
    return features


def speed_features(data, output):
    """Extra shared rotor-speed/flow terms; no run, clock or target features."""
    state = data["state"]
    scale = data["moment_scale"]
    # Use smoothed scale-consistent speed; caller stores it explicitly.
    speed = data["rms_speed_ratio"] - 1
    products = (
        (
            ("mu_x",),
            ("mu_z",),
            ("mu_x", "mu_x"),
            ("mu_x", "mu_z"),
            ("mu_z", "mu_z"),
            ("abs_mu_y",),
            ("qbar",),
            ("mu_x", "qbar"),
        )
        if output == "My"
        else (
            ("mu_y",),
            ("mu_x", "mu_y"),
            ("mu_z",),
            ("mu_y", "mu_z"),
            ("ur",),
            ("rbar",),
            ("local_yaw_linear",),
            ("local_yaw_axial",),
        )
    )
    feature = {
        f"speed_dev__{'_'.join(keys)}": scale * speed * np.prod([state[k] for k in keys], axis=0)
        for keys in products
    }
    # Common RPM-squared term permits a shared rotor-group imbalance. The
    # differential term corrects the fixed thrust/drag coefficient, not zeros.
    feature["rotor_common_scale"] = scale.copy()
    gain = "uq" if output == "My" else "ur"
    feature["rotor_differential_gain"] = scale * state[gain]
    return feature


def extra_feature_names(output, spec):
    """小さな物理量辞書。時刻・run番号・荷重・零点は説明変数にしない。"""
    if spec == "sun_inplane":
        # Sun 2019 Eq.29--30: one common-flow polynomial, without a second
        # per-rotor representation of the same in-plane drag effect.
        variables = {"Fx": ("mu_x", "abs_mu_y", "mu_z"),
                     "Fy": ("abs_mu_x", "mu_y", "mu_z")}[output]
        return ["*".join(k + (f"^{p}" if p != 1 else "") for k, p in term) or "1"
                for term in monomials(variables, 3)]
    if spec not in ("compact", "expanded"):
        raise ValueError("Unsupported extra-axis dictionary")
    theory = False
    spec = spec.removeprefix("theory_")
    variables, rates = {
        "Fx": (("mu_x", "abs_mu_y", "mu_z"), ("qbar", "rbar")),
        "Fy": (("abs_mu_x", "mu_y", "mu_z"), ("pbar", "rbar")),
        "Mx": (("abs_mu_x", "mu_y", "mu_z"), ("pbar", "up")),
    }[output]
    terms = monomials(variables, 2 if spec == "compact" else 3)
    terms += [a + ((rate, 1),) for a in monomials(variables, 1) for rate in rates]
    names = ["*".join(k + (f"^{p}" if p != 1 else "") for k, p in t) or "1" for t in terms]
    if output.startswith("F"):
        names += ["rotor_inplane_linear", "rotor_inplane_quadratic"]
    return list(dict.fromkeys(names))


def extra_feature_bank(data, output, spec, cfg=None):
    scale = data["force_scale"] if output.startswith("F") else data["moment_scale"]
    names = extra_feature_names(output, spec)
    bank = evaluate_names(data["state"], scale, [k for k in names if not k.startswith(("rotor_", "theory__"))])
    if output.startswith("F") and spec != "sun_inplane":
        # 各ローターの局所流 / (|Omega_i| R)。機体角速度の効果も含む。
        f = data["rotor_flow"]
        j = OUTPUTS.index(output)
        mu = f["mu"][:, :, j]
        weighted = -f["omega_squared_weight"] * mu
        bank["rotor_inplane_linear"] = scale * weighted.mean(axis=1)
        bank["rotor_inplane_quadratic"] = scale * (f["mu_h"] * weighted).mean(axis=1)
    return bank


def spline_weights(beta):
    knots = np.sin(np.deg2rad([0.0, 30.0, 60.0, 90.0])) ** 2
    coordinate = np.clip(np.sin(beta) ** 2, 0.0, 1.0)
    return CubicSpline(knots, np.eye(4), bc_type="natural")(coordinate)


def continuous_bank(data):
    """Shared degree<=3 flow model, rate/input terms, and smooth local changes.

    The omitted first spline basis is represented by the corresponding shared
    term: partition of unity prevents adding four duplicate constant effects.
    Front/back and left/right remain distinguishable through signed mu_x/y.
    """
    s = data["state"]
    scale = data["moment_scale"]
    bank = {}
    for i in range(4):
        for j in range(4 - i):
            for k in range(4 - i - j):
                name = f"flow_{i}{j}{k}"
                value = scale * s["mu_x"] ** i * s["mu_y"] ** j * s["mu_z"] ** k
                bank[name] = value
                if i + j + k <= 2:
                    bank[name + "_r"] = value * s["rbar"]
                    bank[name + "_u"] = value * s["ur"]
    for key in ("local_yaw_linear", "local_yaw_quadratic", "local_yaw_axial", "rbar_absrbar"):
        bank[key] = scale * s[key]
    speed = data["rms_speed_ratio"] - 1
    for key in ("mu_y", "ur", "rbar", "local_yaw_axial"):
        bank["speed_" + key] = scale * speed * s[key]
    weights = spline_weights(s["sun_beta"])
    high = 1 - s["sun_low"]
    for key in ("mu_y", "mu_x", "mu_z", "ur", "rbar", "local_yaw_linear", "local_yaw_axial"):
        for j in (1, 2, 3):
            bank[f"local{j}_{key}"] = scale * s[key] * high * weights[:, j]
    return bank


def physics_bank(data, cfg, signs):
    """Return dimensional moment regressors; empirical coefficients are shared."""
    flow = data["rotor_flow"]
    omega = flow["omega_abs_rad_s"]
    weight = cfg.rho * np.pi * cfg.rotor_radius_m**5 * omega**2
    axial = flow["lambda_axial"]
    horizontal = flow["mu_h"]
    result = {}
    # Permit repeatable motor/propeller imbalance without any run-specific gain.
    for i in range(4):
        result[f"rotor{i + 1}_torque"] = signs[i] * weight[:, i]
    for name, value in (
        ("axial", axial),
        ("axial2", axial**2),
        ("horizontal2", horizontal**2),
        ("horizontal_axial", horizontal * axial),
        ("horizontal3", horizontal**3),
    ):
        result["torque_" + name] = np.sum(signs * weight * value, axis=1)
    # Flow-direction terms use the FULL azimuth, including front/back. Dynamic
    # pressure scaling makes them vanish at zero flow; no direction discontinuity
    # remains in the moment at zero horizontal speed.
    s = data["state"]
    mx, my, mz = (s[k] for k in ("mu_x", "mu_y", "mu_z"))
    angle = np.arctan2(my, mx)
    pressure_scale = data["moment_scale"] * (mx**2 + my**2)
    for n in (1, 2, 3, 4):
        for trig, fun in (("sin", np.sin), ("cos", np.cos)):
            result[f"body_h{n}_{trig}"] = pressure_scale * fun(n * angle)
    for name, value in (("axial", mz), ("yaw_input", s["ur"]), ("yaw_rate", s["rbar"])):
        for n in (1, 2):
            result[f"body_h{n}_sin_{name}"] = pressure_scale * np.sin(n * angle) * value
            result[f"body_h{n}_cos_{name}"] = pressure_scale * np.cos(n * angle) * value
    return result


def instantaneous_data(air, rate, rotor, rotor_dot, cfg, physical):
    """Same nondimensional variables as analysis.route_state, without log filters."""
    rotor = np.asarray(rotor, float)
    if not np.isfinite(rotor).all() or np.any(rotor <= 0):
        raise ValueError("Positive finite simulation RPM is required; no RPM floor")
    air, rate = np.asarray(air, float), np.asarray(rate, float)
    if not np.isfinite(np.r_[air, rate, rotor_dot]).all():
        raise ValueError("Nonfinite simulated state")
    bar = np.sqrt(np.mean(rotor**2))
    denominator = bar*cfg.rotor_radius_m
    mu = air[None, :]/denominator
    pqr = rate[None, :]*cfg.arm_xy_m/denominator
    state = dict(zip(("mu_x", "mu_y", "mu_z"), mu.T))
    state.update(zip(("pbar", "qbar", "rbar"), pqr.T))
    for k, signs in (("up", ROLL), ("uq", PITCH), ("ur", SIGNS)):
        state[k] = np.array([rotor**2 @ signs/bar**2])
    for k in ("mu_x", "mu_y", "pbar", "qbar", "rbar", "up", "uq", "ur"):
        state["abs_"+k] = np.abs(state[k])
    raw = dict(air=air[None, :], omega=rate[None, :])
    flow = rotor_local_flow(raw, rotor[None, :], cfg)
    state.update(rotor_local_yaw(raw, rotor[None, :], cfg, flow=flow))
    state.update(sun_yaw_regions(mu, pqr[:, 2], state["ur"], raw["omega"][:, 2], cfg))
    induced = inflow(mu[0], cfg.kf/(cfg.rho*np.pi*cfg.rotor_radius_m**4))
    state.update(inflow=np.array([induced]), mu_h2=mu[:, 0]**2+mu[:, 1]**2)
    scale = np.array([cfg.rho*4*np.pi*cfg.rotor_radius_m**2*denominator**2])
    return dict(state=state, physical=np.asarray(physical)[None, :],
        force_scale=scale, moment_scale=scale*cfg.arm_xy_m,
        rotor_flow=flow, air_frd_m_s=raw["air"], omega_body_rad_s=raw["omega"],
        rpm=rotor[None, :]*60/(2*np.pi),
        rms_speed_ratio=np.array([bar/(4000*2*np.pi/60)]))


def selected_features(data, cfg, output, model):
    """Evaluate only saved terms; reuse original generators for special families."""
    names = model["feature_names"]
    state = dict(data["state"])
    scale = data["force_scale"] if output.startswith("F") else data["moment_scale"]
    state["inflow_delta"] = state["inflow"]-np.sqrt(
        cfg.kf/(cfg.rho*np.pi*cfg.rotor_radius_m**4)/2)
    state["mu_total2"] = sum(state[k]**2 for k in ("mu_x", "mu_y", "mu_z"))
    bank = dict(intercept=np.ones(1), prop_gain=data["physical"][:, OUTPUTS.index(output)])
    if output in EXTRA_CHANNELS:
        bank.update(extra_feature_bank(data, output, model["spec"], cfg))
    if output in ("My", "Mz"):
        bank.update(speed_features(data, output))
    if output == "Mz":
        bank.update(continuous_bank(data))
        bank.update(physics_bank(data, cfg, SIGNS))
    if any(k.startswith(("theory__", "individual__", "rotor_sum__", "coupled__")) for k in names):
        raise ValueError("This public evaluator supports only the exported frozen model")
    for key in names:
        if key in bank:
            continue
        if key.startswith("localflow__"):
            _, i, j, motion = key.split("__")
            flow = data["rotor_flow"]
            value = scale*flow["omega_squared_weight"][:, int(i)]/4*flow["mu"][:, int(i), int(j)]
            bank[key] = value if motion == "1" else value*state[motion]
        else:
            plain = key.removeprefix("cross__")
            bank[key] = evaluate_names(state, scale, [plain])[plain]
    return {k: bank[k] for k in names}


def correction(data, cfg, output, model):
    bank = selected_features(data, cfg, output, model)
    return float(sum(value*bank[key][0] for key, value in model["coefficient_by_name"].items()))
