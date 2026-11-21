"""Own-route, dimensional six-axis candidates; alternatives, not mandatory terms.

The frozen propulsion prediction remains external. Thrust/torque columns here
are corrections to it, not a second baseline. Intercepts are regression terms,
never instructions to adjust independently established sensor zeros.
"""
from functools import lru_cache

import numpy as np

import analysis as a
from scripts import refit_theory_forces as forces

VERSION = "six-axis-dimensional-theory-v1"
MOTIONS = forces.MOTIONS


def _reference(data, cfg, output):
    """Declared generator families, never previously selected or fitted terms."""
    if output == "Mx":
        return a.extra_feature_bank(data, output, "expanded", cfg)
    if output == "Mz":
        # Do not concatenate complete overlapping regional/global expansions.
        return a.continuous_bank(data)
    scale = data["force_scale"] if output == "Fz" else data["moment_scale"]
    bank = a.evaluate_features(data["state"], data["physical"][:, a.OUTPUTS.index(output)],
                               scale, output, "expanded", cfg)
    if output == "My":
        bank.update(a.speed_features(data, output))
    return bank


def _rotor_columns(data, cfg, output):
    """Individual forces and r_i cross F_i, retaining geometry-supported axes."""
    flow = data["rotor_flow"]
    velocity, omega = flow["velocity_frd_m_s"], flow["omega_abs_rad_s"]
    positions = flow["positions_frd_m"]
    area, radius = np.pi*cfg.rotor_radius_m**2, cfg.rotor_radius_m
    axis = "xyz".index(output[-1].lower())
    is_force = output.startswith("F")
    plane = velocity.copy()
    plane[:, :, 2] = 0.
    horizontal = np.linalg.norm(plane, axis=2)
    values = dict(
        plane_linear=-cfg.rho*area*radius*omega[:, :, None]*plane,
        plane_quadratic=-cfg.rho*area*horizontal[:, :, None]*plane,
    )
    for name, scalar in (
        ("thrust", radius**2*omega**2),
        ("axial", radius*omega*velocity[:, :, 2]),
        ("horizontal2", horizontal**2),
        ("axial2", velocity[:, :, 2]*np.abs(velocity[:, :, 2])),
    ):
        values[name] = np.zeros_like(velocity)
        values[name][:, :, 2] = -cfg.rho*area*scalar
    individual, sums = {}, {}
    for name, force in values.items():
        # This test uses geometry, not whether a particular run excites a term.
        directions = np.eye(3)[:2] if name.startswith("plane_") else np.eye(3)[2:]
        supported = np.array([
            any((direction if is_force else np.cross(position, direction))[axis] != 0.
                for direction in directions) for position in positions])
        load = force if is_force else np.cross(positions, force)
        keys = []
        for i in np.flatnonzero(supported):
            key = f"individual__{i+1}__{name}"
            individual[key] = load[:, i, axis]
            keys.append(key)
        if keys:
            sums["rotor_sum__"+name] = sum(individual[k] for k in keys)
    if output == "Mz":
        # Aerodynamic propeller reaction torque: Q_i ~ rho A R^3 Omega_i^2.
        # Axial/horizontal flow corrections are evaluated dimensionally so no
        # artificial RPM floor or cross-route replacement enters these columns.
        for name, scalar in (
            ("torque", radius**3*omega**2),
            ("torque_axial", radius**2*omega*velocity[:, :, 2]),
            ("torque_horizontal2", radius*horizontal**2),
            ("torque_axial2", radius*velocity[:, :, 2]**2),
        ):
            keys = []
            for i in range(4):
                key = f"individual__{i+1}__{name}"
                individual[key] = a.SIGNS[i]*cfg.rho*area*scalar[:, i]
                keys.append(key)
            sums["rotor_sum__"+name] = sum(individual[k] for k in keys)
    return sums, individual


def _physical_candidates(data, cfg, output):
    theory = a.theory_feature_bank(data, cfg, output)
    if output == "Fz":
        body = {k: v for k, v in theory.items() if k.startswith("theory__body_drag__")}
        damping = {}
        body_base = "theory__body_drag__1"
    else:
        body = {k: v for k, v in theory.items() if k.startswith("theory__body_pressure__")}
        damping = {k: v for k, v in theory.items() if k.startswith("theory__damping_")}
        body_base = "theory__body_pressure__1"
    sums, individual = _rotor_columns(data, cfg, output)
    # A single summed rotor axial-flow correction is an ordinary candidate;
    # individual motor coefficients are offered in a separate alternative.
    core = {body_base: body[body_base], **damping, **sums}
    direction = {**body, **damping, **sums}
    coupled = {f"coupled__{base}__{motion}": value*data["state"][motion]
               for base, value in core.items() for motion in MOTIONS}
    bank = {**direction, **coupled, **individual}
    # Core first is a basis-representation convention, not mandatory selection.
    pools = dict(physical_core=list(core),
                 physical_direction=list(dict.fromkeys([*core, *direction])),
                 physical_coupled=list(dict.fromkeys([*core, *direction, *coupled])),
                 per_rotor=[*body, *damping, *individual])
    return bank, pools


def _geometry_key(cfg):
    return (cfg.rho, cfg.rotor_radius_m, cfg.arm_xy_m,
            tuple(cfg.cg_flu_m), tuple(cfg.propulsion_reference_cg_flu_m))


@lru_cache(maxsize=32)
def _independent_physical_pools(key, output):
    """Remove exact geometry aliases on broad *synthetic* states, not data fits.

    For example, signed sum_i Vh_i^2 is zero for the coplanar symmetric rotor
    layout. Its four individual columns consequently have one linear identity.
    This deterministic numerical identity audit does not screen high empirical
    correlations or use experimental samples/targets. Rank is checked again by
    the ordinary training-only selector on the actual experiment.
    """
    rho, radius, arm, cg, reference = key
    cfg = a.Settings(rho=rho, rotor_radius_m=radius, arm_xy_m=arm,
                     cg_flu_m=cg, propulsion_reference_cg_flu_m=reference)
    rng = np.random.default_rng(761421)
    velocity, rate = rng.normal(size=(384, 3))*7, rng.normal(size=(384, 3))*2
    omega = rng.uniform(180., 850., size=(384, 4))
    bar = np.sqrt(np.mean(omega**2, axis=1))
    pqr = rate*arm/(bar[:, None]*radius)
    state = dict(zip(("pbar", "qbar", "rbar"), pqr.T))
    for name, signs in (("up", [-1, 1, 1, -1]), ("uq", [1, -1, 1, -1]), ("ur", a.SIGNS)):
        state[name] = omega**2 @ np.asarray(signs)/bar**2
    data = dict(air_frd_m_s=velocity, omega_body_rad_s=rate, state=state,
                rotor_flow=a.rotor_local_flow(dict(air=velocity, omega=rate), omega, cfg))
    bank, pools = _physical_candidates(data, cfg, output)
    retained, removed = {}, {}
    for pool, names in pools.items():
        q = np.ones((len(velocity), 1))/np.sqrt(len(velocity))
        keep, omit = [], []
        # Values of a theoretically cancelling sum can be roundoff noise.
        typical = max(float(np.linalg.norm(bank[k])) for k in names)
        for name in names:
            value = bank[name]
            scale = np.linalg.norm(value)
            if scale <= max(1e-28, typical*1e-13):
                omit.append(name)
                continue
            residual = value/scale
            for _ in range(2):
                residual -= q @ (q.T @ residual)
            length = np.linalg.norm(residual)
            if length < 1e-10:
                omit.append(name)
            else:
                q = np.column_stack([q, residual/length])
                keep.append(name)
        retained[pool], removed[pool] = tuple(keep), tuple(omit)
    return retained, removed


def candidate_audit(cfg, output):
    if output in ("Fx", "Fy"):
        return dict(policy="Exact prior Fx/Fy dictionaries retained", removed={})
    _, removed = _independent_physical_pools(_geometry_key(cfg), output)
    return dict(policy="Independent synthetic-state algebraic rank audit; no observed-load screening",
                removed={k: list(v) for k, v in removed.items()})


def coefficient_bounds(cfg, output, poolbank):
    """Preserve the existing ±50% Fz propulsion correction convention."""
    if output != "Fz":
        return {}
    bound = .5*cfg.kf/(cfg.rho*np.pi*cfg.rotor_radius_m**4)
    result = {"prop_gain": (-.5, .5)} if "prop_gain" in poolbank else {}
    for name in ("rotor_sum__thrust", *(f"individual__{i}__thrust" for i in range(1, 5))):
        if name in poolbank:
            result[name] = (-bound, bound)
    return result


def feature_bank(data, cfg, output):
    """Return dimensional bank and five deterministic candidate alternatives.

    Body moments use q S l times direction functions. Drag-arm columns are not
    added: their basic forms already equal signed direction functions in that
    pressure family. Per-rotor alternatives replace aggregate loads, never add
    sums beside their summands. No observed wrench, run label or zero is read.
    """
    if output not in a.OUTPUTS:
        raise ValueError("Unknown wrench component: "+str(output))
    if output in ("Fx", "Fy"):
        bank, pools = forces.feature_bank(data, cfg, output)
        return bank, {"reference" if k == "sun" else k: v for k, v in pools.items()}
    bank = _reference(data, cfg, output)
    reference = {"intercept", *bank}
    physical, _ = _physical_candidates(data, cfg, output)
    retained, _ = _independent_physical_pools(_geometry_key(cfg), output)
    bank.update(physical)
    pools = {pool: sorted({"intercept", *names}) for pool, names in retained.items()}
    pools["reference"] = sorted(reference)
    return bank, pools
