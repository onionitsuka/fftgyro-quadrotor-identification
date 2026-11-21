from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


SUN_FORCE_BETA_EDGES_DEG = (0.0, 30.0, 60.0, 90.0)
SUN_MOMENT_BETA_EDGES_DEG = (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0)
SUN_TRUNCATED_FORCE_BETA_EDGES_DEG = (0.0, 30.0, 45.0)
SUN_TRUNCATED_MOMENT_BETA_EDGES_DEG = (0.0, 15.0, 30.0, 45.0)
SUN_ALPHA_DIAGNOSTIC_BANDS_DEG = (
    (-45.0, -40.0),
    (-20.0, -15.0),
    (-5.0, 0.0),
)


@dataclass(frozen=True, slots=True)
class AngleGridProfile:
    name: str
    alpha_level_count: int
    alpha_candidate_deg: tuple[float, ...]
    beta_deg: tuple[float, ...]
    beta_edges_deg: tuple[float, ...]
    purpose: str


PROFILES = {
    "pdf-5x5": AngleGridProfile(
        name="pdf-5x5",
        alpha_level_count=5,
        alpha_candidate_deg=(-40.0, -20.0, 0.0, 20.0, 40.0),
        beta_deg=(-40.0, -20.0, 0.0, 20.0, 40.0),
        beta_edges_deg=SUN_FORCE_BETA_EDGES_DEG,
        purpose="current PDF first-candidate fixed grid",
    ),
    "sun-beta45-coarse": AngleGridProfile(
        name="sun-beta45-coarse",
        alpha_level_count=5,
        alpha_candidate_deg=(-40.0, -20.0, 0.0, 20.0, 40.0),
        # Centers of the available parts of Sun's 30-degree partitions.
        # +/-45 deg are envelope-edge holdouts, not fit points.
        beta_deg=(-45.0, -37.5, -15.0, 0.0, 15.0, 37.5, 45.0),
        beta_edges_deg=SUN_TRUNCATED_FORCE_BETA_EDGES_DEG,
        purpose="cover Sun 30-degree |beta| partitions truncated at 45 deg",
    ),
    "sun-beta45-moment": AngleGridProfile(
        name="sun-beta45-moment",
        alpha_level_count=5,
        alpha_candidate_deg=(-40.0, -20.0, 0.0, 20.0, 40.0),
        # Centers of three 15-degree bins plus +/-45 deg envelope holdouts.
        beta_deg=(
            -45.0,
            -37.5,
            -22.5,
            -7.5,
            0.0,
            7.5,
            22.5,
            37.5,
            45.0,
        ),
        beta_edges_deg=SUN_TRUNCATED_MOMENT_BETA_EDGES_DEG,
        purpose="cover Sun roll/pitch moment 15-degree bins truncated at 45 deg",
    ),
}


@dataclass(frozen=True, slots=True)
class AngleGridSummary:
    profile: str
    purpose: str
    alpha_deg: tuple[float, ...]
    alpha_level_count: int
    alpha_candidate_deg: tuple[float, ...]
    beta_deg: tuple[float, ...]
    pose_count: int
    beta_partition_labels: tuple[str, ...]
    covered_beta_partitions: tuple[str, ...]
    missing_beta_partitions: tuple[str, ...]
    symmetric_signed_beta_coverage: bool
    required_abs_alpha_clearance_deg: float | None
    required_abs_beta_clearance_deg: float
    physical_clearance_verified: bool
    beta_envelope_limit_deg: float
    boundary_holdout_beta_deg: tuple[float, ...]
    sun_full_90deg_envelope_intentionally_truncated: bool
    sun_alpha_is_continuous_not_piecewise: bool
    sun_alpha_diagnostic_bands_deg: tuple[tuple[float, float], ...]
    beta_control_mode: str
    beta_is_fixed_per_block: bool
    dynamic_beta_command_allowed: bool
    beta_block_count: int
    tare_required_after_each_beta_change: bool
    alpha_levels_are_provisional: bool
    alpha_fixture_levels_verified: bool
    all_rotational_dof_mechanically_fixed: bool
    fixed_attitude_axes: tuple[str, ...]
    executable_as_fixed_pose_plan: bool

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        for key in (
            "alpha_deg",
            "alpha_candidate_deg",
            "beta_deg",
            "beta_partition_labels",
            "covered_beta_partitions",
            "missing_beta_partitions",
            "sun_alpha_diagnostic_bands_deg",
            "boundary_holdout_beta_deg",
        ):
            row[key] = list(row[key])
        return row


def _partition_labels(edges: tuple[float, ...]) -> tuple[str, ...]:
    return tuple(
        f"abs_beta_[{lower:g},{upper:g}{']' if index == len(edges) - 2 else ')'}"
        for index, (lower, upper) in enumerate(zip(edges, edges[1:]))
    )


def beta_partition(beta_deg: float, edges: tuple[float, ...]) -> str:
    value = abs(beta_deg)
    if value < edges[0] or value > edges[-1]:
        raise ValueError("beta angle is outside configured partition range")
    labels = _partition_labels(edges)
    for index, (lower, upper) in enumerate(zip(edges, edges[1:])):
        if lower <= value < upper or (
            index == len(edges) - 2 and lower <= value <= upper
        ):
            return labels[index]
    raise AssertionError("unreachable beta partition")


def summarize(
    profile_name: str,
    *,
    verified_abs_alpha_clearance_deg: float | None = None,
    verified_abs_beta_clearance_deg: float | None = None,
    alpha_fixture_levels_verified: bool = False,
    alpha_fixture_levels_deg: tuple[float, ...] | None = None,
) -> AngleGridSummary:
    try:
        profile = PROFILES[profile_name]
    except KeyError as exc:
        raise ValueError(f"unknown angle-grid profile: {profile_name}") from exc

    labels = _partition_labels(profile.beta_edges_deg)
    covered = {
        beta_partition(value, profile.beta_edges_deg) for value in profile.beta_deg
    }
    nonzero_abs_values = {abs(value) for value in profile.beta_deg if value != 0}
    symmetric = all(
        value in profile.beta_deg and -value in profile.beta_deg
        for value in nonzero_abs_values
    )
    if (
        alpha_fixture_levels_deg is not None
        and len(alpha_fixture_levels_deg) != profile.alpha_level_count
    ):
        raise ValueError(
            f"exactly {profile.alpha_level_count} measured alpha levels are required"
        )
    alpha_values = alpha_fixture_levels_deg or ()
    alpha_required = (
        max(abs(value) for value in alpha_values) if alpha_values else None
    )
    beta_required = max(abs(value) for value in profile.beta_deg)
    clearance_verified = (
        verified_abs_alpha_clearance_deg is not None
        and verified_abs_beta_clearance_deg is not None
        and alpha_required is not None
        and verified_abs_alpha_clearance_deg >= alpha_required
        and verified_abs_beta_clearance_deg >= beta_required
        and alpha_fixture_levels_verified
    )
    return AngleGridSummary(
        profile=profile.name,
        purpose=profile.purpose,
        alpha_deg=alpha_values,
        alpha_level_count=profile.alpha_level_count,
        alpha_candidate_deg=profile.alpha_candidate_deg,
        beta_deg=profile.beta_deg,
        pose_count=profile.alpha_level_count * len(profile.beta_deg),
        beta_partition_labels=labels,
        covered_beta_partitions=tuple(label for label in labels if label in covered),
        missing_beta_partitions=tuple(label for label in labels if label not in covered),
        symmetric_signed_beta_coverage=symmetric,
        required_abs_alpha_clearance_deg=alpha_required,
        required_abs_beta_clearance_deg=beta_required,
        physical_clearance_verified=clearance_verified,
        beta_envelope_limit_deg=profile.beta_edges_deg[-1],
        boundary_holdout_beta_deg=tuple(
            value
            for value in profile.beta_deg
            if abs(value) == profile.beta_edges_deg[-1]
        ),
        sun_full_90deg_envelope_intentionally_truncated=(
            profile.beta_edges_deg[-1] < 90.0
        ),
        sun_alpha_is_continuous_not_piecewise=True,
        sun_alpha_diagnostic_bands_deg=SUN_ALPHA_DIAGNOSTIC_BANDS_DEG,
        beta_control_mode="mechanically_fixed_apparatus_rotation",
        beta_is_fixed_per_block=True,
        dynamic_beta_command_allowed=False,
        beta_block_count=len(profile.beta_deg),
        tare_required_after_each_beta_change=True,
        alpha_levels_are_provisional=True,
        alpha_fixture_levels_verified=alpha_fixture_levels_verified,
        all_rotational_dof_mechanically_fixed=True,
        fixed_attitude_axes=("roll", "pitch", "yaw"),
        executable_as_fixed_pose_plan=(
            clearance_verified and alpha_fixture_levels_verified
        ),
    )
