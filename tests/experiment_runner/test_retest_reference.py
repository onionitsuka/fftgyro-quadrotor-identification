import json

import pytest

from fftgyro_quad_id.experiment_runner.retest_reference import (
    SCHEMA,
    audit_block,
    prepare_context,
    validate_context,
)


def context(stage="Z-pre", **extra):
    return dict(
        schema=SCHEMA,
        configuration_id="airframe1",
        battery_id="battery1",
        zero_epoch_id="epoch1",
        calibration_block_id="block1",
        split="T",
        location_id="outside",
        wind_m_s=0,
        stage=stage,
        pair_id="a1",
        axis="YAW",
        outside_wind_tunnel=True,
        motor_power_disconnected=True,
        **extra,
    )


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def captured(tmp_path, name, ctx, start=1, end=2, qc="PASS"):
    folder = tmp_path / name
    folder.mkdir()
    write(
        folder / "run.json",
        dict(
            state="CAPTURED",
            retest_context=ctx,
            idle=ctx["stage"].startswith("Z"),
            qc_stationary_required=True,
            capture_setup_utc_ns=start,
            capture_closed_utc_ns=end,
        ),
    )
    write(folder / "immediate-qc.json", dict(decision=qc))
    return folder


def prepare(path, **kwargs):
    return prepare_context(
        path, idle=False, fixed=True, stationary=False, px4_log=False, session={}, **kwargs
    )


def test_zero_needs_motor_off_and_stationary_flags(tmp_path):
    path = write(tmp_path / "ctx.json", context())
    with pytest.raises(ValueError, match="Z requires"):
        prepare(path)
    assert (
        prepare_context(path, idle=True, fixed=True, stationary=True, px4_log=True, session=None)[
            "stage"
        ]
        == "Z-pre"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("wind_m_s", 5),
        ("motor_power_disconnected", False),
        ("outside_wind_tunnel", False),
        ("split", "X"),
    ],
)
def test_bad_zero_context(field, value):
    ctx = context()
    ctx[field] = value
    with pytest.raises(ValueError):
        validate_context(ctx)


def test_powered_requires_same_battery_and_good_pre_zero(tmp_path):
    captured(tmp_path, "pre", context())
    ctx = context("F-before", pre_zero_run="pre")
    path = write(tmp_path / "ctx.json", ctx)
    assert prepare(path)["pre_zero_run"] == str((tmp_path / "pre").resolve())
    ctx["battery_id"] = "replacement"
    write(path, ctx)
    with pytest.raises(ValueError, match="battery_id"):
        prepare(path)


def test_pre_qc_failure_is_not_accepted(tmp_path):
    captured(tmp_path, "pre", context(), qc="REVIEW")
    path = write(tmp_path / "ctx.json", context("M", pre_zero_run="pre"))
    with pytest.raises(ValueError, match="QC not PASS"):
        prepare(path)


def test_moving_to_wind_tunnel_requires_transfer_confirmation(tmp_path):
    captured(tmp_path, "pre", context())
    ctx = context("M", pre_zero_run="pre")
    ctx.update(location_id="tunnel", wind_m_s=5)
    path = write(tmp_path / "ctx.json", ctx)
    with pytest.raises(ValueError, match="transfer"):
        prepare(path)
    ctx["calibration_transfer_confirmed"] = True
    write(path, ctx)
    assert prepare(path)["wind_m_s"] == 5


def test_sequence_cannot_skip_low_speed_or_switch_axis(tmp_path):
    captured(tmp_path, "pre", context())
    captured(tmp_path, "before", context("F-before"))
    ctx = context("D-fast", pre_zero_run="pre", previous_run="before")
    path = write(tmp_path / "ctx.json", ctx)
    with pytest.raises(ValueError, match="must follow"):
        prepare_context(
            path,
            idle=False,
            fixed=False,
            stationary=False,
            px4_log=False,
            session={"pid": {"axis": "YAW"}},
        )
    ctx["stage"] = "D-slow"
    write(path, ctx)
    with pytest.raises(ValueError, match="axis/session"):
        prepare_context(
            path,
            idle=False,
            fixed=False,
            stationary=False,
            px4_log=False,
            session={"pid": {"axis": "PITCH"}},
        )


def test_metadata_complete_is_never_physics_pass(tmp_path):
    pre = captured(tmp_path, "pre", context())
    path = write(tmp_path / "ctx.json", context("M", pre_zero_run="pre"))
    motor = captured(tmp_path, "motor", prepare(path), 3, 4)
    post = captured(tmp_path, "post", context("Z-post"), 5, 6)
    report = audit_block([pre, motor, post])
    assert report["decision"] == "METADATA_COMPLETE", report
    assert not report["absolute_values_validated"]
    assert audit_block([pre, motor])["decision"] == "REVIEW"
    assert audit_block([pre, post, motor])["decision"] == "REVIEW"
    write(pre / "run.json", {**json.loads((pre / "run.json").read_text()), "extra": 1})
    assert any("changed evidence" in reason for reason in audit_block([pre, motor, post])["issues"])


def test_complete_a_sequence_and_console_forwarding(tmp_path):
    from fftgyro_quad_id.experiment_runner.production_console import command

    folders = [captured(tmp_path, "pre", context())]
    previous = None
    for index, stage in enumerate(("F-before", "D-slow", "D-fast", "F-after")):
        ctx = context(stage, pre_zero_run="pre")
        if previous:
            ctx["previous_run"] = previous.name
        path = write(tmp_path / f"{stage}.json", ctx)
        dynamic = stage.startswith("D-")
        resolved = prepare_context(
            path,
            idle=False,
            fixed=not dynamic,
            stationary=False,
            px4_log=False,
            session={"pid": {"axis": "YAW"}} if dynamic else {},
        )
        previous = captured(tmp_path, stage, resolved, 3 + 2 * index, 4 + 2 * index)
        folders.append(previous)
    folders.append(captured(tmp_path, "post", context("Z-post"), 11, 12))
    assert audit_block(folders)["decision"] == "METADATA_COMPLETE"
    settings = dict(
        executor="/qualified/executor",
        retest_context=str(path),
        qc_profile=str(tmp_path / "qc.json"),
    )
    for key in (
        "aircraft_host",
        "leptrino_host",
        "aircraft_host_key_alias",
        "leptrino_host_key_alias",
        "clock_server_ip",
        "clock_client_visible_ip",
        "windows_client_visible_ip",
    ):
        settings[key] = "test-placeholder"
    args = command(settings, {"axis": "fixed"}, tmp_path / "qualified.json", 120, tmp_path / "next")
    assert args[args.index("--retest-context") + 1] == str(path)
    assert args[args.index("--qc-profile") + 1] == str(tmp_path / "qc.json")
