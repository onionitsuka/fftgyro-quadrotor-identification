import json

import pytest

from fftgyro_quad_id.experiment_runner.sd_log_import import import_one


def setup_files(tmp_path, recorded="/fs/microsd/log/day/test.ulg"):
    sd = tmp_path / "sd"
    source = sd / "log/day/test.ulg"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"ULog\x01\x12\x35" + bytes(100))
    run = tmp_path / "run"
    (run / "flightlog").mkdir(parents=True)
    (run / "flightlog/start.json").write_text(json.dumps({"ulog_path": recorded}))
    return sd, run, source


def test_import_and_repeat_preserve_sd(tmp_path):
    sd, run, source = setup_files(tmp_path)
    result = import_one(sd, run)
    assert result["state"] == "COPIED_HASH_VERIFIED"
    assert source.exists()
    assert not result["content_decoded"]
    assert import_one(sd, run)["sha256"] == result["sha256"]


def test_conflicting_destination_not_overwritten(tmp_path):
    sd, run, source = setup_files(tmp_path)
    target = run / "flightlog/test.ulg"
    target.write_bytes(b"keep me")
    with pytest.raises(ValueError, match="no overwrite"):
        import_one(sd, run)
    assert target.read_bytes() == b"keep me" and source.exists()


def test_bad_header_not_copied(tmp_path):
    sd, run, source = setup_files(tmp_path)
    source.write_bytes(b"invalid")
    with pytest.raises(ValueError, match="header"):
        import_one(sd, run)
    assert not (run / "flightlog/test.ulg").exists()


def test_parent_traversal_rejected(tmp_path):
    sd, run, _ = setup_files(tmp_path, "/fs/microsd/../other.ulg")
    with pytest.raises(ValueError):
        import_one(sd, run)


def test_deferred_receipt_imports_without_logger_start(tmp_path):
    sd, run, source = setup_files(tmp_path)
    (run / "flightlog/start.json").unlink()
    (run / "flightlog").rmdir()
    (run / "operator-result.json").write_text(
        json.dumps({"ulog_path": "/fs/microsd/log/day/test.ulg"}), encoding="utf-8"
    )
    result = import_one(sd, run)
    assert result["state"] == "COPIED_HASH_VERIFIED"
    assert (run / "flightlog/test.ulg").exists()
    assert source.exists()


def test_operator_receipt_takes_priority(tmp_path):
    sd, run, source = setup_files(tmp_path, "/fs/microsd/log/day/wrong.ulg")
    (run / "operator-result.json").write_text(
        json.dumps({"ulog_path": "/fs/microsd/log/day/test.ulg"}), encoding="utf-8"
    )
    assert import_one(sd, run)["source"] == str(source.resolve())
