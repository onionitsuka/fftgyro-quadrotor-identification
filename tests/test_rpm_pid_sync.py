import csv
from pathlib import Path

import pytest

from fftgyro_quad_id.sync.rpm_pid import MODULUS, Exchange, align, fit_clock, read_capture


def exchanges():
    return [Exchange(i, 10_000_000 + i * 1_000_000,
                     900_000_000_000 + i * 999_800_000 - 200_000,
                     900_000_000_000 + i * 999_800_000 + 800_000) for i in range(8)]


def test_skew_asymmetry_bounds_include_true_clock():
    model = fit_clock(exchanges(), slack_ns=0)
    assert model.slope == pytest.approx(.9998)
    for i in range(15):
        nano = 10_000_000 + i * 500_000
        truth = 900_000_000_000 + i * 499_900_000
        estimate, lower, upper = model.map(nano)
        assert lower - 1 <= truth <= upper + 1
        assert lower - 1 <= estimate <= upper + 1
    # A midpoint residual of zero does not eliminate asymmetric-link uncertainty.
    assert model.map(13_000_000)[2] - model.map(13_000_000)[1] >= 999_999


def test_inconsistent_clock_is_rejected():
    pairs = exchanges()
    pairs[3] = Exchange(3, pairs[3].nano_us, pairs[3].sent_ns + 50_000_000,
                        pairs[3].received_ns + 50_000_000)
    with pytest.raises(ValueError, match="affine"):
        fit_clock(pairs)


def write_capture(path: Path, *, wrap=False):
    origin = MODULUS - 500_000 if wrap else 1_000_000
    rows = []
    sequence = 0

    def frame(host, kind, nano, value=0):
        nonlocal sequence
        rows.append([host, kind, 123, sequence, nano % MODULUS, value])
        sequence += 1

    frame(1_000_000_000, "B", origin)
    for token in range(1, 4):
        host = 1_000_000_000 + token * 1_000_000_000
        nano = origin + token * 1_000_000
        rows.append([host - 200_000, "Q", 123, 0, 0, token])
        frame(host + 800_000, "T", nano, token)
        # A queued edge older than T must NOT add another 32-bit epoch.
        frame(host + 800_000, "E", nano - 1000, 1)
    frame(4_100_000_000, "Z", origin + 3_100_000)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["host_ns", "kind", "capture_id", "sequence", "nano_us", "value"])
        writer.writerows(rows)
    return origin


@pytest.mark.parametrize("wrap", [False, True])
def test_wrap_and_old_queued_edges(tmp_path, wrap):
    source = tmp_path / "rpm.csv"
    origin = write_capture(source, wrap=wrap)
    events, pairs = read_capture(source)
    edges = [e for e in events if e.kind == "E"]
    assert [e.unwrapped_us for e in edges] == [origin + i * 1_000_000 - 1000 for i in (1, 2, 3)]
    assert len(pairs) == 3


def test_unanswered_exchange_requires_explicit_audit(tmp_path):
    source = tmp_path / "rpm.csv"
    write_capture(source)
    with source.open(newline="") as stream:
        rows = list(csv.reader(stream))
    rows.insert(-1, [4_050_000_000, "Q", 123, 0, 0, 77])
    with source.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError, match="unanswered"):
        read_capture(source)
    unanswered = []
    events, pairs = read_capture(source, unanswered=unanswered)
    assert unanswered == [(77, 4_050_000_000)]
    assert len(pairs) == 3
    assert events[-1].kind == "Z"


def test_unanswered_audit_never_allows_missing_end(tmp_path):
    source = tmp_path / "rpm.csv"
    write_capture(source)
    lines = source.read_text().splitlines()
    source.write_text("\n".join(lines[:-1])+"\n")
    with pytest.raises(ValueError, match="incomplete"):
        read_capture(source, unanswered=[])


@pytest.mark.parametrize("mutation", ["gap", "capture", "missing_end", "drop", "duplicate_q"])
def test_bad_capture_rejected(tmp_path, mutation):
    source = tmp_path / "rpm.csv"
    write_capture(source)
    with source.open(newline="") as stream:
        rows = list(csv.reader(stream))
    if mutation == "gap":
        rows[3][3] = "9"
    elif mutation == "capture":
        rows[3][2] = "456"
    elif mutation == "missing_end":
        rows.pop()
    elif mutation == "drop":
        rows[-1][-1] = "1"
    else:
        rows.insert(3, rows[2])
    with source.open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    with pytest.raises(ValueError):
        read_capture(source)


def test_alignment_requires_clock_confirmation_and_preserves_sources(tmp_path):
    source, pid = tmp_path / "rpm.csv", tmp_path / "pid.csv"
    write_capture(source)
    pid.write_text("sent_ns\n1500000000\n2000000000\n2500000000\n3500000000\n")
    original = source.read_bytes()
    with pytest.raises(ValueError, match="same Pi"):
        align(source, pid, tmp_path / "no", same_pi_boot=False)
    output = tmp_path / "aligned"
    summary = align(source, pid, output, same_pi_boot=True)
    assert summary["edge_count"] == 3
    assert summary["status"] == "ALIGNED_CONDITIONAL_NOT_QUALIFIED"
    assert source.read_bytes() == original
    with pytest.raises(FileExistsError):
        align(source, pid, output, same_pi_boot=True)


def test_no_pid_overlap_rejected(tmp_path):
    source, pid = tmp_path / "rpm.csv", tmp_path / "pid.csv"
    write_capture(source)
    pid.write_text("sent_ns\n1\n2\n")
    with pytest.raises(ValueError, match="overlap"):
        align(source, pid, tmp_path / "aligned", same_pi_boot=True)


@pytest.mark.parametrize("contents", ["", "INCOMPLETE\n"])
def test_incomplete_logger_result_rejected(tmp_path, contents):
    source, pid = tmp_path / "rpm.csv", tmp_path / "pid.csv"
    write_capture(source)
    (tmp_path / "result.txt").write_text(contents)
    with pytest.raises(ValueError, match="COMPLETE"):
        align(source, pid, tmp_path / "aligned", same_pi_boot=True)


def test_many_anchors():
    pairs = [Exchange(i, i * 1_000_000, 10**12 + i * 10**9 - 100_000,
                      10**12 + i * 10**9 + 200_000) for i in range(1800)]
    model = fit_clock(pairs)
    assert model.slope == pytest.approx(1)
    _, lower, upper = model.map(900_000_000)
    assert lower <= 10**12 + 900 * 10**9 <= upper
