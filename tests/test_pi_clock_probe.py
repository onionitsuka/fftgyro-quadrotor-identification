import pytest
import csv
import json
import socket
import threading
import time
from types import SimpleNamespace

from fftgyro_quad_id.sync.pi_clock_probe import bounds
from fftgyro_quad_id.sync import pi_clock_probe


def test_asymmetric_exchange_contains_true_offset():
    # Client send=100, forward=7, offset=1000, server processing=3, reverse=20.
    assert bounds(100, 1107, 1110, 130) == (980, 1007)


def test_zero_path_delay():
    assert bounds(100, 1100, 1110, 110) == (1000, 1000)


@pytest.mark.parametrize("timestamps", [(10, 20, 21, 9), (10, 21, 20, 30), (10, 20, 40, 11)])
def test_impossible_exchange(timestamps):
    with pytest.raises(ValueError):
        bounds(*timestamps)


def test_coordinator_stop_request_is_visible():
    pi_clock_probe._STOP_REQUESTED = False
    pi_clock_probe.request_stop(15, None)
    assert pi_clock_probe._STOP_REQUESTED
    pi_clock_probe._STOP_REQUESTED = False


def test_delayed_replies_are_retained_without_breaking_affine_contract(tmp_path, monkeypatch):
    from fftgyro_quad_id.sync.pi_affine import fit, read_exchanges
    monkeypatch.setattr(pi_clock_probe, "boot_id", lambda: "client")
    monkeypatch.setattr(pi_clock_probe, "_STOP_REQUESTED", False)
    done = threading.Event()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(.005)

        def respond():
            pending = []
            while not done.is_set():
                try:
                    data, peer = sock.recvfrom(2048)
                    request = json.loads(data)
                    stamp = time.monotonic_ns()
                    reply = json.dumps(dict(request, t2=stamp, t3=stamp, boot_id="server")).encode()
                    pending.append((time.monotonic() + .15, reply, peer))
                except socket.timeout:
                    pass
                for entry in pending[:]:
                    if time.monotonic() >= entry[0]:
                        sock.sendto(entry[1], entry[2])
                        pending.remove(entry)

        worker = threading.Thread(target=respond)
        worker.start()
        output = tmp_path / "clock"
        try:
            pi_clock_probe.probe(SimpleNamespace(peer="127.0.0.1", port=sock.getsockname()[1],
                                                seconds=1.5, hz=10, output=output))
        finally:
            done.set()
            worker.join(timeout=2)
        assert not worker.is_alive()
    summary = json.loads((output / "summary.json").read_text())
    assert summary["received"] >= 5
    assert summary["late_received"] == summary["received"]
    assert summary["overlapping_replies"] > 0
    rows = read_exchanges(output / "exchanges.csv")
    fit(rows)


@pytest.mark.parametrize("drop_tail", [False, True])
def test_stop_requires_a_new_valid_exchange_after_force_closed(tmp_path, monkeypatch, drop_tail):
    """Real local UDP, no Pi/device access; loss of the tail cannot be COMPLETE."""
    monkeypatch.setattr(pi_clock_probe, "boot_id", lambda: "test-client-boot")
    monkeypatch.setattr(pi_clock_probe, "_STOP_REQUESTED", False)
    stop = tmp_path / "force-closed"
    ready = tmp_path / "clock-ready"
    done = threading.Event()
    errors = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.settimeout(.05)

        def respond():
            try:
                while not done.is_set():
                    try:
                        data, peer = sock.recvfrom(2048)
                    except socket.timeout:
                        continue
                    if drop_tail and stop.exists():
                        continue
                    stamp = time.monotonic_ns()
                    request = json.loads(data)
                    sock.sendto(json.dumps(dict(request, t2=stamp, t3=time.monotonic_ns(),
                                                boot_id="test-server-boot")).encode(), peer)
            except Exception as error:
                errors.append(error)

        def close_force():
            while not done.wait(.01):
                if ready.exists():
                    stop.write_text(str(time.monotonic_ns()))
                    return

        workers = [threading.Thread(target=fn) for fn in (respond, close_force)]
        for worker in workers:
            worker.start()
        args = SimpleNamespace(peer="127.0.0.1", port=sock.getsockname()[1], seconds=5, hz=10,
                               output=tmp_path / "clock", stop_file=stop, ready_file=ready)
        try:
            if drop_tail:
                with pytest.raises(SystemExit, match="1"):
                    pi_clock_probe.probe(args)
            else:
                pi_clock_probe.probe(args)
        finally:
            done.set()
            for worker in workers:
                worker.join(timeout=2)
            assert not any(worker.is_alive() for worker in workers)
    assert not errors
    summary = json.loads((args.output / "summary.json").read_text())
    if drop_tail:
        assert summary["state"] == "INCOMPLETE" and "final clock exchange" in summary["error"]
    else:
        assert summary["state"] == "COMPLETE" and summary["final_exchange_after_stop"] and summary["coordinated_stop"]
        with (args.output / "exchanges.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        assert int(rows[-1]["client_send_ns"]) >= int(stop.read_text())
