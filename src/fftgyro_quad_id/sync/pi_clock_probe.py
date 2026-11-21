"""Temporary two-Pi UDP clock probe. No system-clock or motor changes.

Client t1/t4 and server t2/t3 use monotonic_ns. Offset is server minus client.
Each causal interval describes one exchange, not the whole run or sensor latency.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import secrets
import signal
import socket
import time


_STOP_REQUESTED = False


def request_stop(signum: int, unused: object) -> None:
    """Finish the current clock bracket cleanly when the coordinator is done."""
    del signum, unused
    global _STOP_REQUESTED
    _STOP_REQUESTED = True


def stop_requested(args: argparse.Namespace) -> bool:
    path = getattr(args, "stop_file", None)
    return _STOP_REQUESTED or bool(path and Path(path).exists())


def bounds(t1: int, t2: int, t3: int, t4: int) -> tuple[int, int]:
    if t4 < t1 or t3 < t2:
        raise ValueError("Nonmonotonic timestamps")
    low, high = t3 - t4, t2 - t1
    if low > high:
        raise ValueError("Negative network round-trip time")
    return low, high


def boot_id() -> str:
    linux_id = Path("/proc/sys/kernel/random/boot_id")
    if linux_id.is_file():
        return linux_id.read_text().strip()
    # The Windows encoder probe only needs a stable identity for this process.
    # Approximate boot epoch avoids an optional platform-specific dependency.
    boot_epoch_s = round(time.time() - time.monotonic())
    return f"windows-{socket.gethostname()}-{boot_epoch_s}"


def serve(args: argparse.Namespace) -> None:
    identity = boot_id()
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.bind((args.bind, args.port))
        sock.settimeout(0.2)
        end = time.monotonic() + args.seconds
        print("Temporary clock responder ready", flush=True)
        while time.monotonic() < end and not stop_requested(args):
            try:
                data, peer = sock.recvfrom(1024)
            except socket.timeout:
                continue
            t2 = time.monotonic_ns()
            if peer[0] != args.peer:
                continue
            try:
                message = json.loads(data)
                nonce, sequence, t1 = message["nonce"], message["sequence"], message["t1"]
                if not isinstance(nonce, str) or len(nonce) != 32 or type(sequence) is not int or type(t1) is not int:
                    continue
            except (ValueError, KeyError, TypeError):
                continue
            t3 = time.monotonic_ns()
            sock.sendto(json.dumps(dict(nonce=nonce, sequence=sequence, t1=t1, t2=t2, t3=t3, boot_id=identity)).encode(), peer)


def probe(args: argparse.Namespace) -> None:
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(state="INCOMPLETE", client_hostname=socket.gethostname(), client_boot_id=boot_id(),
                   server_ip=args.peer, clock="CLOCK_MONOTONIC via Python monotonic_ns",
                   offset_direction="server minus client", hz=args.hz, seconds=args.seconds,
                   assumptions="Per-exchange constant offset and nonnegative path delay; software timestamps; unauthenticated local UDP",
                   qualified=False, sensor_latency_included=False)
    def save() -> None:
        with (out / "summary.json").open("w", encoding="utf-8") as output:
            json.dump(summary, output, indent=2)
            output.flush()
            os.fsync(output.fileno())
    save()
    rows = []
    sent = 0
    rejected = 0
    pending = {}
    late_received = 0
    received = 0
    nonce = secrets.token_hex(16)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, (out / "exchanges.csv").open("x", newline="") as output, (out / "all_replies.csv").open("x", newline="") as replies:
            sock.connect((args.peer, args.port))
            writer = csv.writer(output)
            writer.writerow(["sequence", "client_send_ns", "server_receive_ns", "server_send_ns", "client_receive_ns", "offset_low_ns", "offset_high_ns", "network_rtt_ns"])
            reply_writer = csv.writer(replies)
            reply_writer.writerow(["sequence", "client_send_ns", "server_receive_ns", "server_send_ns", "client_receive_ns", "offset_low_ns", "offset_high_ns", "network_rtt_ns"])
            start = time.monotonic()
            deadline = start + args.seconds
            next_send = start
            final_deadline = None
            final_exchange = False
            coordinated_stop = False
            while time.monotonic() < deadline:
                stopping = stop_requested(args)
                if stopping and final_deadline is None:
                    final_deadline = time.monotonic() + 2
                    stop_file = getattr(args, "stop_file", None)
                    coordinated_stop = bool(stop_file and Path(stop_file).exists())
                if final_deadline is not None and time.monotonic() >= final_deadline:
                    raise TimeoutError("No final clock exchange after stop request")
                delay = next_send - time.monotonic()
                if delay > 0:
                    time.sleep(min(delay, max(0, deadline - time.monotonic())))
                if time.monotonic() >= deadline:
                    break
                t1 = time.monotonic_ns()
                sequence = sent
                pending = {key: value for key, value in pending.items()
                           if t1 - value[0] <= 2_000_000_000}
                pending[sequence] = (t1, stopping)
                sock.send(json.dumps(dict(nonce=nonce, sequence=sequence, t1=t1)).encode())
                sent += 1
                response_deadline = min(deadline, time.monotonic() + 0.09)
                # Do not interrupt an in-flight exchange. A coordinated stop must
                # include a NEW exchange sent after the force collector closed.
                while time.monotonic() < response_deadline:
                    sock.settimeout(max(0.001, response_deadline - time.monotonic()))
                    try:
                        data = sock.recv(2048)
                    except socket.timeout:
                        break
                    t4 = time.monotonic_ns()
                    try:
                        message = json.loads(data)
                        reply_sequence = message["sequence"]
                        if type(reply_sequence) is not int:
                            raise ValueError("Invalid sequence")
                        expected = pending.get(reply_sequence)
                        if (message["nonce"] != nonce or expected is None or
                                message["t1"] != expected[0]):
                            rejected += 1
                            continue
                        t2, t3 = message["t2"], message["t3"]
                        if type(t2) is not int or type(t3) is not int or not isinstance(message["boot_id"], str):
                            raise ValueError("Invalid response types")
                        reply_t1, sent_after_stop = expected
                        if t4 - reply_t1 > 2_000_000_000:
                            raise ValueError("Expired response")
                        low, high = bounds(reply_t1, t2, t3, t4)
                    except (ValueError, TypeError, KeyError):
                        rejected += 1
                        continue
                    if "server_boot_id" in summary and summary["server_boot_id"] != message["boot_id"]:
                        raise ValueError("Server rebooted")
                    summary["server_boot_id"] = message["boot_id"]
                    del pending[reply_sequence]
                    received += 1
                    late_received += int(t4 - reply_t1 > 90_000_000)
                    row = [reply_sequence, reply_t1, t2, t3, t4, low, high, high - low]
                    reply_writer.writerow(row)
                    # Preserve the affine fitter's non-overlapping exchange
                    # contract. Keep every valid reply in the diagnostic file.
                    if rows and (reply_t1 <= rows[-1][4] or t2 <= rows[-1][3]):
                        continue
                    rows.append(row)
                    writer.writerow(row)
                    ready = getattr(args, "ready_file", None)
                    if len(rows) == 1 and ready:
                        Path(ready).write_text(str(t4), encoding="ascii")
                    if sent_after_stop:
                        final_exchange = True
                    if final_exchange or reply_sequence == sequence:
                        break
                if final_exchange:
                    break
                next_send = max(next_send + 1 / args.hz, time.monotonic())
            output.flush()
            os.fsync(output.fileno())
            replies.flush()
            os.fsync(replies.fileno())
        if not rows:
            raise TimeoutError("No valid clock exchanges")
        if stop_requested(args) and not final_exchange:
            raise TimeoutError("Clock deadline reached without final exchange")
        best = min(rows, key=lambda row: row[-1])
        widths = sorted(row[-1] / 1e6 for row in rows)
        summary.update(state="COMPLETE", best_exchange=best, network_rtt_min_ms=widths[0],
                       network_rtt_median_ms=widths[len(widths) // 2], network_rtt_max_ms=widths[-1],
                       stopped_early=stop_requested(args), final_exchange_after_stop=final_exchange,
                       coordinated_stop=coordinated_stop,
                       note="Minimum RTT interval is only a local snapshot, NOT achieved full-run synchronization accuracy")
    except (Exception, KeyboardInterrupt) as exc:
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        summary.update(sent=sent, received=received, unanswered=sent-received, rejected=rejected,
                       usable_exchanges=len(rows), overlapping_replies=received-len(rows),
                       late_received=late_received, reply_retention_s=2.0)
        save()
    print(json.dumps(summary, indent=2))
    if summary["state"] != "COMPLETE":
        raise SystemExit(1)


def main() -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = False
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["serve", "probe"])
    parser.add_argument("--peer", required=True, help="Exact peer IPv4 address")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=43187)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--hz", type=float, default=10)
    parser.add_argument("--output")
    parser.add_argument("--stop-file", help="Owned per-run request; probe finishes with a new exchange")
    parser.add_argument("--ready-file", help="Created only after the first valid clock exchange")
    args = parser.parse_args()
    socket.inet_aton(args.peer)
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 3600 or not math.isfinite(args.hz) or not 1 <= args.hz <= 10:
        parser.error("seconds must be (0,3600], hz [1,10]")
    if not 1 <= args.port <= 65535:
        parser.error("Invalid UDP port")
    if args.mode == "probe" and not args.output:
        parser.error("probe requires --output")
    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        serve(args) if args.mode == "serve" else probe(args)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)


if __name__ == "__main__":
    main()
