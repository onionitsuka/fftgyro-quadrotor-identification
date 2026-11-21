"""Direct LAN clock mapping for multi-computer experiment logs."""

from __future__ import annotations

import json
import secrets
import socket
import statistics
import time
from dataclasses import asdict, dataclass
from typing import Any

PROTOCOL = "fftgyro-clock-probe-v1"


@dataclass(frozen=True)
class ClockExchange:
    host: str
    port: int
    nonce: str
    pc_send_epoch_ns: int
    pc_send_monotonic_ns: int
    remote_receive_realtime_ns: int
    remote_receive_monotonic_ns: int
    remote_send_realtime_ns: int
    remote_send_monotonic_ns: int
    pc_receive_epoch_ns: int
    pc_receive_monotonic_ns: int

    @property
    def server_processing_ns(self) -> int:
        return self.remote_send_monotonic_ns - self.remote_receive_monotonic_ns

    @property
    def network_rtt_ns(self) -> int:
        return max(
            0,
            self.pc_receive_monotonic_ns
            - self.pc_send_monotonic_ns
            - self.server_processing_ns,
        )

    @property
    def pc_midpoint_epoch_ns(self) -> int:
        elapsed = self.pc_receive_monotonic_ns - self.pc_send_monotonic_ns
        return self.pc_send_epoch_ns + elapsed // 2

    @property
    def remote_midpoint_realtime_ns(self) -> int:
        return (
            self.remote_receive_realtime_ns + self.remote_send_realtime_ns
        ) // 2

    @property
    def remote_midpoint_monotonic_ns(self) -> int:
        return (
            self.remote_receive_monotonic_ns + self.remote_send_monotonic_ns
        ) // 2

    @property
    def pc_minus_remote_realtime_ns(self) -> int:
        return self.pc_midpoint_epoch_ns - self.remote_midpoint_realtime_ns

    @property
    def remote_monotonic_to_pc_epoch_ns(self) -> int:
        return self.pc_midpoint_epoch_ns - self.remote_midpoint_monotonic_ns

    def to_document(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "server_processing_ns": self.server_processing_ns,
            "network_rtt_ns": self.network_rtt_ns,
            "pc_minus_remote_realtime_ns": self.pc_minus_remote_realtime_ns,
            "remote_monotonic_to_pc_epoch_ns": (
                self.remote_monotonic_to_pc_epoch_ns
            ),
            "one_way_uncertainty_ns": self.network_rtt_ns // 2,
        }


def probe_udp_clock(
    host: str,
    port: int,
    *,
    samples: int = 20,
    timeout_s: float = 1.0,
) -> list[ClockExchange]:
    """Measure a remote probe server without trusting either wall clock."""
    if samples < 1:
        raise ValueError("samples must be positive")
    exchanges: list[ClockExchange] = []
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as channel:
        channel.settimeout(timeout_s)
        channel.connect((host, port))
        for _ in range(samples):
            nonce = secrets.token_hex(8)
            request = json.dumps({"protocol": PROTOCOL, "nonce": nonce}).encode()
            send_epoch_ns = time.time_ns()
            send_monotonic_ns = time.monotonic_ns()
            channel.send(request)
            payload = channel.recv(4096)
            receive_monotonic_ns = time.monotonic_ns()
            receive_epoch_ns = time.time_ns()
            response = json.loads(payload)
            if response.get("protocol") != PROTOCOL or response.get("nonce") != nonce:
                raise RuntimeError("clock probe response identity mismatch")
            exchanges.append(
                ClockExchange(
                    host=host,
                    port=port,
                    nonce=nonce,
                    pc_send_epoch_ns=send_epoch_ns,
                    pc_send_monotonic_ns=send_monotonic_ns,
                    remote_receive_realtime_ns=int(response["receive_realtime_ns"]),
                    remote_receive_monotonic_ns=int(response["receive_monotonic_ns"]),
                    remote_send_realtime_ns=int(response["send_realtime_ns"]),
                    remote_send_monotonic_ns=int(response["send_monotonic_ns"]),
                    pc_receive_epoch_ns=receive_epoch_ns,
                    pc_receive_monotonic_ns=receive_monotonic_ns,
                )
            )
    return exchanges


def probe_tcp_clock(
    host: str,
    port: int,
    *,
    samples: int = 20,
    timeout_s: float = 1.0,
) -> list[ClockExchange]:
    """Measure through one persistent TCP connection; setup latency is excluded."""
    if samples < 1:
        raise ValueError("samples must be positive")
    exchanges: list[ClockExchange] = []
    with socket.create_connection((host, port), timeout=timeout_s) as channel:
        channel.settimeout(timeout_s)
        channel.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        receive_buffer = b""
        for sample_index in range(samples):
            nonce = secrets.token_hex(8)
            request = (
                json.dumps({"protocol": PROTOCOL, "nonce": nonce}).encode() + b"\n"
            )
            send_epoch_ns = time.time_ns()
            send_monotonic_ns = time.monotonic_ns()
            channel.sendall(request)
            while b"\n" not in receive_buffer:
                try:
                    chunk = channel.recv(4096)
                except TimeoutError as error:
                    raise TimeoutError(
                        f"clock probe timed out: host={host} port={port} "
                        f"sample={sample_index + 1}/{samples}"
                    ) from error
                if not chunk:
                    raise RuntimeError("clock probe server closed the TCP stream")
                receive_buffer += chunk
            payload, receive_buffer = receive_buffer.split(b"\n", 1)
            receive_monotonic_ns = time.monotonic_ns()
            receive_epoch_ns = time.time_ns()
            response = json.loads(payload)
            if response.get("protocol") != PROTOCOL or response.get("nonce") != nonce:
                raise RuntimeError("clock probe response identity mismatch")
            exchanges.append(
                ClockExchange(
                    host=host,
                    port=port,
                    nonce=nonce,
                    pc_send_epoch_ns=send_epoch_ns,
                    pc_send_monotonic_ns=send_monotonic_ns,
                    remote_receive_realtime_ns=int(response["receive_realtime_ns"]),
                    remote_receive_monotonic_ns=int(response["receive_monotonic_ns"]),
                    remote_send_realtime_ns=int(response["send_realtime_ns"]),
                    remote_send_monotonic_ns=int(response["send_monotonic_ns"]),
                    pc_receive_epoch_ns=receive_epoch_ns,
                    pc_receive_monotonic_ns=receive_monotonic_ns,
                )
            )
    return exchanges


def summarize_exchanges(
    exchanges: list[ClockExchange], *, best_samples: int = 5
) -> dict[str, Any]:
    if not exchanges:
        raise ValueError("at least one exchange is required")
    chosen = sorted(exchanges, key=lambda item: item.network_rtt_ns)[
        : min(best_samples, len(exchanges))
    ]
    mappings = [item.remote_monotonic_to_pc_epoch_ns for item in chosen]
    realtime_offsets = [item.pc_minus_remote_realtime_ns for item in chosen]
    mapping = round(statistics.median(mappings))
    offset = round(statistics.median(realtime_offsets))
    spread = max(abs(value - mapping) for value in mappings)
    best_half_rtt = min(item.network_rtt_ns for item in chosen) // 2
    return {
        "protocol": PROTOCOL,
        "host": exchanges[0].host,
        "port": exchanges[0].port,
        "samples": len(exchanges),
        "selected_samples": len(chosen),
        "remote_monotonic_to_pc_epoch_ns": mapping,
        "pc_minus_remote_realtime_ns": offset,
        "best_network_rtt_ns": min(item.network_rtt_ns for item in chosen),
        "mapping_spread_ns": spread,
        "estimated_uncertainty_ns": max(best_half_rtt, spread),
        "exchanges": [item.to_document() for item in exchanges],
    }
