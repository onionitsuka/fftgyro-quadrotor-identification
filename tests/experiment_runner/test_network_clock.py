from fftgyro_quad_id.experiment_runner.network_clock import (
    ClockExchange,
    summarize_exchanges,
)


def exchange(rtt_ns: int, mapping_ns: int) -> ClockExchange:
    send_mono = 1_000_000_000
    send_epoch = 10_000_000_000
    remote_receive_mono = 2_000_000_000
    remote_send_mono = remote_receive_mono + 1_000
    receive_mono = send_mono + rtt_ns + 1_000
    pc_mid = send_epoch + (receive_mono - send_mono) // 2
    remote_mid_mono = (remote_receive_mono + remote_send_mono) // 2
    remote_mid_real = pc_mid - 123_000
    return ClockExchange(
        host="pi",
        port=39761,
        nonce=str(rtt_ns),
        pc_send_epoch_ns=send_epoch,
        pc_send_monotonic_ns=send_mono,
        remote_receive_realtime_ns=remote_mid_real - 500,
        remote_receive_monotonic_ns=remote_receive_mono,
        remote_send_realtime_ns=remote_mid_real + 500,
        remote_send_monotonic_ns=remote_send_mono,
        pc_receive_epoch_ns=send_epoch + rtt_ns + 1_000,
        pc_receive_monotonic_ns=receive_mono,
    )


def test_exchange_mapping_and_rtt() -> None:
    item = exchange(2_000_000, 0)
    assert item.network_rtt_ns == 2_000_000
    assert item.pc_minus_remote_realtime_ns == 123_000
    assert item.remote_monotonic_to_pc_epoch_ns == (
        item.pc_midpoint_epoch_ns - item.remote_midpoint_monotonic_ns
    )


def test_summary_uses_lowest_rtt_samples() -> None:
    values = [exchange(value, 0) for value in (9_000_000, 1_000_000, 3_000_000)]
    summary = summarize_exchanges(values, best_samples=2)
    assert summary["selected_samples"] == 2
    assert summary["best_network_rtt_ns"] == 1_000_000
    assert summary["estimated_uncertainty_ns"] >= 500_000
