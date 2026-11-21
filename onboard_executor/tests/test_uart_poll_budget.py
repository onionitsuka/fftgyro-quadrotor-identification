from pathlib import Path


def test_uart_poll_has_bounded_time_and_byte_budgets() -> None:
    source = (Path(__file__).resolve().parents[1] / "src/mavlink_uart.cpp").read_text()
    assert "kPollBudgetNs = 2'000'000" in source
    assert "kPollByteBudget = 4096" in source
    assert "consumed >= kPollByteBudget" in source
    assert "monotonic_now_ns() - started_ns >= kPollBudgetNs" in source
