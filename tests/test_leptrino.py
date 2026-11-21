import struct

import pytest

from fftgyro_quad_id.io.leptrino import (Decoder, checked, frame, parse_info,
                                         parse_limits, parse_sample, status_flags)


def test_command_known_bytes():
    assert frame(bytes([4, 255, 0x32, 0])) == bytes.fromhex("100204ff32001003ca")


@pytest.mark.parametrize("split", range(12))
def test_escaped_fragmented_frames(split):
    payload = bytes([4, 255, 16, 0])
    encoded = frame(payload)
    decoder = Decoder()
    assert decoder.feed(encoded[:split]) + decoder.feed(encoded[split:]) == [payload]
    assert decoder.errors == 0


def test_bad_checksum_recovers():
    payload = bytes([4, 255, 0x33, 0])
    good = frame(payload)
    decoder = Decoder()
    assert decoder.feed(good[:-1] + bytes([good[-1] ^ 1]) + good) == [payload]
    assert decoder.errors == 1


def test_bounded_decoder():
    decoder = Decoder()
    decoder.feed(b"\x10\x02" + b"a" * 300)
    assert decoder.errors == 1
    assert len(decoder.payload) <= 255


def test_sensor_info():
    payload = bytes([38, 255, 0x2A, 0]) + b"ASM0551012R0A00".ljust(16, b"\0") + b"2506002\0" + b"1.00" + b"1200\0\0"
    assert parse_info(payload)["serial"] == "2506002"
    assert parse_info(payload)["frequency_hz"] == "1200"


def test_signed_scaling_and_status():
    limits = parse_limits(bytes([28, 255, 0x2B, 0]) + struct.pack("<6f", 100, 100, 100, 2, 2, 2))
    payload = bytes([20, 255, 0x32, 0]) + struct.pack("<6h", -10000, 0, 10000, -10000, 16, 5000) + bytes([0, 0, 1, 0])
    raw, values, status = parse_sample(payload, limits)
    assert values == pytest.approx((-100, 0, 100, -2, 0.0032, 1))
    assert status == 1
    assert raw[0] == -10000


def test_documented_status_bits_and_reserved_bits():
    assert status_flags(0x44) == {
        "compensation_rom_error": False,
        "sensor_error": False,
        "rated_overrange": True,
        "external_input": False,
        "reserved_bits": 0x40,
    }


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 0, -1])
def test_reject_bad_limits(value):
    with pytest.raises(ValueError):
        parse_limits(bytes([28, 255, 0x2B, 0]) + struct.pack("<6f", *([value] * 6)))


@pytest.mark.parametrize("payload", [b"", bytes([4, 255, 0x32, 1]), bytes([4, 255, 0x33, 0])])
def test_reject_bad_response(payload):
    with pytest.raises(ValueError):
        checked(payload, 0x32, 4)
