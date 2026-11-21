"""Pi-side PX4 UTC + standard logger lifecycle. No motor or arming commands.

Runs standalone through the Pi's pymavlink interpreter; no package install needed.
The UART must be free: invoke start before executor launch, stop after executor exit.
"""

import argparse
import json
import math
from pathlib import Path
import re
import secrets
import statistics
import struct
import subprocess
import time


MIN_SD_FREE_BYTES = 1024 ** 3
SD_MOUNTPOINT = "/fs/microsd"
LIGHTWEIGHT_LOGGING_PARAMETERS = {
    # Per-run capture is explicitly bracketed by `logger on` / `logger off`.
    "SDLOG_MODE": 0,
    # Default analysis topics + full-rate system-identification inputs/outputs.
    "SDLOG_PROFILE": 9,
    # Store on the Pixhawk SD only; bulk retrieval is intentionally deferred.
    "SDLOG_BACKEND": 1,
    "SDLOG_MISSION": 0,
}


def install_pymavlink_instance_workaround(mavutil):
    """Avoid pymavlink 2.4.49 crashing when a type changes to instance form.

    Some PX4 message streams first populate mavutil's cache with a plain
    message whose ``_instances`` member is None, then deliver an instance-keyed
    form of the same type.  Upstream ``add_message`` assumes the dictionary
    already exists.  Drop only that stale cache entry and let the original
    function recreate the documented instance cache.
    """
    original = mavutil.add_message

    def add_message(messages, message_type, message):
        instance_field = getattr(message, "_instance_field", None)
        instance_value = getattr(message, instance_field, None) if instance_field else None
        cached = messages.get(message_type)
        if instance_value is not None and cached is not None and getattr(cached, "_instances", None) is None:
            del messages[message_type]
        return original(messages, message_type, message)

    mavutil.add_message = add_message


def require_synced_utc():
    result = subprocess.run(["timedatectl", "show", "-p", "NTPSynchronized", "--value"],
                            capture_output=True, text=True, timeout=3, check=True)
    if result.stdout.strip().lower() not in ("yes", "true"):
        raise RuntimeError("Pi UTC is not NTP-synchronized; no PX4 clock change or logger on")


def _battery_voltage_v(message):
    """Return the primary PX4 battery voltage, rejecting MAVLink sentinels."""
    message_type = message.get_type()
    if message_type == "SYS_STATUS":
        millivolts = getattr(message, "voltage_battery", None)
    elif message_type == "BATTERY_STATUS":
        # PX4 reports pack voltage in element zero when cell voltages are unavailable.
        if getattr(message, "id", 0) not in (0, None):
            return None
        voltages = getattr(message, "voltages", None)
        millivolts = voltages[0] if voltages else None
    else:
        return None
    if isinstance(millivolts, bool) or not isinstance(millivolts, (int, float)):
        return None
    if millivolts in (0, 65535):
        return None
    voltage = float(millivolts) / 1000.0
    return voltage if math.isfinite(voltage) and 1.0 <= voltage <= 100.0 else None


def sample_battery_voltage(shell, timeout_s=4.0, minimum_samples=3, monotonic=time.monotonic):
    """Measure the connected primary battery from the live PX4 MAVLink stream."""
    if timeout_s <= 0 or minimum_samples < 2:
        raise ValueError("invalid battery sampling request")
    values = []
    sources = set()
    deadline = monotonic() + timeout_s
    while monotonic() < deadline and len(values) < minimum_samples:
        message = shell.link.recv_match(
            type=["HEARTBEAT", "SYS_STATUS", "BATTERY_STATUS"], blocking=True, timeout=.25)
        if not message or message.get_srcSystem() != 1 or message.get_srcComponent() != 1:
            continue
        if message.get_type() == "HEARTBEAT" and message.base_mode & 128:
            raise RuntimeError("PX4 is armed during battery preflight")
        voltage = _battery_voltage_v(message)
        if voltage is not None:
            values.append(voltage)
            sources.add(message.get_type())
    if len(values) < minimum_samples:
        raise RuntimeError(
            f"PX4 battery voltage unavailable: {len(values)}/{minimum_samples} samples")
    return {
        "voltage_v": statistics.median(values),
        "observed_min_v": min(values),
        "observed_max_v": max(values),
        "samples": len(values),
        "sources": sorted(sources),
    }


def verify_battery_preflight(shell, record, save, minimum_v, maximum_v):
    if not (math.isfinite(minimum_v) and math.isfinite(maximum_v) and
            0 < minimum_v < maximum_v):
        raise ValueError("invalid battery voltage limits")
    measurement = sample_battery_voltage(shell)
    passed = minimum_v <= measurement["observed_min_v"] and measurement["observed_max_v"] <= maximum_v
    measurement.update(allowed_min_v=minimum_v, allowed_max_v=maximum_v, passed=passed)
    record["battery_preflight"] = measurement
    save()
    if not passed:
        raise RuntimeError(
            f"battery voltage {measurement['voltage_v']:.3f} V outside "
            f"{minimum_v:.2f}..{maximum_v:.2f} V; no logger or motor start")
    return measurement


def parse_sd_capacity(response):
    """Select only the mounted PX4 microSD row from NuttX `df -h`."""
    rows = []
    for line in response.splitlines():
        fields = line.split()
        if len(fields) != 5 or fields[-1] != SD_MOUNTPOINT:
            continue
        if fields[0].lower() not in ("vfat", "fat"):
            continue
        match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)([BKMGTP])", fields[3], re.IGNORECASE)
        if not match:
            continue
        magnitude = float(match.group(1))
        exponent = {"B": 0, "K": 1, "M": 2, "G": 3, "T": 4, "P": 5}[match.group(2).upper()]
        available_bytes = int(magnitude * (1024 ** exponent))
        rows.append({
            "filesystem": fields[0],
            "mountpoint": SD_MOUNTPOINT,
            "available_reported": fields[3],
            "available_bytes_approx": available_bytes,
            "precision": "rounded human-readable df output",
        })
    if len(rows) != 1:
        raise RuntimeError("PX4 microSD capacity missing or ambiguous in df -h response")
    return rows[0]


def verify_sd_capacity_preflight(shell, record, save):
    """Fail closed before logger on when the flight-controller SD is low.

    NuttX ``df -h`` reports byte counts through a 32-bit value on this build,
    so a 32 GB FAT32 card wraps roughly every 4 GiB (28.87 GiB appeared as
    883 MiB).  STORAGE_INFORMATION carries capacities in MiB without that
    overflow and is therefore the authoritative preflight source.
    """
    message = shell.storage_information()
    status = int(message.status)
    total_mib = float(message.total_capacity)
    used_mib = float(message.used_capacity)
    available_mib = float(message.available_capacity)
    if status != 2 or not all(math.isfinite(value) and value >= 0
                              for value in (total_mib, used_mib, available_mib)):
        raise RuntimeError("PX4 microSD STORAGE_INFORMATION is not ready or invalid")
    capacity = {
        "source": "MAVLink STORAGE_INFORMATION",
        "storage_id": int(message.storage_id),
        "storage_count": int(message.storage_count),
        "status": status,
        "name": str(getattr(message, "name", "")),
        "total_mib": total_mib,
        "used_mib": used_mib,
        "available_mib": available_mib,
        "available_reported": f"{available_mib:.0f} MiB",
        "available_bytes_approx": int(available_mib * 1024 ** 2),
        "precision": "MAVLink float capacity in MiB",
    }
    passed = capacity["available_bytes_approx"] > MIN_SD_FREE_BYTES
    capacity.update(minimum_free_bytes=MIN_SD_FREE_BYTES, passed=passed)
    record["sd_capacity_preflight"] = capacity
    save()
    if not passed:
        raise RuntimeError(
            f"PX4 microSD free space {capacity['available_reported']} at or below 1 GiB; "
            "no logger or motor start"
        )
    return capacity


def verify_logging_profile_preflight(shell, record, save):
    """Require the lean PX4 profile before creating an experiment ULog.

    These parameters are reboot-required in PX4.  Silently changing them in a
    run would therefore be misleading: read the live values and fail before
    acquisition or motor output when the booted logger still uses another
    profile.
    """
    actual = {
        name: shell.parameter(name)
        for name in LIGHTWEIGHT_LOGGING_PARAMETERS
    }
    mismatches = {
        name: {"actual": actual[name], "expected": expected}
        for name, expected in LIGHTWEIGHT_LOGGING_PARAMETERS.items()
        if actual[name] != expected
    }
    result = {
        "actual": actual,
        "expected": dict(LIGHTWEIGHT_LOGGING_PARAMETERS),
        "passed": not mismatches,
        "mismatches": mismatches,
        "profile_note": (
            "SDLOG_PROFILE=9 keeps the default analysis set and full-rate "
            "system-identification topics without the broad high-rate/debug sets"
        ),
    }
    record["logging_profile_preflight"] = result
    save()
    if mismatches:
        details = ", ".join(
            f"{name}={values['actual']} expected {values['expected']}"
            for name, values in mismatches.items()
        )
        raise RuntimeError(
            "PX4 lightweight logging parameters not active (" + details
            + "); set parameters and reboot Pixhawk; no logger or motor start"
        )
    return result


class Shell:
    def __init__(self, port, baud):
        from pymavlink import mavutil
        install_pymavlink_instance_workaround(mavutil)
        self.link = mavutil.mavlink_connection(port, baud=baud,
                                             source_system=246, source_component=191)
        try:
            deadline = time.monotonic() + 5
            while time.monotonic() < deadline:
                message = self.link.recv_match(type="HEARTBEAT", blocking=True, timeout=.2)
                if message and message.get_srcSystem() == 1 and message.get_srcComponent() == 1:
                    if message.base_mode & 128:
                        raise RuntimeError("PX4 is armed; trial logger setup requires stopped executor")
                    break
            else:
                raise RuntimeError("PX4 heartbeat missing")
            # An interrupted previous client can leave PX4's SERIAL_CONTROL
            # shell channel allocated even though no process owns the UART.
            # Release that stale channel before claiming it for this run.  The
            # first NSH banner/echo can also take longer than later commands.
            self.link.mav.serial_control_send(10, 0, 0, 0, 0, [0] * 70)
            time.sleep(.1)
            self.command("", timeout_s=8)
        except Exception:
            self.close()
            raise

    def command(self, command, timeout_s=3):
        # A unique standalone echo line separates actual output from command echo and stale replies.
        token = "FG" + secrets.token_hex(8)
        wire = (command + "\necho " + token + "\n").encode("ascii")
        for index in range(0, len(wire), 70):
            chunk = wire[index:index + 70]
            # Request responses without the EXCLUSIVE bit.  In the 0268
            # preflight, RESPOND|EXCLUSIVE (6) yielded no shell replies;
            # RESPOND alone (2) answered after the same channel release.
            self.link.mav.serial_control_send(10, 2, 0, 0, len(chunk),
                                              list(chunk) + [0] * (70 - len(chunk)))
        output = ""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self.link.recv_match(blocking=True, timeout=.1)
            if not message or message.get_srcSystem() != 1 or message.get_srcComponent() != 1:
                continue
            if message.get_type() == "HEARTBEAT" and message.base_mode & 128:
                raise RuntimeError("PX4 became armed during logger setup")
            if message.get_type() == "SERIAL_CONTROL" and message.device == 10:
                output += bytes(message.data[:message.count]).decode("ascii", errors="replace")
                if len(output) > 65536:
                    raise RuntimeError("PX4 shell response too large")
                clean = re.sub(r"\x1b\[[0-9;]*[A-Za-z]", "", output).replace("\r", "")
                if re.search(r"(?m)^" + token + r"$", clean):
                    return clean
        raise RuntimeError("PX4 shell response timeout: " + command)

    def close(self):
        try:
            self.link.mav.serial_control_send(10, 0, 0, 0, 0, [0] * 70)
        finally:
            self.link.close()

    def parameter(self, name, timeout_s=3):
        """Read one PX4 parameter with its declared MAVLink scalar type."""
        from pymavlink import mavutil

        self.link.mav.param_request_read_send(1, 1, name.encode("ascii"), -1)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            message = self.link.recv_match(blocking=True, timeout=.2)
            if not message or message.get_srcSystem() != 1 or message.get_srcComponent() != 1:
                continue
            if message.get_type() == "HEARTBEAT" and message.base_mode & 128:
                raise RuntimeError("PX4 became armed during parameter preflight")
            if message.get_type() != "PARAM_VALUE":
                continue
            received = message.param_id
            if isinstance(received, bytes):
                received = received.split(b"\0", 1)[0].decode("ascii")
            else:
                received = received.rstrip("\0")
            if received != name:
                continue
            parameter_type = int(message.param_type)
            if parameter_type == mavutil.mavlink.MAV_PARAM_TYPE_REAL32:
                return float(message.param_value)
            formats = {
                mavutil.mavlink.MAV_PARAM_TYPE_UINT8: "B",
                mavutil.mavlink.MAV_PARAM_TYPE_INT8: "b",
                mavutil.mavlink.MAV_PARAM_TYPE_UINT16: "H",
                mavutil.mavlink.MAV_PARAM_TYPE_INT16: "h",
                mavutil.mavlink.MAV_PARAM_TYPE_UINT32: "I",
                mavutil.mavlink.MAV_PARAM_TYPE_INT32: "i",
            }
            code = formats.get(parameter_type)
            if code is None:
                raise RuntimeError(f"unsupported MAVLink parameter type for {name}")
            raw = struct.pack("<f", float(message.param_value))
            return int(struct.unpack("<" + code, raw[:struct.calcsize(code)])[0])
        raise RuntimeError(f"PX4 parameter read timeout: {name}")

    def storage_information(self, timeout_s=5, attempts=3, monotonic=time.monotonic):
        """Request the primary microSD capacity without NuttX df overflow.

        PX4 can occasionally miss this request while the MAVLink stream is
        settling after another short-lived client disconnects.  Retry the
        read-only request, but keep the preflight fail-closed if no valid
        response arrives.
        """
        from pymavlink import mavutil

        if timeout_s <= 0 or attempts < 1:
            raise ValueError("invalid PX4 storage information retry request")
        for _ in range(attempts):
            self.link.mav.command_long_send(
                1, 1, mavutil.mavlink.MAV_CMD_REQUEST_STORAGE_INFORMATION,
                0, 2, 0, 0, 0, 0, 0, 0,
            )
            deadline = monotonic() + timeout_s
            while monotonic() < deadline:
                message = self.link.recv_match(blocking=True, timeout=.2)
                if not message or message.get_srcSystem() != 1 or message.get_srcComponent() != 1:
                    continue
                if message.get_type() == "HEARTBEAT" and message.base_mode & 128:
                    raise RuntimeError("PX4 became armed during storage preflight")
                # PX4 currently reports the only mounted microSD as storage_id=1
                # even when MAV_CMD_REQUEST_STORAGE_INFORMATION is requested with
                # parameter 1 set to 2.  With one storage device, the first ready
                # STORAGE_INFORMATION response is authoritative.
                if (message.get_type() == "STORAGE_INFORMATION"
                        and int(message.storage_count) == 1):
                    return message
        raise RuntimeError(
            f"PX4 microSD STORAGE_INFORMATION timeout after {attempts} attempts")

    def system_time_anchor(self, timeout_s=3.0, monotonic_ns=time.monotonic_ns):
        """Bracket one PX4 boot-time observation on the aircraft Pi clock.

        ``SYSTEM_TIME.time_boot_ms`` and ULog ``timestamp[_sample]`` share the
        PX4 boot-time domain.  The Pi timestamps bracket message generation;
        retaining the interval avoids pretending that MAVLink transport delay
        is known or symmetric.
        """
        from pymavlink import mavutil

        before_ns = monotonic_ns()
        self.link.mav.command_long_send(
            1, 1, mavutil.mavlink.MAV_CMD_REQUEST_MESSAGE,
            0, mavutil.mavlink.MAVLINK_MSG_ID_SYSTEM_TIME, 0, 0, 0, 0, 0, 0,
        )
        deadline_ns = before_ns + round(timeout_s * 1e9)
        while monotonic_ns() < deadline_ns:
            message = self.link.recv_match(blocking=True, timeout=.2)
            if not message or message.get_srcSystem() != 1 or message.get_srcComponent() != 1:
                continue
            if message.get_type() == "HEARTBEAT" and message.base_mode & 128:
                raise RuntimeError("PX4 became armed during clock-anchor request")
            if message.get_type() != "SYSTEM_TIME":
                continue
            after_ns = monotonic_ns()
            boot_ms = int(message.time_boot_ms)
            unix_us = int(message.time_unix_usec)
            if boot_ms <= 0 or unix_us <= 0 or after_ns < before_ns:
                raise RuntimeError("PX4 SYSTEM_TIME clock anchor is invalid")
            return {
                "px4_boot_us": boot_ms * 1000,
                "px4_unix_us": unix_us,
                "aircraft_monotonic_lower_ns": before_ns,
                "aircraft_monotonic_upper_ns": after_ns,
                "aircraft_monotonic_estimate_ns": (before_ns + after_ns) // 2,
                "interval_width_ms": (after_ns - before_ns) / 1e6,
                "method": "MAV_CMD_REQUEST_MESSAGE SYSTEM_TIME; causal round-trip interval",
            }
        raise RuntimeError("PX4 SYSTEM_TIME clock-anchor timeout")


def start_logger(shell, record, save, now=time.time, monotonic=time.monotonic):
    status = shell.command("logger status")
    record["status_before"] = status
    if "Not logging" not in status or "Logging Running" in status:
        raise RuntimeError("PX4 logger is already recording or its status is unknown")
    sent = now()
    sent_mono = monotonic()
    if not 1577836800 <= sent < 2145916800:
        raise RuntimeError("Pi UTC outside supported 2020..2037 range")
    epoch = int(sent)
    record.update(utc_source="Pi CLOCK_REALTIME (NTP synchronized)", sent_unix_s=epoch)
    response = shell.command("system_time set " + str(epoch))
    record["set_response"] = response
    if "Successfully set system time" not in response:
        raise RuntimeError("PX4 rejected UTC setting")
    response = shell.command("system_time get")
    record["get_response"] = response
    values = re.findall(r"Unix epoch time:\s*(\d+)", response)
    if len(values) != 1:
        raise RuntimeError("PX4 UTC readback missing or ambiguous")
    received, elapsed = now(), monotonic() - sent_mono
    if abs((received - sent) - elapsed) > .5:
        raise RuntimeError("Pi wall clock stepped during UTC exchange")
    readback = int(values[0])
    record.update(readback_unix_s=readback, readback_error_s=readback - received,
                  exchange_seconds=elapsed)
    if not epoch <= readback <= int(received) + 1 or abs(readback - received) > 2:
        raise RuntimeError("PX4 UTC readback differs by more than 2 seconds")
    record["utc_verified"] = True
    # Persist ownership intent before sending; cleanup may stop our partially started logger.
    record["logger_on_attempted"] = True
    save()
    record["on_response"] = shell.command("logger on")
    deadline = monotonic() + 3
    while monotonic() < deadline:
        status = shell.command("logger status")
        record["status_after"] = status
        paths = re.findall(r"Log file:\s*(/\S+\.ulg)\b", status)
        if "Full File Logging Running:" in status and paths:
            record.update(ulog_path=paths[0], logger_running=True)
            record["px4_clock_anchor"] = shell.system_time_anchor()
            save()
            return
        time.sleep(.1)
    raise RuntimeError("PX4 full ULog recording did not start")


def verify_sd_log(shell, record, start_record):
    """Metadata evidence only; ULog decoding happens after physical SD import."""
    path = start_record.get("ulog_path", "")
    if not re.fullmatch(r"/fs/microsd/log/[A-Za-z0-9_./-]+\.ulg", path) or ".." in path.split("/"):
        record.update(sd_log_verified=False, sd_check_error="missing or invalid owned ULog path")
        return
    record["ulog_path"] = path
    response = shell.command("ls -l " + path)
    record["sd_listing"] = response
    # NuttX may print the size immediately after the ten mode characters
    # ("-rw-rw-rw-13877786 ...") or separate it with whitespace.
    sizes = []
    for line in response.splitlines():
        match = re.match(r"^\s*-[rwx-]{9}\s*(\d+)\s+(\S+)\s*$", line)
        if not match:
            continue
        if match.group(2) not in (path, Path(path).name):
            continue
        sizes.append(int(match.group(1)))
    record["sd_file_bytes"] = sizes[0] if len(sizes) == 1 else None
    record["sd_log_verified"] = len(sizes) == 1 and sizes[0] > 16
    record["content_verified"] = False
    record["retrieval"] = "physical SD import after experiments"


def stop_logger(shell, record, monotonic=time.monotonic, start_record=None):
    # A failed diagnostic must never prevent logger off.
    try:
        status = shell.command("logger status")
        record["status_before_stop"] = status
        record["written_kib"] = [float(value) * (1024 if unit == "MiB" else 1)
                                 for value, unit in re.findall(r"Wrote\s+([0-9.]+)\s+(KiB|MiB)", status)]
        record["dropouts_reported"] = [int(v) for v in re.findall(r"dropouts:\s*(\d+)", status)]
        paths = re.findall(r"Log file:\s*(/\S+\.ulg)\b", status)
        record["same_log_running_before_stop"] = bool(start_record and
            "Full File Logging Running:" in status and paths == [start_record.get("ulog_path")])
    except Exception as error:
        record["status_check_error"] = str(error)
    record["off_response"] = shell.command("logger off")
    deadline = monotonic() + 3
    while monotonic() < deadline:
        status = shell.command("logger status")
        record["status_after"] = status
        if "Not logging" in status and "Logging Running" not in status:
            record["logger_stopped"] = True
            record["px4_clock_anchor"] = shell.system_time_anchor()
            if start_record is not None:
                try:
                    verify_sd_log(shell, record, start_record)
                except Exception as error:
                    record.update(sd_log_verified=False, sd_check_error=str(error))
            return
        time.sleep(.1)
    raise RuntimeError("PX4 logger stop not confirmed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "stop"))
    parser.add_argument("--port", default="/dev/serial0")
    parser.add_argument("--baud", type=int, default=921600)
    parser.add_argument("--battery-min-v", type=float, default=14.5)
    parser.add_argument("--battery-max-v", type=float, default=16.9)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an earlier trial's UTC evidence.
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write("{}")
    record = dict(action=args.action, logger_on_attempted=False, utc_verified=False,
                  complete=False, motor_commands_sent=False)

    def save():
        pending = args.output.with_suffix(".pending")
        pending.write_text(json.dumps(record, indent=2), encoding="utf-8")
        pending.replace(args.output)

    shell = None
    try:
        if args.action == "start":
            require_synced_utc()
        shell = Shell(args.port, args.baud)
        if args.action == "start":
            verify_logging_profile_preflight(shell, record, save)
            verify_sd_capacity_preflight(shell, record, save)
            verify_battery_preflight(
                shell, record, save, args.battery_min_v, args.battery_max_v)
            start_logger(shell, record, save)
        else:
            start_path = args.output.with_name("start.json")
            try:
                owned = json.loads(start_path.read_text(encoding="utf-8")) if start_path.exists() else {}
            except (ValueError, OSError) as error:
                record["start_record_error"] = str(error)
                owned = {}
            stop_logger(shell, record, start_record=owned)
            try:
                record["battery_postflight"] = sample_battery_voltage(shell)
            except Exception as error:
                # SD verification already completed. Preserve missing end voltage
                # as evidence without undoing a safely closed PX4 logger.
                record["battery_postflight_error"] = str(error)
        record["complete"] = True
    except Exception as error:
        record["error"] = str(error)
        raise
    finally:
        try:
            if shell:
                shell.close()
        finally:
            save()


if __name__ == "__main__":
    main()
