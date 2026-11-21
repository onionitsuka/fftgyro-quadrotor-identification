"""Flash the approved FMUv6X image, following the device across USB renaming."""
import argparse
import importlib.util
import time

import serial
from serial.tools import list_ports
from pymavlink.dialects.v20 import ardupilotmega

parser = argparse.ArgumentParser(description="Flash an explicitly selected FMUv6X image while disarmed")
parser.add_argument("device", help="COM port, for example COM7")
parser.add_argument("firmware", help="Local .px4 firmware file")
parser.add_argument("--uploader", required=True, help="Official PX4 Tools/px_uploader.py file")
parser.add_argument("--serial", default="280042000551323436333634", help="USB bootloader serial of the controller")
args = parser.parse_args()
SERIAL = args.serial
FIRMWARE = args.firmware
ports = [p for p in list_ports.comports() if p.device == args.device]
if len(ports) != 1:
    raise SystemExit("Selected serial port not present")
LOCATION = ports[0].location
spec = importlib.util.spec_from_file_location('px_upload', args.uploader)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ports = [p for p in list_ports.comports() if p.device == args.device and p.vid == 12677 and p.pid == 53]
if not ports:
    raise SystemExit('Target Pixhawk not connected')
with serial.Serial(sorted(ports, key=lambda p: p.device)[0].device, 115200, timeout=0.2) as connection:
    link = ardupilotmega.MAVLink(connection, srcSystem=246, srcComponent=191)
    link.robust_parsing = True
    heartbeat = None
    deadline = time.monotonic() + 12
    while time.monotonic() < deadline and heartbeat is None:
        for message in link.parse_buffer(connection.read(max(1, connection.in_waiting))) or []:
            if message.get_type() == 'HEARTBEAT' and message.get_srcComponent() == 1:
                heartbeat = message
    if heartbeat is None or heartbeat.base_mode & 128:
        raise SystemExit('Missing heartbeat or armed: refusing reboot')
    print('Disarmed heartbeat; requesting bootloader on exact system/component',
          heartbeat.get_srcSystem(), heartbeat.get_srcComponent(), flush=True)
    link.command_int_send(heartbeat.get_srcSystem(), heartbeat.get_srcComponent(), 0,
                          246, 0, 0, 3, 0, 0, 0, 0, 0, 0)
    connection.flush()
    ack_deadline = time.monotonic() + 2
    accepted = False
    while time.monotonic() < ack_deadline:
        try:
            messages = link.parse_buffer(connection.read(max(1, connection.in_waiting))) or []
        except (serial.SerialException, OSError):
            break
        for message in messages:
            if message.get_type() == 'COMMAND_ACK' and message.command == 246:
                print('Reboot ACK:', message.to_dict(), flush=True)
                if message.result != 0:
                    raise SystemExit('Reboot rejected; no upload attempted')
                accepted = True
        if accepted:
            break

deadline = time.monotonic() + 25
last_error = None
while time.monotonic() < deadline:
    for port in list_ports.comports():
        # Only follow this physical controller, never scan unrelated serial devices.
        if port.serial_number != SERIAL or (LOCATION is not None and port.location != LOCATION):
            continue
        uploader = None
        try:
            uploader = module.uploader(port.device, 115200, [115200])
            uploader.identify()
        except (RuntimeError, serial.SerialException, OSError) as error:
            description = f'{port.device}: {error}'
            if description != last_error:
                print('Waiting:', description, flush=True)
                last_error = description
            if uploader:
                uploader.close()
            continue
        try:
            print('Bootloader board:', uploader.board_type, 'revision:', uploader.bl_rev, flush=True)
            if uploader.board_type != 53:
                raise SystemExit('Wrong board: refusing upload')
            uploader.upload([FIRMWARE], force=False)
            print('UPLOAD VERIFIED AND REBOOTED', flush=True)
            raise SystemExit(0)
        finally:
            uploader.close()
    time.sleep(0.1)
raise SystemExit('Bootloader not identified within 25 seconds; no erase performed')
