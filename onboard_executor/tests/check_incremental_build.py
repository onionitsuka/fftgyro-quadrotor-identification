"""Native compiler check in a temporary directory. No SSH or hardware access."""

import argparse
import base64
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parents[2]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mavlink-root", required=True)
    args = parser.parse_args()
    client = load("deploy_client", "scripts/deploy_pi_executor.py")
    remote = load("deploy_remote", "onboard_executor/deploy/pi_release.py")
    with tempfile.TemporaryDirectory(prefix="fftgyro-incremental-") as temp:
        source = Path(temp) / "source"
        source.mkdir()
        package = client.source_files(ROOT)
        remote.sync_sources(source, package)
        build = Path(temp) / "build"
        configure = ["cmake", "-S", str(source / "onboard_executor"), "-B", str(build),
                     "-DFFTGYRO_BUILD_TESTS=OFF", "-DFFTGYRO_MAVLINK_ROOT=" + args.mavlink_root]
        command = ["cmake", "--build", str(build), "--target", "fftgyro-executor", "--parallel", "2"]
        subprocess.run(configure, check=True)
        started = time.monotonic()
        subprocess.run(command, check=True)
        full_seconds = time.monotonic() - started
        _, changed, removed = remote.sync_sources(source, package)
        assert changed == removed == 0
        started = time.monotonic()
        subprocess.run(configure, check=True)
        second = subprocess.run(command, check=True, capture_output=True, text=True)
        unchanged_seconds = time.monotonic() - started
        assert "Building CXX object" not in second.stdout, second.stdout
        assert "Generating FFTGYRO" not in second.stdout, second.stdout
        name = "onboard_executor/src/axis_controller.cpp"
        package[name] = base64.b64encode(base64.b64decode(package[name]) + b"\n// synthetic rebuild test\n").decode()
        _, changed, removed = remote.sync_sources(source, package)
        assert changed == 1 and removed == 0
        third = subprocess.run(command, check=True, capture_output=True, text=True)
        assert third.stdout.count("Building CXX object") == 1, third.stdout
        assert "axis_controller.cpp.o" in third.stdout and "Generating FFTGYRO" not in third.stdout
        subprocess.run([str(build / "fftgyro-executor"), "session-validate",
                        str(client.DEFAULT_CONFIG)], check=True)
        print(f"PASS: initial native build {full_seconds:.2f}s; unchanged configure/build {unchanged_seconds:.2f}s")
        print("PASS: unchanged=0 compile, one changed .cpp=1 compile; cascade config validated")
        print("Synthetic local validation only; not Pi deployment or Pi timing")


if __name__ == "__main__":
    main()
