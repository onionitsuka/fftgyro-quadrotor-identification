"""Pi-side staging helper, Python standard library only. Never runs a session."""

import base64
import hashlib
import json
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import subprocess
import sys
import time

MARKER = ".fftgyro-deploy-v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(root, name):
    relative = PurePosixPath(name)
    if (not name or relative.is_absolute() or ".." in relative.parts or
            "\\" in name or ":" in name or str(relative) != name):
        raise ValueError("Unsafe relative path: " + name)
    path = root.joinpath(*relative.parts)
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Symlink is not allowed: " + str(path))
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("Path escapes deployment root")
    return path


def source_name(name):
    return (name in ("onboard_executor/CMakeLists.txt", "protocol/fftgyro.xml",
                     "firmware/rpm_capture_nano/rpm_wire.h") or
            name.startswith(("onboard_executor/src/", "onboard_executor/include/",
                             "onboard_executor/cmake/")))


def sync_sources(source_root, encoded):
    """Preserve unchanged file timestamps; remove only previously managed source files."""
    decoded = {}
    for name, data in encoded.items():
        if not source_name(name):
            raise ValueError("Not a build input: " + name)
        safe_path(source_root, name)
        decoded[name] = base64.b64decode(data, validate=True)
    if not {"onboard_executor/CMakeLists.txt", "protocol/fftgyro.xml"} <= decoded.keys():
        raise ValueError("Incomplete source package")
    manifest = safe_path(source_root, ".sources.json")
    old = json.loads(manifest.read_text()) if manifest.exists() else {}
    # Validate all stale targets before making any changes.
    for name in old:
        if not source_name(name):
            raise ValueError("Invalid managed-source manifest")
        safe_path(source_root, name)
    hashes = {name: digest(data) for name, data in decoded.items()}
    # Track pending writes too, so an interrupted sync cannot leave an unowned stale header.
    pending = safe_path(source_root, ".sources.pending")
    pending.write_text(json.dumps({**old, **hashes}, sort_keys=True), encoding="utf-8")
    pending.replace(manifest)
    changed = 0
    for name, data in decoded.items():
        path = safe_path(source_root, name)
        if not path.exists() or path.read_bytes() != data:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            changed += 1
    removed = 0
    for name in old.keys() - decoded.keys():
        path = safe_path(source_root, name)
        if path.exists():
            path.unlink()
            removed += 1
    pending.write_text(json.dumps(hashes, sort_keys=True), encoding="utf-8")
    pending.replace(manifest)
    return hashes, changed, removed


def checked(command, log=None):
    subprocess.run([str(v) for v in command], check=True, stdout=log,
                   stderr=subprocess.STDOUT if log else None, timeout=1800)


def prepare_root(raw):
    root = Path(raw)
    if not root.is_absolute() or root == Path.home() or len(root.parts) < 4:
        raise ValueError("Use a dedicated absolute deployment directory, not a broad root")
    if any(p.is_symlink() for p in (root, *root.parents)):
        raise ValueError("Deployment root may not contain symlinks")
    root = root.resolve()
    marker = root / MARKER
    if root.exists() and not marker.is_file() and any(root.iterdir()):
        raise ValueError("Existing nonempty directory is not managed by this tool")
    root.mkdir(parents=True, exist_ok=True)
    safe_path(root, MARKER)
    if not marker.exists():
        marker.write_text("Dedicated FFTGYRO deployment workspace v1\n")
    return root


def deploy(request):
    mode = request["mode"]
    if mode not in ("code", "config"):
        raise ValueError("Unknown mode")
    for key in ("release", "base_release"):
        if key in request and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", request[key]):
            raise ValueError("Invalid release identifier")
    data = base64.b64decode(request["config"], validate=True)
    config = json.loads(data.decode("utf-8-sig"))
    if config.get("schema") != "fftgyro-axis-session-v1":
        raise ValueError("Expected session config")
    if mode == "config" and ("sources" in request or not request.get("base_release")):
        raise ValueError("Config mode accepts no source files and needs a base release")
    root = prepare_root(request["root"])
    lock = safe_path(root, ".deploy-lock")
    lock.mkdir()  # Exclusive; fail rather than race another build. Never remove someone else's lock.
    try:
        release = safe_path(root, "releases/" + request["release"])
        release.mkdir(parents=True, exist_ok=False)
        try:
            config_path = release / "session.json"
            config_path.write_bytes(data)
            metadata = {"schema": "fftgyro-release-v1", "mode": mode,
                        "release": request["release"], "created_unix_s": time.time(),
                        "config_sha256": digest(data), "pid_law": config["pid"].get("law", "ANGLE_PID"),
                        "machine": platform.machine(), "python": platform.python_version()}
            binary = release / "fftgyro-executor"
            rpm_binary = release / "fftgyro-rpm"
            if mode == "code":
                jobs = request.get("jobs", 2)
                if type(jobs) is not int or not 1 <= jobs <= 8:
                    raise ValueError("jobs must be 1..8")
                mavlink = Path(request["mavlink_root"])
                if not mavlink.is_absolute() or not (mavlink / "pymavlink/tools/mavgen.py").is_file():
                    raise ValueError("Provide the Pi's absolute PX4 MAVLink checkout path")
                python_executable = Path(request.get("python_executable", "/usr/bin/python3"))
                if not python_executable.is_absolute() or not python_executable.is_file():
                    raise ValueError("Provide an absolute Pi Python executable")
                source = safe_path(root, "work/source")
                build = safe_path(root, "work/build")
                source.mkdir(parents=True, exist_ok=True)
                build.mkdir(parents=True, exist_ok=True)
                hashes, changed, removed = sync_sources(source, request["sources"])
                metadata.update(source_sha256=hashes, changed_files=changed, removed_files=removed,
                                mavlink_root=str(mavlink.resolve()),
                                python_executable=str(python_executable.resolve()), jobs=jobs)
                print(f"Source: changed={changed}, removed={removed}; fixed build={build}", flush=True)
                with (release / "build.log").open("w") as log:
                    # Build production programs only, not the test suite.
                    checked(["cmake", "-S", source / "onboard_executor", "-B", build,
                             "-G", "Unix Makefiles", "-DFFTGYRO_BUILD_TESTS=OFF",
                             "-DFFTGYRO_REQUIRE_GPIOD=ON",
                             "-DPython3_EXECUTABLE=" + str(python_executable),
                             "-DFFTGYRO_MAVLINK_ROOT=" + str(mavlink)], log)
                    checked(["cmake", "--build", build, "--target", "fftgyro-executor", "fftgyro-rpm",
                             "--parallel", str(jobs)], log)
                shutil.copy2(safe_path(build, "fftgyro-executor"), binary)
                shutil.copy2(safe_path(build, "fftgyro-rpm"), rpm_binary)
            else:
                base = safe_path(root, "releases/" + request["base_release"])
                previous = json.loads(safe_path(base, "manifest.json").read_text())
                original = safe_path(base, "fftgyro-executor")
                if previous.get("schema") != "fftgyro-release-v1" or not previous.get("validated"):
                    raise ValueError("Base release did not finish validation")
                if digest(original.read_bytes()) != previous["binary_sha256"]:
                    raise ValueError("Base binary hash mismatch")
                shutil.copy2(original, binary)  # Independent copy, never a mutable build symlink.
                original_rpm = safe_path(base, "fftgyro-rpm")
                if digest(original_rpm.read_bytes()) != previous["rpm_binary_sha256"]:
                    raise ValueError("Base RPM binary hash mismatch")
                shutil.copy2(original_rpm, rpm_binary)
                metadata.update(base_release=request["base_release"],
                                source_sha256=previous["source_sha256"])
            with (release / "validation.log").open("w") as log:
                checked([binary, "session-validate", config_path], log)
            metadata.update(binary_sha256=digest(binary.read_bytes()),
                            rpm_binary_sha256=digest(rpm_binary.read_bytes()), validated=True)
            # Manifest is the completion marker; failed/partial releases cannot be config bases.
            pending = release / "manifest.pending"
            pending.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            pending.replace(release / "manifest.json")
            print("STAGED (not started): " + str(release), flush=True)
            return release
        except Exception as error:
            (release / "FAILED.txt").write_text(str(error), encoding="utf-8")
            raise
    finally:
        lock.rmdir()


if __name__ == "__main__":
    try:
        deploy(json.load(sys.stdin))
    except Exception as error:
        print("DEPLOY FAILED: " + str(error), file=sys.stderr)
        sys.exit(1)
