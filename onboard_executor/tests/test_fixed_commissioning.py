"""Generate a temporary draft, then exercise only synthetic fixed commissioning."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from fftgyro_quad_id.experiment_runner.commissioning import build_commissioning


def main():
    executable = str(Path(sys.argv[1]).resolve(strict=True))
    source = json.loads(Path(sys.argv[2]).read_text())
    tests = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="fixed-commissioning-tests-") as directory:
        config = Path(directory) / "fixed.json"
        config.write_text(json.dumps(build_commissioning(source, "fixed")))
        for script in ("test_props_off_guard.py", "test_session_transport.py"):
            subprocess.run([sys.executable, str(tests / script), executable, str(config),
                            "--fixed-commissioning"], check=True, timeout=90)


if __name__ == "__main__":
    main()
