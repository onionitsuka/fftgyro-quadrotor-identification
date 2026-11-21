#!/usr/bin/env python3
"""Generate C headers from the repository's canonical FFTGYRO MAVLink dialect."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mavgen", type=Path, required=True)
    parser.add_argument("--definitions", type=Path, required=True)
    parser.add_argument("--dialect", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    for path in (args.mavgen, args.definitions, args.dialect):
        if not path.exists():
            parser.error(f"required path does not exist: {path}")

    args.output.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="fftgyro-mavlink-") as temporary:
        definitions = Path(temporary)
        for xml in args.definitions.glob("*.xml"):
            shutil.copy2(xml, definitions / xml.name)
        shutil.copy2(args.dialect, definitions / "fftgyro.xml")

        subprocess.run(
            [
                sys.executable,
                str(args.mavgen),
                "--lang=C",
                "--wire-protocol=2.0",
                f"--output={args.output}",
                str(definitions / "fftgyro.xml"),
            ],
            check=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
