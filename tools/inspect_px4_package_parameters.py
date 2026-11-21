#!/usr/bin/env python3
"""Print selected parameter definitions embedded in a PX4 firmware package."""

from __future__ import annotations

import argparse
import base64
import json
import zlib
from pathlib import Path
from xml.etree import ElementTree


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("firmware", type=Path)
    parser.add_argument("names", nargs="+")
    args = parser.parse_args()

    package = json.loads(args.firmware.read_text(encoding="utf-8"))
    root = ElementTree.fromstring(
        zlib.decompress(base64.b64decode(package["parameter_xml"]))
    )
    wanted = set(args.names)
    found = {}
    for parameter in root.iter("parameter"):
        name = parameter.attrib.get("name")
        if name in wanted:
            found[name] = ElementTree.tostring(parameter, encoding="unicode")
    for name in args.names:
        print(found.get(name, f"{name}: NOT FOUND"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
