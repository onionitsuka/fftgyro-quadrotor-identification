"""Match physical PX4 SD files to recorded trial paths; never delete SD files."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil


def recorded_ulog_path(run: Path) -> PurePosixPath | None:
    """Use the operator receipt for deferred runs, with the older logger receipt as fallback."""
    for record in (run / "operator-result.json", run / "flightlog/start.json"):
        if record.is_file():
            value = json.loads(record.read_text(encoding="utf-8")).get("ulog_path")
            if value:
                return PurePosixPath(value)
    return None


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def import_one(sd_root: Path, run: Path) -> dict:
    path = recorded_ulog_path(run)
    if path is None:
        raise ValueError("trial has no recorded ULog path")
    relative = path.relative_to("/fs/microsd")
    if ".." in relative.parts or path.suffix.lower() != ".ulg":
        raise ValueError("invalid recorded SD path")
    source = sd_root.joinpath(*relative.parts).resolve()
    if not source.is_relative_to(sd_root.resolve()):
        raise ValueError("SD path escapes selected root")
    with source.open("rb") as stream:
        if stream.read(7) != b"ULog\x01\x12\x35":
            raise ValueError("ULog header missing")
    target = run / "flightlog" / path.name
    target.parent.mkdir(parents=True, exist_ok=True)
    source_hash = digest(source)
    if target.exists():
        if digest(target) != source_hash:
            raise ValueError("different file already exists; no overwrite")
    else:
        # x mode protects existing data even when two import commands race.
        with source.open("rb") as inp, target.open("xb") as out:
            shutil.copyfileobj(inp, out)
    if digest(target) != source_hash:
        raise ValueError("copy checksum mismatch")
    return dict(state="COPIED_HASH_VERIFIED", source=str(source), target=str(target),
                sha256=source_hash, size_bytes=target.stat().st_size,
                header_verified=True, content_decoded=False,
                limitation="Header and copy identity only; analyze ULog topics/dropouts separately")


def main() -> int:
    parser = argparse.ArgumentParser(description="SDカードから試験ごとのULogを照合してコピー（元ファイル保持）")
    parser.add_argument("sd_root", type=Path, help="logフォルダーのあるSDルート。例 E:/")
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--date", help="対象runの年月日。例 20260915")
    args = parser.parse_args()
    if args.data_root is None:
        from .production_console import REPO, read_json, resolve
        args.data_root = resolve(read_json(REPO / "configs/experiment_runner/production.json")["data_root"])
    failures = 0
    count = 0
    if args.date and (len(args.date) != 8 or not args.date.isdecimal()):
        parser.error("--date must be YYYYMMDD")
    runs = {record.parent for record in args.data_root.glob("*/operator-result.json")}
    runs.update(record.parent.parent for record in args.data_root.glob("*/flightlog/start.json"))
    for run in sorted(runs):
        if args.date and not run.name.startswith(args.date + "_"):
            continue
        if recorded_ulog_path(run) is None:
            continue
        try:
            result = import_one(args.sd_root, run)
            # Save each import as separate evidence, preserving earlier attempts.
            record_dir = run / "flightlog"
            i = 1
            while (record_dir / f"sd-import-{i:03d}.json").exists():
                i += 1
            with (record_dir / f"sd-import-{i:03d}.json").open("x", encoding="utf-8") as stream:
                json.dump(result, stream, indent=2)
            count += 1
            print(f"回収OK: {run.name} → {Path(result['target']).name}")
        except (ValueError, OSError, KeyError) as error:
            failures += 1
            print(f"未回収: {run.name} ／ {error}")
    print(f"回収 {count} 件、未回収 {failures} 件。SD元ファイルは保持。内容解析は別途実施。")
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
