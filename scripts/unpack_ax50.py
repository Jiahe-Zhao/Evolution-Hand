#!/usr/bin/env python3
"""Verify and extract the bundled AX50 release. Requires only Python's standard library."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    repo = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=repo / "local_assets")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    releases = repo / "assets" / "releases"
    manifest = json.loads((releases / "ax50_visualskin_v3.0.0.json").read_text())
    archive = releases / manifest["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != manifest["sha256"]:
        raise SystemExit("AX50 archive checksum mismatch. Fetch the complete repository again.")
    with zipfile.ZipFile(archive) as z:
        if z.testzip() is not None:
            raise SystemExit("AX50 archive is damaged.")
        prefix = manifest["root"] + "/"
        checks = json.loads(z.read(prefix + "SHA256SUMS.json"))
        for name, expected in checks.items():
            if hashlib.sha256(z.read(prefix + name)).hexdigest() != expected:
                raise SystemExit("AX50 file checksum mismatch: " + name)
        print(f"Verified archive and {len(checks)} files.")
        if args.verify_only:
            return
        dest = args.dest.expanduser().resolve()
        # The archive hash pins the trusted release; reject unsafe paths too.
        for name in z.namelist():
            if not (dest / name).resolve().is_relative_to(dest):
                raise SystemExit("Unsafe archive path: " + name)
        z.extractall(dest)
    root = dest / manifest["root"]
    print("Extracted:", root)
    print("Read:", root / "README.md")
    print("Install in a matching Newton/CUDA environment:")
    print(f'python -m pip install --no-deps "{root}/dist/ax50_visualskin-3.0.0-py3-none-any.whl"')
    print(f'python "{root}/examples/basic.py" --side right --wrist six_axis')


if __name__ == "__main__":
    main()
