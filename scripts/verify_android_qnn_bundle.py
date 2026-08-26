#!/usr/bin/env python3
"""Fail closed unless an Android arm64 ORT/QNN dependency bundle is complete."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

REQUIRED_LIBRARIES = (
    "libonnxruntime.so",
    "libQnnHtp.so",
    "libQnnSystem.so",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_android_qnn_bundle(root: Path) -> dict[str, Any]:
    if not root.is_absolute():
        raise ValueError("QNN bundle root must be an absolute path")
    header = root / "include" / "onnxruntime_cxx_api.h"
    if not header.is_file():
        raise ValueError("missing ONNX Runtime C++ headers")
    library_dir = root / "lib" / "arm64-v8a"
    missing = [name for name in REQUIRED_LIBRARIES if not (library_dir / name).is_file()]
    if missing:
        raise ValueError(f"missing Android arm64 runtime libraries: {', '.join(missing)}")
    files = [header, *(library_dir / name for name in REQUIRED_LIBRARIES)]
    return {
        "schema_version": 1,
        "target": "android-arm64-v8a",
        "root": str(root),
        "files": [
            {
                "path": path.relative_to(root).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in files
        ],
        "claim": "Dependency completeness only; this is not NPU placement evidence.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = verify_android_qnn_bundle(args.root)
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
