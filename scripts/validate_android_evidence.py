#!/usr/bin/env python3
"""Validate an Android export and render machine/human-readable summaries."""

from __future__ import annotations

import argparse
import json
import runpy
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-results", type=int, default=3)
    args = parser.parse_args()
    release_script = Path(__file__).with_name("build_release_evidence.py")
    functions = runpy.run_path(release_script)
    summary = functions["validate_android_device_bundle"](
        args.evidence, min_results=args.min_results
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    functions["write_android_device_report"](summary, args.output_dir / "report.md")
    print(f"Android reference evidence validated: {summary_path}")


if __name__ == "__main__":
    main()
