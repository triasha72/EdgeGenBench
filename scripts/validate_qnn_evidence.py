#!/usr/bin/env python3
"""Validate an EdgeGenBench physical-device QNN evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_release_evidence import validate_qnn_evidence_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        summary = validate_qnn_evidence_bundle(args.evidence)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        # Keep CI and device operators on the same fail-closed path, while
        # still returning a useful one-line diagnosis instead of a traceback.
        parser.error(str(exc))
    rendered = json.dumps(summary, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")


if __name__ == "__main__":
    main()
