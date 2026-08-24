#!/usr/bin/env python3
"""Build and gate the Qualcomm-native INT8/QDQ evidence bundle."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgegenbench.deployment.qualcomm_int8_report import build_qualcomm_int8_report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/qualcomm_int8.json"))
    parser.add_argument("--max-normalized-drift", type=float, default=0.01)
    args = parser.parse_args()
    artifacts = build_qualcomm_int8_report(
        args.manifest, args.output, max_normalized_drift=args.max_normalized_drift
    )
    print(f"REPORT={artifacts.report_path}")
    print(f"ACCEPTED={str(artifacts.accepted).lower()}")
    for reason in artifacts.rejection_reasons:
        print(f"REJECTION={reason}")
    return 0 if artifacts.accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
