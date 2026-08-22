#!/usr/bin/env python3
"""Export the EdgeGenBench neural surrogate for the native iOS demo."""

from __future__ import annotations

import argparse
from pathlib import Path

from edgegenbench.deployment.coreml_export import export_neural_surrogate_coreml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--preprocessing", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/coreml"))
    args = parser.parse_args()
    artifacts = export_neural_surrogate_coreml(
        model_path=args.model,
        preprocessing_path=args.preprocessing,
        output_dir=args.output_dir,
    )
    print(f"Core ML model: {artifacts.model_path}")
    print(f"iOS contract: {artifacts.contract_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
