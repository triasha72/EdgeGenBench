#!/usr/bin/env python3
"""Export and stage the current Core ML model for the native iOS target."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from edgegenbench.deployment.coreml_export import export_neural_surrogate_coreml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("artifacts/neural_surrogate/model.pt"))
    parser.add_argument(
        "--preprocessing",
        type=Path,
        default=Path("artifacts/neural_surrogate/preprocessing.npz"),
    )
    parser.add_argument("--target", type=Path, default=Path("ios/EdgeGenBenchDemo/Resources"))
    args = parser.parse_args()

    build_dir = args.target.parent / ".generated-coreml"
    shutil.rmtree(build_dir, ignore_errors=True)
    shutil.rmtree(args.target, ignore_errors=True)
    artifacts = export_neural_surrogate_coreml(
        model_path=args.model,
        preprocessing_path=args.preprocessing,
        output_dir=build_dir,
    )
    args.target.mkdir(parents=True, exist_ok=True)
    shutil.copytree(artifacts.model_path, args.target / artifacts.model_path.name)
    shutil.copy2(artifacts.contract_path, args.target / artifacts.contract_path.name)
    shutil.rmtree(build_dir)
    print(f"Staged Core ML resources in {args.target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
