#!/usr/bin/env python3
"""Select a real-flight candidate from validation metrics without reading test results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from edgegenbench.real_data.candidate_selection import select_candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.candidates.read_text())
    selected = select_candidate(payload["candidates"])
    result = {
        "schema_version": "1.0",
        "selection_partition": "validation",
        "frozen_gates": {
            "minimum_macro_f1": 0.75,
            "minimum_anomaly_recall": 0.60,
            "minimum_coverage": 0.80,
        },
        "selected": selected,
        "decision": "candidate_selected" if selected else "no_eligible_candidate",
        "test_data_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["decision"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
