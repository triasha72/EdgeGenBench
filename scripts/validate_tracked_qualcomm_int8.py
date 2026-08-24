#!/usr/bin/env python3
"""Re-run the frozen gate over the tracked Qualcomm INT8 device report."""

from pathlib import Path

from edgegenbench.deployment.qualcomm_int8_report import validate_tracked_qualcomm_int8_report


def main() -> int:
    accepted, reasons = validate_tracked_qualcomm_int8_report(
        Path("reports/qualcomm_int8_qnn_v0_1.json")
    )
    print(f"ACCEPTED={str(accepted).lower()}")
    for reason in reasons:
        print(f"REJECTION={reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
