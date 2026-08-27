import json
import runpy
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/build_portfolio_acceptance.py"
FUNCTIONS = runpy.run_path(SCRIPT)
ValidateQnn = Callable[[Path, Path], dict[str, object]]
validate_ai_hub_qnn = cast(ValidateQnn, FUNCTIONS["validate_ai_hub_qnn"])
Validate16Kb = Callable[[Path, Path], dict[str, object]]
validate_android_16kb_runtime = cast(Validate16Kb, FUNCTIONS["validate_android_16kb_runtime"])
ValidateIOS = Callable[[Path], dict[str, object]]
validate_ios_coreml_implementation = cast(
    ValidateIOS, FUNCTIONS["validate_ios_coreml_implementation"]
)


def test_validates_tracked_ai_hub_qnn_evidence() -> None:
    root = Path(__file__).parents[1]
    result = validate_ai_hub_qnn(root / "reports/qualcomm_qnn_v0_1.json", root)
    assert result["status"] == "validated_ai_hub_physical_qnn"
    assert result["backend"] == "QNN HTP"
    assert result["source_model_matches_repository"] is True
    assert result["source_model_hash_origin"] == "committed_git_blob"
    assert result["context_matches_repository"] is True
    assert result["context_hash_origin"] == "committed_git_blob"
    assert len(cast(list[object], result["graphs"])) == 3


def test_rejects_cpu_compute_unit_in_qnn_report(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    report = json.loads((root / "reports/qualcomm_qnn_v0_1.json").read_text())
    report["linked_multigraph"]["validation"]["graphs"]["edgegenbench_batch1"]["profile"][
        "compute_units"
    ] = {"CPU": 1, "NPU": 8}
    path = tmp_path / "qnn.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exclusive NPU"):
        validate_ai_hub_qnn(path, root)


def test_rejects_mismatched_qnn_context_hash(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    report = json.loads((root / "reports/qualcomm_qnn_v0_1.json").read_text())
    report["linked_multigraph"]["serialized_model_sha256"] = "0" * 64
    path = tmp_path / "qnn.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="context binary"):
        validate_ai_hub_qnn(path, root)


def test_validates_tracked_android_16kb_runtime_evidence() -> None:
    root = Path(__file__).parents[1]
    result = validate_android_16kb_runtime(
        root / "reports/device/android-16kb-api35-reference-10-runs",
        root / "reports/android_16kb_emulator_reference_v0_1_7.md",
    )
    assert result["status"] == "validated_16kb_emulator_runtime"
    assert result["page_size_bytes"] == 16384
    assert result["runs"] == 10


def test_rejects_non_16kb_android_runtime_evidence(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    source = root / "reports/device/android-16kb-api35-reference-10-runs"
    evidence = tmp_path / "evidence"
    shutil.copytree(source, evidence)
    device_report = evidence / "device-report.txt"
    device_report.write_text(
        device_report.read_text(encoding="utf-8").replace("page_size=16384", "page_size=4096"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="ARM64 16 KB"):
        validate_android_16kb_runtime(
            evidence,
            root / "reports/android_16kb_emulator_reference_v0_1_7.md",
        )


def test_validates_ios_coreml_implementation_contract() -> None:
    root = Path(__file__).parents[1]
    result = validate_ios_coreml_implementation(root)
    assert result["status"] == "ci_build_and_simulator_test_configured"
    assert result["backend"] == "CoreML"
    assert result["physical_device_status"] == "evidence_pending"
