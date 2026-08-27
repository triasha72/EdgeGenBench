from __future__ import annotations

import hashlib
import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/validate_ios_evidence.py"
ValidateIOS = Callable[..., dict[str, object]]
validate_ios_evidence = cast(ValidateIOS, runpy.run_path(SCRIPT)["validate_ios_evidence"])


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(model: Path, preprocessing: Path) -> dict[str, object]:
    return {
        "schemaVersion": "1.0",
        "capturedAtUTC": "2026-08-27T00:00:00Z",
        "appVersion": "0.1.0",
        "backend": "CoreML",
        "requestedComputeUnits": "all",
        "neuralEnginePlacement": "not_measured",
        "powerMeasurement": "not_measured",
        "thermalStateBefore": "nominal",
        "thermalStateAfter": "fair",
        "lowPowerMode": False,
        "sourceModelSha256": _hash(model),
        "preprocessingSha256": _hash(preprocessing),
        "contractSha256": "c" * 64,
        "device": {
            "model": "iPhone15,4",
            "systemName": "iOS",
            "systemVersion": "17.6",
            "simulator": False,
        },
        "latency": {"coldMs": 3.0, "warmMeanMs": 1.0, "warmP95Ms": 1.2, "warmRuns": 100},
        "outputMaxAbsDrift": 0.0,
        "outputs": [{"name": "mass", "value": 1.0}],
    }


def test_validates_physical_iphone_coreml_evidence(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    preprocessing = tmp_path / "preprocessing.npz"
    model.write_bytes(b"model")
    preprocessing.write_bytes(b"preprocessing")
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(_evidence(model, preprocessing)))
    result = validate_ios_evidence(evidence, model_path=model, preprocessing_path=preprocessing)
    assert result["status"] == "validated_physical_iphone_coreml"
    assert result["neural_engine_placement"] == "not_measured"


def test_rejects_simulator_as_physical_evidence(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    preprocessing = tmp_path / "preprocessing.npz"
    model.write_bytes(b"model")
    preprocessing.write_bytes(b"preprocessing")
    payload = _evidence(model, preprocessing)
    cast(dict[str, object], payload["device"])["simulator"] = True
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="physical-iPhone"):
        validate_ios_evidence(evidence, model_path=model, preprocessing_path=preprocessing)


def test_rejects_unproven_ane_claim(tmp_path: Path) -> None:
    model = tmp_path / "model.pt"
    preprocessing = tmp_path / "preprocessing.npz"
    model.write_bytes(b"model")
    preprocessing.write_bytes(b"preprocessing")
    payload = _evidence(model, preprocessing)
    payload["neuralEnginePlacement"] = "ANE"
    evidence = tmp_path / "evidence.json"
    evidence.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="ANE placement"):
        validate_ios_evidence(evidence, model_path=model, preprocessing_path=preprocessing)
