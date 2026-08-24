from __future__ import annotations

import json
import math
from pathlib import Path


def test_native_golden_fixture_contract() -> None:
    payload = json.loads(Path("tests/fixtures/native_model_golden.json").read_text())
    assert payload["input_width"] == 10
    assert payload["output_width"] == 6
    assert len(payload["model_sha256"]) == 64
    assert len(payload["preprocessing_sha256"]) == 64
    assert len(payload["cases"]) == 4
    for case in payload["cases"]:
        assert len(case["encoded"]) == 10
        assert len(case["normalized_prediction"]) == 6
        assert len(case["physical_prediction"]) == 6
        assert all(math.isfinite(value) for value in case["encoded"])
        assert all(math.isfinite(value) for value in case["normalized_prediction"])
        assert all(math.isfinite(value) for value in case["physical_prediction"])
