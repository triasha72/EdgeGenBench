import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from edgegenbench.real_data.dashlink import summarize_windows, validate_runtime_windows
from scripts.assess_dashlink_release import assess


def test_window_summary_shape_and_values():
    data = np.arange(2 * 160 * 20, dtype=float).reshape(2, 160, 20)
    result = summarize_windows(data)
    assert result.shape == (2, 100)
    np.testing.assert_allclose(result[:, 80:100], data[:, -1, :] - data[:, 0, :])


def test_published_real_model_and_onnx_evidence_are_bound_together():
    root = Path(__file__).parents[1]
    artifact = json.loads((root / "artifacts/dashlink_real_baseline_v1.json").read_text())
    model_path = root / "artifacts" / artifact["onnx"]["path"]
    assert artifact["contains_synthetic_data"] is False
    assert artifact["rows"]["test"] == 17_780
    assert artifact["onnx"]["label_agreement"] == 1.0
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == artifact["onnx"]["sha256"]


def test_runtime_window_validation_fails_closed():
    data = np.ones((2, 160, 20))
    assert validate_runtime_windows(data)["missing_fraction"] == 0
    data[:, 10:30, 0] = np.nan
    with pytest.raises(ValueError, match="channel missing-value"):
        validate_runtime_windows(data)


def test_current_real_model_is_not_release_eligible():
    root = Path(__file__).parents[1]
    artifact = json.loads((root / "artifacts/dashlink_real_baseline_v1.json").read_text())
    result = assess(artifact)
    assert result["decision"] == "rejected"
    assert not result["checks"]["minimum_macro_f1"]["passed"]
