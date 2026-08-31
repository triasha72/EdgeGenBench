import hashlib
import json
from pathlib import Path

import numpy as np

from edgegenbench.real_data.dashlink import summarize_windows


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
