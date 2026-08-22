from __future__ import annotations

import numpy as np

from edgegenbench.deployment.coreml_export import build_ios_contract
from edgegenbench.models.neural_preprocessing import NeuralPreprocessor


def test_ios_contract_preserves_preprocessing_and_output_scaling() -> None:
    preprocessor = NeuralPreprocessor(
        categories=("battery_electric", "hybrid", "hydrogen"),
        feature_mean=np.arange(6, dtype=np.float32),
        feature_scale=np.ones(6, dtype=np.float32),
        target_mean=np.arange(6, dtype=np.float32),
        target_scale=np.ones(6, dtype=np.float32) * 2,
        targets=("a", "b", "c", "d", "e", "f"),
    )
    contract = build_ios_contract(preprocessor)
    assert contract["inputName"] == "features"
    assert contract["outputName"] == "predictions"
    assert contract["inputDimension"] == 9
    assert contract["outputDimension"] == 6
    assert contract["categories"] == ["battery_electric", "hybrid", "hydrogen"]
    assert contract["targetScale"] == [2.0] * 6

