import json
from pathlib import Path

import numpy as np

from scripts.benchmark_dashlink_corruptions import grouped_test_indices


def test_corruption_benchmark_reuses_grouped_test_contract():
    labels = np.asarray([0, 1] * 20)
    groups = np.repeat(np.arange(10), 4)
    features = np.zeros((40, 1))
    first = grouped_test_indices(features, labels, groups, 42)
    second = grouped_test_indices(features, labels, groups, 42)
    np.testing.assert_array_equal(first, second)
    assert set(groups[first]).isdisjoint(set(groups[np.setdiff1d(np.arange(40), first)]))


def test_published_corruption_evidence_uses_real_held_out_flights():
    root = Path(__file__).parents[1]
    artifact = json.loads((root / "artifacts/dashlink_corruption_benchmark_v1.json").read_text())
    assert artifact["contains_synthetic_flights"] is False
    assert artifact["dataset_rows"] == 17_780
    assert (
        artifact["conditions"]["four_timestep_dropout"]["prediction_consistency_with_clean"] > 0.99
    )
