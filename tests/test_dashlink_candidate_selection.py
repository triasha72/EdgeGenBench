import numpy as np

from edgegenbench.real_data.candidate_selection import select_candidate, selective_metrics


def test_selective_metrics_reports_abstention_and_minority_recall():
    expected = np.array([0, 1, 2, 3, 3])
    probabilities = np.array([
        [0.9, 0.05, 0.03, 0.02], [0.1, 0.8, 0.05, 0.05],
        [0.1, 0.1, 0.7, 0.1], [0.1, 0.1, 0.1, 0.7],
        [0.3, 0.2, 0.2, 0.3],
    ])
    result = selective_metrics(expected, probabilities, 0.6)
    assert result == {"coverage": 0.8, "macro_f1": 1.0, "minimum_anomaly_recall": 1.0}


def test_selection_requires_all_validation_gates():
    records = [
        {"name": "high-score-low-recall", "validation": {
            "macro_f1": 0.8, "minimum_anomaly_recall": 0.59, "coverage": 0.9,
        }},
        {"name": "eligible", "validation": {
            "macro_f1": 0.76, "minimum_anomaly_recall": 0.65, "coverage": 0.85,
        }},
    ]
    assert select_candidate(records)["name"] == "eligible"
