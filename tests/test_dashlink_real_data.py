import numpy as np

from edgegenbench.real_data.dashlink import summarize_windows


def test_window_summary_shape_and_values():
    data = np.arange(2 * 160 * 20, dtype=float).reshape(2, 160, 20)
    result = summarize_windows(data)
    assert result.shape == (2, 100)
    np.testing.assert_allclose(result[:, 80:100], data[:, -1, :] - data[:, 0, :])
