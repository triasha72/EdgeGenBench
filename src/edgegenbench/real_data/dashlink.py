"""NASA DASHlink four-class recorded-flight dataset adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "nominal", 1: "speed_high", 2: "path_high", 3: "flaps_late"}


def validate_runtime_windows(data, maximum_missing_fraction: float = 0.05) -> dict[str, float]:
    """Fail closed on malformed or excessively incomplete approach windows."""
    array = np.asarray(data)
    if array.ndim != 3 or array.shape[1:] != (160, 20):
        raise ValueError(f"Expected [rows, 160, 20], received {array.shape}")
    if len(array) == 0:
        raise ValueError("At least one approach window is required")
    if not 0 <= maximum_missing_fraction < 1:
        raise ValueError("maximum_missing_fraction must be in [0, 1)")
    if np.isinf(array).any():
        raise ValueError("Approach windows cannot contain infinite values")
    missing_fraction = float(np.isnan(array).mean())
    worst_channel_missing_fraction = float(np.isnan(array).mean(axis=(0, 1)).max())
    if missing_fraction > maximum_missing_fraction:
        raise ValueError("Batch missing-value fraction exceeds runtime limit")
    if worst_channel_missing_fraction > maximum_missing_fraction:
        raise ValueError("A channel missing-value fraction exceeds runtime limit")
    if np.isnan(array[:, (0, -1), :]).any():
        raise ValueError("Window endpoints must be present for trend features")
    return {
        "missing_fraction": missing_fraction,
        "worst_channel_missing_fraction": worst_channel_missing_fraction,
    }


def load_dashlink(npz_path, metadata_path):
    archive = np.load(npz_path)
    data = archive["data"]
    labels = archive["label"].reshape(-1).astype(np.int64)
    metadata = pd.read_csv(metadata_path, dtype={"flight_record": str})
    if data.ndim != 3 or data.shape[1:] != (160, 20):
        raise ValueError(f"Expected [rows, 160, 20], received {data.shape}")
    if len(data) != len(labels) or len(data) != len(metadata):
        raise ValueError("Feature, label, and metadata row counts differ")
    if set(np.unique(labels)) - set(CLASS_NAMES):
        raise ValueError("Unexpected DASHlink class label")
    return data, labels, metadata


def summarize_windows(data):
    """Convert each recorded 160x20 approach to 100 edge-friendly features."""
    validate_runtime_windows(data)
    return np.concatenate(
        (
            np.nanmean(data, axis=1),
            np.nanstd(data, axis=1),
            np.nanmin(data, axis=1),
            np.nanmax(data, axis=1),
            data[:, -1, :] - data[:, 0, :],
        ),
        axis=1,
    ).astype(np.float32)


def aircraft_groups(metadata):
    """Derive de-identified aircraft groups from NASA flight-record identifiers."""
    return metadata["flight_record"].str.slice(0, 3).to_numpy()
