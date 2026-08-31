"""NASA DASHlink four-class recorded-flight dataset adapter."""

from __future__ import annotations

import numpy as np
import pandas as pd

CLASS_NAMES = {0: "nominal", 1: "speed_high", 2: "path_high", 3: "flaps_late"}


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
