from pathlib import Path

import pandas as pd
import yaml

from edgegenbench.data.generate import generate_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_CONFIG_PATH = PROJECT_ROOT / "configs" / "v0_1.yaml"


def _small_config(tmp_path: Path, sample_count: int = 64) -> Path:
    config = yaml.safe_load(BASE_CONFIG_PATH.read_text(encoding="utf-8"))
    config["dataset"]["n_samples"] = sample_count
    config["dataset"]["filename"] = "test_designs.csv"

    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    return config_path


def test_generation_creates_expected_files_and_columns(tmp_path: Path) -> None:
    config_path = _small_config(tmp_path)
    artifacts = generate_dataset(config_path=config_path, output_dir=tmp_path / "generated")
    dataset = pd.read_csv(artifacts.data_path)

    expected_columns = {
        "passenger_capacity",
        "propulsion_architecture",
        "estimated_takeoff_mass_kg",
        "mission_energy_kwh",
        "is_feasible",
        "split",
    }

    assert artifacts.data_path.exists()
    assert artifacts.metadata_path.exists()
    assert expected_columns.issubset(dataset.columns)
    assert len(dataset) == 64
    assert dataset.isna().sum().sum() == 0
    assert {"train", "validation", "test"}.issubset(set(dataset["split"]))


def test_generation_is_reproducible_for_a_fixed_seed(tmp_path: Path) -> None:
    config_path = _small_config(tmp_path)

    first = generate_dataset(config_path=config_path, output_dir=tmp_path / "first")
    second = generate_dataset(config_path=config_path, output_dir=tmp_path / "second")

    assert first.data_path.read_bytes() == second.data_path.read_bytes()
