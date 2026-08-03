"""Physics-informed synthetic regional-aircraft design model.

This module creates transparent synthetic targets for benchmarking. Its outputs
are illustrative and must not be interpreted as certified aircraft-performance
predictions.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

LIQUID_HYDROGEN_DENSITY_KG_PER_M3 = 70.8
HYDROGEN_LOWER_HEATING_VALUE_KWH_PER_KG = 33.3

REQUIRED_INPUT_COLUMNS = (
    "passenger_capacity",
    "design_range_km",
    "cruise_speed_kmh",
    "battery_specific_energy_wh_per_kg",
    "hydrogen_storage_efficiency",
    "hybridization_ratio",
    "propulsion_architecture",
)

ARCHITECTURE_FACTORS: Mapping[str, Mapping[str, float]] = {
    "conventional_turboprop": {
        "energy_multiplier": 1.00,
        "battery_utilization": 0.00,
        "hydrogen_energy_share": 0.00,
        "hydrogen_conversion_efficiency": 0.00,
        "emissions_intensity": 0.245,
        "energy_cost_per_kwh": 0.095,
        "noise_offset_db": 0.0,
        "empty_mass_multiplier": 1.00,
    },
    "parallel_hybrid": {
        "energy_multiplier": 0.93,
        "battery_utilization": 1.00,
        "hydrogen_energy_share": 0.25,
        "hydrogen_conversion_efficiency": 0.50,
        "emissions_intensity": 0.180,
        "energy_cost_per_kwh": 0.110,
        "noise_offset_db": -3.0,
        "empty_mass_multiplier": 1.05,
    },
    "series_hybrid": {
        "energy_multiplier": 0.89,
        "battery_utilization": 0.80,
        "hydrogen_energy_share": 0.60,
        "hydrogen_conversion_efficiency": 0.54,
        "emissions_intensity": 0.120,
        "energy_cost_per_kwh": 0.125,
        "noise_offset_db": -5.0,
        "empty_mass_multiplier": 1.10,
    },
    "fuel_cell_electric": {
        "energy_multiplier": 0.86,
        "battery_utilization": 0.20,
        "hydrogen_energy_share": 1.00,
        "hydrogen_conversion_efficiency": 0.58,
        "emissions_intensity": 0.045,
        "energy_cost_per_kwh": 0.145,
        "noise_offset_db": -8.0,
        "empty_mass_multiplier": 1.17,
    },
}


def _validate_designs(designs: pd.DataFrame) -> None:
    missing = set(REQUIRED_INPUT_COLUMNS).difference(designs.columns)
    if missing:
        raise ValueError(f"Missing required design columns: {sorted(missing)}")

    architectures = set(designs["propulsion_architecture"])
    unknown = architectures.difference(ARCHITECTURE_FACTORS)
    if unknown:
        raise ValueError(f"Unknown propulsion architecture(s): {sorted(unknown)}")

    numeric_columns = REQUIRED_INPUT_COLUMNS[:-1]
    if (designs.loc[:, numeric_columns] < 0).any().any():
        raise ValueError("Design inputs must be non-negative.")


def _architecture_array(designs: pd.DataFrame, key: str) -> np.ndarray:
    architectures = designs["propulsion_architecture"]
    return np.asarray(
        [ARCHITECTURE_FACTORS[architecture][key] for architecture in architectures],
        dtype=float,
    )


def simulate_designs(
    designs: pd.DataFrame,
    constraints: Mapping[str, float],
    seed: int,
) -> pd.DataFrame:
    """Calculate synthetic performance, constraint, and feasibility outputs."""
    _validate_designs(designs)

    passenger_capacity = designs["passenger_capacity"].to_numpy(dtype=float)
    design_range_km = designs["design_range_km"].to_numpy(dtype=float)
    cruise_speed_kmh = designs["cruise_speed_kmh"].to_numpy(dtype=float)
    battery_specific_energy = designs["battery_specific_energy_wh_per_kg"].to_numpy(dtype=float)
    hydrogen_storage_efficiency = designs["hydrogen_storage_efficiency"].to_numpy(dtype=float)
    hybridization_ratio = designs["hybridization_ratio"].to_numpy(dtype=float)

    energy_multiplier = _architecture_array(designs, "energy_multiplier")
    battery_utilization = _architecture_array(designs, "battery_utilization")
    hydrogen_energy_share = _architecture_array(designs, "hydrogen_energy_share")
    hydrogen_conversion_efficiency = _architecture_array(
        designs,
        "hydrogen_conversion_efficiency",
    )
    emissions_intensity = _architecture_array(designs, "emissions_intensity")
    energy_cost_per_kwh = _architecture_array(designs, "energy_cost_per_kwh")
    noise_offset_db = _architecture_array(designs, "noise_offset_db")
    empty_mass_multiplier = _architecture_array(designs, "empty_mass_multiplier")

    reserve_range_km = design_range_km * 1.12
    baseline_energy_per_km = (
        6.4
        + 0.085 * passenger_capacity
        + 0.014 * (cruise_speed_kmh - 420.0)
        + 0.000004 * (design_range_km - 850.0) ** 2
    )
    mission_energy_kwh = (
        reserve_range_km
        * baseline_energy_per_km
        * energy_multiplier
        * (1.0 - 0.07 * hybridization_ratio)
    )

    battery_energy_kwh = mission_energy_kwh * hybridization_ratio * battery_utilization
    battery_mass_kg = battery_energy_kwh * 1000.0 / battery_specific_energy

    hydrogen_energy_kwh = mission_energy_kwh * hydrogen_energy_share
    hydrogen_mass_kg = np.divide(
        hydrogen_energy_kwh,
        HYDROGEN_LOWER_HEATING_VALUE_KWH_PER_KG * hydrogen_conversion_efficiency,
        out=np.zeros_like(hydrogen_energy_kwh),
        where=hydrogen_conversion_efficiency > 0,
    )
    hydrogen_tank_mass_kg = hydrogen_mass_kg * (1.0 / hydrogen_storage_efficiency - 0.15)
    hydrogen_tank_volume_m3 = (
        hydrogen_mass_kg / LIQUID_HYDROGEN_DENSITY_KG_PER_M3 / hydrogen_storage_efficiency
    )

    payload_mass_kg = passenger_capacity * 95.0
    baseline_empty_mass_kg = (
        9300.0 + 72.0 * passenger_capacity + 1.25 * design_range_km + 3.2 * cruise_speed_kmh
    )
    energy_system_mass_kg = battery_mass_kg + hydrogen_mass_kg + hydrogen_tank_mass_kg
    estimated_takeoff_mass_kg = (
        baseline_empty_mass_kg * empty_mass_multiplier
        + payload_mass_kg
        + 1.08 * energy_system_mass_kg
    )

    normalized_range = (design_range_km - 400.0) / 1100.0
    normalized_hybridization = hybridization_ratio / 0.65
    observation_noise_scale = 0.004 + 0.020 * np.clip(
        0.60 * normalized_range + 0.40 * normalized_hybridization,
        0.0,
        1.0,
    )
    random_generator = np.random.default_rng(seed)

    estimated_takeoff_mass_kg *= random_generator.normal(
        loc=1.0,
        scale=observation_noise_scale,
        size=len(designs),
    )
    mission_energy_kwh *= random_generator.normal(
        loc=1.0,
        scale=observation_noise_scale,
        size=len(designs),
    )

    energy_per_passenger_km = mission_energy_kwh / (design_range_km * passenger_capacity)
    lifecycle_emissions_proxy_kgco2e = mission_energy_kwh * emissions_intensity
    operating_cost_proxy_usd = (
        mission_energy_kwh * energy_cost_per_kwh + 0.018 * estimated_takeoff_mass_kg
    )
    noise_proxy_db = (
        85.0 + 0.020 * (cruise_speed_kmh - 420.0) + 0.030 * passenger_capacity + noise_offset_db
    )

    battery_mass_fraction = battery_mass_kg / estimated_takeoff_mass_kg
    max_takeoff_mass_kg = float(
        constraints["max_takeoff_mass_base_kg"]
    ) + passenger_capacity * float(constraints["max_takeoff_mass_per_passenger_kg"])
    max_hydrogen_tank_volume_m3 = float(
        constraints["max_hydrogen_tank_volume_base_m3"]
    ) + passenger_capacity * float(constraints["max_hydrogen_tank_volume_per_passenger_m3"])

    battery_fraction_margin = (
        float(constraints["max_battery_mass_fraction"]) - battery_mass_fraction
    )
    takeoff_mass_margin_kg = max_takeoff_mass_kg - estimated_takeoff_mass_kg
    hydrogen_tank_volume_margin_m3 = max_hydrogen_tank_volume_m3 - hydrogen_tank_volume_m3
    is_feasible = (
        (battery_fraction_margin >= 0.0)
        & (takeoff_mass_margin_kg >= 0.0)
        & (hydrogen_tank_volume_margin_m3 >= 0.0)
    )

    return pd.DataFrame(
        {
            "estimated_takeoff_mass_kg": estimated_takeoff_mass_kg,
            "mission_energy_kwh": mission_energy_kwh,
            "energy_per_passenger_km_kwh": energy_per_passenger_km,
            "lifecycle_emissions_proxy_kgco2e": lifecycle_emissions_proxy_kgco2e,
            "operating_cost_proxy_usd": operating_cost_proxy_usd,
            "noise_proxy_db": noise_proxy_db,
            "battery_mass_kg": battery_mass_kg,
            "hydrogen_mass_kg": hydrogen_mass_kg,
            "hydrogen_tank_volume_m3": hydrogen_tank_volume_m3,
            "battery_fraction_margin": battery_fraction_margin,
            "takeoff_mass_margin_kg": takeoff_mass_margin_kg,
            "hydrogen_tank_volume_margin_m3": hydrogen_tank_volume_margin_m3,
            "is_feasible": is_feasible,
        },
        index=designs.index,
    )
