import pandas as pd

from edgegenbench.physics.synthetic_aircraft import simulate_designs


def test_simulation_returns_sensible_outputs() -> None:
    designs = pd.DataFrame(
        {
            "passenger_capacity": [50, 70, 85, 60],
            "design_range_km": [500, 800, 1200, 1000],
            "cruise_speed_kmh": [440, 500, 600, 560],
            "battery_specific_energy_wh_per_kg": [450, 550, 650, 500],
            "hydrogen_storage_efficiency": [0.50, 0.60, 0.65, 0.55],
            "hybridization_ratio": [0.0, 0.30, 0.50, 0.40],
            "propulsion_architecture": [
                "conventional_turboprop",
                "parallel_hybrid",
                "series_hybrid",
                "fuel_cell_electric",
            ],
        }
    )

    constraints = {
        "max_battery_mass_fraction": 0.32,
        "max_takeoff_mass_base_kg": 18000,
        "max_takeoff_mass_per_passenger_kg": 130,
        "max_hydrogen_tank_volume_base_m3": 5.0,
        "max_hydrogen_tank_volume_per_passenger_m3": 0.10,
    }

    outputs = simulate_designs(designs, constraints=constraints, seed=7)

    assert len(outputs) == len(designs)
    assert (outputs["estimated_takeoff_mass_kg"] > 0).all()
    assert (outputs["mission_energy_kwh"] > 0).all()
    assert outputs["is_feasible"].isin([True, False]).all()
    assert outputs.isna().sum().sum() == 0
