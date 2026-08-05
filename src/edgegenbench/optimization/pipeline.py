"""End-to-end constrained multi-objective optimization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from edgegenbench.evaluation.optimization import (
    plot_objective_tradeoff,
    plot_parallel_coordinates,
    select_representative_designs,
)
from edgegenbench.optimization.design_space import (
    generate_candidate_designs,
    load_optimization_config,
)
from edgegenbench.optimization.pareto import (
    extract_pareto_front,
)
from edgegenbench.optimization.search import (
    run_candidate_search_from_paths,
)


@dataclass(frozen=True)
class OptimizationArtifacts:
    """Artifacts produced by one optimization run."""

    candidate_designs_path: Path
    feasible_candidates_path: Path
    pareto_front_path: Path
    representative_designs_path: Path
    summary_path: Path
    plot_paths: tuple[Path, ...]
    candidate_count: int
    feasible_count: int
    pareto_count: int
    representative_count: int
    feasibility_threshold: float

    @property
    def feasible_fraction(self) -> float:
        """Return the accepted candidate fraction."""
        if self.candidate_count == 0:
            return 0.0

        return self.feasible_count / self.candidate_count


def optimize_designs(
    config_path: Path,
    surrogate_model_path: Path,
    feasibility_model_path: Path,
    output_dir: Path | None = None,
) -> OptimizationArtifacts:
    """Run constrained surrogate-assisted optimization."""
    config = load_optimization_config(config_path)

    candidates = generate_candidate_designs(config)

    search_result = run_candidate_search_from_paths(
        candidates=candidates,
        surrogate_model_path=(surrogate_model_path),
        feasibility_model_path=(feasibility_model_path),
        objectives=config.objectives,
    )

    if search_result.feasible_count == 0:
        raise RuntimeError("No candidate designs satisfied the stored feasibility threshold.")

    pareto_front = extract_pareto_front(
        frame=search_result.feasible_candidates,
        objectives=config.objectives,
        directions=config.objective_directions,
    )

    if pareto_front.empty:
        raise RuntimeError("Pareto-front calculation returned no designs.")

    representative_designs = select_representative_designs(
        pareto_front=pareto_front,
        objectives=config.objectives,
        directions=(config.objective_directions),
    )

    destination = output_dir if output_dir is not None else config.output_directory

    destination.mkdir(
        parents=True,
        exist_ok=True,
    )

    candidate_designs_path = destination / "candidate_designs.csv"
    feasible_candidates_path = destination / "feasible_candidates.csv"
    pareto_front_path = destination / "pareto_front.csv"
    representative_designs_path = destination / "representative_designs.csv"
    summary_path = destination / "optimization_summary.json"

    search_result.scored_candidates.to_csv(
        candidate_designs_path,
        index=False,
    )

    search_result.feasible_candidates.to_csv(
        feasible_candidates_path,
        index=False,
    )

    pareto_front.to_csv(
        pareto_front_path,
        index=False,
    )

    representative_designs.to_csv(
        representative_designs_path,
        index=False,
    )

    cost_objective = "operating_cost_proxy_usd"
    emissions_objective = "lifecycle_emissions_proxy_kgco2e"
    noise_objective = "noise_proxy_db"

    required_plot_objectives = {
        cost_objective,
        emissions_objective,
        noise_objective,
    }

    missing_plot_objectives = sorted(required_plot_objectives.difference(config.objectives))

    if missing_plot_objectives:
        raise ValueError(
            f"Optimization configuration is missing plot objectives: {missing_plot_objectives}"
        )

    cost_emissions_path = destination / "cost_vs_emissions.png"
    cost_noise_path = destination / "cost_vs_noise.png"
    emissions_noise_path = destination / "emissions_vs_noise.png"
    parallel_coordinates_path = destination / "parallel_coordinates.png"

    plot_paths = (
        plot_objective_tradeoff(
            feasible_candidates=(search_result.feasible_candidates),
            pareto_front=pareto_front,
            x_objective=cost_objective,
            y_objective=emissions_objective,
            output_path=cost_emissions_path,
            title=("Operating Cost vs Lifecycle Emissions"),
        ),
        plot_objective_tradeoff(
            feasible_candidates=(search_result.feasible_candidates),
            pareto_front=pareto_front,
            x_objective=cost_objective,
            y_objective=noise_objective,
            output_path=cost_noise_path,
            title="Operating Cost vs Noise",
        ),
        plot_objective_tradeoff(
            feasible_candidates=(search_result.feasible_candidates),
            pareto_front=pareto_front,
            x_objective=emissions_objective,
            y_objective=noise_objective,
            output_path=emissions_noise_path,
            title=("Lifecycle Emissions vs Noise"),
        ),
        plot_parallel_coordinates(
            reference_frame=pareto_front,
            representative_designs=(representative_designs),
            columns=(
                "cruise_speed_kmh",
                ("battery_specific_energy_wh_per_kg"),
                "hydrogen_storage_efficiency",
                "hybridization_ratio",
                emissions_objective,
                cost_objective,
                noise_objective,
                "feasibility_probability",
            ),
            output_path=(parallel_coordinates_path),
        ),
    )

    representative_records = {
        str(row.representative_role): str(row.candidate_id)
        for row in representative_designs.itertuples(index=False)
    }

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "configuration_path": str(config_path),
        "optimization_name": config.name,
        "schema_version": (config.schema_version),
        "mission": {
            "passenger_capacity": (config.passenger_capacity),
            "design_range_km": (config.design_range_km),
        },
        "candidate_count": (search_result.candidate_count),
        "feasible_count": (search_result.feasible_count),
        "feasible_fraction": (search_result.feasible_fraction),
        "pareto_count": int(len(pareto_front)),
        "representative_count": int(len(representative_designs)),
        "feasibility_threshold": (search_result.feasibility_threshold),
        "objectives": list(config.objectives),
        "objective_directions": list(config.objective_directions),
        "surrogate_model_path": str(surrogate_model_path),
        "feasibility_model_path": str(feasibility_model_path),
        "representative_designs": (representative_records),
        "candidate_designs_path": str(candidate_designs_path),
        "feasible_candidates_path": str(feasible_candidates_path),
        "pareto_front_path": str(pareto_front_path),
        "representative_designs_path": str(representative_designs_path),
    }

    summary_path.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    return OptimizationArtifacts(
        candidate_designs_path=(candidate_designs_path),
        feasible_candidates_path=(feasible_candidates_path),
        pareto_front_path=pareto_front_path,
        representative_designs_path=(representative_designs_path),
        summary_path=summary_path,
        plot_paths=plot_paths,
        candidate_count=(search_result.candidate_count),
        feasible_count=(search_result.feasible_count),
        pareto_count=len(pareto_front),
        representative_count=len(representative_designs),
        feasibility_threshold=(search_result.feasibility_threshold),
    )
