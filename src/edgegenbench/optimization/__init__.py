"""Constrained optimization tools for EdgeGenBench."""

from edgegenbench.optimization.design_space import (
    OptimizationConfig,
    generate_candidate_designs,
    load_optimization_config,
)
from edgegenbench.optimization.pareto import (
    calculate_pareto_mask,
    extract_pareto_front,
)
from edgegenbench.optimization.pipeline import (
    OptimizationArtifacts,
    optimize_designs,
)
from edgegenbench.optimization.search import (
    CandidateSearchResult,
    filter_feasible_candidates,
    load_search_models,
    run_candidate_search,
    run_candidate_search_from_paths,
    score_candidate_designs,
)

__all__ = [
    "CandidateSearchResult",
    "OptimizationArtifacts",
    "OptimizationConfig",
    "calculate_pareto_mask",
    "extract_pareto_front",
    "filter_feasible_candidates",
    "generate_candidate_designs",
    "load_optimization_config",
    "load_search_models",
    "optimize_designs",
    "run_candidate_search",
    "run_candidate_search_from_paths",
    "score_candidate_designs",
]
