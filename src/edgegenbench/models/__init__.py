"""Surrogate-model implementations for EdgeGenBench."""

from edgegenbench.models.fp32_linear import (
    DEFAULT_TARGETS,
    FP32LinearSurrogate,
)
from edgegenbench.models.tree_surrogate import (
    HIST_GRADIENT_BOOSTING,
    RANDOM_FOREST,
    TreeSurrogate,
)

__all__ = [
    "DEFAULT_TARGETS",
    "FP32LinearSurrogate",
    "HIST_GRADIENT_BOOSTING",
    "RANDOM_FOREST",
    "TreeSurrogate",
]
