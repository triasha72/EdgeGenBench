# EdgeGenBench

**Uncertainty-aware surrogate modeling and edge-ready inference for hybrid-electric and hydrogen regional-aircraft design.**

## Objective

EdgeGenBench is an independent, reproducible benchmark for comparing surrogate models used in early aircraft design trade studies. It evaluates predictive accuracy, uncertainty behavior, optimization usefulness, and deployment efficiency.

The project uses public or synthetic data only. It does not use proprietary ATR, R&T-team, or industry code or data.

## What it will benchmark

- Gaussian-process, radial-basis-function, and neural surrogate models
- Physics-informed synthetic aircraft-design data generation
- Uncertainty estimation and calibration
- Constrained multi-objective optimization and Pareto-front analysis
- ONNX export, quantization, model size, and inference-latency comparisons

## Initial design variables

- Passenger capacity
- Design range
- Cruise speed
- Battery specific energy
- Hydrogen storage efficiency
- Hybridization ratio
- Propulsion architecture

## Roadmap

1. Build a reproducible synthetic hybrid-electric/hydrogen aircraft design dataset.
2. Train and validate surrogate-model baselines.
3. Compare uncertainty and failure behavior across models.
4. Run constrained multi-objective optimization.
5. Export the best deployable model to ONNX and benchmark edge inference.

## Development

```bash
python -m pip install -e ".[dev]"
pytest
ruff check .
edgegenbench info

## Generate the benchmark dataset

```bash
edgegenbench generate-data --config configs/v0_1.yaml

## 9. Train the FP32 baseline

Generate the synthetic dataset first:

```bash
edgegenbench generate-data --config configs/v0_1.yaml

## 10. Run everything

From the repository root:

```bash
ruff format .
ruff check .
pytest
edgegenbench info
edgegenbench generate-data --config configs/v0_1.yaml
ls -lh data/raw
git status

# 11. Final verification

Run the complete workflow:

```bash
ruff format .
ruff check .
pytest

edgegenbench info

edgegenbench generate-data \
  --config configs/v0_1.yaml

edgegenbench train-fp32-baseline \
  --dataset data/raw/edgegenbench_v0_1.csv \
  --output-dir artifacts/fp32_baseline
  git status
git diff --stat