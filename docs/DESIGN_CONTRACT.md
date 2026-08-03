# EdgeGenBench v0.1 Design Contract

## Purpose

EdgeGenBench is an independent, public benchmark for comparing surrogate models
on simplified hybrid-electric and hydrogen regional-aircraft design trade-offs.

The v0.1 dataset is synthetic and physics-informed. It is not a certified
aircraft-sizing model and must not be interpreted as a real aircraft-performance
or manufacturer-design prediction.

## Design variables

- Passenger capacity: 40–90 passengers
- Design range: 400–1,500 km
- Cruise speed: 420–650 km/h
- Battery specific energy: 300–750 Wh/kg
- Hydrogen-storage efficiency: 0.45–0.70
- Hybridization ratio: 0.00–0.65
- Propulsion architecture:
  - Conventional turboprop reference
  - Parallel hybrid
  - Series hybrid
  - Fuel-cell electric

## Synthetic outputs

- Estimated takeoff mass
- Mission energy demand
- Energy per passenger-kilometre
- Lifecycle-emissions proxy
- Operating-cost proxy
- Noise proxy
- Battery mass, hydrogen mass, and tank volume
- Constraint margins and a feasibility flag

## Feasibility constraints

A configuration is feasible only when all three conditions hold:

1. Battery mass fraction is below the configured limit.
2. Estimated takeoff mass is below the passenger-scaled mass limit.
3. Hydrogen tank volume is below the passenger-scaled volume limit.

## Reproducibility

The generator uses a fixed random seed, Latin-hypercube sampling, architecture-balanced samples, and deterministic train/validation/test splits.

Generated data are intentionally excluded from Git. The configuration, source code, tests, metadata, and commands required to recreate them are version-controlled.

## Limitations

This project uses no proprietary ATR, R&T-team, or industry data or code. Its outputs are useful for machine-learning benchmarking, uncertainty analysis, optimization, and deployment experiments—not operational aircraft decisions.