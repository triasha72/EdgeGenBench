# EdgeGenBench project overview

## The problem

An offline classifier score does not establish that an edge model is suitable
for release. The export must preserve predictions, handle malformed or degraded
sensor inputs, and meet declared quality gates.

## What I built

I trained a class-balanced Random Forest on NASA DASHlink approaches using an
aircraft-grouped split, exported the selected model to ONNX, and connected model
quality, parity, size, latency, and corrupted-input checks to a release policy.
The repository also contains a separate synthetic aircraft-design benchmark;
those results are not presented as measured-flight performance.

## What the evidence says

The real DASHlink model achieved 0.9496 accuracy but only 0.7380 macro F1 on
17,780 aircraft-disjoint test approaches. Macro F1 and late-flap recall missed
their thresholds, so the release gate blocks it. Under the recorded-flight
corruption tests, ONNX prediction agreement remained above 99.55%.

## Reproduce the software checks

```bash
python -m pip install -e ".[dev,edge,neural]"
edgegenbench info
pytest -q
```

The DASHlink training command and the checksummed measurement artifacts are in
the [README](../README.md). The public data is not redistributed here.

## Next validation

The highest-value next step is improving minority anomaly detection without
weakening the grouped split, followed by a named physical-device measurement of
latency, energy, and thermal behaviour. Neither is currently claimed complete.
