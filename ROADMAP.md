# EdgeGenBench: what remains

[Read the project overview and measured results](README.md)

The real-flight track is now the center of this project. It trained on 64,887
NASA DASHlink approaches and was tested once on 17,780 approaches from different
de-identified aircraft. The exported ONNX model has also been tested on the same
flights with noise, missing readings, and short sensor dropouts.

The current macro F1 is `0.7380`. That is close to the `0.75` release threshold,
but close is not a pass. Late-flap recall also misses its `0.60` requirement, so
the release assessment correctly blocks this candidate.

## The next model work

I will focus on the minority anomaly classes rather than making the already
strong nominal class even easier. Class weighting, calibrated probabilities,
and an abstention option should be compared on the validation set. The protected
test data stays untouched until a candidate is chosen.

Where the public metadata supports it, I also want to create harder splits by
airport, time, weather, or aircraft subgroup. That would show whether the model
has learned an anomaly pattern or merely the conditions surrounding it.

## The next deployment work

- Measure end-to-end latency, peak memory, and energy on named physical devices.
- Run the exact preprocessing and ONNX graph used in the application, not a
  detached matrix benchmark.
- Add calibrated escalation for inputs that are valid but unfamiliar.
- Conduct a shadow-mode study with aviation-domain review before discussing any
  operational use.

The generated aircraft-design track will remain as a useful optimization and
hardware benchmark. Its results will continue to be labeled as generated-data
results, separate from the recorded-flight evidence.
