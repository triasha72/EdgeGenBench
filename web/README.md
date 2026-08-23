# iPhone browser deployment

The browser app turns the validated FP32 ONNX surrogate into an installable
iPhone experience without Xcode or an App Store release. It loads the model in
Safari, applies the same frozen preprocessing contract used by Python, runs
inference through ONNX Runtime Web, and converts all six outputs back to
physical units. Inputs and predictions stay on the device.

## Publish

The `Deploy browser demo` GitHub Actions workflow stages this directory with
`artifacts/neural_onnx/neural_surrogate.onnx` and publishes it to GitHub Pages.
In the repository's **Settings → Pages**, set **Source** to **GitHub Actions**,
then run the workflow or push a change under `web/`.

The expected public URL is:

```text
https://triasha72.github.io/EdgeGenBench/
```

## Install on an iPhone

1. Open the public URL in Safari.
2. Tap **Share** and then **Add to Home Screen**.
3. Open EdgeGenBench from the new Home Screen icon.
4. Run one design point while online so the runtime and model enter the cache.

The app shell and model are cached for later use. ONNX Runtime Web is supplied
by a pinned CDN release, so the first launch needs a network connection.

## Verify locally

Stage the same files used by the deployment workflow and serve them over HTTP:

```bash
mkdir -p /tmp/edgegenbench-site/model
cp -R web/. /tmp/edgegenbench-site/
cp artifacts/neural_onnx/neural_surrogate.onnx /tmp/edgegenbench-site/model/
python -m http.server 8000 --directory /tmp/edgegenbench-site
```

Open `http://localhost:8000`, wait for “Model ready,” and run a design point.
Localhost supports service workers for development; an iPhone installation
requires the HTTPS address supplied by GitHub Pages.

## Evidence boundary

This closes the browser-delivery gap with a real deployable artifact and a
tested model contract. It does not create an App Store binary or establish
native Core ML, Apple Neural Engine, iPhone latency, or iPhone energy results.
Those claims require the signed-device route described in `ios/README.md`.
