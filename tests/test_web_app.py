import json
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_web_contract_matches_deployed_onnx_metadata() -> None:
    contract = json.loads((ROOT / "web/model-contract.json").read_text())
    metadata = json.loads((ROOT / "artifacts/neural_onnx/metadata.json").read_text())
    assert contract["inputName"] == metadata["input_name"]
    assert contract["outputName"] == metadata["output_name"]
    assert contract["inputDimension"] == metadata["input_dim"]
    assert contract["outputDimension"] == metadata["output_dim"]
    assert contract["targets"] == metadata["targets"]
    assert len(contract["featureMean"]) + len(contract["categories"]) == contract["inputDimension"]


def test_web_app_references_installable_local_assets() -> None:
    html = (ROOT / "web/index.html").read_text()
    worker = (ROOT / "web/service-worker.js").read_text()
    assert 'rel="manifest"' in html
    assert "neural_surrogate.onnx" in worker
    assert (ROOT / "web/manifest.webmanifest").is_file()
    assert (ROOT / "artifacts/neural_onnx/neural_surrogate.onnx").is_file()
