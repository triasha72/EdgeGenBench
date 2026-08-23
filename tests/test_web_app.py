import json
from pathlib import Path

import onnx

ROOT = Path(__file__).parents[1]


def test_web_contract_matches_deployed_onnx_metadata() -> None:
    contract = json.loads((ROOT / "web/model-contract.json").read_text())
    model = onnx.load(ROOT / "web/model/neural_surrogate.onnx")
    model_input = model.graph.input[0]
    model_output = model.graph.output[0]
    assert contract["inputName"] == model_input.name
    assert contract["outputName"] == model_output.name
    assert contract["inputDimension"] == model_input.type.tensor_type.shape.dim[1].dim_value
    assert contract["outputDimension"] == model_output.type.tensor_type.shape.dim[1].dim_value
    assert len(contract["featureMean"]) + len(contract["categories"]) == contract["inputDimension"]
    assert len(contract["targets"]) == contract["outputDimension"]


def test_web_app_references_installable_local_assets() -> None:
    html = (ROOT / "web/index.html").read_text()
    worker = (ROOT / "web/service-worker.js").read_text()
    assert 'rel="manifest"' in html
    assert "neural_surrogate.onnx" in worker
    assert (ROOT / "web/manifest.webmanifest").is_file()
    assert (ROOT / "web/model/neural_surrogate.onnx").is_file()
