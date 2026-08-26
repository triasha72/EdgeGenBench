import json
import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/build_portfolio_acceptance.py"
FUNCTIONS = runpy.run_path(SCRIPT)
ValidateQnn = Callable[[Path, Path], dict[str, object]]
validate_ai_hub_qnn = cast(ValidateQnn, FUNCTIONS["validate_ai_hub_qnn"])


def test_validates_tracked_ai_hub_qnn_evidence() -> None:
    root = Path(__file__).parents[1]
    result = validate_ai_hub_qnn(root / "reports/qualcomm_qnn_v0_1.json", root)
    assert result["status"] == "validated_ai_hub_physical_qnn"
    assert result["backend"] == "QNN HTP"
    assert result["source_model_matches_repository"] is True
    assert len(cast(list[object], result["graphs"])) == 3


def test_rejects_cpu_compute_unit_in_qnn_report(tmp_path: Path) -> None:
    root = Path(__file__).parents[1]
    report = json.loads((root / "reports/qualcomm_qnn_v0_1.json").read_text())
    report["linked_multigraph"]["validation"]["graphs"]["edgegenbench_batch1"]["profile"][
        "compute_units"
    ] = {"CPU": 1, "NPU": 8}
    path = tmp_path / "qnn.json"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="exclusive NPU"):
        validate_ai_hub_qnn(path, root)
