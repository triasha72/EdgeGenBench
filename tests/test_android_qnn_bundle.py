import runpy
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "verify_android_qnn_bundle.py"
VerifyBundle = Callable[[Path], dict[str, object]]
verify_android_qnn_bundle = cast(VerifyBundle, runpy.run_path(SCRIPT)["verify_android_qnn_bundle"])


def _bundle(tmp_path: Path) -> Path:
    root = tmp_path.resolve() / "qnn"
    (root / "include").mkdir(parents=True)
    (root / "lib/arm64-v8a").mkdir(parents=True)
    (root / "include/onnxruntime_cxx_api.h").write_text("// pinned test header")
    for name in ("libonnxruntime.so", "libQnnHtp.so", "libQnnSystem.so"):
        (root / "lib/arm64-v8a" / name).write_bytes(f"test-{name}".encode())
    return root


def test_verifies_and_hashes_android_qnn_bundle(tmp_path: Path) -> None:
    report = verify_android_qnn_bundle(_bundle(tmp_path))
    assert report["target"] == "android-arm64-v8a"
    assert len(cast(list[object], report["files"])) == 4
    assert "not NPU placement evidence" in cast(str, report["claim"])


def test_rejects_missing_qnn_runtime_library(tmp_path: Path) -> None:
    root = _bundle(tmp_path)
    (root / "lib/arm64-v8a/libQnnSystem.so").unlink()
    with pytest.raises(ValueError, match="libQnnSystem.so"):
        verify_android_qnn_bundle(root)


def test_rejects_relative_bundle_root() -> None:
    with pytest.raises(ValueError, match="absolute path"):
        verify_android_qnn_bundle(Path("qnn"))
