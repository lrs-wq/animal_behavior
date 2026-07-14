"""Security-boundary tests for model artifact loading."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pytest

import eating_behavior.models as models


def test_validate_model_artifact_accepts_regular_file_with_root_and_hash(
    tmp_path: Path,
) -> None:
    """A pinned regular artifact inside an allowed root is accepted."""
    model_path = tmp_path / "trusted" / "model.onnx"
    model_path.parent.mkdir()
    model_path.write_bytes(b"trusted-model")
    expected = hashlib.sha256(b"trusted-model").hexdigest().upper()

    validated = models.validate_model_artifact(
        model_path,
        allowed_roots=[model_path.parent],
        expected_sha256=expected,
        allow_pickle_weights=False,
    )

    assert validated == model_path.resolve()


@pytest.mark.parametrize(
    "invalid_digest",
    ["0" * 63, "0" * 65, "g" * 64, " " + "0" * 64],
)
def test_validate_model_artifact_requires_exact_sha256_format(
    tmp_path: Path,
    invalid_digest: str,
) -> None:
    """Malformed pins fail as policy errors instead of reaching the loader."""
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(b"model")

    with pytest.raises(models.ModelArtifactSecurityError, match="policy is invalid"):
        models.validate_model_artifact(model_path, expected_sha256=invalid_digest)


def test_validate_model_artifact_rejects_hash_mismatch_without_digest_leak(
    tmp_path: Path,
) -> None:
    """Integrity failures do not disclose either the path or calculated hash."""
    model_path = tmp_path / "private-model.onnx"
    model_path.write_bytes(b"model")
    actual = hashlib.sha256(b"model").hexdigest()

    with pytest.raises(models.ModelArtifactSecurityError) as error:
        models.validate_model_artifact(model_path, expected_sha256="0" * 64)

    assert str(model_path) not in str(error.value)
    assert actual not in str(error.value)


def test_validate_model_artifact_rejects_file_outside_allowed_roots(
    tmp_path: Path,
) -> None:
    """Resolved containment is based on path components, not string prefixes."""
    allowed_root = tmp_path / "models"
    allowed_root.mkdir()
    outside_model = tmp_path / "models-untrusted" / "model.onnx"
    outside_model.parent.mkdir()
    outside_model.write_bytes(b"model")

    with pytest.raises(models.ModelArtifactSecurityError, match="permitted"):
        models.validate_model_artifact(outside_model, allowed_roots=[allowed_root])


def test_validate_model_artifact_rejects_pickle_weights_when_disabled(
    tmp_path: Path,
) -> None:
    """Strict callers can opt out of pickle-capable PyTorch checkpoints."""
    model_path = tmp_path / "model.PT"
    model_path.write_bytes(b"not-loaded")

    with pytest.raises(models.ModelArtifactSecurityError, match="Pickle-based"):
        models.validate_model_artifact(model_path, allow_pickle_weights=False)


def test_validate_model_artifact_keeps_pickle_default_backwards_compatible(
    tmp_path: Path,
) -> None:
    """Existing .pt configurations remain valid until strict mode is enabled."""
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"not-loaded")

    assert models.validate_model_artifact(model_path) == model_path.resolve()


def test_validate_model_artifact_rejects_symlink(tmp_path: Path) -> None:
    """A symlink must never be treated as a regular trusted artifact."""
    target = tmp_path / "target.onnx"
    target.write_bytes(b"model")
    link = tmp_path / "model.onnx"
    try:
        link.symlink_to(target)
    except (NotImplementedError, OSError):
        pytest.skip("Creating symlinks is not available to this test process")

    with pytest.raises(models.ModelArtifactSecurityError, match="non-linked"):
        models.validate_model_artifact(link)


@pytest.mark.parametrize("constructor", [models.CatDetector, models.CoatClassifier])
def test_model_constructor_validates_before_yolo(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    constructor: type[Any],
) -> None:
    """Both wrappers complete all artifact checks before constructing YOLO."""
    validated_path = tmp_path / "validated.onnx"
    calls: list[tuple[str, Any]] = []

    def fake_validate(path: str | Path, **kwargs: Any) -> Path:
        calls.append(("validate", (path, kwargs)))
        return validated_path

    def fake_yolo(path: str) -> object:
        calls.append(("yolo", path))
        return object()

    monkeypatch.setattr(models, "validate_model_artifact", fake_validate)
    monkeypatch.setattr(models, "YOLO", fake_yolo)

    instance = constructor(
        "untrusted-input.pt",
        allowed_roots=[tmp_path],
        expected_sha256="a" * 64,
        allow_pickle_weights=False,
    )

    assert [call[0] for call in calls] == ["validate", "yolo"]
    assert calls[0][1][1] == {
        "allowed_roots": [tmp_path],
        "expected_sha256": "a" * 64,
        "allow_pickle_weights": False,
    }
    assert calls[1] == ("yolo", str(validated_path))
    assert instance.model_path == validated_path


@pytest.mark.parametrize("constructor", [models.CatDetector, models.CoatClassifier])
def test_model_constructor_never_calls_yolo_after_validation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    constructor: type[Any],
) -> None:
    """Invalid artifacts cannot reach the external deserializer."""
    yolo_calls: list[str] = []
    monkeypatch.setattr(models, "YOLO", lambda path: yolo_calls.append(path))

    with pytest.raises(FileNotFoundError, match="unavailable"):
        constructor(tmp_path / "missing.pt")

    assert yolo_calls == []


def test_ultralytics_auto_install_is_disabled() -> None:
    """Importing this module leaves Ultralytics auto-install disabled."""
    assert os.environ["YOLO_AUTOINSTALL"] == "false"
