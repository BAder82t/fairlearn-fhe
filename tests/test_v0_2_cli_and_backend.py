# Copyright 2026 Vaultbytes (Bader Alissaei)
# SPDX-License-Identifier: Apache-2.0

"""Tests for the v0.2.0 CLI subcommands, JSON Schema export, and the
OpenFHE ``noise_flooding`` wiring.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from fairlearn_fhe import (
    ENVELOPE_JSON_SCHEMA,
    ENVELOPE_SCHEMA,
    audit_metric,
    build_context,
    encrypt,
    envelope_json_schema,
)
from fairlearn_fhe._backends import list_backends
from fairlearn_fhe._backends.openfhe_backend import (
    _apply_native_noise_flooding,
    _normalize_flooding_label,
)
from fairlearn_fhe.cli import main, main_verify_legacy

# ---------------------------------------------------------------------------
# JSON Schema export
# ---------------------------------------------------------------------------


def test_envelope_json_schema_is_well_formed():
    schema = envelope_json_schema()
    assert isinstance(schema, dict)
    assert schema["$id"] == ENVELOPE_SCHEMA
    assert schema["type"] == "object"
    assert "metric_name" in schema["properties"]
    assert "parameter_set" in schema["properties"]


def test_envelope_json_schema_constant_matches_factory():
    assert envelope_json_schema() == ENVELOPE_JSON_SCHEMA
    # Mutating the returned copy must not affect the constant.
    snapshot = envelope_json_schema()
    snapshot["title"] = "tampered"
    assert ENVELOPE_JSON_SCHEMA["title"] != "tampered"


# ---------------------------------------------------------------------------
# CLI subcommands
# ---------------------------------------------------------------------------


@pytest.fixture()
def envelope_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 32
    y_true = (rng.random(n) > 0.5).astype(int)
    y_pred = y_true.copy()
    sf = rng.choice(["A", "B"], size=n).astype(object)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, y_pred.astype(float))
    env = audit_metric(
        "demographic_parity_difference",
        y_true=y_true,
        y_pred=yp_enc,
        sensitive_features=sf,
        ctx=ctx,
        min_group_size=8,
    )
    out = tmp_path / "envelope.json"
    out.write_text(json.dumps(env.to_dict()))
    return out


def test_cli_verify_subcommand(envelope_path: Path, capsys):
    rc = main(["verify", str(envelope_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out


def test_cli_verify_legacy_entry_point(envelope_path: Path, capsys):
    rc = main_verify_legacy([str(envelope_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out


def test_cli_legacy_invocation_through_main(envelope_path: Path, capsys):
    # The new ``main`` should still accept the legacy positional form
    # (envelope path without an explicit subcommand).
    rc = main([str(envelope_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out


def test_cli_inspect_subcommand(envelope_path: Path, capsys):
    rc = main(["inspect", str(envelope_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "metric:" in out
    assert "backend:" in out
    assert "demographic_parity_difference" in out


def test_cli_inspect_subcommand_json(envelope_path: Path, capsys):
    rc = main(["inspect", str(envelope_path), "--json"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["metric"] == "demographic_parity_difference"
    assert "backend" in payload


def test_cli_inspect_subcommand_handles_missing_file(tmp_path: Path, capsys):
    rc = main(["inspect", str(tmp_path / "missing.json")])
    err = capsys.readouterr().err
    assert rc == 2
    assert "failed to read" in err


def test_cli_schema_subcommand(capsys):
    rc = main(["schema"])
    out = capsys.readouterr().out
    assert rc == 0
    payload = json.loads(out)
    assert payload["$id"] == ENVELOPE_SCHEMA


def test_cli_schema_subcommand_pretty(capsys):
    rc = main(["schema", "--pretty"])
    out = capsys.readouterr().out
    assert rc == 0
    # Pretty output has line breaks; compact does not.
    assert "\n" in out
    assert json.loads(out)["$id"] == ENVELOPE_SCHEMA


def test_cli_doctor_subcommand(capsys):
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "backends:" in out
    for name in list_backends():
        assert name in out


def test_cli_no_subcommand_and_no_positional_prints_help(capsys):
    rc = main([])
    err = capsys.readouterr().err
    assert rc == 2
    assert "usage:" in err.lower()


# ---------------------------------------------------------------------------
# OpenFHE noise_flooding wiring (no live openfhe install required)
# ---------------------------------------------------------------------------


class _StubParams:
    def __init__(self) -> None:
        self.execution_mode = None
        self.decrypt_mode = None

    def SetExecutionMode(self, mode):  # noqa: N802 - mirrors openfhe API
        self.execution_mode = mode

    def SetDecryptionNoiseMode(self, mode):  # noqa: N802 - mirrors openfhe API
        self.decrypt_mode = mode


class _StubFHE:
    EXEC_NOISE_FLOODING = "EXEC_NOISE_FLOODING"
    NOISE_FLOODING_DECRYPT = "NOISE_FLOODING_DECRYPT"


class _StubFHEMissingAPI:
    pass


@pytest.mark.parametrize(
    "label,expected",
    [
        ("openfhe-NOISE_FLOODING_DECRYPT", "openfhe-noise-flooding-decrypt"),
        ("openfhe-noise_flooding_decrypt", "openfhe-noise-flooding-decrypt"),
        ("noise-flooding", "noise-flooding"),
        ("Noise Flooding", "noise-flooding"),
        (None, ""),
        (True, "noise-flooding"),
        (False, ""),
    ],
)
def test_normalize_flooding_label(label, expected):
    assert _normalize_flooding_label(label) == expected


def test_apply_native_noise_flooding_recognized_label():
    p = _StubParams()
    enabled = _apply_native_noise_flooding(_StubFHE(), p, "openfhe-noise-flooding-decrypt")
    assert enabled is True
    assert p.execution_mode == "EXEC_NOISE_FLOODING"
    assert p.decrypt_mode == "NOISE_FLOODING_DECRYPT"


def test_apply_native_noise_flooding_unrecognized_label_is_noop():
    p = _StubParams()
    enabled = _apply_native_noise_flooding(_StubFHE(), p, "unknown-strategy")
    assert enabled is False
    assert p.execution_mode is None


def test_apply_native_noise_flooding_missing_api_raises():
    p = _StubParams()
    with pytest.raises(RuntimeError, match="SetExecutionMode"):
        _apply_native_noise_flooding(_StubFHEMissingAPI(), p, "noise-flooding")


def test_build_context_threads_noise_flooding_through(monkeypatch):
    # Stub the openfhe import so we can verify the wiring without a
    # live native build. The test patches the backend module directly.
    from fairlearn_fhe._backends import openfhe_backend as oh

    class _StubKeys:
        secretKey = object()
        publicKey = object()

    class _StubCC:
        def Enable(self, _f):
            pass

        def KeyGen(self):
            return _StubKeys()

        def EvalMultKeyGen(self, _sk):
            pass

        def EvalRotateKeyGen(self, _sk, _steps):
            pass

        def EvalSumKeyGen(self, _sk):
            pass

        def GetRingDimension(self):
            return 16384

    captured = {}

    class _StubModule:
        EXEC_NOISE_FLOODING = "EXEC_NOISE_FLOODING"
        NOISE_FLOODING_DECRYPT = "NOISE_FLOODING_DECRYPT"

        class CCParamsCKKSRNS:
            def __init__(self):
                self.execution_mode = None
                self.decrypt_mode = None

            def SetMultiplicativeDepth(self, _d):
                pass

            def SetScalingModSize(self, _s):
                pass

            def SetBatchSize(self, _b):
                pass

            def SetExecutionMode(self, mode):
                self.execution_mode = mode
                captured["execution_mode"] = mode

            def SetDecryptionNoiseMode(self, mode):
                self.decrypt_mode = mode
                captured["decrypt_mode"] = mode

        class PKESchemeFeature:
            PKE = "PKE"
            KEYSWITCH = "KEYSWITCH"
            LEVELEDSHE = "LEVELEDSHE"
            ADVANCEDSHE = "ADVANCEDSHE"

        @staticmethod
        def GenCryptoContext(_p):
            return _StubCC()

    import sys

    monkeypatch.setitem(sys.modules, "openfhe", _StubModule)
    ctx = oh.build_context(noise_flooding="openfhe-NOISE_FLOODING_DECRYPT")
    assert ctx.noise_flooding is True
    assert captured.get("execution_mode") == "EXEC_NOISE_FLOODING"
    assert captured.get("decrypt_mode") == "NOISE_FLOODING_DECRYPT"
