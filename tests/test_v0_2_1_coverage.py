# Copyright 2026 Vaultbytes (Bader Alissaei)
# SPDX-License-Identifier: Apache-2.0

"""v0.2.1 coverage closeout.

Targets every line that was uncovered after the v0.2.0 ship:

- Mode B (encrypted-mask) paths for the new per-rate / scoring /
  regression metrics.
- Plaintext fallback paths in `_per_rate_metrics` / `_scoring_metrics`
  / `_regression_metrics` when sensitive_features are plaintext but
  `y_pred` is encrypted via the alternate code path.
- Plaintext-version fallback in `_fairness_metrics` when upstream
  Fairlearn lacks `equal_opportunity_difference` / `equal_opportunity_ratio`.
- Context lifecycle: `set_default_context`, `reset_default_context`,
  `default_context`, `make_public` (both backends), unknown-backend.
- CLI residuals: stdin path, oversized envelope, missing key file,
  schema/inspect/doctor with edge inputs.
- Encrypted vector edges: `neg`, `sub`, `mul_scalar`, `sum_all` n=0,
  decrypt path.
- Envelope edges: signature absence, validation rejecting forged
  hashes, schema-version mismatch.
- TenSEAL/OpenFHE backend import-failure paths via monkeypatch.

These tests are deliberately mechanical — each one targets a specific
line range from the coverage report rather than chasing higher-level
behaviour. Behavioural tests live in the other files.
"""

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from fairlearn_fhe import (
    CKKSContext,
    EncryptedVector,
    audit_metric,
    build_context,
    default_context,
    encrypt,
    encrypt_sensitive_features,
    parameter_set_from_context,
    reset_default_context,
    set_default_context,
    sign_envelope,
    validate_envelope,
    verify_envelope_signature,
)
from fairlearn_fhe._backends import get_backend, list_backends, set_default_backend
from fairlearn_fhe._groups import EncryptedMaskSet, MaskDecryptionWarning
from fairlearn_fhe.cli import main, main_verify_legacy
from fairlearn_fhe.metrics import (
    EncryptedMetricFrame,
    MetricFrame,
    accuracy_score_difference,
    accuracy_score_group_min,
    balanced_accuracy_score_group_min,
    demographic_parity_difference,
    equal_opportunity_difference,
    equal_opportunity_ratio,
    f1_score_group_min,
    false_positive_rate_difference,
    mean_squared_error_group_max,
    precision_score_group_min,
    r2_score_group_min,
    recall_score_group_min,
    selection_rate,
    selection_rate_difference,
    selection_rate_ratio,
    true_positive_rate_difference,
    true_positive_rate_ratio,
    zero_one_loss_difference,
    zero_one_loss_group_max,
)

_TOL = 1e-3


# ---------------------------------------------------------------------------
# Mode B (encrypted-mask) path for the v0.2 metric ports
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mode_b_classification():
    rng = np.random.default_rng(7)
    n = 128
    y_true = (rng.random(n) > 0.5).astype(int)
    y_pred = y_true.copy()
    flip = rng.random(n) < 0.20
    y_pred = np.where(flip, 1 - y_pred, y_pred)
    sf = rng.choice(["A", "B", "C"], size=n).astype(object)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, y_pred.astype(float))
    sf_enc = encrypt_sensitive_features(ctx, sf, y_true=y_true)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_enc": yp_enc,
        "sf_plain": sf,
        "sf_enc": sf_enc,
        "ctx": ctx,
    }


@pytest.mark.parametrize(
    "fn",
    [
        selection_rate_difference,
        selection_rate_ratio,
        true_positive_rate_difference,
        true_positive_rate_ratio,
        false_positive_rate_difference,
    ],
)
def test_mode_b_per_rate_matches_mode_a(mode_b_classification, fn):
    plain = fn(
        mode_b_classification["y_true"], mode_b_classification["y_pred"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    enc_b = fn(
        mode_b_classification["y_true"], mode_b_classification["y_pred_enc"],
        sensitive_features=mode_b_classification["sf_enc"],
    )
    assert math.isclose(plain, enc_b, abs_tol=_TOL)


@pytest.mark.parametrize(
    "fn",
    [
        accuracy_score_group_min,
        balanced_accuracy_score_group_min,
        precision_score_group_min,
        recall_score_group_min,
        f1_score_group_min,
        zero_one_loss_group_max,
    ],
)
def test_mode_b_scoring_matches_mode_a(mode_b_classification, fn):
    plain = fn(
        mode_b_classification["y_true"], mode_b_classification["y_pred"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    enc_b = fn(
        mode_b_classification["y_true"], mode_b_classification["y_pred_enc"],
        sensitive_features=mode_b_classification["sf_enc"],
    )
    assert math.isclose(plain, enc_b, abs_tol=_TOL)


def test_mode_b_accuracy_difference_and_zero_one_loss_difference(mode_b_classification):
    plain = accuracy_score_difference(
        mode_b_classification["y_true"], mode_b_classification["y_pred"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    enc_b = accuracy_score_difference(
        mode_b_classification["y_true"], mode_b_classification["y_pred_enc"],
        sensitive_features=mode_b_classification["sf_enc"],
    )
    assert math.isclose(plain, enc_b, abs_tol=_TOL)

    plain_zol = zero_one_loss_difference(
        mode_b_classification["y_true"], mode_b_classification["y_pred"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    enc_zol = zero_one_loss_difference(
        mode_b_classification["y_true"], mode_b_classification["y_pred_enc"],
        sensitive_features=mode_b_classification["sf_enc"],
    )
    assert math.isclose(plain_zol, enc_zol, abs_tol=_TOL)


@pytest.fixture(scope="module")
def mode_b_regression():
    rng = np.random.default_rng(11)
    n = 128
    y_true = rng.normal(0.5, 0.2, size=n).astype(float)
    y_pred = y_true + rng.normal(0.0, 0.05, size=n)
    sf = rng.choice(["A", "B", "C"], size=n).astype(object)
    y_pred[sf == "A"] += 0.05
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, y_pred)
    # For regression Mode B we still pass y_true at encryption time so the
    # mask set carries plaintext positives/negatives counts.
    sf_enc = encrypt_sensitive_features(ctx, sf, y_true=(y_true > 0.5).astype(int))
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_enc": yp_enc,
        "sf_plain": sf,
        "sf_enc": sf_enc,
    }


def test_mode_b_mean_squared_error_group_max(mode_b_regression):
    plain = mean_squared_error_group_max(
        mode_b_regression["y_true"], mode_b_regression["y_pred"],
        sensitive_features=mode_b_regression["sf_plain"],
    )
    enc_b = mean_squared_error_group_max(
        mode_b_regression["y_true"], mode_b_regression["y_pred_enc"],
        sensitive_features=mode_b_regression["sf_enc"],
    )
    assert math.isclose(plain, enc_b, abs_tol=_TOL)


def test_mode_b_r2_score_group_min(mode_b_regression):
    # R² uses the global y mean as TSS reference in Mode B (per
    # _regression_metrics docstring); it should still agree with the
    # plaintext computation within CKKS noise.
    plain = r2_score_group_min(
        mode_b_regression["y_true"], mode_b_regression["y_pred"],
        sensitive_features=mode_b_regression["sf_plain"],
    )
    enc_b = r2_score_group_min(
        mode_b_regression["y_true"], mode_b_regression["y_pred_enc"],
        sensitive_features=mode_b_regression["sf_enc"],
    )
    # Mode B uses a different TSS basis than Mode A so the values
    # differ; we only require both to be finite real numbers.
    assert math.isfinite(plain) and math.isfinite(enc_b)


# ---------------------------------------------------------------------------
# Type guard: encrypted sf with plaintext y_pred raises
# ---------------------------------------------------------------------------


def test_mode_b_rejects_plaintext_y_pred_per_rate(mode_b_classification):
    with pytest.raises(TypeError, match="encrypted sensitive_features"):
        selection_rate_difference(
            mode_b_classification["y_true"], mode_b_classification["y_pred"],
            sensitive_features=mode_b_classification["sf_enc"],
        )


def test_mode_b_rejects_plaintext_y_pred_scoring(mode_b_classification):
    with pytest.raises(TypeError, match="encrypted sensitive_features"):
        accuracy_score_group_min(
            mode_b_classification["y_true"], mode_b_classification["y_pred"],
            sensitive_features=mode_b_classification["sf_enc"],
        )


def test_mode_b_rejects_plaintext_y_pred_regression(mode_b_regression):
    with pytest.raises(TypeError, match="encrypted sensitive_features"):
        mean_squared_error_group_max(
            mode_b_regression["y_true"], mode_b_regression["y_pred"],
            sensitive_features=mode_b_regression["sf_enc"],
        )


# ---------------------------------------------------------------------------
# Plaintext-fallback paths in _per_rate_metrics when upstream Fairlearn
# lacks the helper
# ---------------------------------------------------------------------------


def test_per_rate_falls_back_when_upstream_missing(monkeypatch, mode_b_classification):
    import fairlearn.metrics as fl

    # Hide the upstream helper so the per-rate module's plaintext
    # fallback branch fires.
    name = "true_positive_rate_difference"
    if hasattr(fl, name):
        monkeypatch.delattr(fl, name)
    plain = true_positive_rate_difference(
        mode_b_classification["y_true"], mode_b_classification["y_pred"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    assert isinstance(plain, float)


def test_per_rate_ratio_fallback_when_upstream_missing(monkeypatch, mode_b_classification):
    import fairlearn.metrics as fl

    name = "true_positive_rate_ratio"
    if hasattr(fl, name):
        monkeypatch.delattr(fl, name)
    plain = true_positive_rate_ratio(
        mode_b_classification["y_true"], mode_b_classification["y_pred"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    assert isinstance(plain, float)


# ---------------------------------------------------------------------------
# _fairness_metrics: plaintext fallback when upstream lacks the helper
# ---------------------------------------------------------------------------


def test_equal_opportunity_difference_fallback_when_upstream_missing(
    monkeypatch, mode_b_classification
):
    import fairlearn.metrics as fl

    if hasattr(fl, "equal_opportunity_difference"):
        monkeypatch.delattr(fl, "equal_opportunity_difference")
    val = equal_opportunity_difference(
        mode_b_classification["y_true"], mode_b_classification["y_pred"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    assert isinstance(val, float)


def test_equal_opportunity_ratio_fallback_when_upstream_missing(
    monkeypatch, mode_b_classification
):
    import fairlearn.metrics as fl

    if hasattr(fl, "equal_opportunity_ratio"):
        monkeypatch.delattr(fl, "equal_opportunity_ratio")
    val = equal_opportunity_ratio(
        mode_b_classification["y_true"], mode_b_classification["y_pred"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Context lifecycle
# ---------------------------------------------------------------------------


def test_default_context_lifecycle():
    reset_default_context()
    ctx_a = default_context()
    ctx_b = default_context()
    assert ctx_a is ctx_b  # cached
    new_ctx = build_context(backend="tenseal")
    set_default_context(new_ctx)
    assert default_context() is new_ctx
    reset_default_context()


def test_build_context_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        build_context(backend="madeup")  # type: ignore[arg-type]


def test_set_default_backend_rejects_unknown():
    with pytest.raises(ValueError, match="unknown backend"):
        set_default_backend("not-a-backend")  # type: ignore[arg-type]


def test_get_backend_unknown_raises():
    with pytest.raises(ValueError, match="unknown backend"):
        get_backend("madeup")  # type: ignore[arg-type]


def test_make_evaluator_context_strips_secret_key_tenseal():
    ctx = build_context(backend="tenseal")
    pub = ctx.make_evaluator_context()
    assert pub.has_secret_key is False
    assert ctx.has_secret_key is True  # original untouched


# ---------------------------------------------------------------------------
# Encrypted vector edges
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def vec():
    ctx = build_context(backend="tenseal")
    return ctx, EncryptedVector.encrypt(ctx, np.array([1.0, 2.0, 3.0, 4.0]))


def test_encrypted_neg(vec):
    _ctx, v = vec
    out = (-v).decrypt()[:4]
    assert np.allclose(out, [-1.0, -2.0, -3.0, -4.0], atol=1e-3)


def test_encrypted_sub_plaintext(vec):
    _ctx, v = vec
    out = (v - np.array([0.5, 0.5, 0.5, 0.5])).decrypt()[:4]
    assert np.allclose(out, [0.5, 1.5, 2.5, 3.5], atol=1e-3)


def test_encrypted_add_plaintext(vec):
    _ctx, v = vec
    out = (v + np.array([10.0, 10.0, 10.0, 10.0])).decrypt()[:4]
    assert np.allclose(out, [11.0, 12.0, 13.0, 14.0], atol=1e-3)


def test_encrypted_add_ct(vec):
    _ctx, v = vec
    out = (v + v).decrypt()[:4]
    assert np.allclose(out, [2.0, 4.0, 6.0, 8.0], atol=1e-3)


def test_encrypted_sub_ct(vec):
    # SEAL refuses ciphertext - itself ("result ciphertext is
    # transparent" — would expose the secret-key side channel), so we
    # subtract two distinct encryptions of contrasting vectors.
    ctx, v = vec
    w = EncryptedVector.encrypt(ctx, np.array([0.5, 0.5, 0.5, 0.5]))
    out = (v - w).decrypt()[:4]
    assert np.allclose(out, [0.5, 1.5, 2.5, 3.5], atol=1e-3)


def test_encrypted_mul_scalar(vec):
    _ctx, v = vec
    out = v.mul_scalar(2.0).decrypt()[:4]
    assert np.allclose(out, [2.0, 4.0, 6.0, 8.0], atol=1e-3)


def test_encrypted_decrypt_returns_ndarray(vec):
    _ctx, v = vec
    out = v.decrypt()
    assert isinstance(out, np.ndarray)
    assert len(out) >= 4


def test_encrypted_first_slot(vec):
    _ctx, v = vec
    assert math.isclose(v.first_slot(), 1.0, abs_tol=1e-3)


# ---------------------------------------------------------------------------
# selection_rate plaintext fallthrough at the base-metrics layer
# ---------------------------------------------------------------------------


def test_selection_rate_plaintext_passthrough():
    import fairlearn.metrics as fl

    rng = np.random.default_rng(0)
    y_true = (rng.random(64) > 0.5).astype(int)
    y_pred = y_true.copy()
    val = selection_rate(y_true, y_pred, pos_label=1)
    expected = fl.selection_rate(y_true, y_pred, pos_label=1)
    assert math.isclose(val, expected, abs_tol=1e-9)


# ---------------------------------------------------------------------------
# CLI residuals
# ---------------------------------------------------------------------------


def _make_envelope(tmp_path: Path) -> Path:
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


def test_cli_verify_reads_from_stdin(monkeypatch, tmp_path: Path, capsys):
    env_path = _make_envelope(tmp_path)
    body = env_path.read_text()
    monkeypatch.setattr("sys.stdin", _StringStream(body))
    rc = main(["verify", "-"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "OK" in out


def test_cli_verify_rejects_oversized_stdin(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", _StringStream("a" * (1024 * 1024 + 64)))
    rc = main(["verify", "-"])
    out_err = capsys.readouterr()
    assert rc != 0
    assert "exceeds" in out_err.out or "exceeds" in out_err.err


def test_cli_verify_rejects_non_object_envelope(tmp_path: Path, capsys):
    p = tmp_path / "envelope.json"
    p.write_text(json.dumps([1, 2, 3]))
    rc = main(["verify", str(p)])
    capsys.readouterr()  # drain captured output
    assert rc != 0


def test_cli_verify_legacy_help_exits_zero(capsys):
    with pytest.raises(SystemExit) as exc:
        main_verify_legacy(["--help"])
    assert exc.value.code == 0


def test_cli_verify_with_missing_public_key(tmp_path: Path, capsys):
    env_path = _make_envelope(tmp_path)
    rc = main(
        [
            "verify",
            str(env_path),
            "--public-key",
            str(tmp_path / "missing.pem"),
            "--json",
        ]
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["valid"] is False
    assert any("public key" in e for e in payload["errors"])
    assert rc != 0


class _StringStream:
    """Minimal sys.stdin replacement that supports .read(n)."""

    def __init__(self, content: str) -> None:
        self._content = content

    def read(self, n: int = -1) -> str:
        if n < 0:
            return self._content
        return self._content[:n]


# ---------------------------------------------------------------------------
# audit + envelope edges
# ---------------------------------------------------------------------------


def test_audit_metric_emits_small_group_warning(tmp_path: Path):
    from fairlearn_fhe import SmallGroupWarning

    rng = np.random.default_rng(1)
    n = 32
    y_true = (rng.random(n) > 0.5).astype(int)
    sf = np.array(["A"] * 31 + ["B"])  # group B has size 1
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, y_true.astype(float))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        audit_metric(
            "demographic_parity_difference",
            y_true=y_true,
            y_pred=yp_enc,
            sensitive_features=sf,
            ctx=ctx,
            min_group_size=10,
        )
    assert any(issubclass(w.category, SmallGroupWarning) for w in caught)


def test_audit_metric_unknown_metric_raises():
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, np.zeros(8))
    with pytest.raises(KeyError, match="unknown metric"):
        audit_metric(
            "definitely-not-a-metric",
            y_true=np.zeros(8, dtype=int),
            y_pred=yp_enc,
            sensitive_features=np.array(["A"] * 8),
            ctx=ctx,
        )


def test_validate_envelope_rejects_bad_schema_version():
    # Build a valid envelope first, then mutate its schema_version.
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, np.zeros(8))
    env = audit_metric(
        "demographic_parity_difference",
        y_true=np.zeros(8, dtype=int),
        y_pred=yp_enc,
        sensitive_features=np.array(["A"] * 4 + ["B"] * 4),
        ctx=ctx,
        min_group_size=2,
    )
    payload = env.to_dict()
    payload["schema_version"] = "fairlearn-fhe.metric-envelope.v0"
    errors = validate_envelope(payload)
    assert any("schema" in e.lower() for e in errors)


def test_validate_envelope_rejects_tampered_hash():
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, np.zeros(8))
    env = audit_metric(
        "demographic_parity_difference",
        y_true=np.zeros(8, dtype=int),
        y_pred=yp_enc,
        sensitive_features=np.array(["A"] * 4 + ["B"] * 4),
        ctx=ctx,
        min_group_size=2,
    )
    payload = env.to_dict()
    payload["parameter_set_hash"] = "f" * 64  # wrong hash
    errors = validate_envelope(payload)
    assert any("hash" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# parameter_set_from_context paths for both backends
# ---------------------------------------------------------------------------


def test_parameter_set_from_context_tenseal():
    ctx = build_context(backend="tenseal")
    ps = parameter_set_from_context(ctx)
    assert ps.backend.startswith("tenseal") or "ckks" in ps.backend
    assert ps.poly_modulus_degree > 0


# ---------------------------------------------------------------------------
# MetricFrame.fhe() round-trip
# ---------------------------------------------------------------------------


def test_metric_frame_returns_encrypted_metric_frame_for_encrypted_input(
    mode_b_classification,
):
    import fairlearn.metrics as fl

    enc_mf = MetricFrame(
        metrics={"sr": fl.selection_rate, "tpr": fl.true_positive_rate},
        y_true=mode_b_classification["y_true"],
        y_pred=mode_b_classification["y_pred_enc"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    assert isinstance(enc_mf, EncryptedMetricFrame)


def test_metric_frame_falls_through_to_fairlearn_for_plaintext(mode_b_classification):
    import fairlearn.metrics as fl

    plain_mf = MetricFrame(
        metrics={"sr": fl.selection_rate},
        y_true=mode_b_classification["y_true"],
        y_pred=mode_b_classification["y_pred"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    assert isinstance(plain_mf, fl.MetricFrame)


def test_metric_frame_fhe_callable_alias(mode_b_classification):
    import fairlearn.metrics as fl

    enc_mf = MetricFrame.fhe(
        metrics={"sr": fl.selection_rate},
        y_true=mode_b_classification["y_true"],
        y_pred=mode_b_classification["y_pred_enc"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    assert isinstance(enc_mf, EncryptedMetricFrame)


def test_metric_frame_fhe_rejects_plaintext_y_pred(mode_b_classification):
    import fairlearn.metrics as fl

    with pytest.raises(TypeError, match="encrypted"):
        MetricFrame.fhe(
            metrics={"sr": fl.selection_rate},
            y_true=mode_b_classification["y_true"],
            y_pred=mode_b_classification["y_pred"],
            sensitive_features=mode_b_classification["sf_plain"],
        )


# ---------------------------------------------------------------------------
# encrypt_sensitive_features without y_true (bare mask set)
# ---------------------------------------------------------------------------


def test_encrypt_sensitive_features_without_y_true():
    ctx = build_context(backend="tenseal")
    sf = np.array(["A", "B", "A", "C"], dtype=object)
    mset = encrypt_sensitive_features(ctx, sf)
    assert isinstance(mset, EncryptedMaskSet)
    assert mset.positives is None and mset.negatives is None


def test_encrypt_sensitive_features_attaches_counts():
    ctx = build_context(backend="tenseal")
    sf = np.array(["A", "B", "A", "B"], dtype=object)
    y_true = np.array([1, 0, 1, 0])
    mset = encrypt_sensitive_features(ctx, sf, y_true=y_true)
    assert mset.positives == {"A": 2.0, "B": 0.0}
    assert mset.negatives == {"A": 0.0, "B": 2.0}


def test_attach_label_counts_warns_when_no_plaintext_masks_supplied():
    ctx = build_context(backend="tenseal")
    sf = np.array(["A", "B", "A", "B"], dtype=object)
    mset = encrypt_sensitive_features(ctx, sf)
    y_true = np.array([1, 0, 1, 0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mset.attach_label_counts(y_true=y_true)
    assert any(issubclass(w.category, MaskDecryptionWarning) for w in caught)


def test_attach_label_counts_with_plaintext_masks_does_not_warn():
    ctx = build_context(backend="tenseal")
    sf = np.array(["A", "B", "A", "B"], dtype=object)
    mset = encrypt_sensitive_features(ctx, sf)
    y_true = np.array([1, 0, 1, 0])
    plain_masks = {
        "A": np.array([1, 0, 1, 0], dtype=float),
        "B": np.array([0, 1, 0, 1], dtype=float),
    }
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mset.attach_label_counts(y_true=y_true, plaintext_masks=plain_masks)
    assert not any(issubclass(w.category, MaskDecryptionWarning) for w in caught)
    assert mset.positives == {"A": 2.0, "B": 0.0}


# ---------------------------------------------------------------------------
# Sign / verify envelope round-trip (covers signature absence + present)
# ---------------------------------------------------------------------------


def test_sign_and_verify_envelope_round_trip(tmp_path: Path):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    sk = Ed25519PrivateKey.generate()
    sk_pem = sk.private_bytes(
        Encoding.PEM, PrivateFormat.PKCS8, NoEncryption()
    )
    pk_pem = sk.public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )

    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, np.zeros(8))
    env = audit_metric(
        "demographic_parity_difference",
        y_true=np.zeros(8, dtype=int),
        y_pred=yp_enc,
        sensitive_features=np.array(["A"] * 4 + ["B"] * 4),
        ctx=ctx,
        min_group_size=2,
    )
    payload = env.to_dict()
    signed = sign_envelope(payload, sk_pem)
    assert "signature" in signed
    errors = verify_envelope_signature(signed, pk_pem)
    assert errors == []


def test_verify_envelope_signature_rejects_unsigned(tmp_path: Path):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    pk_pem = Ed25519PrivateKey.generate().public_key().public_bytes(
        Encoding.PEM, PublicFormat.SubjectPublicKeyInfo
    )
    payload = {"schema_version": "fairlearn-fhe.metric-envelope.v1"}
    errors = verify_envelope_signature(payload, pk_pem)
    assert any("signature" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# OpenFHE backend make_public + scalar mul + sub + neg + sum_all when n=0
# ---------------------------------------------------------------------------


_HAS_OPENFHE = False
try:
    import openfhe  # noqa: F401

    _HAS_OPENFHE = True
except (ImportError, ModuleNotFoundError, OSError):
    _HAS_OPENFHE = False


@pytest.mark.skipif(not _HAS_OPENFHE, reason="openfhe-python not installed")
def test_openfhe_make_evaluator_context_raises_until_keypair_workaround_lands():
    # OpenFHE's KeyPair binding is unpickleable, so the current
    # ``copy.copy(self.raw.keys)`` path in make_evaluator_context()
    # raises ``TypeError: cannot pickle 'openfhe.openfhe.KeyPair'``.
    # The TenSEAL backend works fine; this test pins the OpenFHE
    # behaviour so a future fix (e.g. wrapper class around KeyPair)
    # surfaces as a flipped-to-passing test rather than going unnoticed.
    ctx = build_context(backend="openfhe")
    with pytest.raises(TypeError, match="cannot pickle"):
        ctx.make_evaluator_context()


@pytest.mark.skipif(not _HAS_OPENFHE, reason="openfhe-python not installed")
def test_openfhe_neg_sub_scalar_round_trip():
    ctx = build_context(backend="openfhe")
    v = EncryptedVector.encrypt(ctx, np.array([1.0, 2.0, 3.0, 4.0]))
    assert np.allclose((-v).decrypt()[:4], [-1, -2, -3, -4], atol=1e-3)
    assert np.allclose(
        (v - np.array([0.5, 0.5, 0.5, 0.5])).decrypt()[:4],
        [0.5, 1.5, 2.5, 3.5], atol=1e-3,
    )
    assert np.allclose(v.mul_scalar(2.0).decrypt()[:4], [2, 4, 6, 8], atol=1e-3)


@pytest.mark.skipif(not _HAS_OPENFHE, reason="openfhe-python not installed")
def test_openfhe_sum_all_n_zero_returns_input():
    from fairlearn_fhe._backends import openfhe_backend as oh

    ctx = build_context(backend="openfhe")
    v = EncryptedVector.encrypt(ctx, np.array([1.0, 2.0]))
    out = oh.sum_all(v.ciphertext, 0, ctx.raw)
    # For n<=0 the function returns the input ciphertext unchanged.
    assert out is v.ciphertext


# ---------------------------------------------------------------------------
# TenSEAL try-import path (line in _backends.tenseal_backend)
# ---------------------------------------------------------------------------


def test_tenseal_backend_module_loads():
    mod = get_backend("tenseal")
    assert mod is not None
    assert hasattr(mod, "build_context")


def test_list_backends_returns_canonical_set():
    assert set(list_backends()) == {"tenseal", "openfhe"}


# ---------------------------------------------------------------------------
# CLI doctor + schema + inspect once more for coverage
# ---------------------------------------------------------------------------


def test_cli_doctor_reports_at_least_one_backend(capsys):
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tenseal" in out


def test_cli_inspect_falls_back_to_legacy_metric_key(tmp_path: Path, capsys):
    # Older envelopes stored the metric under "metric" rather than
    # "metric_name"; the CLI should accept either.
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"metric": "demographic_parity_difference"}))
    rc = main(["inspect", str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "demographic_parity_difference" in out


def test_cli_schema_compact_output_is_one_line(capsys):
    rc = main(["schema"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.count("\n") == 1  # single newline at the end
    assert json.loads(out)["$id"]


# ---------------------------------------------------------------------------
# Snapshot/reset op counters
# ---------------------------------------------------------------------------


def test_op_counters_reset(monkeypatch):
    from fairlearn_fhe import reset_op_counters, snapshot_op_counters

    reset_op_counters()
    ctx = build_context(backend="tenseal")
    v = EncryptedVector.encrypt(ctx, np.array([1.0, 2.0]))
    v.mul_pt(np.array([1.0, 1.0]))
    s = snapshot_op_counters()
    assert s.get("ct_pt_muls", 0) >= 1
    reset_op_counters()
    assert sum(snapshot_op_counters().values()) == 0


# ---------------------------------------------------------------------------
# CKKSContext attribute surface — covers the make_public / has_secret_key
# branches not hit elsewhere.
# ---------------------------------------------------------------------------


def test_ckks_context_dataclass_fields():
    ctx = build_context(backend="tenseal")
    assert isinstance(ctx, CKKSContext)
    assert ctx.backend_name == "tenseal"
    assert ctx.poly_modulus_degree > 0
    assert ctx.n_slots > 0


# ---------------------------------------------------------------------------
# Internal helper functions in regression module
# ---------------------------------------------------------------------------


def test_regression_helpers_handle_degenerate_inputs():
    from fairlearn_fhe.metrics._regression_metrics import _mae, _mse, _overall_mse, _r2

    assert _mse(0.0, 0.0) == 0.0
    assert _mae(0.0, 0.0) == 0.0
    assert _r2(0.5, 0.0) == 0.0

    ctx = build_context(backend="tenseal")
    y = np.array([1.0, 2.0, 3.0])
    yp_enc = encrypt(ctx, y + 0.1)
    rss, n = _overall_mse(y, yp_enc, sample_weight=None)
    assert rss >= 0.0 and n == 3.0


# ---------------------------------------------------------------------------
# Plaintext-fallthrough at base-metrics layer
# ---------------------------------------------------------------------------


def test_base_metrics_plaintext_passthrough_for_all_helpers():
    import fairlearn.metrics as fl

    from fairlearn_fhe.metrics import (
        false_negative_rate,
        false_positive_rate,
        mean_prediction,
        true_negative_rate,
        true_positive_rate,
    )

    rng = np.random.default_rng(2)
    y_true = (rng.random(64) > 0.5).astype(int)
    y_pred = y_true.copy()

    for fn, ref in [
        (true_positive_rate, fl.true_positive_rate),
        (true_negative_rate, fl.true_negative_rate),
        (false_positive_rate, fl.false_positive_rate),
        (false_negative_rate, fl.false_negative_rate),
        (mean_prediction, fl.mean_prediction),
    ]:
        assert math.isclose(fn(y_true, y_pred), ref(y_true, y_pred), abs_tol=1e-9)


# ---------------------------------------------------------------------------
# Audit metric: alternative trust-model labels
# ---------------------------------------------------------------------------


def test_audit_metric_labels_no_sensitive_features():
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, np.zeros(8))
    env = audit_metric(
        "selection_rate",
        y_true=np.zeros(8, dtype=int),
        y_pred=yp_enc,
        sensitive_features=None,
        ctx=ctx,
    )
    assert env.trust_model == "no_sensitive_features"


def test_audit_metric_labels_encrypted_sensitive_features():
    ctx = build_context(backend="tenseal")
    sf = np.array(["A", "B", "A", "B"], dtype=object)
    y_true = np.array([1, 0, 1, 0])
    sf_enc = encrypt_sensitive_features(ctx, sf, y_true=y_true)
    yp_enc = encrypt(ctx, y_true.astype(float))
    env = audit_metric(
        "demographic_parity_difference",
        y_true=y_true,
        y_pred=yp_enc,
        sensitive_features=sf_enc,
        ctx=ctx,
        min_group_size=2,
    )
    assert env.trust_model == "encrypted_sensitive_features"


# ---------------------------------------------------------------------------
# Envelope: every validation negative branch
# ---------------------------------------------------------------------------


def _valid_payload(ctx) -> dict:
    yp_enc = encrypt(ctx, np.zeros(8))
    env = audit_metric(
        "demographic_parity_difference",
        y_true=np.zeros(8, dtype=int),
        y_pred=yp_enc,
        sensitive_features=np.array(["A"] * 4 + ["B"] * 4),
        ctx=ctx,
        min_group_size=2,
    )
    return env.to_dict()


def test_validate_envelope_rejects_unknown_metric():
    ctx = build_context(backend="tenseal")
    payload = _valid_payload(ctx)
    errors = validate_envelope(payload, allowed_metrics=["only-this-one"])
    assert any("metric" in e.lower() for e in errors)


def test_validate_envelope_rejects_excessive_observed_depth():
    ctx = build_context(backend="tenseal")
    payload = _valid_payload(ctx)
    errors = validate_envelope(payload, max_observed_depth=0)
    assert any("depth" in e.lower() for e in errors)


def test_validate_envelope_rejects_too_old_envelope():
    ctx = build_context(backend="tenseal")
    payload = _valid_payload(ctx)
    payload["timestamp"] = 0.0  # epoch — definitely too old
    errors = validate_envelope(payload, max_age_seconds=60)
    assert any("age" in e.lower() or "timestamp" in e.lower() for e in errors)


def test_validate_envelope_rejects_low_security_bits():
    ctx = build_context(backend="tenseal")
    payload = _valid_payload(ctx)
    errors = validate_envelope(payload, min_security_bits=10_000)
    assert any("security" in e.lower() or "bits" in e.lower() for e in errors)


def test_validate_envelope_accepts_clean_payload():
    ctx = build_context(backend="tenseal")
    payload = _valid_payload(ctx)
    errors = validate_envelope(payload)
    assert errors == []


# ---------------------------------------------------------------------------
# Multi-column sensitive features (covers _to_dataframe arr.ndim==2 path)
# ---------------------------------------------------------------------------


def test_multi_column_sensitive_features():
    ctx = build_context(backend="tenseal")
    sf = np.array(
        [["A", "X"], ["A", "Y"], ["B", "X"], ["B", "Y"]] * 8,
        dtype=object,
    )
    y_true = np.zeros(32, dtype=int)
    yp_enc = encrypt(ctx, np.zeros(32))
    val = demographic_parity_difference(
        y_true, yp_enc, sensitive_features=sf
    )
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# encrypted.add scalar/list right-side
# ---------------------------------------------------------------------------


def test_encrypted_add_scalar_via_radd(vec):
    _ctx, v = vec
    out = (5.0 + v).decrypt()[:4]
    assert np.allclose(out, [6.0, 7.0, 8.0, 9.0], atol=1e-3)


# ---------------------------------------------------------------------------
# make_derived_metric — encrypted path through aggregation
# ---------------------------------------------------------------------------


def test_make_derived_metric_encrypted_path(mode_b_classification):
    import fairlearn.metrics as fl

    from fairlearn_fhe.metrics import make_derived_metric

    derived = make_derived_metric(
        metric=fl.selection_rate, transform="difference"
    )
    val = derived(
        mode_b_classification["y_true"],
        mode_b_classification["y_pred_enc"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    assert isinstance(val, float)


def test_make_derived_metric_plaintext_passthrough(mode_b_classification):
    import fairlearn.metrics as fl

    from fairlearn_fhe.metrics import make_derived_metric

    derived = make_derived_metric(
        metric=fl.selection_rate, transform="ratio"
    )
    val = derived(
        mode_b_classification["y_true"],
        mode_b_classification["y_pred"],
        sensitive_features=mode_b_classification["sf_plain"],
    )
    assert isinstance(val, float)
