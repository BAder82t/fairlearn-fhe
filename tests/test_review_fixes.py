"""Regression tests for the v0.1.0 review-feedback fixes.

Each test exercises one of the issues raised in the pre-release review:

* clamp_lower in ``_safe_div`` so CKKS-noise negatives don't propagate.
* thread-safe ``OP_COUNTERS`` updates.
* ``mul_scalar`` consumes a multiplicative level.
* ``MetricFrame`` is a callable factory (returns the genuine fairlearn
  object on the plaintext path).
* ``estimate_security_bits`` is derived rather than hardcoded.
* ``validate_envelope`` enforces ``max_age_seconds`` and
  ``min_security_bits``.
* ``audit_metric`` warns on small groups.
* ``attach_label_counts`` warns when it has to decrypt masks.
* ``cli`` rejects oversized envelopes.
* sign + tamper-detect for the ``value`` field.
"""

from __future__ import annotations

import threading
import time
import warnings

import fairlearn.metrics as _fl
import numpy as np
import pytest

from fairlearn_fhe import (
    MaskDecryptionWarning,
    SmallGroupWarning,
    audit_metric,
    encrypt,
    encrypt_sensitive_features,
    estimate_security_bits,
    sign_envelope,
    validate_envelope,
    verify_envelope_signature,
)
from fairlearn_fhe._circuits import _safe_div
from fairlearn_fhe.encrypted import OP_COUNTERS, EncryptedVector, reset_op_counters
from fairlearn_fhe.metrics import MetricFrame

# ---------------------------------------------------------------------------
# _safe_div clamps to [0, 1] under CKKS noise
# ---------------------------------------------------------------------------


def test_safe_div_clamps_negative_numerator_to_zero():
    # Mimic CKKS noise pushing a true-zero numerator slightly negative.
    assert _safe_div(-1e-7, 1.0) == 0.0


def test_safe_div_clamps_overshoot_to_one():
    assert _safe_div(1.0 + 1e-7, 1.0) == 1.0


def test_safe_div_passes_through_normal_values():
    assert _safe_div(0.3, 0.6) == pytest.approx(0.5)


def test_safe_div_zero_denominator_returns_zero():
    assert _safe_div(0.0, 0.0) == 0.0
    assert _safe_div(1.0, 0.0) == 0.0


def test_safe_div_disable_clamping():
    assert _safe_div(-0.5, 1.0, clip_lower=None, clip_upper=None) == -0.5
    assert _safe_div(2.0, 1.0, clip_lower=None, clip_upper=None) == 2.0


# ---------------------------------------------------------------------------
# OP_COUNTERS is thread-safe
# ---------------------------------------------------------------------------


def test_op_counters_thread_safe(ctx):
    """100 threads × 100 ops should produce exactly 10_000 increments."""
    reset_op_counters()
    vec = encrypt(ctx, np.zeros(8))
    n_threads = 20
    per_thread = 50

    barrier = threading.Barrier(n_threads)

    def worker():
        barrier.wait()
        for _ in range(per_thread):
            _ = vec + vec  # __add__ → _inc("additions")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert OP_COUNTERS["additions"] == n_threads * per_thread


# ---------------------------------------------------------------------------
# mul_scalar consumes a multiplicative level
# ---------------------------------------------------------------------------


def test_mul_scalar_increments_depth(ctx):
    vec = encrypt(ctx, np.ones(4))
    assert vec.depth == 0
    scaled = vec.mul_scalar(2.0)
    assert scaled.depth == 1
    twice = scaled.mul_scalar(0.5)
    assert twice.depth == 2


# ---------------------------------------------------------------------------
# MetricFrame factory returns the real fairlearn type on the
# plaintext path so isinstance checks work.
# ---------------------------------------------------------------------------


def test_metric_frame_plaintext_returns_real_fairlearn(small_dataset):
    y_true, y_pred, sf = small_dataset
    mf = MetricFrame(
        metrics=_fl.selection_rate,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sf,
    )
    assert isinstance(mf, _fl.MetricFrame)


def test_metric_frame_encrypted_returns_encrypted_frame(small_dataset, ctx, encrypted_pred):
    y_true, _y_pred, sf = small_dataset
    from fairlearn_fhe.metrics import EncryptedMetricFrame
    mf = MetricFrame(
        metrics=_fl.selection_rate,
        y_true=y_true,
        y_pred=encrypted_pred,
        sensitive_features=sf,
    )
    assert isinstance(mf, EncryptedMetricFrame)


# ---------------------------------------------------------------------------
# security_bits is derived from parameters
# ---------------------------------------------------------------------------


def test_estimate_security_bits_known_safe_pair():
    # N=8192 with a 200-bit modulus chain is within the 128-bit table row.
    assert estimate_security_bits(8192, 200) == 128


def test_estimate_security_bits_unsafe_pair_returns_zero():
    # N=8192 with a wildly oversized chain falls below 128-bit security.
    assert estimate_security_bits(8192, 1000) == 0


def test_estimate_security_bits_unknown_ring_returns_zero():
    assert estimate_security_bits(123, 200) == 0


def test_audit_metric_records_derived_security_bits(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    # The conftest ctx is N=8192, coeff_mod_bit_sizes=[60,40,40,60] (200 bits)
    # → 128-bit row of the HE-standard table.
    assert env.parameter_set.security_bits == 128


# ---------------------------------------------------------------------------
# validate_envelope checks staleness + min_security_bits
# ---------------------------------------------------------------------------


def test_validate_envelope_rejects_stale(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    payload = env.to_dict()
    payload["timestamp"] = time.time() - 7200  # 2 hours old
    errors = validate_envelope(payload, max_age_seconds=3600)
    assert "envelope timestamp is older than max_age_seconds" in errors


def test_validate_envelope_rejects_future(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    payload = env.to_dict()
    payload["timestamp"] = time.time() + 7200
    errors = validate_envelope(payload, max_age_seconds=3600)
    assert "envelope timestamp is in the future" in errors


def test_validate_envelope_min_security_bits(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    payload = env.to_dict()
    errors = validate_envelope(payload, min_security_bits=192)
    assert any("security_bits" in e for e in errors)


# ---------------------------------------------------------------------------
# small group warning
# ---------------------------------------------------------------------------


def test_audit_metric_warns_on_small_group(ctx):
    rng = np.random.default_rng(42)
    n = 50
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    # Group "C" has only 3 samples.
    sensitive = np.array(["A"] * 30 + ["B"] * 17 + ["C"] * 3)
    with pytest.warns(SmallGroupWarning):
        audit_metric(
            "demographic_parity_difference",
            y_true, y_pred,
            sensitive_features=sensitive,
            ctx=ctx,
        )


def test_audit_metric_no_warning_when_disabled(ctx):
    rng = np.random.default_rng(42)
    n = 50
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sensitive = np.array(["A"] * 30 + ["B"] * 17 + ["C"] * 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error", SmallGroupWarning)
        audit_metric(
            "demographic_parity_difference",
            y_true, y_pred,
            sensitive_features=sensitive,
            ctx=ctx,
            min_group_size=0,
        )


# ---------------------------------------------------------------------------
# attach_label_counts warns when it decrypts
# ---------------------------------------------------------------------------


def test_attach_label_counts_warns_on_decrypt(ctx, small_dataset):
    y_true, _, sf = small_dataset
    mask_set = encrypt_sensitive_features(ctx, sf)  # no y_true → counts not stamped
    with pytest.warns(MaskDecryptionWarning):
        mask_set.attach_label_counts(y_true)


def test_attach_label_counts_silent_with_plaintext_masks(ctx, small_dataset):
    y_true, _, sf = small_dataset
    from fairlearn_fhe._groups import group_masks
    labels, plaintext_masks = group_masks(sf)
    mask_set = encrypt_sensitive_features(ctx, sf)
    with warnings.catch_warnings():
        warnings.simplefilter("error", MaskDecryptionWarning)
        mask_set.attach_label_counts(y_true, plaintext_masks=plaintext_masks)
    assert mask_set.positives is not None
    assert set(mask_set.positives) == set(labels)


# ---------------------------------------------------------------------------
# CLI rejects oversized envelopes
# ---------------------------------------------------------------------------


def test_cli_rejects_oversized_envelope(tmp_path, capsys):
    from fairlearn_fhe.cli import main
    big = tmp_path / "huge.json"
    big.write_text("{" + ", ".join(f'"k{i}":{i}' for i in range(200_000)) + "}")
    rc = main([str(big)])
    captured = capsys.readouterr().out
    assert rc == 1
    assert "exceeds" in captured


# ---------------------------------------------------------------------------
# Test gap — tampered ``value`` is detected by signature verification
# ---------------------------------------------------------------------------


def test_signed_envelope_value_tamper_detected(small_dataset, ctx):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    sk = Ed25519PrivateKey.generate()
    pk = sk.public_key()
    private_pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = pk.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signed = sign_envelope(env, private_pem)
    assert verify_envelope_signature(signed, public_pem) == []
    signed["value"] = 0.0  # the metric scalar tamper
    assert verify_envelope_signature(signed, public_pem) == ["signature verification failed"]


# ---------------------------------------------------------------------------
# Test gap — encrypted-mask + sample_weight produces correct rates
# ---------------------------------------------------------------------------


def test_encrypted_mask_with_sample_weight_matches_plaintext(ctx, tol):
    rng = np.random.default_rng(7)
    n = 120
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sensitive = rng.choice(["A", "B"], size=n)
    sw = rng.uniform(0.5, 1.5, size=n)

    plaintext = _fl.demographic_parity_difference(
        y_true, y_pred,
        sensitive_features=sensitive, sample_weight=sw,
    )

    enc_pred = encrypt(ctx, y_pred)
    mask_set = encrypt_sensitive_features(
        ctx, sensitive, y_true=y_true, sample_weight=sw,
    )
    from fairlearn_fhe.metrics import demographic_parity_difference
    encrypted = demographic_parity_difference(
        y_true, enc_pred,
        sensitive_features=mask_set, sample_weight=sw,
    )
    assert encrypted == pytest.approx(plaintext, abs=tol)


# ---------------------------------------------------------------------------
# Test gap — depth budget exhaustion is observable via OP_COUNTERS
# ---------------------------------------------------------------------------


def test_depth_accounting_after_repeated_mul_pt(ctx):
    # The conftest ctx has only 4 coefficient moduli (depth budget ≈2);
    # one mul_pt + one mul_scalar lets us assert the counter without
    # exhausting the modulus chain.
    vec = encrypt(ctx, np.ones(8))
    once = vec.mul_pt(np.full(8, 0.5))
    assert once.depth == 1
    twice = once.mul_scalar(2.0)
    assert twice.depth == 2


# ---------------------------------------------------------------------------
# Test gap — make_evaluator_context drops decrypt capability
# ---------------------------------------------------------------------------


def test_evaluator_context_cannot_decrypt(ctx):
    eval_ctx = ctx.make_evaluator_context()
    assert eval_ctx.has_secret_key is False
    enc = EncryptedVector.encrypt(eval_ctx, [1.0, 2.0, 3.0])
    with pytest.raises(RuntimeError, match="no secret key"):
        enc.decrypt()
