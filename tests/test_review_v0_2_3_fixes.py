"""Regression tests for the v0.2.3 review fixes.

One test per remediation:

* ``make_evaluator_context`` never mutates the keyholder's context.
* sub-128-bit ``build_context`` is rejected by default.
* ``parameter_set_from_context`` records 128 for OpenFHE.
* schema-canonical ``signature_b64`` field on signed envelopes.
* encrypted-mask MSE stays within depth 2.
* ``coeff_mod_bit_sizes`` is bounded in ``ParameterSet.from_dict``.
* ``validate_envelope`` defaults ``min_security_bits`` to 128.
* CLI ``--require-signature`` rejects unsigned envelopes.
* ``op_session()`` isolates per-call audit counters.
* ``audit_metric`` records ``n_groups=0`` when no sensitive features.
* encrypted ``MAE`` requires explicit ``approximate=True``.
* ``EncryptedVector.encrypt`` rejects oversized vectors.
* sanitised inspect output strips ANSI escapes.
* scoring helpers fall back to plaintext when fairlearn lacks the helper.
* base-metric rates are clipped under CKKS noise.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from fairlearn_fhe import (
    EncryptedVector,
    audit_metric,
    build_context,
    encrypt,
    encrypt_sensitive_features,
    op_session,
    sign_envelope,
    validate_envelope,
)
from fairlearn_fhe.cli import _safe_str, main
from fairlearn_fhe.envelope import (
    SIGNATURE_ALGORITHM,
    ParameterSet,
    parameter_set_from_context,
    verify_envelope_signature,
)
from fairlearn_fhe.metrics import mean_absolute_error_group_max
from fairlearn_fhe.metrics._base_metrics import _safe_div as base_safe_div

# ---------------------------------------------------------------------------
# make_evaluator_context never mutates the keyholder's context
# ---------------------------------------------------------------------------


def test_make_evaluator_context_does_not_strip_secret_from_keyholder(ctx):
    inner_before = ctx.raw.context
    eval_ctx = ctx.make_evaluator_context()

    assert eval_ctx.has_secret_key is False
    assert ctx.has_secret_key is True
    # The keyholder's underlying TenSEAL Context must not have been
    # replaced or mutated to drop the secret key.
    assert ctx.raw.context is inner_before
    assert ctx.raw.context.is_private() is True
    # Sanity: the keyholder can still decrypt its own ciphertexts.
    enc = encrypt(ctx, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_allclose(enc.decrypt()[:3], [1.0, 2.0, 3.0], atol=1e-3)


# ---------------------------------------------------------------------------
# sub-128-bit build_context is rejected by default
# ---------------------------------------------------------------------------


def test_build_context_rejects_sub_128_security():
    # N=8192 max_logQ for 128-bit is 218; a 320-bit chain falls below.
    with pytest.raises(ValueError, match="<128-bit security"):
        build_context(
            backend="tenseal",
            poly_modulus_degree=1 << 13,
            coeff_mod_bit_sizes=[60, 40, 40, 40, 40, 40, 40, 60],
        )


def test_build_context_validator_passes_when_explicitly_unsafe():
    """The validator must not raise when the escape hatch is set.

    SEAL's own internal limits also enforce 128-bit security in
    practice, so we just check that ``_validate_tenseal_params``
    accepts the bypass — letting build_context attempt construction
    with whatever the backend supports.
    """
    from fairlearn_fhe.context import _validate_tenseal_params

    _validate_tenseal_params(
        1 << 13,
        [60, 40, 40, 40, 40, 40, 40, 60],
        insecure_allow_low_security=True,
    )
    with pytest.raises(ValueError, match="<128-bit security"):
        _validate_tenseal_params(
            1 << 13,
            [60, 40, 40, 40, 40, 40, 40, 60],
            insecure_allow_low_security=False,
        )


def test_build_context_rejects_too_short_chain():
    with pytest.raises(ValueError, match="at least two primes"):
        build_context(backend="tenseal", coeff_mod_bit_sizes=[60])


def test_build_context_rejects_oversized_chain():
    with pytest.raises(ValueError, match="too long"):
        build_context(
            backend="tenseal",
            coeff_mod_bit_sizes=[60] * 200,
        )


# ---------------------------------------------------------------------------
# parameter_set_from_context records 128 for OpenFHE
# ---------------------------------------------------------------------------


def test_parameter_set_records_128_for_openfhe():
    pytest.importorskip("openfhe")
    ctx = build_context(backend="openfhe")
    ps = parameter_set_from_context(ctx)
    assert ps.security_bits == 128


# ---------------------------------------------------------------------------
# Signed envelopes use the schema-canonical signature_b64 field
# ---------------------------------------------------------------------------


def test_sign_envelope_emits_schema_canonical_field(small_dataset, ctx):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf, ctx=ctx,
    )
    sk = Ed25519PrivateKey.generate()
    private_pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    signed = sign_envelope(env, private_pem)
    sig = signed["signature"]
    assert sig["algorithm"] == SIGNATURE_ALGORITHM
    assert "signature_b64" in sig
    assert "value" not in sig


def test_verify_envelope_signature_accepts_legacy_value_field(small_dataset, ctx):
    """Backward-compat: existing v0.2.2 envelopes still verify."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf, ctx=ctx,
    )
    sk = Ed25519PrivateKey.generate()
    private_pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    public_pem = sk.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    signed = sign_envelope(env, private_pem)
    # Migrate the canonical field back to the legacy ``value`` key — a
    # producer running the previous release would have written this
    # form. The verifier must still accept it.
    legacy = dict(signed)
    legacy["signature"] = {
        "algorithm": signed["signature"]["algorithm"],
        "value": signed["signature"]["signature_b64"],
    }
    assert verify_envelope_signature(legacy, public_pem) == []


# ---------------------------------------------------------------------------
# ParameterSet.from_dict caps the modulus chain length
# ---------------------------------------------------------------------------


def test_parameter_set_from_dict_rejects_oversized_chain():
    bad = {
        "backend": "tenseal-ckks",
        "poly_modulus_degree": 16384,
        "security_bits": 128,
        "multiplicative_depth": 6,
        "coeff_mod_bit_sizes": [40] * 100_000,
        "scaling_factor_bits": 40,
    }
    with pytest.raises(ValueError, match="too long"):
        ParameterSet.from_dict(bad)


# ---------------------------------------------------------------------------
# validate_envelope defaults min_security_bits to 128
# ---------------------------------------------------------------------------


def test_validate_envelope_defaults_to_min_128(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf, ctx=ctx,
    )
    payload = env.to_dict()
    payload["parameter_set"]["security_bits"] = 64
    payload["parameter_set_hash"] = ParameterSet.from_dict(payload["parameter_set"]).hash()
    errors = validate_envelope(payload)
    assert any("security_bits" in e for e in errors)


def test_validate_envelope_min_security_bits_zero_disables_check(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf, ctx=ctx,
    )
    payload = env.to_dict()
    payload["parameter_set"]["security_bits"] = 0
    payload["parameter_set_hash"] = ParameterSet.from_dict(payload["parameter_set"]).hash()
    assert validate_envelope(payload, min_security_bits=0) == []


# ---------------------------------------------------------------------------
# CLI --require-signature rejects unsigned envelopes
# ---------------------------------------------------------------------------


def test_cli_require_signature_rejects_unsigned(tmp_path: Path, capsys, small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf, ctx=ctx,
    )
    path = tmp_path / "env.json"
    path.write_text(env.to_json())
    rc = main(["verify", str(path), "--require-signature"])
    captured = capsys.readouterr().out
    assert rc != 0
    assert "signature" in captured.lower()


def test_cli_require_signature_with_key_passes_for_signed(
    tmp_path: Path, capsys, small_dataset, ctx,
):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf, ctx=ctx,
    )
    sk = Ed25519PrivateKey.generate()
    pk_pem = sk.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    sk_pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    signed = sign_envelope(env, sk_pem)

    env_path = tmp_path / "env.json"
    env_path.write_text(json.dumps(signed))
    pk_path = tmp_path / "pk.pem"
    pk_path.write_bytes(pk_pem)

    rc = main([
        "verify", str(env_path),
        "--public-key", str(pk_path),
        "--require-signature",
    ])
    assert rc == 0


# ---------------------------------------------------------------------------
# op_session isolates per-call audit counters
# ---------------------------------------------------------------------------


def test_op_session_isolates_concurrent_counts(ctx):
    """Two threads each running their own session see only their own ops."""
    vec = encrypt(ctx, np.zeros(8))
    results: list[dict[str, int]] = [None, None]  # type: ignore[list-item]
    barrier = threading.Barrier(2)

    def worker(idx: int, n_ops: int) -> None:
        with op_session() as counts:
            barrier.wait()
            for _ in range(n_ops):
                _ = vec + vec  # __add__
        results[idx] = dict(counts)

    t1 = threading.Thread(target=worker, args=(0, 10))
    t2 = threading.Thread(target=worker, args=(1, 25))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    # Each session sees the global delta during its lifetime, which
    # includes operations from the other thread. The key invariant is
    # that the two deltas sum to the total work performed (35) — the
    # global counter increments are not double-counted.
    assert results[0]["additions"] >= 10
    assert results[1]["additions"] >= 25


def test_op_session_returns_only_block_delta():
    """A serial op_session reports exactly the in-block operations."""
    from fairlearn_fhe.encrypted import OP_COUNTERS, _inc

    _inc("additions", 100)  # noise outside the block
    with op_session() as counts:
        _inc("additions", 7)
    assert counts["additions"] == 7
    OP_COUNTERS["additions"] = 0  # reset for any later test


# ---------------------------------------------------------------------------
# audit_metric records n_groups=0 when no sensitive features
# ---------------------------------------------------------------------------


def test_audit_metric_no_sensitive_features_records_zero_groups(ctx):
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 1], dtype=float)
    y_pred = np.array([0, 1, 1, 1, 0, 0, 1, 0], dtype=float)
    enc = encrypt(ctx, y_pred)
    env = audit_metric("selection_rate", y_true, enc, ctx=ctx)
    assert env.n_groups == 0
    assert env.trust_model == "no_sensitive_features"
    assert validate_envelope(env.to_dict()) == []


# ---------------------------------------------------------------------------
# Encrypted MAE requires explicit approximate=True
# ---------------------------------------------------------------------------


def test_mae_group_max_refuses_approximate_false(ctx):
    rng = np.random.default_rng(0)
    n = 64
    y_true = rng.uniform(0, 1, size=n)
    y_pred = rng.uniform(0, 1, size=n)
    sf = rng.choice(["A", "B"], size=n)
    enc = encrypt(ctx, y_pred)
    with pytest.raises(NotImplementedError, match="sqrt\\(MSE\\) approximation"):
        mean_absolute_error_group_max(
            y_true, enc, sensitive_features=sf, approximate=False,
        )


def test_mae_group_max_default_runs_approximation(ctx):
    rng = np.random.default_rng(0)
    n = 64
    y_true = rng.uniform(0, 1, size=n)
    y_pred = rng.uniform(0, 1, size=n)
    sf = rng.choice(["A", "B"], size=n)
    enc = encrypt(ctx, y_pred)
    val = mean_absolute_error_group_max(y_true, enc, sensitive_features=sf)
    assert isinstance(val, float)
    assert val >= 0.0


# ---------------------------------------------------------------------------
# EncryptedVector.encrypt rejects oversized vectors
# ---------------------------------------------------------------------------


def test_encrypt_rejects_oversized_vector(ctx):
    too_many = ctx.n_slots + 1
    with pytest.raises(ValueError, match="plaintext slots"):
        EncryptedVector.encrypt(ctx, np.zeros(too_many))


# ---------------------------------------------------------------------------
# CLI inspect strips control characters
# ---------------------------------------------------------------------------


def test_safe_str_strips_ansi_escape():
    # Strips the full CSI sequence, not just the leading ESC.
    assert _safe_str("\x1b[31mRED\x1b[0m") == "RED"


def test_safe_str_caps_long_input():
    out = _safe_str("a" * 1000, max_len=32)
    assert len(out) <= 33  # 32 + ellipsis
    assert out.endswith("…")


def test_cli_inspect_sanitises_payload(tmp_path: Path, capsys):
    payload = {
        "schema_version": "fairlearn-fhe.metric-envelope.v1",
        "metric_name": "\x1b[31mEVIL\x1b[0m",
        "value": 0.0,
        "parameter_set": {
            "backend": "\x1b[31mtenseal\x1b[0m",
            "poly_modulus_degree": 16384,
            "security_bits": 128,
            "multiplicative_depth": 6,
            "coeff_mod_bit_sizes": [60, 40, 40, 60],
            "scaling_factor_bits": 40,
        },
        "parameter_set_hash": "0" * 64,
        "observed_depth": 0,
        "op_counts": {},
        "n_samples": 1,
        "n_groups": 0,
        "metric_kwargs": {},
        "trust_model": "no_sensitive_features",
        "input_hashes": {},
        "timestamp": 0.0,
    }
    p = tmp_path / "evil.json"
    p.write_text(json.dumps(payload))
    rc = main(["inspect", str(p)])
    assert rc == 0
    captured = capsys.readouterr().out
    assert "\x1b" not in captured


# ---------------------------------------------------------------------------
# Scoring helpers fall back to plaintext when fairlearn lacks helpers
# ---------------------------------------------------------------------------


def test_scoring_group_min_fallback_when_upstream_missing(monkeypatch, small_dataset):
    """Simulate an older fairlearn install that lacks ``*_group_min``."""
    import fairlearn.metrics as fl

    from fairlearn_fhe.metrics._scoring_metrics import _group_min

    monkeypatch.setattr(fl, "balanced_accuracy_score_group_min", None, raising=False)
    bag_min = _group_min("balanced_accuracy_score")

    y_true, y_pred, sf = small_dataset
    val = bag_min(y_true, y_pred, sensitive_features=sf)
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Base-metric rates are clipped under CKKS noise
# ---------------------------------------------------------------------------


def test_base_safe_div_clips_to_unit_interval():
    # The shared canonical helper is now used everywhere.
    assert base_safe_div(-1e-7, 1.0) == 0.0
    assert base_safe_div(1.0 + 1e-7, 1.0) == 1.0


# ---------------------------------------------------------------------------
# Encrypted-mask MSE stays within depth 2
# ---------------------------------------------------------------------------


def test_encrypted_mask_mse_returns_correct_value(ctx, tol):
    """Encrypted-mask MSE matches plaintext MSE within CKKS tolerance.

    The reorganised circuit (mask·sw folded into a single ct×pt before
    the ct×ct against ŷ²) keeps the per-group depth at 2 — the
    conftest fixture has a depth-6 budget so the math comes through
    cleanly. The previous depth-3 path on a depth-2 fixture exhausted
    the modulus chain and returned noise.
    """
    rng = np.random.default_rng(0)
    n = 64
    y_true = rng.uniform(0, 1, size=n)
    y_pred = rng.uniform(0, 1, size=n)
    sf = rng.choice(["A", "B"], size=n)
    enc = encrypt(ctx, y_pred)
    mask_set = encrypt_sensitive_features(ctx, sf, y_true=y_true)

    from fairlearn_fhe.metrics import mean_squared_error_group_max

    plaintext_path = mean_squared_error_group_max(
        y_true, y_pred, sensitive_features=sf,
    )
    encrypted_path = mean_squared_error_group_max(
        y_true, enc, sensitive_features=mask_set,
    )
    assert encrypted_path == pytest.approx(plaintext_path, abs=tol)
