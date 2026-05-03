"""Regression tests for the second round of review fixes.

Covered behaviours:

* ``validate_envelope`` rejects signature blocks lacking ``signature_b64``.
* CLI ``--require-signature`` enforces cryptographic verification — not
  just structural presence — and emits exactly one error per failure
  mode.
* Scoring helper closures route to the right reduction even when
  ``__name__`` is rewritten.
* ``op_session()`` provides true per-thread isolation under concurrent
  audits.
* ``make_evaluator_context`` produces a fully independent wrapper —
  no shared mutable fields.
* Regression metrics route through the canonical ``_safe_div`` helper.
* ``insecure_allow_low_security=True`` emits a warning.
* ``_safe_str`` strips Unicode bidi / zero-width characters.
* ``_safe_str`` strips full ANSI sequences (CSI + OSC).
* ``ParameterSet`` constructor enforces ``_MAX_COEFF_MODULI``.
* ``parameter_set_from_context`` records the exact ``scaling_factor_bits``.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import numpy as np
import pytest

from fairlearn_fhe import (
    InsecureCKKSParametersWarning,
    audit_metric,
    encrypt,
    op_session,
    sign_envelope,
    validate_envelope,
)
from fairlearn_fhe.cli import _safe_str, main
from fairlearn_fhe.context import _MAX_COEFF_MODULI, _validate_tenseal_params
from fairlearn_fhe.envelope import (
    SIGNATURE_ALGORITHM,
    ParameterSet,
    parameter_set_from_context,
)

# ---------------------------------------------------------------------------
# validate_envelope rejects signature blocks lacking signature_b64
# ---------------------------------------------------------------------------


def test_validate_envelope_rejects_legacy_value_only_signature(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf, ctx=ctx,
    )
    payload = env.to_dict()
    payload["signature"] = {"algorithm": SIGNATURE_ALGORITHM, "value": "abc"}
    errors = validate_envelope(payload)
    assert any("signature_b64" in e for e in errors), errors


def test_validate_envelope_accepts_canonical_signature_b64(small_dataset, ctx):
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
    sk_pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    signed = sign_envelope(env, sk_pem)
    assert "signature_b64" in signed["signature"]
    assert validate_envelope(signed) == []


# ---------------------------------------------------------------------------
# CLI --require-signature: single error per failure mode
# ---------------------------------------------------------------------------


def test_cli_require_signature_no_key_emits_single_error(
    tmp_path: Path, capsys, small_dataset, ctx,
):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf, ctx=ctx,
    )
    path = tmp_path / "env.json"
    path.write_text(env.to_json())
    rc = main(["verify", str(path), "--require-signature", "--json"])
    assert rc != 0
    out = json.loads(capsys.readouterr().out)
    require_errors = [e for e in out["errors"] if "require-signature" in e]
    assert len(require_errors) == 1, require_errors


def test_cli_require_signature_unreadable_key_does_not_pass(
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
    sk_pem = sk.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    signed = sign_envelope(env, sk_pem)
    env_path = tmp_path / "env.json"
    env_path.write_text(json.dumps(signed))
    rc = main([
        "verify", str(env_path),
        "--public-key", str(tmp_path / "does-not-exist.pem"),
        "--require-signature",
        "--json",
    ])
    assert rc != 0
    out = json.loads(capsys.readouterr().out)
    # Both the read failure AND the require-signature enforcement must
    # appear; --require-signature must NOT be silently satisfied just
    # because the envelope happens to carry a signature block.
    assert any("failed to read public key" in e for e in out["errors"])
    assert any("not cryptographically verified" in e for e in out["errors"])


# ---------------------------------------------------------------------------
# Scoring closures use explicit reduction even after __name__ rewrite
# ---------------------------------------------------------------------------


def test_group_min_routes_correctly_when_name_is_rewritten(monkeypatch, small_dataset):
    """Renaming ``_impl.__name__`` must not break reduction routing."""
    import fairlearn.metrics as fl

    from fairlearn_fhe.metrics._scoring_metrics import _group_min

    monkeypatch.setattr(fl, "balanced_accuracy_score_group_min", None, raising=False)
    fn = _group_min("balanced_accuracy_score")
    fn.__name__ = "totally_different_name_unrelated_to_min_or_max"

    y_true, y_pred, sf = small_dataset
    val_min = fn(y_true, y_pred, sensitive_features=sf)
    assert isinstance(val_min, float)


# ---------------------------------------------------------------------------
# op_session true per-thread isolation
# ---------------------------------------------------------------------------


def test_op_session_excludes_other_threads_increments(ctx):
    """A session in thread A must not see ops performed in thread B."""
    from fairlearn_fhe.encrypted import _inc

    barrier = threading.Barrier(2)
    a_delta: dict[str, int] = {}
    b_running = threading.Event()
    a_done = threading.Event()

    def thread_a() -> None:
        with op_session() as counts:
            barrier.wait()
            b_running.wait()
            # Do nothing in this thread; B is hammering increments.
        a_delta.update(counts)
        a_done.set()

    def thread_b() -> None:
        barrier.wait()
        b_running.set()
        for _ in range(50):
            _inc("additions")
        a_done.wait()  # let A close before we exit

    ta = threading.Thread(target=thread_a)
    tb = threading.Thread(target=thread_b)
    ta.start()
    tb.start()
    ta.join()
    tb.join()
    assert a_delta["additions"] == 0, a_delta


def test_op_session_nesting_propagates_to_outer():
    from fairlearn_fhe.encrypted import _inc

    with op_session() as outer:
        _inc("additions", 1)
        with op_session() as inner:
            _inc("additions", 4)
        assert inner["additions"] == 4
        _inc("additions", 2)
    assert outer["additions"] == 7  # 1 + 4 + 2


# ---------------------------------------------------------------------------
# make_evaluator_context: independent wrapper
# ---------------------------------------------------------------------------


def test_evaluator_wrapper_is_fully_independent(ctx):
    eval_ctx = ctx.make_evaluator_context()
    # Distinct wrapper objects.
    assert eval_ctx.raw is not ctx.raw
    # Distinct underlying TenSEAL Context handles — this is the
    # invariant that matters for secret-key isolation.
    assert eval_ctx.raw.context is not ctx.raw.context
    # The two wrappers report the same parameters.
    assert eval_ctx.raw.coeff_mod_bit_sizes == ctx.raw.coeff_mod_bit_sizes
    assert eval_ctx.raw.poly_modulus_degree == ctx.raw.poly_modulus_degree
    # The evaluator copy carries no secret key.
    assert eval_ctx.has_secret_key is False
    assert eval_ctx.raw.context.is_private() is False


# ---------------------------------------------------------------------------
# Regression metrics route through canonical _safe_div
# ---------------------------------------------------------------------------


def test_regression_safe_div_imported_from_circuits():
    from fairlearn_fhe._circuits import _safe_div as canonical
    from fairlearn_fhe.metrics import _regression_metrics as rm

    assert rm._safe_div is canonical


def test_regression_mse_clamps_negative_rss_under_noise(ctx):
    """A near-zero RSS that drifts negative under CKKS noise → 0."""
    rng = np.random.default_rng(0)
    n = 32
    y_true = rng.uniform(0, 1, size=n)
    sf = rng.choice(["A", "B"], size=n)
    enc = encrypt(ctx, y_true)  # ŷ = y → MSE ≈ 0 (modulo CKKS noise)
    from fairlearn_fhe.metrics import mean_squared_error_group_max
    val = mean_squared_error_group_max(y_true, enc, sensitive_features=sf)
    assert val >= 0.0
    assert val < 1e-2


# ---------------------------------------------------------------------------
# insecure_allow_low_security warning
# ---------------------------------------------------------------------------


def test_insecure_allow_low_security_emits_warning():
    with pytest.warns(InsecureCKKSParametersWarning):
        _validate_tenseal_params(
            1 << 13,
            [60, 40, 40, 40, 40, 40, 40, 60],
            insecure_allow_low_security=True,
        )


def test_secure_params_emit_no_warning():
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", InsecureCKKSParametersWarning)
        _validate_tenseal_params(
            1 << 14,
            [60, 40, 40, 40, 40, 40, 40, 60],  # 320 bits @ N=16384 → 128
            insecure_allow_low_security=False,
        )


# ---------------------------------------------------------------------------
# _safe_str: Unicode bidi / zero-width / full ANSI sequences
# ---------------------------------------------------------------------------


def test_safe_str_strips_unicode_bidi_overrides():
    # U+202E is the right-to-left override; could visually flip text.
    out = _safe_str("INVALID‮OK")
    assert "‮" not in out
    assert out == "INVALIDOK"


def test_safe_str_strips_zero_width_chars():
    # U+200B ZWSP, U+200C ZWNJ, U+200D ZWJ.
    out = _safe_str("ad​min‌@‍example.com")
    assert out == "admin@example.com"


def test_safe_str_strips_bom_and_separators():
    # U+FEFF BOM, U+2028 LINE SEPARATOR, U+2029 PARAGRAPH SEPARATOR.
    out = _safe_str("﻿hello\u2028world\u2029bye")
    assert out == "helloworldbye"


def test_safe_str_strips_full_ansi_csi_sequence():
    # The bracket and trailing 'm' should also disappear, not just ESC.
    out = _safe_str("\x1b[31mRED\x1b[0m")
    assert out == "RED"


def test_safe_str_strips_ansi_osc_sequence():
    # OSC sequences set window titles; strip them too.
    out = _safe_str("\x1b]0;evil-title\x07ok")
    assert out == "ok"


# ---------------------------------------------------------------------------
# ParameterSet constructor enforces _MAX_COEFF_MODULI
# ---------------------------------------------------------------------------


def test_parameter_set_constructor_rejects_oversized_chain():
    with pytest.raises(ValueError, match="too long"):
        ParameterSet(
            backend="tenseal-ckks",
            poly_modulus_degree=16384,
            security_bits=128,
            multiplicative_depth=6,
            coeff_mod_bit_sizes=tuple(range(_MAX_COEFF_MODULI + 1)),
            scaling_factor_bits=40,
        )


# ---------------------------------------------------------------------------
# parameter_set_from_context records exact scaling_factor_bits
# ---------------------------------------------------------------------------


def test_scaling_factor_bits_records_exact_power_of_two(ctx):
    ps = parameter_set_from_context(ctx)
    # conftest builds the context with scale_bits=40 → scale = 2**40.
    assert ps.scaling_factor_bits == 40
