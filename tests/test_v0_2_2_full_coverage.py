# Copyright 2026 Vaultbytes (Bader Alissaei)
# SPDX-License-Identifier: Apache-2.0

"""Final 6% of coverage to reach 100%.

Each block targets a specific line range from the v0.2.1 coverage gap.
Tests are deliberately mechanical — defensive branches, plaintext
fallbacks, and validation edge cases that v0.2.1's behavioural tests
did not exercise.
"""

from __future__ import annotations

import json
from base64 import b64encode
from pathlib import Path

import fairlearn.metrics as fl
import numpy as np
import pytest

from fairlearn_fhe import (
    EncryptedVector,
    audit_metric,
    build_context,
    encrypt,
    encrypt_sensitive_features,
    parameter_set_from_context,
    sign_envelope,
    validate_envelope,
    verify_envelope_signature,
)
from fairlearn_fhe._circuits import (
    aggregate_difference,
    aggregate_ratio,
    positive_negative_counts,
    selection_rate_per_group,
)
from fairlearn_fhe.audit import _json_safe
from fairlearn_fhe.cli import main
from fairlearn_fhe.envelope import (
    ENVELOPE_SCHEMA,
    SIGNATURE_ALGORITHM,
    MetricEnvelope,
    estimate_security_bits,
)
from fairlearn_fhe.metrics import (
    EncryptedMetricFrame,
    MetricFrame,
    count,
    equal_opportunity_difference,
    equal_opportunity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    false_positive_rate,
    make_derived_metric,
    mean_prediction,
    r2_score_group_min,
    true_negative_rate,
)

# ---------------------------------------------------------------------------
# _circuits.py
# ---------------------------------------------------------------------------


def test_selection_rate_per_group_rejects_bad_pos_label():
    ctx = build_context(backend="tenseal")
    yp = encrypt(ctx, np.array([1.0, 0.0, 1.0]))
    masks = {"a": np.array([1.0, 0.0, 1.0]), "b": np.array([0.0, 1.0, 0.0])}
    with pytest.raises(NotImplementedError, match="pos_label"):
        selection_rate_per_group(yp, masks, pos_label=2)


def test_selection_rate_per_group_pos_label_zero_inverts_rate():
    ctx = build_context(backend="tenseal")
    yp = encrypt(ctx, np.array([1.0, 0.0, 1.0, 0.0]))
    masks = {"a": np.array([1.0, 1.0, 1.0, 1.0])}
    rates_one = selection_rate_per_group(yp, masks, pos_label=1)
    rates_zero = selection_rate_per_group(yp, masks, pos_label=0)
    assert pytest.approx(rates_one["a"] + rates_zero["a"], abs=1e-3) == 1.0


def test_positive_negative_counts_rejects_encrypted_masks():
    ctx = build_context(backend="tenseal")
    sf = np.array(["A", "B", "A", "B"], dtype=object)
    sf_enc = encrypt_sensitive_features(ctx, sf, y_true=np.array([1, 0, 1, 0]))
    with pytest.raises(ValueError, match="encrypted masks"):
        positive_negative_counts(np.array([1, 0, 1, 0]), sf_enc)


def test_aggregate_difference_empty_returns_zero():
    assert aggregate_difference([]) == 0.0


def test_aggregate_difference_to_overall_with_default_ref():
    # When `overall` is None the function falls back to the mean of
    # ``values`` as the reference.
    val = aggregate_difference([0.1, 0.3, 0.5], method="to_overall")
    assert pytest.approx(val) == 0.2


def test_aggregate_difference_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        aggregate_difference([0.1, 0.2], method="bogus")


def test_aggregate_ratio_empty_returns_one():
    assert aggregate_ratio([]) == 1.0


def test_aggregate_ratio_between_groups_zero_high_returns_zero():
    assert aggregate_ratio([0.0, 0.0]) == 0.0


def test_aggregate_ratio_to_overall_with_default_ref():
    # ref = mean(values) when `overall` is None. For values=[0.4, 0.6]
    # the ref is 0.5; the per-value min(v/ref, ref/v) is 0.8 for both;
    # the min across values is 0.8.
    val = aggregate_ratio([0.4, 0.6], method="to_overall")
    assert pytest.approx(val, abs=1e-3) == 0.8


def test_aggregate_ratio_to_overall_zero_ref_returns_zero():
    assert aggregate_ratio([0.0, 0.0], method="to_overall") == 0.0


def test_aggregate_ratio_to_overall_zero_value_collapses_to_zero():
    val = aggregate_ratio([0.0, 0.4], method="to_overall", overall=0.5)
    assert val == 0.0


def test_aggregate_ratio_unknown_method_raises():
    with pytest.raises(ValueError, match="unknown method"):
        aggregate_ratio([0.1, 0.2], method="bogus")


# ---------------------------------------------------------------------------
# _groups.py
# ---------------------------------------------------------------------------


def test_encrypted_mask_set_items_yields_triples():
    ctx = build_context(backend="tenseal")
    sf = np.array(["A", "B", "A", "B"], dtype=object)
    mset = encrypt_sensitive_features(ctx, sf, y_true=np.array([1, 0, 1, 0]))
    yielded = list(mset.items())
    assert len(yielded) == 2
    for label, mask, n_in_group in yielded:
        assert label in ("A", "B")
        assert isinstance(mask, EncryptedVector)
        assert n_in_group == 2.0


# ---------------------------------------------------------------------------
# audit.py — _json_safe
# ---------------------------------------------------------------------------


def test_json_safe_handles_numpy_scalars():
    assert _json_safe(np.int64(42)) == 42
    assert _json_safe(np.float32(0.5)) == pytest.approx(0.5)


def test_json_safe_handles_nested_containers():
    out = _json_safe({"a": [1, np.int64(2)], "b": (3.0, "x")})
    assert out == {"a": [1, 2], "b": [3.0, "x"]}


def test_json_safe_falls_back_to_repr_for_unknown():
    class _Custom:
        def __repr__(self):
            return "<Custom>"

    assert _json_safe(_Custom()) == "<Custom>"


# ---------------------------------------------------------------------------
# cli.py — public-key error handling
# ---------------------------------------------------------------------------


@pytest.fixture()
def _envelope_path(tmp_path: Path) -> Path:
    rng = np.random.default_rng(0)
    n = 16
    y_true = (rng.random(n) > 0.5).astype(int)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, y_true.astype(float))
    env = audit_metric(
        "demographic_parity_difference",
        y_true=y_true,
        y_pred=yp_enc,
        sensitive_features=sf,
        ctx=ctx,
        min_group_size=2,
    )
    p = tmp_path / "envelope.json"
    p.write_text(json.dumps(env.to_dict()))
    return p


def test_cli_verify_with_invalid_public_key_pem(tmp_path: Path, _envelope_path: Path):
    # PEM file present on disk but not actually a valid PEM. Should
    # surface as `invalid public key:` in the errors list.
    pk_path = tmp_path / "invalid.pem"
    pk_path.write_text("this is not a pem")
    rc = main(
        [
            "verify",
            str(_envelope_path),
            "--public-key",
            str(pk_path),
            "--json",
        ]
    )
    assert rc != 0


def test_cli_verify_with_unsigned_envelope_and_public_key(
    tmp_path: Path, _envelope_path: Path
):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    pk_pem = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
    pk_path = tmp_path / "pk.pem"
    pk_path.write_bytes(pk_pem)
    rc = main(
        [
            "verify",
            str(_envelope_path),
            "--public-key",
            str(pk_path),
            "--json",
        ]
    )
    # Envelope is unsigned → verify_envelope_signature returns
    # ['missing signature']; rc != 0.
    assert rc != 0


# ---------------------------------------------------------------------------
# context.py — TenSEAL make_evaluator_context AttributeError fallback
# ---------------------------------------------------------------------------


def test_make_evaluator_context_handles_inner_copy_attribute_error(monkeypatch):
    # The TenSEAL context exposes `.copy()`; some forks/builds don't.
    # The fallback path uses `make_context_public()` directly on the
    # original inner context. We simulate that by removing `.copy` from
    # the inner context object.
    ctx = build_context(backend="tenseal")
    inner = ctx.raw.context

    class _FakeInner:
        """Wraps the real inner but raises AttributeError on .copy."""

        def __init__(self, real):
            self._real = real

        def copy(self):
            raise AttributeError("no copy method")

        def __getattr__(self, name):
            return getattr(self._real, name)

        def make_context_public(self):
            self._real.make_context_public()

    ctx.raw.context = _FakeInner(inner)
    pub = ctx.make_evaluator_context()
    assert pub.has_secret_key is False


# ---------------------------------------------------------------------------
# encrypted.py — TenSEAL ct+ct, ct−ct paths
# ---------------------------------------------------------------------------


def test_encrypted_add_two_ciphertexts_tenseal():
    ctx = build_context(backend="tenseal")
    a = encrypt(ctx, np.array([1.0, 2.0, 3.0, 4.0]))
    b = encrypt(ctx, np.array([0.5, 0.5, 0.5, 0.5]))
    out = (a + b).decrypt()[:4]
    assert np.allclose(out, [1.5, 2.5, 3.5, 4.5], atol=1e-3)


def test_encrypted_sub_two_ciphertexts_tenseal():
    ctx = build_context(backend="tenseal")
    a = encrypt(ctx, np.array([1.0, 2.0, 3.0, 4.0]))
    b = encrypt(ctx, np.array([0.5, 0.5, 0.5, 0.5]))
    out = (a - b).decrypt()[:4]
    assert np.allclose(out, [0.5, 1.5, 2.5, 3.5], atol=1e-3)


def test_encrypted_mul_ct_two_ciphertexts_tenseal():
    ctx = build_context(backend="tenseal")
    a = encrypt(ctx, np.array([1.0, 2.0, 3.0, 4.0]))
    b = encrypt(ctx, np.array([2.0, 2.0, 2.0, 2.0]))
    out = a.mul_ct(b).decrypt()[:4]
    assert np.allclose(out, [2.0, 4.0, 6.0, 8.0], atol=1e-2)


# ---------------------------------------------------------------------------
# envelope.py — signature serialisation, validation negatives,
# parameter_set_from_context error paths.
# ---------------------------------------------------------------------------


def test_metric_envelope_to_dict_includes_signature_when_present():
    pytest.importorskip("cryptography")
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
    payload["signature"] = {"algorithm": SIGNATURE_ALGORITHM, "value": "abc"}
    env2 = MetricEnvelope.from_dict(payload)
    assert env2.signature == {"algorithm": SIGNATURE_ALGORITHM, "value": "abc"}
    assert "signature" in env2.to_dict()


def test_verify_envelope_signature_rejects_unsupported_algorithm():
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    pk_pem = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
    payload = {
        "schema_version": ENVELOPE_SCHEMA,
        "signature": {"algorithm": "RSA", "value": "abc"},
    }
    errors = verify_envelope_signature(payload, pk_pem)
    assert any("unsupported signature algorithm" in e for e in errors)


def test_verify_envelope_signature_rejects_non_base64_value():
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    pk_pem = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
    payload = {
        "schema_version": ENVELOPE_SCHEMA,
        "signature": {"algorithm": SIGNATURE_ALGORITHM, "value": "@@@@"},
    }
    errors = verify_envelope_signature(payload, pk_pem)
    assert any("base64" in e for e in errors)


def test_verify_envelope_signature_rejects_tampered_payload():
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    sk = Ed25519PrivateKey.generate()
    sk_pem = sk.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    pk_pem = sk.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

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
    signed = sign_envelope(env.to_dict(), sk_pem)
    signed["value"] = 999.0  # mutate after signing → InvalidSignature
    errors = verify_envelope_signature(signed, pk_pem)
    assert any("verification failed" in e for e in errors)


def test_validate_envelope_rejects_negative_observed_depth():
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
    payload["observed_depth"] = -1
    errors = validate_envelope(payload)
    assert any("non-negative" in e for e in errors)


def test_validate_envelope_rejects_observed_depth_above_param_set():
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
    payload["observed_depth"] = payload["parameter_set"]["multiplicative_depth"] + 99
    errors = validate_envelope(payload)
    assert any("multiplicative_depth" in e for e in errors)


def test_validate_envelope_rejects_non_integer_observed_depth():
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
    payload["observed_depth"] = "not-an-int"
    errors = validate_envelope(payload)
    assert any("integer" in e for e in errors)


def test_validate_envelope_rejects_bad_op_count_value():
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
    payload["op_counts"]["ct_pt_muls"] = -1
    errors = validate_envelope(payload)
    assert any("op_counts" in e for e in errors)


def test_validate_envelope_rejects_unsigned_signature_block():
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
    payload["signature"] = "not-a-mapping"
    errors = validate_envelope(payload)
    assert any("signature must be a mapping" in e for e in errors)

    payload["signature"] = {"algorithm": "RSA", "value": "abc"}
    errors = validate_envelope(payload)
    assert any("unsupported signature algorithm" in e for e in errors)


def test_estimate_security_bits_unknown_ring_returns_zero():
    assert estimate_security_bits(99999, 1000) == 0


def test_estimate_security_bits_192_bit_band():
    # 16384-degree ring with sum-of-bits in the 192-bit band.
    val = estimate_security_bits(16384, 412)
    assert val in (128, 192, 256)


def test_estimate_security_bits_256_bit_band():
    val = estimate_security_bits(16384, 1)
    assert val == 256


def test_parameter_set_from_context_openfhe_path():
    pytest.importorskip("openfhe")
    ctx = build_context(backend="openfhe")
    ps = parameter_set_from_context(ctx)
    assert "openfhe" in ps.backend.lower() or "ckks" in ps.backend.lower()


def test_parameter_set_from_context_explicit_depth_override():
    ctx = build_context(backend="tenseal")
    ps = parameter_set_from_context(ctx, depth=3)
    assert ps.multiplicative_depth == 3


# ---------------------------------------------------------------------------
# _base_metrics.py — count + plaintext passthrough
# ---------------------------------------------------------------------------


def test_count_with_encrypted_y_pred_returns_n():
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, np.array([1.0, 0.0, 1.0]))
    assert count(np.array([1, 0, 1]), yp_enc) == 3


def test_count_plaintext_passthrough():
    val = count(np.array([1, 0, 1]), np.array([1, 0, 1]))
    # fairlearn.metrics.count returns the number of samples
    assert val == fl.count(np.array([1, 0, 1]), np.array([1, 0, 1]))


def test_mean_prediction_plaintext_passthrough():
    val = mean_prediction(np.array([1, 0, 1, 0]), np.array([0.5, 0.4, 0.6, 0.7]))
    assert pytest.approx(val) == 0.55


def test_true_negative_rate_plaintext_passthrough():
    y = np.array([0, 0, 1, 1])
    yp = np.array([0, 0, 1, 1])
    assert pytest.approx(true_negative_rate(y, yp)) == fl.true_negative_rate(y, yp)


def test_false_positive_rate_plaintext_passthrough():
    y = np.array([0, 0, 1, 1])
    yp = np.array([0, 1, 1, 1])
    assert pytest.approx(false_positive_rate(y, yp)) == fl.false_positive_rate(y, yp)


# ---------------------------------------------------------------------------
# _fairness_metrics.py
# ---------------------------------------------------------------------------


def test_call_fairlearn_var_kw_path():
    """A function with **kwargs should receive every kwarg unchanged."""
    from fairlearn_fhe.metrics._fairness_metrics import _call_fairlearn

    def kw(**kwargs):
        return kwargs

    out = _call_fairlearn(kw, agg="x", method="y")
    assert out == {"agg": "x", "method": "y"}


def test_call_fairlearn_handles_signature_failure(monkeypatch):
    """When inspect.signature can't introspect the callable, fall through."""
    from fairlearn_fhe.metrics import _fairness_metrics as fm

    class _Builtin:
        # Built-in functions sometimes raise ValueError on inspect.signature.
        def __call__(self, *args, **kwargs):
            return ("called", args, kwargs)

    obj = _Builtin()
    out = fm._call_fairlearn(obj, "a", x=1)
    assert out[0] == "called"


def test_call_fairlearn_drops_unknown_kwargs():
    from fairlearn_fhe.metrics._fairness_metrics import _call_fairlearn

    def explicit(a, b):
        return (a, b)

    out = _call_fairlearn(explicit, a=1, b=2, ignored=3)
    assert out == (1, 2)


def test_pos_neg_counts_encrypted_mask_without_counts_raises():
    """Edge case: an EncryptedMaskSet without positives/negatives
    attrs should surface a ValueError, not a silent default."""
    ctx = build_context(backend="tenseal")
    sf = np.array(["A", "B", "A", "B"], dtype=object)
    mset = encrypt_sensitive_features(ctx, sf)  # no y_true → no counts
    yp_enc = encrypt(ctx, np.array([1.0, 0.0, 1.0, 0.0]))
    with pytest.raises((ValueError, TypeError)):
        equalized_odds_difference(
            np.array([1, 0, 1, 0]), yp_enc, sensitive_features=mset
        )


def test_equalized_odds_difference_rejects_bad_agg(mode_b_classification=None):
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, np.zeros(8))
    with pytest.raises(ValueError, match="agg must be"):
        equalized_odds_difference(
            np.zeros(8, dtype=int),
            yp_enc,
            sensitive_features=np.array(["A"] * 4 + ["B"] * 4),
            agg="invalid",  # type: ignore[arg-type]
        )


def test_equalized_odds_ratio_rejects_bad_agg():
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, np.zeros(8))
    with pytest.raises(ValueError, match="agg must be"):
        equalized_odds_ratio(
            np.zeros(8, dtype=int),
            yp_enc,
            sensitive_features=np.array(["A"] * 4 + ["B"] * 4),
            agg="invalid",  # type: ignore[arg-type]
        )


def test_equalized_odds_difference_mean_agg_path():
    ctx = build_context(backend="tenseal")
    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    val = equalized_odds_difference(
        y, yp_enc, sensitive_features=sf, agg="mean"
    )
    assert isinstance(val, float)


def test_equalized_odds_ratio_mean_agg_path():
    ctx = build_context(backend="tenseal")
    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    val = equalized_odds_ratio(y, yp_enc, sensitive_features=sf, agg="mean")
    assert isinstance(val, float)


def test_equal_opportunity_difference_plaintext_fallback_compute(monkeypatch):
    import fairlearn.metrics as flmod

    if hasattr(flmod, "equal_opportunity_difference"):
        monkeypatch.delattr(flmod, "equal_opportunity_difference")
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(int)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    val = equal_opportunity_difference(y, yp, sensitive_features=sf)
    assert isinstance(val, float)


def test_equal_opportunity_ratio_plaintext_fallback_compute(monkeypatch):
    import fairlearn.metrics as flmod

    if hasattr(flmod, "equal_opportunity_ratio"):
        monkeypatch.delattr(flmod, "equal_opportunity_ratio")
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(int)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    val = equal_opportunity_ratio(y, yp, sensitive_features=sf)
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# _make_derived_metric.py
# ---------------------------------------------------------------------------


def test_make_derived_metric_rejects_non_callable():
    with pytest.raises(ValueError, match="callable"):
        make_derived_metric(metric="not-callable", transform="difference")  # type: ignore[arg-type]


def test_make_derived_metric_rejects_unknown_transform():
    with pytest.raises(ValueError, match="transform"):
        make_derived_metric(metric=fl.selection_rate, transform="bogus")


def test_make_derived_metric_rejects_metric_with_reserved_arg():
    # Only ``method`` is reserved (it collides with the transform's
    # own `method=...` kwarg). A metric that defines its own
    # ``method=...`` parameter must be wrapped in functools.partial first.
    def naughty(y_true, y_pred, method=None):
        return 0.0

    with pytest.raises(ValueError, match="method"):
        make_derived_metric(metric=naughty, transform="difference")


def test_make_derived_metric_group_max_transform():
    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    derived = make_derived_metric(metric=fl.selection_rate, transform="group_max")
    val = derived(y, yp_enc, sensitive_features=sf)
    assert isinstance(val, float)


def test_make_derived_metric_group_min_transform():
    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    derived = make_derived_metric(metric=fl.selection_rate, transform="group_min")
    val = derived(y, yp_enc, sensitive_features=sf)
    assert isinstance(val, float)


def test_make_derived_metric_with_sample_param_names():
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    derived = make_derived_metric(
        metric=fl.selection_rate,
        transform="difference",
        sample_param_names=["sample_weight"],
    )
    val = derived(
        y, yp_enc, sensitive_features=sf, sample_weight=np.ones(n)
    )
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# _metric_frame.py
# ---------------------------------------------------------------------------


def test_encrypted_metric_frame_with_mean_prediction(mode_b_classification=None):
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    mf = MetricFrame(
        metrics={"mp": fl.mean_prediction},
        y_true=y,
        y_pred=yp_enc,
        sensitive_features=sf,
    )
    assert isinstance(mf, EncryptedMetricFrame)


def test_encrypted_metric_frame_with_count_metric():
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    mf = MetricFrame(
        metrics={"n": fl.count},
        y_true=y,
        y_pred=yp_enc,
        sensitive_features=sf,
    )
    assert isinstance(mf, EncryptedMetricFrame)


def test_encrypted_metric_frame_unknown_metric_without_decrypt_raises():
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)

    def custom_metric(yt, yp_arr):
        return float(yp_arr.mean())

    with pytest.raises(ValueError, match="not in the encrypted catalogue"):
        MetricFrame(
            metrics={"custom": custom_metric},
            y_true=y,
            y_pred=yp_enc,
            sensitive_features=sf,
        )


def test_encrypted_metric_frame_unknown_metric_with_encrypted_mask_raises():
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    mset = encrypt_sensitive_features(ctx, sf, y_true=y)

    def custom_metric(yt, yp_arr):
        return float(yp_arr.mean())

    with pytest.raises(ValueError, match="encrypted"):
        MetricFrame.fhe(
            metrics={"custom": custom_metric},
            y_true=y,
            y_pred=yp_enc,
            sensitive_features=mset,
            allow_decrypt=True,
        )


def test_encrypted_metric_frame_decrypt_fallback_for_custom_metric():
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)

    def custom_metric(yt, yp_arr):
        return float(np.mean(yp_arr))

    mf = MetricFrame.fhe(
        metrics={"custom": custom_metric},
        y_true=y,
        y_pred=yp_enc,
        sensitive_features=sf,
        allow_decrypt=True,
    )
    assert isinstance(mf, EncryptedMetricFrame)


def test_encrypted_metric_frame_handles_nested_sample_params():
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    sample_weight = np.ones(n)
    mf = MetricFrame.fhe(
        metrics={"sr": fl.selection_rate},
        y_true=y,
        y_pred=yp_enc,
        sensitive_features=sf,
        sample_params={"sr": {"sample_weight": sample_weight}},
    )
    assert isinstance(mf, EncryptedMetricFrame)


# ---------------------------------------------------------------------------
# _regression_metrics.py — r2 plaintext path
# ---------------------------------------------------------------------------


def test_r2_score_group_min_plaintext_passthrough():
    rng = np.random.default_rng(0)
    n = 32
    y = rng.normal(0.5, 0.2, size=n)
    yp = y + rng.normal(0.0, 0.05, size=n)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    val = r2_score_group_min(y, yp, sensitive_features=sf)
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# _scoring_metrics.py — group_max plaintext path
# ---------------------------------------------------------------------------


def test_zero_one_loss_group_max_plaintext_passthrough():
    from fairlearn_fhe.metrics import zero_one_loss_group_max

    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(int)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    val = zero_one_loss_group_max(y, yp, sensitive_features=sf)
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# OpenFHE backend edge ops
# ---------------------------------------------------------------------------


_HAS_OPENFHE = False
try:
    import openfhe  # noqa: F401

    _HAS_OPENFHE = True
except (ImportError, ModuleNotFoundError, OSError):
    _HAS_OPENFHE = False


@pytest.mark.skipif(not _HAS_OPENFHE, reason="openfhe-python not installed")
def test_openfhe_add_two_ciphertexts():
    ctx = build_context(backend="openfhe")
    a = encrypt(ctx, np.array([1.0, 2.0, 3.0, 4.0]))
    b = encrypt(ctx, np.array([0.5, 0.5, 0.5, 0.5]))
    out = (a + b).decrypt()[:4]
    assert np.allclose(out, [1.5, 2.5, 3.5, 4.5], atol=1e-3)


@pytest.mark.skipif(not _HAS_OPENFHE, reason="openfhe-python not installed")
def test_openfhe_sub_two_ciphertexts():
    ctx = build_context(backend="openfhe")
    a = encrypt(ctx, np.array([1.0, 2.0, 3.0, 4.0]))
    b = encrypt(ctx, np.array([0.5, 0.5, 0.5, 0.5]))
    out = (a - b).decrypt()[:4]
    assert np.allclose(out, [0.5, 1.5, 2.5, 3.5], atol=1e-3)


@pytest.mark.skipif(not _HAS_OPENFHE, reason="openfhe-python not installed")
def test_openfhe_add_plaintext():
    ctx = build_context(backend="openfhe")
    a = encrypt(ctx, np.array([1.0, 2.0, 3.0, 4.0]))
    out = (a + np.array([10.0, 10.0, 10.0, 10.0])).decrypt()[:4]
    assert np.allclose(out, [11.0, 12.0, 13.0, 14.0], atol=1e-3)


@pytest.mark.skipif(not _HAS_OPENFHE, reason="openfhe-python not installed")
def test_openfhe_mul_ct_two_ciphertexts():
    ctx = build_context(backend="openfhe")
    a = encrypt(ctx, np.array([1.0, 2.0, 3.0, 4.0]))
    b = encrypt(ctx, np.array([2.0, 2.0, 2.0, 2.0]))
    out = a.mul_ct(b).decrypt()[:4]
    assert np.allclose(out, [2.0, 4.0, 6.0, 8.0], atol=1e-2)


# ---------------------------------------------------------------------------
# Final cleanup batch
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _HAS_OPENFHE, reason="openfhe-python not installed")
def test_openfhe_sum_all_window_clamps_to_n_slots():
    """When the requested sum window exceeds the configured batch size,
    the backend silently clamps it. Construct a context whose
    ``n_slots`` is small enough for ``n > n_slots`` to be possible."""
    from fairlearn_fhe._backends import openfhe_backend as oh

    ctx = build_context(backend="openfhe", batch_size=8)
    v = encrypt(ctx, np.array([1.0] * 8))
    # Pass an n larger than n_slots to force the `window > ctx.n_slots`
    # clamp on line 186.
    out = oh.sum_all(v.ciphertext, 32, ctx.raw)
    assert out is not None  # the EvalSum call returned a ciphertext


def test_cli_inspect_with_legacy_envelope_no_metric_field(tmp_path: Path, capsys):
    p = tmp_path / "legacy.json"
    p.write_text(json.dumps({"foo": "bar"}))
    rc = main(["inspect", str(p)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "<unknown>" in out


def test_cli_verify_with_valid_signature_extends_no_errors(tmp_path: Path, capsys):
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    sk = Ed25519PrivateKey.generate()
    sk_pem = sk.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    pk_pem = sk.public_key().public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

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
    signed = sign_envelope(env.to_dict(), sk_pem)
    env_path = tmp_path / "envelope.json"
    env_path.write_text(json.dumps(signed))
    pk_path = tmp_path / "pk.pem"
    pk_path.write_bytes(pk_pem)
    rc = main(
        [
            "verify",
            str(env_path),
            "--public-key",
            str(pk_path),
            "--json",
        ]
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc == 0
    assert payload["valid"] is True
    assert payload["errors"] == []


def test_cli_verify_invalid_format_envelope_returns_error(tmp_path: Path, capsys):
    p = tmp_path / "bad.json"
    p.write_text("not-json-at-all")
    rc = main(["verify", str(p), "--json"])
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc != 0
    assert payload["valid"] is False


def test_cli_doctor_imports_both_backends(capsys):
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    # Both backends should be on the list. The probe outcome (available
    # vs missing) depends on the local install; what we care about is
    # that line 195/201/206 were executed for both names.
    assert "tenseal" in out
    assert "openfhe" in out


def test_envelope_to_json_round_trip_via_from_dict():
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
    text = env.to_json()
    loaded = MetricEnvelope.from_dict(json.loads(text))
    assert loaded.metric_name == env.metric_name


def test_envelope_signature_missing_value_field_returns_error():
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    pk_pem = (
        Ed25519PrivateKey.generate()
        .public_key()
        .public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    )
    payload = {
        "schema_version": ENVELOPE_SCHEMA,
        "signature": {"algorithm": SIGNATURE_ALGORITHM},  # no `value`
    }
    errors = verify_envelope_signature(payload, pk_pem)
    assert any("base64" in e for e in errors)


def test_validate_envelope_rejects_non_integer_n_samples():
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
    payload["n_samples"] = "not-an-int"
    errors = validate_envelope(payload)
    assert any("n_samples" in e for e in errors)


def test_validate_envelope_rejects_non_integer_op_count():
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
    payload["op_counts"]["ct_pt_muls"] = "not-an-int"
    errors = validate_envelope(payload)
    assert any("integer" in e for e in errors)


def test_estimate_security_bits_below_128_returns_zero():
    # 16384-degree ring, q exceeding the 128-bit cap → return 0
    val = estimate_security_bits(16384, 100_000)
    assert val == 0


def test_parameter_set_from_context_records_backend_version():
    pytest.importorskip("tenseal")
    ctx = build_context(backend="tenseal")
    ps = parameter_set_from_context(ctx)
    # backend_version should be a non-empty string when tenseal is installed
    assert isinstance(ps.backend_version, str)


def test_demographic_parity_ratio_plaintext_passthrough():
    from fairlearn_fhe.metrics import demographic_parity_ratio

    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(int)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    val = demographic_parity_ratio(y, yp, sensitive_features=sf)
    assert isinstance(val, float)


def test_equalized_odds_difference_plaintext_passthrough():
    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(int)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    val = equalized_odds_difference(y, yp, sensitive_features=sf)
    assert isinstance(val, float)


def test_equalized_odds_ratio_plaintext_passthrough():
    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(int)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    val = equalized_odds_ratio(y, yp, sensitive_features=sf)
    assert isinstance(val, float)


def test_equal_opportunity_difference_plaintext_passthrough():
    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(int)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    val = equal_opportunity_difference(y, yp, sensitive_features=sf)
    assert isinstance(val, float)


def test_equal_opportunity_ratio_plaintext_passthrough():
    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(int)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    val = equal_opportunity_ratio(y, yp, sensitive_features=sf)
    assert isinstance(val, float)


def test_metric_frame_count_metric_with_encrypted_mask():
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    sf_enc = encrypt_sensitive_features(ctx, sf, y_true=y)
    mf = MetricFrame(
        metrics={"n": fl.count},
        y_true=y,
        y_pred=yp_enc,
        sensitive_features=sf_enc,
    )
    assert isinstance(mf, EncryptedMetricFrame)


def test_metric_frame_tpr_with_encrypted_mask_missing_counts():
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    # Construct an EncryptedMaskSet without the positives/negatives counts.
    mset = encrypt_sensitive_features(ctx, sf)
    with pytest.raises((ValueError, TypeError)):
        MetricFrame(
            metrics={"tpr": fl.true_positive_rate},
            y_true=y,
            y_pred=yp_enc,
            sensitive_features=mset,
        )


def test_make_derived_metric_with_other_param_passes_through():
    """`other_params` keys that aren't reserved or sample-params get
    threaded into the dispatch via functools.partial. Hits lines 67-70."""
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)

    def metric_with_extra(y_true, y_pred, **extra):
        # Force decrypt-fallback by being a non-catalogue metric.
        return float(np.mean(y_pred))

    derived = make_derived_metric(
        metric=metric_with_extra, transform="difference"
    )
    val = derived(
        y, yp_enc, sensitive_features=sf, custom_kwarg=42
    )
    assert isinstance(val, float)


def test_make_derived_metric_unknown_transform_at_dispatch_time():
    """Sneak an unknown transform past the constructor's check by
    mutating the instance attribute. Hits the final `raise ValueError`
    at the end of `__call__`."""
    derived = make_derived_metric(metric=fl.selection_rate, transform="difference")
    # Bypass the constructor's transform validation by mutating the
    # private attribute on the returned instance.
    derived._transform = "made-up-transform"
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    with pytest.raises(ValueError, match="unknown transform"):
        derived(y, yp_enc, sensitive_features=sf)


def test_zero_one_loss_group_max_plaintext_with_sample_weight():
    """Hits the `_group_max` plaintext-fallthrough path (lines 147, 153
    in _scoring_metrics)."""
    from fairlearn_fhe.metrics import zero_one_loss_group_max

    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(int)
    sf = rng.choice(["A", "B"], size=n).astype(object)
    sw = np.ones(n) * 1.0
    val = zero_one_loss_group_max(
        y, yp, sensitive_features=sf, sample_weight=sw
    )
    assert isinstance(val, float)


def test_r2_score_group_min_plaintext_with_empty_group():
    """Hits the `n_g <= 0` branch in r2_score_group_min plaintext path."""
    rng = np.random.default_rng(0)
    n = 16
    y = rng.normal(0.5, 0.2, size=n)
    yp = y + rng.normal(0.0, 0.05, size=n)
    sf = np.array(["A"] * n, dtype=object)  # all in one group
    val = r2_score_group_min(y, yp, sensitive_features=sf)
    assert isinstance(val, float)


# ---------------------------------------------------------------------------
# Final cleanup batch 2 — defensive-branch hits
# ---------------------------------------------------------------------------


def test_cli_verify_signed_envelope_with_non_ed25519_public_key(tmp_path: Path, capsys):
    """Hits cli.py:105-106 — verify_envelope_signature raises TypeError
    when the supplied public-key PEM isn't an Ed25519 key."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
        PublicFormat,
    )

    sk = Ed25519PrivateKey.generate()
    sk_pem = sk.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    # Public key in RSA flavour — not Ed25519. verify_envelope_signature
    # will raise TypeError, the CLI catches it as "invalid public key".
    rsa_pk = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    rsa_pk_pem = rsa_pk.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)

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
    signed = sign_envelope(env.to_dict(), sk_pem)
    env_path = tmp_path / "envelope.json"
    env_path.write_text(json.dumps(signed))
    pk_path = tmp_path / "rsa.pem"
    pk_path.write_bytes(rsa_pk_pem)
    rc = main(
        [
            "verify",
            str(env_path),
            "--public-key",
            str(pk_path),
            "--json",
        ]
    )
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert rc != 0
    assert any("invalid public key" in e for e in payload["errors"])


def test_cli_doctor_when_tenseal_import_fails(monkeypatch, capsys):
    """Hits cli.py:203-204 — the tenseal-missing branch in _probe_backend."""
    import builtins

    real_import = builtins.__import__

    def _failing_import(name, *args, **kwargs):
        if name == "tenseal":
            raise ImportError("simulated missing tenseal")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_import)
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "tenseal" in out
    assert "missing" in out


def test_cli_doctor_when_openfhe_import_fails(monkeypatch, capsys):
    """Hits cli.py:209-210 — the openfhe-missing branch in _probe_backend."""
    import builtins

    real_import = builtins.__import__

    def _failing_import(name, *args, **kwargs):
        if name == "openfhe":
            raise ImportError("simulated missing openfhe")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_import)
    rc = main(["doctor"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "openfhe" in out


def test_sign_envelope_rejects_non_ed25519_private_key():
    """Hits envelope.py:247 — sign_envelope raises TypeError for an RSA key."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import (
        Encoding,
        NoEncryption,
        PrivateFormat,
    )

    rsa_sk = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    rsa_sk_pem = rsa_sk.private_bytes(Encoding.PEM, PrivateFormat.PKCS8, NoEncryption())
    payload = {"schema_version": ENVELOPE_SCHEMA, "metric_name": "x", "value": 0.0}
    with pytest.raises(TypeError, match="Ed25519"):
        sign_envelope(payload, rsa_sk_pem)


def test_verify_envelope_signature_rejects_non_ed25519_public_key():
    """Hits envelope.py:290."""
    pytest.importorskip("cryptography")
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

    rsa_pk = rsa.generate_private_key(public_exponent=65537, key_size=2048).public_key()
    rsa_pk_pem = rsa_pk.public_bytes(Encoding.PEM, PublicFormat.SubjectPublicKeyInfo)
    payload = {
        "schema_version": ENVELOPE_SCHEMA,
        "signature": {
            "algorithm": SIGNATURE_ALGORITHM,
            "value": b64encode(b"\x00" * 64).decode("ascii"),
        },
    }
    with pytest.raises(TypeError, match="Ed25519"):
        verify_envelope_signature(payload, rsa_pk_pem)


def test_validate_envelope_rejects_non_mapping_op_counts():
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
    payload["op_counts"] = ["not", "a", "mapping"]
    errors = validate_envelope(payload)
    assert any("op_counts must be a mapping" in e for e in errors)


def test_estimate_security_bits_192_band_explicit():
    # 16384-degree ring: caps are (438, 305, 237) for (128, 192, 256).
    # q=300 falls in the 192-bit band (above 237, at-or-below 305).
    val = estimate_security_bits(16384, 300)
    assert val == 192


def test_parameter_set_from_context_tenseal_import_failure(monkeypatch):
    """Hits envelope.py:477-478 — tenseal.__version__ lookup raises."""
    ctx = build_context(backend="tenseal")
    import builtins

    real_import = builtins.__import__

    def _failing_import(name, *args, **kwargs):
        if name == "tenseal":
            raise RuntimeError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_import)
    ps = parameter_set_from_context(ctx)
    assert ps.backend_version == ""


def test_parameter_set_from_context_openfhe_import_failure(monkeypatch):
    """Hits envelope.py:484-485 — openfhe.__version__ lookup raises."""
    pytest.importorskip("openfhe")
    ctx = build_context(backend="openfhe")
    import builtins

    real_import = builtins.__import__

    def _failing_import(name, *args, **kwargs):
        if name == "openfhe":
            raise RuntimeError("simulated")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _failing_import)
    ps = parameter_set_from_context(ctx)
    assert ps.backend_version == ""


def test_call_fairlearn_handles_signature_typeerror(monkeypatch):
    """Hits _fairness_metrics.py:36-37 — inspect.signature raises
    TypeError on certain built-ins; the helper falls through and
    invokes ``fn(*args, **kwargs)`` directly. We pass kwargs the
    function accepts so the fallthrough call succeeds."""
    from fairlearn_fhe.metrics import _fairness_metrics as fm

    real_signature = fm.inspect.signature

    def _failing_signature(fn):
        if getattr(fn, "_no_introspect", False):
            raise TypeError("cannot introspect")
        return real_signature(fn)

    monkeypatch.setattr(fm.inspect, "signature", _failing_signature)

    def my_fn(a, b=2):
        return a + b

    my_fn._no_introspect = True
    out = fm._call_fairlearn(my_fn, 5, b=10)
    assert out == 15


def test_make_derived_metric_method_kwarg_routed_to_transform():
    """Hits _make_derived_metric.py:68 — passing ``method=...`` as a kwarg
    routes to the transform's method= parameter."""
    rng = np.random.default_rng(0)
    n = 32
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)

    def my_metric(y_true, y_pred):
        return float(np.mean(y_pred))

    derived = make_derived_metric(metric=my_metric, transform="difference")
    val = derived(
        y, yp_enc, sensitive_features=sf, method="to_overall"
    )
    assert isinstance(val, float)


def test_metric_frame_fhe_falls_back_when_metric_rejects_sample_weight():
    """Hits _metric_frame.py:161-162 — TypeError on sample_weight kwarg
    is caught and the metric is called without it."""
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(float)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    sf = rng.choice(["A", "B"], size=n).astype(object)

    def metric_no_sw(y_true, y_pred):
        return float(np.mean(y_pred))

    mf = MetricFrame.fhe(
        metrics={"m": metric_no_sw},
        y_true=y,
        y_pred=yp_enc,
        sensitive_features=sf,
        sample_params={"sample_weight": np.ones(n)},
        allow_decrypt=True,
    )
    assert isinstance(mf, EncryptedMetricFrame)


def test_metric_frame_factory_rejects_encrypted_sf_with_plaintext_y_pred():
    """Hits _metric_frame.py:190 — the MetricFrame factory itself
    (vs the .fhe alias) rejects encrypted sf with plaintext y_pred."""
    rng = np.random.default_rng(0)
    n = 16
    y = (rng.random(n) > 0.5).astype(int)
    yp = (rng.random(n) > 0.5).astype(int)
    ctx = build_context(backend="tenseal")
    sf = rng.choice(["A", "B"], size=n).astype(object)
    sf_enc = encrypt_sensitive_features(ctx, sf, y_true=y)
    with pytest.raises(TypeError, match="encrypted"):
        MetricFrame(
            metrics={"sr": fl.selection_rate},
            y_true=y,
            y_pred=yp,
            sensitive_features=sf_enc,
        )


def test_r2_score_group_min_handles_zero_weight_group():
    """Hits _regression_metrics.py:236-237 — n_g <= 0 branch when a
    group's effective weight collapses to zero."""
    rng = np.random.default_rng(0)
    n = 16
    y = rng.normal(0.5, 0.2, size=n)
    yp = y.copy()
    sf = np.array(["A"] * 8 + ["B"] * 8, dtype=object)
    sw = np.array([1.0] * 8 + [0.0] * 8)
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, yp)
    # Encrypted path — the in-tree compute handles the zero-weight
    # group via the `n_g <= 0: continue` branch.
    val = r2_score_group_min(y, yp_enc, sensitive_features=sf, sample_weight=sw)
    assert isinstance(val, float)


def test_scoring_helpers_handle_zero_division_edges():
    """Hits _scoring_metrics.py:147 (_f1 p+r==0) and 153 (_zero_one_loss
    n_total==0)."""
    from fairlearn_fhe.metrics._scoring_metrics import _f1, _zero_one_loss

    counts_zero = {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n_pos": 0, "n_neg": 0, "n_total": 0}
    assert _f1(counts_zero) == 0.0
    assert _zero_one_loss(counts_zero) == 0.0


def test_plaintext_tpr_overall_zero_denominator():
    """Hits _fairness_metrics.py:323 — _plaintext_tpr_overall when there
    are no positives in y_true."""
    from fairlearn_fhe.metrics._fairness_metrics import _plaintext_tpr_overall

    val = _plaintext_tpr_overall(
        np.array([0, 0, 0]), np.array([0, 1, 1]), None
    )
    assert val == 0.0


def test_plaintext_tpr_per_group_zero_denominator():
    """Hits _fairness_metrics.py:311 — _plaintext_tpr_per_group when a
    group has no positives."""
    from fairlearn_fhe.metrics._fairness_metrics import _plaintext_tpr_per_group

    masks = {"A": np.array([1.0, 1.0, 0.0]), "B": np.array([0.0, 0.0, 1.0])}
    out = _plaintext_tpr_per_group(
        np.array([0, 0, 1]),
        np.array([1, 1, 1]),
        masks,
        sw=None,
        positives={"A": 0.0, "B": 1.0},  # group A has no positives
    )
    assert out["A"] == 0.0
    assert out["B"] > 0


def test_build_context_unknown_backend_raises_directly():
    """Hits context.py:187 — build_context's final raise when the
    backend name doesn't match any known branch. Bypasses the
    set_default_backend validator by passing the bad name directly."""
    from fairlearn_fhe.context import build_context as _bc

    with pytest.raises(ValueError, match="unknown backend"):
        _bc(backend="not-a-backend")  # type: ignore[arg-type]
