"""Envelope output: parameter hash stable, op counters populated."""

import json

import pytest

from fairlearn_fhe import (
    ENVELOPE_SCHEMA,
    MetricEnvelope,
    ParameterSet,
    audit_metric,
    validate_envelope,
)


def test_audit_metric_envelope(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    assert env.metric_name == "demographic_parity_difference"
    assert 0.0 <= env.value <= 1.0
    assert env.n_samples == len(y_true)
    assert env.n_groups == 3
    assert env.observed_depth >= 1
    assert env.op_counts["ct_pt_muls"] >= 3  # one per group
    payload = env.to_dict()
    assert payload["schema_version"] == ENVELOPE_SCHEMA
    assert payload["parameter_set_hash"]
    assert json.loads(env.to_json())["metric_name"] == "demographic_parity_difference"
    assert validate_envelope(payload) == []


def test_parameter_hash_stable(ctx):
    ps1 = ParameterSet(
        backend="tenseal-ckks", poly_modulus_degree=8192,
        security_bits=128, multiplicative_depth=6,
        coeff_mod_bit_sizes=(60, 40, 40, 60),
        scaling_factor_bits=40,
    )
    ps2 = ParameterSet(
        backend="tenseal-ckks", poly_modulus_degree=8192,
        security_bits=128, multiplicative_depth=6,
        coeff_mod_bit_sizes=(60, 40, 40, 60),
        scaling_factor_bits=40,
    )
    assert ps1.hash() == ps2.hash()


def test_envelope_roundtrip(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    clone = MetricEnvelope.from_json(env.to_json())
    assert clone.to_dict() == env.to_dict()


def test_validate_envelope_detects_tampering(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    payload = env.to_dict()
    payload["parameter_set"]["scaling_factor_bits"] = 30
    errors = validate_envelope(payload)
    assert "parameter_set_hash does not match parameter_set" in errors


def test_validate_envelope_rejects_depth_over_budget(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    payload = env.to_dict()
    payload["observed_depth"] = payload["parameter_set"]["multiplicative_depth"] + 1
    errors = validate_envelope(payload)
    assert "observed_depth exceeds parameter_set multiplicative_depth" in errors


def test_validate_envelope_rejects_structural_errors(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true,
        y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    payload = env.to_dict()

    missing = dict(payload)
    missing.pop("metric_name")
    assert validate_envelope(missing) == ["missing required field: metric_name"]

    bad = dict(payload)
    bad["schema_version"] = "wrong"
    bad["metric_name"] = "not_allowed"
    bad["observed_depth"] = -1
    bad["n_samples"] = 0
    bad["n_groups"] = "nan"
    bad["op_counts"] = {"ct_pt_muls": -1, "bad": "nan"}
    bad["value"] = object()
    bad["timestamp"] = object()

    errors = validate_envelope(
        bad,
        allowed_metrics=["demographic_parity_difference"],
        max_observed_depth=0,
    )
    assert "unsupported schema_version 'wrong'" in errors
    assert "metric_name 'not_allowed' is not allowed" in errors
    assert "observed_depth must be non-negative" in errors
    assert "n_samples must be positive" in errors
    assert "n_groups must be an integer" in errors
    assert "op_counts['ct_pt_muls'] must be non-negative" in errors
    assert "op_counts['bad'] must be an integer" in errors
    assert "value must be numeric" in errors
    assert "timestamp must be numeric" in errors


def test_validate_envelope_rejects_invalid_parameter_set(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true,
        y_pred,
        sensitive_features=sf,
        ctx=ctx,
    )
    payload = env.to_dict()
    payload["parameter_set"] = {"backend": "tenseal-ckks"}
    errors = validate_envelope(payload)
    assert any(error.startswith("invalid parameter_set:") for error in errors)


def test_unknown_metric():
    with pytest.raises(KeyError):
        audit_metric("not_a_real_metric", [0, 1], [0, 1])


def test_op_counters_per_metric(small_dataset, ctx):
    y_true, y_pred, sf = small_dataset
    env_dp = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf, ctx=ctx,
    )
    env_eo = audit_metric(
        "equalized_odds_difference",
        y_true, y_pred,
        sensitive_features=sf, ctx=ctx,
    )
    # Equalized odds touches every group twice (TPR + FPR) → more ct_pt mults.
    assert env_eo.op_counts["ct_pt_muls"] > env_dp.op_counts["ct_pt_muls"]
