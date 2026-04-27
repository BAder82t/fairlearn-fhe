"""OpenFHE backend smoke tests.

Skipped if openfhe-python is not importable.
"""

import numpy as np
import pytest

import fairlearn.metrics as fl

pytest.importorskip("openfhe")

from fairlearn_fhe import build_context, encrypt
from fairlearn_fhe.metrics import (
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equal_opportunity_difference,
)


@pytest.fixture(scope="module")
def ofhe_ctx():
    return build_context(backend="openfhe", batch_size=1024, scale_bits=40)


@pytest.fixture(scope="module")
def shared_data():
    rng = np.random.default_rng(42)
    n = 256
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sf = rng.choice(["A", "B", "C"], size=n)
    return y_true, y_pred, sf


_TOL = 5e-3  # OpenFHE rescaling can be slightly noisier than TenSEAL


def test_roundtrip(ofhe_ctx):
    ev = encrypt(ofhe_ctx, [1.0, 2.0, 3.0, 4.0])
    out = ev.decrypt()
    np.testing.assert_allclose(out[:4], [1.0, 2.0, 3.0, 4.0], atol=_TOL)


def test_selection_rate_openfhe(ofhe_ctx, shared_data):
    y_true, y_pred, _ = shared_data
    enc = encrypt(ofhe_ctx, y_pred)
    plain = fl.selection_rate(y_true, y_pred)
    fhe = selection_rate(y_true, enc)
    assert abs(plain - fhe) < _TOL


def test_demographic_parity_openfhe(ofhe_ctx, shared_data):
    y_true, y_pred, sf = shared_data
    enc = encrypt(ofhe_ctx, y_pred)
    plain = fl.demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
    fhe = demographic_parity_difference(y_true, enc, sensitive_features=sf)
    assert abs(plain - fhe) < _TOL


def test_demographic_parity_ratio_openfhe(ofhe_ctx, shared_data):
    y_true, y_pred, sf = shared_data
    enc = encrypt(ofhe_ctx, y_pred)
    plain = fl.demographic_parity_ratio(y_true, y_pred, sensitive_features=sf)
    fhe = demographic_parity_ratio(y_true, enc, sensitive_features=sf)
    assert abs(plain - fhe) < _TOL


def test_tpr_fpr_openfhe(ofhe_ctx, shared_data):
    y_true, y_pred, _ = shared_data
    enc = encrypt(ofhe_ctx, y_pred)
    assert abs(fl.true_positive_rate(y_true, y_pred) - true_positive_rate(y_true, enc)) < _TOL
    assert abs(fl.false_positive_rate(y_true, y_pred) - false_positive_rate(y_true, enc)) < _TOL


def test_equalized_odds_openfhe(ofhe_ctx, shared_data):
    y_true, y_pred, sf = shared_data
    enc = encrypt(ofhe_ctx, y_pred)
    plain = fl.equalized_odds_difference(y_true, y_pred, sensitive_features=sf)
    fhe = equalized_odds_difference(y_true, enc, sensitive_features=sf)
    assert abs(plain - fhe) < _TOL


def test_equal_opportunity_openfhe(ofhe_ctx, shared_data):
    y_true, y_pred, sf = shared_data
    enc = encrypt(ofhe_ctx, y_pred)
    plain = fl.equal_opportunity_difference(y_true, y_pred, sensitive_features=sf)
    fhe = equal_opportunity_difference(y_true, enc, sensitive_features=sf)
    assert abs(plain - fhe) < _TOL


def test_envelope_records_openfhe(ofhe_ctx, shared_data):
    from fairlearn_fhe import audit_metric
    y_true, y_pred, sf = shared_data
    env = audit_metric(
        "demographic_parity_difference",
        y_true, y_pred,
        sensitive_features=sf, ctx=ofhe_ctx,
    )
    assert env.parameter_set.backend == "openfhe-ckks"
    assert env.observed_depth >= 1
