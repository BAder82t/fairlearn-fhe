"""Encrypted fairness metrics match plaintext Fairlearn within CKKS noise."""

import numpy as np
import pytest

import fairlearn.metrics as fl

from fairlearn_fhe.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    equal_opportunity_difference,
    equal_opportunity_ratio,
)


def test_demographic_parity_difference(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
    enc = demographic_parity_difference(y_true, encrypted_pred, sensitive_features=sf)
    assert abs(plain - enc) < tol


def test_demographic_parity_ratio(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.demographic_parity_ratio(y_true, y_pred, sensitive_features=sf)
    enc = demographic_parity_ratio(y_true, encrypted_pred, sensitive_features=sf)
    assert abs(plain - enc) < tol


@pytest.mark.parametrize("agg", ["worst_case", "mean"])
def test_equalized_odds_difference(small_dataset, encrypted_pred, tol, agg):
    y_true, y_pred, sf = small_dataset
    plain = fl.equalized_odds_difference(y_true, y_pred, sensitive_features=sf, agg=agg)
    enc = equalized_odds_difference(y_true, encrypted_pred, sensitive_features=sf, agg=agg)
    assert abs(plain - enc) < tol


@pytest.mark.parametrize("agg", ["worst_case", "mean"])
def test_equalized_odds_ratio(small_dataset, encrypted_pred, tol, agg):
    y_true, y_pred, sf = small_dataset
    plain = fl.equalized_odds_ratio(y_true, y_pred, sensitive_features=sf, agg=agg)
    enc = equalized_odds_ratio(y_true, encrypted_pred, sensitive_features=sf, agg=agg)
    assert abs(plain - enc) < tol


def test_equal_opportunity_difference(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.equal_opportunity_difference(y_true, y_pred, sensitive_features=sf)
    enc = equal_opportunity_difference(y_true, encrypted_pred, sensitive_features=sf)
    assert abs(plain - enc) < tol


def test_equal_opportunity_ratio(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.equal_opportunity_ratio(y_true, y_pred, sensitive_features=sf)
    enc = equal_opportunity_ratio(y_true, encrypted_pred, sensitive_features=sf)
    assert abs(plain - enc) < tol


def test_method_to_overall(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.demographic_parity_difference(y_true, y_pred, sensitive_features=sf, method="to_overall")
    enc = demographic_parity_difference(
        y_true, encrypted_pred, sensitive_features=sf, method="to_overall",
    )
    assert abs(plain - enc) < tol


def test_two_groups(ctx, tol):
    """Binary sensitive feature — worst case for FHE numerical noise."""
    from fairlearn_fhe import encrypt
    rng = np.random.default_rng(7)
    n = 300
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sf = rng.integers(0, 2, size=n)
    enc = encrypt(ctx, y_pred)

    plain = fl.demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
    fhe = demographic_parity_difference(y_true, enc, sensitive_features=sf)
    assert abs(plain - fhe) < tol
