# Copyright 2026 Vaultbytes (Bader Alissaei)
# SPDX-License-Identifier: Apache-2.0

"""Tests for the v0.2.0 metric ports.

Covers:

- The per-rate ``_difference`` / ``_ratio`` family
  (selection_rate, true/false positive/negative rate, …);
- Scoring disaggregations (``accuracy_score_*``,
  ``balanced_accuracy_score_*``, ``precision_*``, ``recall_*``,
  ``f1_*``, ``zero_one_loss_*``);
- Regression disaggregations (``mean_squared_error_group_max``,
  ``mean_absolute_error_group_max``, ``r2_score_group_min``);
- ``selection_rate`` with ``pos_label=0``.

Each encrypted-path verdict is compared against the equivalent
plaintext-path computation; tolerances match Fairlearn's CKKS noise
band (default 1e-3 abs).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import fairlearn_fhe as flfhe
from fairlearn_fhe import build_context, encrypt
from fairlearn_fhe.metrics import (
    accuracy_score_difference,
    accuracy_score_group_min,
    balanced_accuracy_score_group_min,
    f1_score_group_min,
    false_negative_rate_difference,
    false_negative_rate_ratio,
    false_positive_rate_difference,
    false_positive_rate_ratio,
    mean_absolute_error_group_max,
    mean_squared_error_group_max,
    precision_score_group_min,
    r2_score_group_min,
    recall_score_group_min,
    selection_rate,
    selection_rate_difference,
    selection_rate_ratio,
    true_negative_rate_difference,
    true_negative_rate_ratio,
    true_positive_rate_difference,
    true_positive_rate_ratio,
    zero_one_loss_difference,
    zero_one_loss_group_max,
    zero_one_loss_ratio,
)

_TOL = 1e-3


@pytest.fixture(scope="module")
def fixture():
    rng = np.random.default_rng(7)
    n = 256
    y_true = (rng.random(n) > 0.5).astype(int)
    # Predictions correlated with truth but biased per group so we get a
    # measurable fairness gap.
    y_pred = y_true.copy()
    flip = rng.random(n) < 0.20
    y_pred = np.where(flip, 1 - y_pred, y_pred)
    sf = rng.choice(["A", "B", "C"], size=n).astype(object)
    # Bias group "A" toward false positives, "B" toward false negatives.
    for i, g in enumerate(sf):
        if g == "A" and rng.random() < 0.20 and y_true[i] == 0:
            y_pred[i] = 1
        if g == "B" and rng.random() < 0.20 and y_true[i] == 1:
            y_pred[i] = 0
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, y_pred.astype(float))
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_enc": yp_enc,
        "sensitive_features": sf,
        "ctx": ctx,
    }


# ---------------------------------------------------------------------------
# Per-rate difference / ratio family
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        selection_rate_difference,
        true_positive_rate_difference,
        true_negative_rate_difference,
        false_positive_rate_difference,
        false_negative_rate_difference,
    ],
)
def test_diff_family_matches_plaintext(fixture, fn):
    plain = fn(
        fixture["y_true"], fixture["y_pred"],
        sensitive_features=fixture["sensitive_features"],
    )
    enc = fn(
        fixture["y_true"], fixture["y_pred_enc"],
        sensitive_features=fixture["sensitive_features"],
    )
    assert math.isclose(plain, enc, abs_tol=_TOL)


@pytest.mark.parametrize(
    "fn",
    [
        selection_rate_ratio,
        true_positive_rate_ratio,
        true_negative_rate_ratio,
        false_positive_rate_ratio,
        false_negative_rate_ratio,
    ],
)
def test_ratio_family_matches_plaintext(fixture, fn):
    plain = fn(
        fixture["y_true"], fixture["y_pred"],
        sensitive_features=fixture["sensitive_features"],
    )
    enc = fn(
        fixture["y_true"], fixture["y_pred_enc"],
        sensitive_features=fixture["sensitive_features"],
    )
    assert math.isclose(plain, enc, abs_tol=_TOL)


def test_diff_family_supports_method_to_overall(fixture):
    plain = true_positive_rate_difference(
        fixture["y_true"], fixture["y_pred"],
        sensitive_features=fixture["sensitive_features"],
        method="to_overall",
    )
    enc = true_positive_rate_difference(
        fixture["y_true"], fixture["y_pred_enc"],
        sensitive_features=fixture["sensitive_features"],
        method="to_overall",
    )
    assert math.isclose(plain, enc, abs_tol=_TOL)


# ---------------------------------------------------------------------------
# Scoring disaggregations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        accuracy_score_group_min,
        balanced_accuracy_score_group_min,
        precision_score_group_min,
        recall_score_group_min,
        f1_score_group_min,
    ],
)
def test_group_min_scoring_matches_plaintext(fixture, fn):
    plain = fn(
        fixture["y_true"], fixture["y_pred"],
        sensitive_features=fixture["sensitive_features"],
    )
    enc = fn(
        fixture["y_true"], fixture["y_pred_enc"],
        sensitive_features=fixture["sensitive_features"],
    )
    assert math.isclose(plain, enc, abs_tol=_TOL)


def test_zero_one_loss_group_max_matches_plaintext(fixture):
    plain = zero_one_loss_group_max(
        fixture["y_true"], fixture["y_pred"],
        sensitive_features=fixture["sensitive_features"],
    )
    enc = zero_one_loss_group_max(
        fixture["y_true"], fixture["y_pred_enc"],
        sensitive_features=fixture["sensitive_features"],
    )
    assert math.isclose(plain, enc, abs_tol=_TOL)


def test_accuracy_difference_matches_plaintext(fixture):
    plain = accuracy_score_difference(
        fixture["y_true"], fixture["y_pred"],
        sensitive_features=fixture["sensitive_features"],
    )
    enc = accuracy_score_difference(
        fixture["y_true"], fixture["y_pred_enc"],
        sensitive_features=fixture["sensitive_features"],
    )
    assert math.isclose(plain, enc, abs_tol=_TOL)


def test_zero_one_loss_difference_and_ratio_match_plaintext(fixture):
    plain_d = zero_one_loss_difference(
        fixture["y_true"], fixture["y_pred"],
        sensitive_features=fixture["sensitive_features"],
    )
    enc_d = zero_one_loss_difference(
        fixture["y_true"], fixture["y_pred_enc"],
        sensitive_features=fixture["sensitive_features"],
    )
    plain_r = zero_one_loss_ratio(
        fixture["y_true"], fixture["y_pred"],
        sensitive_features=fixture["sensitive_features"],
    )
    enc_r = zero_one_loss_ratio(
        fixture["y_true"], fixture["y_pred_enc"],
        sensitive_features=fixture["sensitive_features"],
    )
    assert math.isclose(plain_d, enc_d, abs_tol=_TOL)
    assert math.isclose(plain_r, enc_r, abs_tol=_TOL)


# ---------------------------------------------------------------------------
# Regression disaggregations
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def regression_fixture():
    rng = np.random.default_rng(11)
    n = 256
    y_true = rng.normal(0.5, 0.2, size=n).astype(float)
    y_pred = y_true + rng.normal(0.0, 0.05, size=n)
    sf = rng.choice(["A", "B", "C"], size=n).astype(object)
    # Bias group A's residuals upward.
    y_pred[sf == "A"] += 0.05
    ctx = build_context(backend="tenseal")
    yp_enc = encrypt(ctx, y_pred)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_enc": yp_enc,
        "sensitive_features": sf,
    }


def test_mean_squared_error_group_max_matches_plaintext(regression_fixture):
    plain = mean_squared_error_group_max(
        regression_fixture["y_true"], regression_fixture["y_pred"],
        sensitive_features=regression_fixture["sensitive_features"],
    )
    enc = mean_squared_error_group_max(
        regression_fixture["y_true"], regression_fixture["y_pred_enc"],
        sensitive_features=regression_fixture["sensitive_features"],
    )
    assert math.isclose(plain, enc, abs_tol=_TOL)


def test_mean_absolute_error_group_max_within_jensen_bound(regression_fixture):
    plain = mean_absolute_error_group_max(
        regression_fixture["y_true"], regression_fixture["y_pred"],
        sensitive_features=regression_fixture["sensitive_features"],
    )
    enc = mean_absolute_error_group_max(
        regression_fixture["y_true"], regression_fixture["y_pred_enc"],
        sensitive_features=regression_fixture["sensitive_features"],
    )
    # The encrypted MAE is sqrt(MSE), an upper bound on the true MAE.
    # Assert it's not below the plaintext value (within CKKS noise).
    assert enc + _TOL >= plain


def test_r2_score_group_min_matches_plaintext(regression_fixture):
    plain = r2_score_group_min(
        regression_fixture["y_true"], regression_fixture["y_pred"],
        sensitive_features=regression_fixture["sensitive_features"],
    )
    enc = r2_score_group_min(
        regression_fixture["y_true"], regression_fixture["y_pred_enc"],
        sensitive_features=regression_fixture["sensitive_features"],
    )
    assert math.isclose(plain, enc, abs_tol=_TOL)


# ---------------------------------------------------------------------------
# pos_label support in selection_rate
# ---------------------------------------------------------------------------


def test_selection_rate_pos_label_zero_matches_inverse(fixture):
    enc_rate_one = selection_rate(
        fixture["y_true"], fixture["y_pred_enc"], pos_label=1
    )
    enc_rate_zero = selection_rate(
        fixture["y_true"], fixture["y_pred_enc"], pos_label=0
    )
    assert math.isclose(enc_rate_one + enc_rate_zero, 1.0, abs_tol=_TOL)


def test_selection_rate_unsupported_pos_label_raises(fixture):
    with pytest.raises(NotImplementedError, match="pos_label"):
        selection_rate(
            fixture["y_true"], fixture["y_pred_enc"], pos_label=2
        )


# ---------------------------------------------------------------------------
# Plaintext fallthrough — encrypted path is never taken
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fn",
    [
        selection_rate_difference,
        true_positive_rate_difference,
        accuracy_score_group_min,
        zero_one_loss_group_max,
        mean_squared_error_group_max,
    ],
)
def test_plaintext_fallthrough_does_not_use_encrypted_path(fixture, fn):
    """When ``y_pred`` is plaintext the function delegates to Fairlearn
    and does not require a CKKS context."""
    flfhe.reset_op_counters()
    if fn is mean_squared_error_group_max:
        rng = np.random.default_rng(0)
        y_true = rng.normal(0, 1, 64)
        y_pred = y_true + rng.normal(0, 0.1, 64)
        sf = rng.choice(["a", "b"], size=64)
        fn(y_true, y_pred, sensitive_features=sf)
    else:
        fn(
            fixture["y_true"], fixture["y_pred"],
            sensitive_features=fixture["sensitive_features"],
        )
    counters = flfhe.snapshot_op_counters()
    # No ciphertext op should have run.
    assert sum(counters.values()) == 0
