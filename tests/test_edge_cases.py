"""Edge cases: multi-column sensitive features, NaN, single group, large n,
sample weights, empty group, all-same-class.
"""

import numpy as np
import pandas as pd
import pytest

import fairlearn.metrics as fl

from fairlearn_fhe import build_context, encrypt
from fairlearn_fhe.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
    selection_rate,
    true_positive_rate,
    MetricFrame,
)


def test_multi_column_sensitive_features(ctx, tol):
    rng = np.random.default_rng(11)
    n = 200
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sf = pd.DataFrame({
        "race": rng.choice(["X", "Y"], size=n),
        "sex": rng.choice(["M", "F"], size=n),
    })
    enc = encrypt(ctx, y_pred)
    plain = fl.demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
    fhe = demographic_parity_difference(y_true, enc, sensitive_features=sf)
    assert abs(plain - fhe) < tol


def test_single_group(ctx, tol):
    rng = np.random.default_rng(12)
    n = 100
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sf = np.zeros(n, dtype=int)  # one group only
    enc = encrypt(ctx, y_pred)
    fhe = demographic_parity_difference(y_true, enc, sensitive_features=sf)
    assert abs(fhe) < tol  # disparity = 0 by construction


def test_sample_weight_passthrough(ctx, tol):
    rng = np.random.default_rng(13)
    n = 150
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sf = rng.choice(["A", "B"], size=n)
    sw = rng.uniform(0.1, 2.0, size=n)
    enc = encrypt(ctx, y_pred)
    plain = fl.demographic_parity_difference(
        y_true, y_pred, sensitive_features=sf, sample_weight=sw,
    )
    fhe = demographic_parity_difference(
        y_true, enc, sensitive_features=sf, sample_weight=sw,
    )
    assert abs(plain - fhe) < tol


def test_large_n(ctx, tol):
    rng = np.random.default_rng(14)
    n = 2048
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sf = rng.choice(["A", "B", "C"], size=n)
    enc = encrypt(ctx, y_pred)
    plain = fl.demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
    fhe = demographic_parity_difference(y_true, enc, sensitive_features=sf)
    assert abs(plain - fhe) < tol


def test_all_positive(ctx, tol):
    """y_true all 1 → no negatives → fpr/tnr undefined; tpr should still match."""
    rng = np.random.default_rng(15)
    n = 100
    y_true = np.ones(n, dtype=float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sf = rng.choice(["A", "B"], size=n)
    enc = encrypt(ctx, y_pred)
    plain = fl.MetricFrame(
        metrics=fl.true_positive_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sf,
    )
    fhe = MetricFrame(
        metrics=fl.true_positive_rate, y_true=y_true, y_pred=enc, sensitive_features=sf,
    )
    for label in plain.by_group.index:
        p = float(plain.by_group.loc[label])
        e = float(fhe.by_group.loc[label].iloc[0])
        assert abs(p - e) < tol


def test_pos_label_not_one_raises(ctx):
    rng = np.random.default_rng(16)
    n = 50
    y_pred = rng.integers(0, 2, size=n).astype(float)
    enc = encrypt(ctx, y_pred)
    with pytest.raises(NotImplementedError):
        selection_rate(np.zeros(n), enc, pos_label=2)


def test_floating_predictions(ctx, tol):
    """Probabilistic predictions in [0, 1] — selection_rate becomes mean prob."""
    rng = np.random.default_rng(17)
    n = 200
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.uniform(0.0, 1.0, size=n)
    sf = rng.choice(["A", "B"], size=n)
    enc = encrypt(ctx, y_pred)
    # Use mean_prediction for soft preds (selection_rate plaintext requires {0,1}).
    from fairlearn_fhe.metrics import mean_prediction
    plain = float(np.mean(y_pred))
    fhe = mean_prediction(y_true, enc)
    assert abs(plain - fhe) < tol


def test_pandas_series_sensitive_features(ctx, tol):
    rng = np.random.default_rng(18)
    n = 100
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sf = pd.Series(rng.choice(["A", "B", "C"], size=n), name="race")
    enc = encrypt(ctx, y_pred)
    plain = fl.demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
    fhe = demographic_parity_difference(y_true, enc, sensitive_features=sf)
    assert abs(plain - fhe) < tol


def test_passthrough_when_plaintext():
    """Plaintext y_pred uses Fairlearn directly — exact equality."""
    rng = np.random.default_rng(19)
    n = 100
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sf = rng.choice(["A", "B"], size=n)
    p1 = fl.demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
    p2 = demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
    assert p1 == p2
