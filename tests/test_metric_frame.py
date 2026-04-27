"""Encrypted MetricFrame mirrors fairlearn.metrics.MetricFrame."""

import numpy as np
import pytest

import fairlearn.metrics as fl

from fairlearn_fhe.metrics import MetricFrame


def test_single_metric(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.MetricFrame(metrics=fl.selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sf)
    enc = MetricFrame(metrics=fl.selection_rate, y_true=y_true, y_pred=encrypted_pred, sensitive_features=sf)

    # Per-group rates within tol.
    for label in plain.by_group.index:
        assert abs(float(plain.by_group.loc[label]) - float(enc.by_group.loc[label].iloc[0])) < tol


def test_multi_metric(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    fns = {"tpr": fl.true_positive_rate, "fpr": fl.false_positive_rate}
    plain = fl.MetricFrame(metrics=fns, y_true=y_true, y_pred=y_pred, sensitive_features=sf)
    enc = MetricFrame(metrics=fns, y_true=y_true, y_pred=encrypted_pred, sensitive_features=sf)
    for col in ("tpr", "fpr"):
        for label in plain.by_group.index:
            p = float(plain.by_group.loc[label, col])
            e = float(enc.by_group.loc[label, col])
            assert abs(p - e) < tol


def test_difference_matches(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    fns = {"tpr": fl.true_positive_rate, "fpr": fl.false_positive_rate}
    plain = fl.MetricFrame(metrics=fns, y_true=y_true, y_pred=y_pred, sensitive_features=sf)
    enc = MetricFrame(metrics=fns, y_true=y_true, y_pred=encrypted_pred, sensitive_features=sf)
    pd_diff = plain.difference()
    fd_diff = enc.difference()
    for col in ("tpr", "fpr"):
        assert abs(float(pd_diff[col]) - float(fd_diff[col])) < tol


def test_fhe_alias_rejects_plaintext(small_dataset):
    y_true, y_pred, sf = small_dataset
    with pytest.raises(TypeError):
        MetricFrame.fhe(metrics=fl.selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sf)


def test_unknown_metric_requires_optin(small_dataset, encrypted_pred):
    y_true, _, sf = small_dataset

    def custom(y_t, y_p):
        return float(np.mean(y_t == y_p))

    with pytest.raises(ValueError):
        MetricFrame(metrics=custom, y_true=y_true, y_pred=encrypted_pred, sensitive_features=sf)

    # With opt-in, falls back to decrypt+plaintext.
    enc = MetricFrame(
        metrics=custom, y_true=y_true, y_pred=encrypted_pred,
        sensitive_features=sf, allow_decrypt=True,
    )
    assert enc.by_group.shape[0] >= 1
