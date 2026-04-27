"""Encrypted-aware make_derived_metric + group_min/group_max."""

import numpy as np
import pytest

import fairlearn.metrics as fl

from fairlearn_fhe.metrics import (
    make_derived_metric,
    MetricFrame,
    selection_rate,
)


def test_derived_difference(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    m_plain = fl.make_derived_metric(metric=fl.selection_rate, transform="difference")
    m_fhe = make_derived_metric(metric=fl.selection_rate, transform="difference")
    p = m_plain(y_true, y_pred, sensitive_features=sf)
    f = m_fhe(y_true, encrypted_pred, sensitive_features=sf)
    assert abs(p - f) < tol


def test_derived_ratio(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    m_plain = fl.make_derived_metric(metric=fl.selection_rate, transform="ratio")
    m_fhe = make_derived_metric(metric=fl.selection_rate, transform="ratio")
    p = m_plain(y_true, y_pred, sensitive_features=sf)
    f = m_fhe(y_true, encrypted_pred, sensitive_features=sf)
    assert abs(p - f) < tol


def test_derived_group_min(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    m_plain = fl.make_derived_metric(metric=fl.selection_rate, transform="group_min")
    m_fhe = make_derived_metric(metric=fl.selection_rate, transform="group_min")
    p = m_plain(y_true, y_pred, sensitive_features=sf)
    f = m_fhe(y_true, encrypted_pred, sensitive_features=sf)
    assert abs(p - f) < tol


def test_derived_group_max(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    m_plain = fl.make_derived_metric(metric=fl.selection_rate, transform="group_max")
    m_fhe = make_derived_metric(metric=fl.selection_rate, transform="group_max")
    p = m_plain(y_true, y_pred, sensitive_features=sf)
    f = m_fhe(y_true, encrypted_pred, sensitive_features=sf)
    assert abs(p - f) < tol


def test_metric_frame_group_min_max(small_dataset, encrypted_pred, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.MetricFrame(metrics=fl.selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sf)
    enc = MetricFrame(metrics=fl.selection_rate, y_true=y_true, y_pred=encrypted_pred, sensitive_features=sf)
    assert abs(float(plain.group_min()) - float(enc.group_min())) < tol
    assert abs(float(plain.group_max()) - float(enc.group_max())) < tol


def test_invalid_transform():
    with pytest.raises(ValueError):
        make_derived_metric(metric=fl.selection_rate, transform="median")


def test_method_arg_rejected():
    def bad(y_true, y_pred, method=None):
        return 0.0
    with pytest.raises(ValueError):
        make_derived_metric(metric=bad, transform="difference")


def test_passthrough_plaintext(small_dataset):
    """Plaintext y_pred → exact equality with Fairlearn."""
    y_true, y_pred, sf = small_dataset
    m_plain = fl.make_derived_metric(metric=fl.selection_rate, transform="difference")
    m_fhe = make_derived_metric(metric=fl.selection_rate, transform="difference")
    assert m_plain(y_true, y_pred, sensitive_features=sf) == m_fhe(y_true, y_pred, sensitive_features=sf)
