"""Encrypted base-metric outputs match plaintext Fairlearn within CKKS noise."""

import numpy as np
import pytest

import fairlearn.metrics as fl

from fairlearn_fhe.metrics import (
    selection_rate, mean_prediction,
    true_positive_rate, true_negative_rate,
    false_positive_rate, false_negative_rate,
    count,
)


def test_selection_rate(small_dataset, encrypted_pred, tol):
    y_true, y_pred, _ = small_dataset
    plain = fl.selection_rate(y_true, y_pred)
    enc = selection_rate(y_true, encrypted_pred)
    assert abs(plain - enc) < tol


def test_mean_prediction(small_dataset, encrypted_pred, tol):
    y_true, y_pred, _ = small_dataset
    plain = fl.mean_prediction(y_true, y_pred)
    enc = mean_prediction(y_true, encrypted_pred)
    assert abs(plain - enc) < tol


@pytest.mark.parametrize(
    "fn,fn_enc",
    [
        (fl.true_positive_rate, true_positive_rate),
        (fl.true_negative_rate, true_negative_rate),
        (fl.false_positive_rate, false_positive_rate),
        (fl.false_negative_rate, false_negative_rate),
    ],
)
def test_confusion_rates(small_dataset, encrypted_pred, tol, fn, fn_enc):
    y_true, y_pred, _ = small_dataset
    plain = fn(y_true, y_pred)
    enc = fn_enc(y_true, encrypted_pred)
    assert abs(plain - enc) < tol


def test_count(small_dataset, encrypted_pred):
    y_true, y_pred, _ = small_dataset
    assert count(y_true, encrypted_pred) == len(y_true)
    assert count(y_true, y_pred) == len(y_true)


def test_passthrough_plaintext(small_dataset):
    y_true, y_pred, _ = small_dataset
    # Ensure plaintext path is byte-equal to Fairlearn (no FHE error).
    assert selection_rate(y_true, y_pred) == fl.selection_rate(y_true, y_pred)
    assert true_positive_rate(y_true, y_pred) == fl.true_positive_rate(y_true, y_pred)
