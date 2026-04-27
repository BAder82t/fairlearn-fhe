"""Encrypted analogues of Fairlearn base metrics.

Each function dispatches: if ``y_pred`` is plaintext we delegate to
Fairlearn (preserving identical numerical behaviour); if encrypted we
run the CKKS circuit. This keeps the API truly drop-in.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import fairlearn.metrics as _fl

from ..encrypted import EncryptedVector

_EPS = 1e-12


def _is_encrypted(x) -> bool:
    return isinstance(x, EncryptedVector)


def _safe_div(num: float, den: float) -> float:
    return num / den if den > _EPS else 0.0


# ---------------------------------------------------------------------------
# selection_rate / mean_prediction — single-mask circuits
# ---------------------------------------------------------------------------


def selection_rate(y_true, y_pred, *, pos_label: Any = 1, sample_weight=None) -> float:
    if not _is_encrypted(y_pred):
        return _fl.selection_rate(y_true, y_pred, pos_label=pos_label, sample_weight=sample_weight)
    if pos_label not in (1, 1.0):
        raise NotImplementedError(
            "Encrypted selection_rate currently requires pos_label=1; "
            "encode predictions as {0, 1} or pre-translate before encrypting."
        )
    n = y_pred.n
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    denom = float(sw.sum())
    numer = float(y_pred.mul_pt(sw).sum_all().first_slot())
    return _safe_div(numer, denom)


def mean_prediction(y_true, y_pred, sample_weight=None) -> float:
    if not _is_encrypted(y_pred):
        return _fl.mean_prediction(y_true, y_pred, sample_weight=sample_weight)
    n = y_pred.n
    sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    denom = float(sw.sum())
    numer = float(y_pred.mul_pt(sw).sum_all().first_slot())
    return _safe_div(numer, denom)


# ---------------------------------------------------------------------------
# Confusion-matrix rates
# ---------------------------------------------------------------------------


def _conf_rates(y_true, y_pred_enc: EncryptedVector, sample_weight) -> dict:
    y = np.asarray(y_true, dtype=float)
    sw = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    n_pos = float((y * sw).sum())
    n_neg = float(((1.0 - y) * sw).sum())
    tp = float(y_pred_enc.mul_pt(y * sw).sum_all().first_slot())
    fp = float(y_pred_enc.mul_pt((1.0 - y) * sw).sum_all().first_slot())
    return {
        "tpr": _safe_div(tp, n_pos),
        "fnr": _safe_div(n_pos - tp, n_pos),
        "fpr": _safe_div(fp, n_neg),
        "tnr": _safe_div(n_neg - fp, n_neg),
    }


def true_positive_rate(y_true, y_pred, sample_weight=None, pos_label=None) -> float:
    if not _is_encrypted(y_pred):
        return _fl.true_positive_rate(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label)
    return _conf_rates(y_true, y_pred, sample_weight)["tpr"]


def true_negative_rate(y_true, y_pred, sample_weight=None, pos_label=None) -> float:
    if not _is_encrypted(y_pred):
        return _fl.true_negative_rate(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label)
    return _conf_rates(y_true, y_pred, sample_weight)["tnr"]


def false_positive_rate(y_true, y_pred, sample_weight=None, pos_label=None) -> float:
    if not _is_encrypted(y_pred):
        return _fl.false_positive_rate(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label)
    return _conf_rates(y_true, y_pred, sample_weight)["fpr"]


def false_negative_rate(y_true, y_pred, sample_weight=None, pos_label=None) -> float:
    if not _is_encrypted(y_pred):
        return _fl.false_negative_rate(y_true, y_pred, sample_weight=sample_weight, pos_label=pos_label)
    return _conf_rates(y_true, y_pred, sample_weight)["fnr"]


def count(y_true, y_pred) -> int:
    """Group size — independent of whether ``y_pred`` is encrypted."""
    if _is_encrypted(y_pred):
        return int(y_pred.n)
    return _fl.count(y_true, y_pred)
