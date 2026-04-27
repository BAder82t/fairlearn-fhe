"""Per-rate ``_difference`` / ``_ratio`` family.

Each function below mirrors the upstream Fairlearn helper of the same
name (e.g. ``selection_rate_difference``, ``true_positive_rate_ratio``).
For plaintext ``y_pred`` we delegate to Fairlearn so the API surface is
truly drop-in. For encrypted ``y_pred`` we run the corresponding
ciphertext circuit, reuse the existing ``aggregate_difference`` /
``aggregate_ratio`` helpers, and return a single float verdict.

These helpers are thin wrappers — the heavy lifting (per-group
selection rate, confusion rates) is already in
:mod:`fairlearn_fhe._circuits`. This module exists so users can pick
the metric they need by name without having to know how to compose the
primitives.
"""

from __future__ import annotations

from typing import Literal

import fairlearn.metrics as _fl
import numpy as np

from .._circuits import (
    aggregate_difference,
    aggregate_ratio,
    confusion_rates_per_group,
    positive_negative_counts,
    selection_rate_per_group,
)
from .._groups import EncryptedMaskSet, group_masks
from ..encrypted import EncryptedVector

_RATE = Literal["tpr", "tnr", "fpr", "fnr"]

_FAIRLEARN_RATE_NAME = {
    "tpr": "true_positive_rate",
    "tnr": "true_negative_rate",
    "fpr": "false_positive_rate",
    "fnr": "false_negative_rate",
}


def _sw(sample_weight):
    return None if sample_weight is None else np.asarray(sample_weight, dtype=float)


def _is_encrypted(x) -> bool:
    return isinstance(x, EncryptedVector)


def _all_ones_mask(n: int) -> dict:
    return {None: np.ones(n, dtype=float)}


def _resolve_masks(sensitive_features):
    if isinstance(sensitive_features, EncryptedMaskSet):
        return list(sensitive_features.labels), sensitive_features
    return group_masks(sensitive_features)


def _pos_neg(sensitive_features, y_true, sw):
    if isinstance(sensitive_features, EncryptedMaskSet):
        return sensitive_features.positives, sensitive_features.negatives
    _labels, masks = group_masks(sensitive_features)
    return positive_negative_counts(y_true, masks, sample_weight=sw)


def _needs_encrypted(y_pred, sensitive_features) -> bool:
    if isinstance(sensitive_features, EncryptedMaskSet):
        if not _is_encrypted(y_pred):
            raise TypeError(
                "encrypted sensitive_features require an encrypted y_pred."
            )
        return True
    return _is_encrypted(y_pred)


def _conf_rates_dict(y_true, y_pred_enc: EncryptedVector, sensitive_features, sw):
    """Per-group {tpr, fpr, tnr, fnr} + overall rates dict."""
    _labels, masks = _resolve_masks(sensitive_features)
    pos, neg = _pos_neg(sensitive_features, y_true, sw)
    rates = confusion_rates_per_group(
        y_true,
        y_pred_enc,
        masks,
        sample_weight=sw,
        positives_per_group=pos,
        negatives_per_group=neg,
    )
    overall = next(
        iter(
            confusion_rates_per_group(
                y_true,
                y_pred_enc,
                _all_ones_mask(y_pred_enc.n),
                sample_weight=sw,
            ).values()
        )
    )
    return rates, overall


def _selection_rates_dict(y_pred_enc, sensitive_features, sw):
    _labels, masks = _resolve_masks(sensitive_features)
    rates = selection_rate_per_group(y_pred_enc, masks, sample_weight=sw)
    overall = next(
        iter(
            selection_rate_per_group(
                y_pred_enc, _all_ones_mask(y_pred_enc.n), sample_weight=sw
            ).values()
        )
    )
    return rates, overall


# ---------------------------------------------------------------------------
# selection_rate
# ---------------------------------------------------------------------------


def selection_rate_difference(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
) -> float:
    if not _needs_encrypted(y_pred, sensitive_features):
        return _fl.selection_rate_difference(
            y_true,
            y_pred,
            sensitive_features=sensitive_features,
            method=method,
            sample_weight=sample_weight,
        )
    sw = _sw(sample_weight)
    rates, overall = _selection_rates_dict(y_pred, sensitive_features, sw)
    return aggregate_difference(list(rates.values()), method=method, overall=overall)


def selection_rate_ratio(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
) -> float:
    if not _needs_encrypted(y_pred, sensitive_features):
        return _fl.selection_rate_ratio(
            y_true,
            y_pred,
            sensitive_features=sensitive_features,
            method=method,
            sample_weight=sample_weight,
        )
    sw = _sw(sample_weight)
    rates, overall = _selection_rates_dict(y_pred, sensitive_features, sw)
    return aggregate_ratio(list(rates.values()), method=method, overall=overall)


# ---------------------------------------------------------------------------
# Confusion-matrix rates: tpr/tnr/fpr/fnr × difference/ratio
# ---------------------------------------------------------------------------


def _conf_diff(rate: _RATE):
    fl_name = _FAIRLEARN_RATE_NAME[rate]

    def _impl(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method: Literal["between_groups", "to_overall"] = "between_groups",
        sample_weight=None,
    ) -> float:
        upstream = getattr(_fl, f"{fl_name}_difference", None)
        if not _needs_encrypted(y_pred, sensitive_features):
            if upstream is None:
                # Older Fairlearn versions lack this exact helper; fall
                # back to the plaintext compute path.
                return _plaintext_conf_aggregate(
                    y_true, y_pred, sensitive_features,
                    rate=rate, sample_weight=sample_weight,
                    method=method, kind="difference",
                )
            return upstream(
                y_true,
                y_pred,
                sensitive_features=sensitive_features,
                method=method,
                sample_weight=sample_weight,
            )
        sw = _sw(sample_weight)
        per_group, overall = _conf_rates_dict(y_true, y_pred, sensitive_features, sw)
        return aggregate_difference(
            [r[rate] for r in per_group.values()],
            method=method,
            overall=overall[rate],
        )

    _impl.__name__ = f"{fl_name}_difference"
    return _impl


def _conf_ratio(rate: _RATE):
    fl_name = _FAIRLEARN_RATE_NAME[rate]

    def _impl(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method: Literal["between_groups", "to_overall"] = "between_groups",
        sample_weight=None,
    ) -> float:
        upstream = getattr(_fl, f"{fl_name}_ratio", None)
        if not _needs_encrypted(y_pred, sensitive_features):
            if upstream is None:
                return _plaintext_conf_aggregate(
                    y_true, y_pred, sensitive_features,
                    rate=rate, sample_weight=sample_weight,
                    method=method, kind="ratio",
                )
            return upstream(
                y_true,
                y_pred,
                sensitive_features=sensitive_features,
                method=method,
                sample_weight=sample_weight,
            )
        sw = _sw(sample_weight)
        per_group, overall = _conf_rates_dict(y_true, y_pred, sensitive_features, sw)
        return aggregate_ratio(
            [r[rate] for r in per_group.values()],
            method=method,
            overall=overall[rate],
        )

    _impl.__name__ = f"{fl_name}_ratio"
    return _impl


def _plaintext_conf_aggregate(
    y_true, y_pred, sensitive_features, *, rate, sample_weight, method, kind
):
    """Plaintext fallback used when upstream Fairlearn lacks the helper.

    Computes per-group {tpr, fpr, tnr, fnr} on numpy arrays and routes
    to ``aggregate_difference`` / ``aggregate_ratio`` so the verdict
    matches what the encrypted path would produce.
    """
    y = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    sw = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    _labels, masks = group_masks(sensitive_features)
    per_group = {}
    for lbl, mask in masks.items():
        m = mask * sw
        m_pos = m * y
        m_neg = m * (1.0 - y)
        n_pos = float(m_pos.sum())
        n_neg = float(m_neg.sum())
        tp = float((m_pos * yp).sum())
        fp = float((m_neg * yp).sum())
        per_group[lbl] = {
            "tpr": tp / n_pos if n_pos > 0 else 0.0,
            "fnr": (n_pos - tp) / n_pos if n_pos > 0 else 0.0,
            "fpr": fp / n_neg if n_neg > 0 else 0.0,
            "tnr": (n_neg - fp) / n_neg if n_neg > 0 else 0.0,
        }
    n_pos = float((y * sw).sum())
    n_neg = float(((1.0 - y) * sw).sum())
    tp = float((y * sw * yp).sum())
    fp = float(((1.0 - y) * sw * yp).sum())
    overall = {
        "tpr": tp / n_pos if n_pos > 0 else 0.0,
        "fnr": (n_pos - tp) / n_pos if n_pos > 0 else 0.0,
        "fpr": fp / n_neg if n_neg > 0 else 0.0,
        "tnr": (n_neg - fp) / n_neg if n_neg > 0 else 0.0,
    }
    values = [r[rate] for r in per_group.values()]
    if kind == "difference":
        return aggregate_difference(values, method=method, overall=overall[rate])
    return aggregate_ratio(values, method=method, overall=overall[rate])


true_positive_rate_difference = _conf_diff("tpr")
true_positive_rate_ratio = _conf_ratio("tpr")
true_negative_rate_difference = _conf_diff("tnr")
true_negative_rate_ratio = _conf_ratio("tnr")
false_positive_rate_difference = _conf_diff("fpr")
false_positive_rate_ratio = _conf_ratio("fpr")
false_negative_rate_difference = _conf_diff("fnr")
false_negative_rate_ratio = _conf_ratio("fnr")
