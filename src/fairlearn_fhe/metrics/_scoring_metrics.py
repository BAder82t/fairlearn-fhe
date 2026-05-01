"""Group-disaggregated classification scores under encryption.

Ports the upstream Fairlearn ``*_group_min``, ``*_difference``, and
``*_ratio`` helpers for accuracy / precision / recall / f1 /
balanced_accuracy / zero_one_loss. All of these reduce to the
ciphertext-friendly TP/FP/TN/FN counts already produced by
:func:`fairlearn_fhe._circuits.confusion_rates_per_group` plus a few
extra ciphertext sums for accuracy.

The encrypted path computes per-group raw counts, decrypts at the audit
boundary, then aggregates as plaintext scalars — matching the
``selection_rate_difference`` style of the rest of the package.

All zero-division guards mirror Fairlearn semantics: undefined rates
collapse to ``0`` rather than ``NaN``.
"""

from __future__ import annotations

from typing import Any, Literal

import fairlearn.metrics as _fl
import numpy as np

from .._circuits import (
    _safe_div,
    aggregate_difference,
    aggregate_ratio,
    positive_negative_counts,
)
from .._groups import EncryptedMaskSet, group_masks
from ..encrypted import EncryptedVector


def _is_encrypted(x) -> bool:
    return isinstance(x, EncryptedVector)


def _sw(sample_weight):
    return None if sample_weight is None else np.asarray(sample_weight, dtype=float)


def _needs_encrypted(y_pred, sensitive_features) -> bool:
    if isinstance(sensitive_features, EncryptedMaskSet):
        if not _is_encrypted(y_pred):
            raise TypeError(
                "encrypted sensitive_features require an encrypted y_pred."
            )
        return True
    return _is_encrypted(y_pred)


def _per_group_confusion_counts(
    y_true: np.ndarray,
    y_pred_enc: EncryptedVector,
    sensitive_features: Any,
    sample_weight: np.ndarray | None,
) -> dict[Any, dict[str, float]]:
    """Per-group ``{tp, fp, tn, fn, n_pos, n_neg, n_total}`` raw counts.

    Different scoring metrics consume different combinations of these
    counts; computing them once and reusing avoids redundant ciphertext
    sums (each tp/fp pair costs one ct×pt + sum_all).
    """
    y = np.asarray(y_true, dtype=float)
    sw = np.ones_like(y) if sample_weight is None else sample_weight

    if isinstance(sensitive_features, EncryptedMaskSet):
        labels = list(sensitive_features.labels)
        get_mask = lambda lbl: sensitive_features.masks[lbl]  # noqa: E731
        is_encrypted_mask = True
        positives = sensitive_features.positives
        negatives = sensitive_features.negatives
    else:
        _labels, plain_masks = group_masks(sensitive_features)
        labels = list(plain_masks.keys())
        get_mask = lambda lbl: plain_masks[lbl] * sw  # noqa: E731
        is_encrypted_mask = False
        positives, negatives = positive_negative_counts(
            y_true, plain_masks, sample_weight=sw
        )

    out: dict[Any, dict[str, float]] = {}
    for lbl in labels:
        if is_encrypted_mask:
            mask_obj = get_mask(lbl)
            n_pos = positives[lbl]
            n_neg = negatives[lbl]
            tp = float(
                y_pred_enc.mul_ct(mask_obj).mul_pt(y * sw).sum_all().first_slot()
            )
            fp = float(
                y_pred_enc.mul_ct(mask_obj)
                .mul_pt((1.0 - y) * sw)
                .sum_all()
                .first_slot()
            )
        else:
            m = get_mask(lbl)
            m_pos = m * y
            m_neg = m * (1.0 - y)
            n_pos = float(m_pos.sum())
            n_neg = float(m_neg.sum())
            tp = float(y_pred_enc.mul_pt(m_pos).sum_all().first_slot())
            fp = float(y_pred_enc.mul_pt(m_neg).sum_all().first_slot())
        fn = max(n_pos - tp, 0.0)
        tn = max(n_neg - fp, 0.0)
        out[lbl] = {
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_total": n_pos + n_neg,
        }
    return out


# ---------------------------------------------------------------------------
# Per-group score functions (act on the raw counts dict)
# ---------------------------------------------------------------------------


def _accuracy(counts: dict[str, float]) -> float:
    return _safe_div(counts["tp"] + counts["tn"], counts["n_total"])


def _balanced_accuracy(counts: dict[str, float]) -> float:
    tpr = _safe_div(counts["tp"], counts["n_pos"])
    tnr = _safe_div(counts["tn"], counts["n_neg"])
    return 0.5 * (tpr + tnr)


def _precision(counts: dict[str, float]) -> float:
    return _safe_div(counts["tp"], counts["tp"] + counts["fp"], clip_upper=1.0)


def _recall(counts: dict[str, float]) -> float:
    return _safe_div(counts["tp"], counts["n_pos"])


def _f1(counts: dict[str, float]) -> float:
    p = _precision(counts)
    r = _recall(counts)
    if p + r <= 0:
        return 0.0
    return float(2 * p * r / (p + r))


def _zero_one_loss(counts: dict[str, float]) -> float:
    if counts["n_total"] <= 0:
        return 0.0
    return float((counts["fp"] + counts["fn"]) / counts["n_total"])


_SCORERS = {
    "accuracy_score": _accuracy,
    "balanced_accuracy_score": _balanced_accuracy,
    "precision_score": _precision,
    "recall_score": _recall,
    "f1_score": _f1,
    "zero_one_loss": _zero_one_loss,
}


def _plaintext_scorer_aggregate(
    y_true,
    y_pred,
    sensitive_features,
    *,
    scorer_name: str,
    sample_weight,
    reduction: str,
    method: str = "between_groups",
) -> float:
    """Plaintext fallback for older fairlearn versions lacking a helper.

    Computes per-group {tp, fp, tn, fn, n_*} on numpy arrays and routes
    through the same scorer + aggregation used by the encrypted path so
    that verdicts are consistent across fairlearn versions.
    """
    y = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    sw = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    _labels, plain_masks = group_masks(sensitive_features)
    scorer = _SCORERS[scorer_name]

    per_group_values: list[float] = []
    for _lbl, mask in plain_masks.items():
        m = mask * sw
        m_pos = m * y
        m_neg = m * (1.0 - y)
        n_pos = float(m_pos.sum())
        n_neg = float(m_neg.sum())
        tp = float((m_pos * yp).sum())
        fp = float((m_neg * yp).sum())
        per_group_values.append(scorer({
            "tp": tp,
            "fp": fp,
            "tn": max(n_neg - fp, 0.0),
            "fn": max(n_pos - tp, 0.0),
            "n_pos": n_pos,
            "n_neg": n_neg,
            "n_total": n_pos + n_neg,
        }))

    if reduction == "min":
        return float(min(per_group_values))
    if reduction == "max":
        return float(max(per_group_values))

    n_pos = float((y * sw).sum())
    n_neg = float(((1.0 - y) * sw).sum())
    tp = float((y * sw * yp).sum())
    fp = float(((1.0 - y) * sw * yp).sum())
    overall = scorer({
        "tp": tp,
        "fp": fp,
        "tn": max(n_neg - fp, 0.0),
        "fn": max(n_pos - tp, 0.0),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_total": n_pos + n_neg,
    })
    if reduction == "difference":
        return aggregate_difference(per_group_values, method=method, overall=overall)
    if reduction == "ratio":
        return aggregate_ratio(per_group_values, method=method, overall=overall)
    raise ValueError(f"unknown reduction {reduction!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _group_min(scorer_name: str):
    upstream = getattr(_fl, f"{scorer_name}_group_min", None)
    reduction = "min"

    def _impl(
        y_true,
        y_pred,
        *,
        sensitive_features,
        sample_weight=None,
    ) -> float:
        if not _needs_encrypted(y_pred, sensitive_features):
            if upstream is None:
                return _plaintext_scorer_aggregate(
                    y_true, y_pred, sensitive_features,
                    scorer_name=scorer_name,
                    sample_weight=sample_weight,
                    reduction=reduction,
                )
            return upstream(
                y_true,
                y_pred,
                sensitive_features=sensitive_features,
                sample_weight=sample_weight,
            )
        sw = _sw(sample_weight)
        per_group = _per_group_confusion_counts(y_true, y_pred, sensitive_features, sw)
        scorer = _SCORERS[scorer_name]
        return float(min(scorer(c) for c in per_group.values()))

    _impl.__name__ = f"{scorer_name}_group_min"
    return _impl


def _group_max(scorer_name: str):
    upstream = getattr(_fl, f"{scorer_name}_group_max", None)
    reduction = "max"

    def _impl(
        y_true,
        y_pred,
        *,
        sensitive_features,
        sample_weight=None,
    ) -> float:
        if not _needs_encrypted(y_pred, sensitive_features):
            if upstream is None:
                return _plaintext_scorer_aggregate(
                    y_true, y_pred, sensitive_features,
                    scorer_name=scorer_name,
                    sample_weight=sample_weight,
                    reduction=reduction,
                )
            return upstream(
                y_true,
                y_pred,
                sensitive_features=sensitive_features,
                sample_weight=sample_weight,
            )
        sw = _sw(sample_weight)
        per_group = _per_group_confusion_counts(y_true, y_pred, sensitive_features, sw)
        scorer = _SCORERS[scorer_name]
        return float(max(scorer(c) for c in per_group.values()))

    _impl.__name__ = f"{scorer_name}_group_max"
    return _impl


def _group_difference(scorer_name: str):
    upstream = getattr(_fl, f"{scorer_name}_difference", None)
    reduction = "difference"

    def _impl(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method: Literal["between_groups", "to_overall"] = "between_groups",
        sample_weight=None,
    ) -> float:
        if not _needs_encrypted(y_pred, sensitive_features):
            if upstream is None:
                return _plaintext_scorer_aggregate(
                    y_true, y_pred, sensitive_features,
                    scorer_name=scorer_name,
                    sample_weight=sample_weight,
                    reduction=reduction,
                    method=method,
                )
            return upstream(
                y_true,
                y_pred,
                sensitive_features=sensitive_features,
                method=method,
                sample_weight=sample_weight,
            )
        sw = _sw(sample_weight)
        per_group = _per_group_confusion_counts(y_true, y_pred, sensitive_features, sw)
        scorer = _SCORERS[scorer_name]
        values = [scorer(c) for c in per_group.values()]
        # Overall = score on the union of all groups (= no group filter).
        sw_arr = (
            np.ones(len(y_true), dtype=float)
            if sw is None
            else sw
        )
        y = np.asarray(y_true, dtype=float)
        n_pos = float((y * sw_arr).sum())
        n_neg = float(((1.0 - y) * sw_arr).sum())
        tp = float(y_pred.mul_pt(y * sw_arr).sum_all().first_slot())
        fp = float(y_pred.mul_pt((1.0 - y) * sw_arr).sum_all().first_slot())
        overall = scorer(
            {
                "tp": tp,
                "fp": fp,
                "tn": max(n_neg - fp, 0.0),
                "fn": max(n_pos - tp, 0.0),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_total": n_pos + n_neg,
            }
        )
        return aggregate_difference(values, method=method, overall=overall)

    _impl.__name__ = f"{scorer_name}_difference"
    return _impl


def _group_ratio(scorer_name: str):
    upstream = getattr(_fl, f"{scorer_name}_ratio", None)
    reduction = "ratio"

    def _impl(
        y_true,
        y_pred,
        *,
        sensitive_features,
        method: Literal["between_groups", "to_overall"] = "between_groups",
        sample_weight=None,
    ) -> float:
        if not _needs_encrypted(y_pred, sensitive_features):
            if upstream is None:
                return _plaintext_scorer_aggregate(
                    y_true, y_pred, sensitive_features,
                    scorer_name=scorer_name,
                    sample_weight=sample_weight,
                    reduction=reduction,
                    method=method,
                )
            return upstream(
                y_true,
                y_pred,
                sensitive_features=sensitive_features,
                method=method,
                sample_weight=sample_weight,
            )
        sw = _sw(sample_weight)
        per_group = _per_group_confusion_counts(y_true, y_pred, sensitive_features, sw)
        scorer = _SCORERS[scorer_name]
        values = [scorer(c) for c in per_group.values()]
        sw_arr = (
            np.ones(len(y_true), dtype=float)
            if sw is None
            else sw
        )
        y = np.asarray(y_true, dtype=float)
        n_pos = float((y * sw_arr).sum())
        n_neg = float(((1.0 - y) * sw_arr).sum())
        tp = float(y_pred.mul_pt(y * sw_arr).sum_all().first_slot())
        fp = float(y_pred.mul_pt((1.0 - y) * sw_arr).sum_all().first_slot())
        overall = scorer(
            {
                "tp": tp,
                "fp": fp,
                "tn": max(n_neg - fp, 0.0),
                "fn": max(n_pos - tp, 0.0),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "n_total": n_pos + n_neg,
            }
        )
        return aggregate_ratio(values, method=method, overall=overall)

    _impl.__name__ = f"{scorer_name}_ratio"
    return _impl


# Group-min / group-max
accuracy_score_group_min = _group_min("accuracy_score")
balanced_accuracy_score_group_min = _group_min("balanced_accuracy_score")
precision_score_group_min = _group_min("precision_score")
recall_score_group_min = _group_min("recall_score")
f1_score_group_min = _group_min("f1_score")
zero_one_loss_group_max = _group_max("zero_one_loss")

# Differences / ratios for the rest
accuracy_score_difference = _group_difference("accuracy_score")
zero_one_loss_difference = _group_difference("zero_one_loss")
zero_one_loss_ratio = _group_ratio("zero_one_loss")
