"""Encrypted analogues of Fairlearn fairness metrics.

Per-group selection_rate / TPR / FPR computed under encryption; the
final aggregation (max-min, min/max, worst_case, mean) runs on the K
plaintext per-group rates after the audit boundary.
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


def _all_ones_mask(n: int) -> dict:
    return {None: np.ones(n, dtype=float)}


def _resolve_masks(sensitive_features):
    """Return ``(labels, masks)`` for either plaintext or encrypted sf."""
    if isinstance(sensitive_features, EncryptedMaskSet):
        return list(sensitive_features.labels), sensitive_features
    return group_masks(sensitive_features)


def _pos_neg_counts(sensitive_features, y_true, sample_weight):
    """Per-group positive/negative counts.

    Required when sensitive_features are encrypted; for plaintext we
    delegate to :func:`positive_negative_counts`.
    """
    if isinstance(sensitive_features, EncryptedMaskSet):
        # Auditor-public metadata stamped on the EncryptedMaskSet.
        if not hasattr(sensitive_features, "positives") or not hasattr(
            sensitive_features, "negatives"
        ):
            raise ValueError(
                "EncryptedMaskSet does not carry positive/negative counts; "
                "use fairlearn_fhe.attach_label_counts(mask_set, y_true, sample_weight)."
            )
        return sensitive_features.positives, sensitive_features.negatives
    _labels, masks = group_masks(sensitive_features)
    return positive_negative_counts(y_true, masks, sample_weight=sample_weight)


def _is_encrypted(x) -> bool:
    return isinstance(x, EncryptedVector)


def _needs_encrypted_path(y_pred, sensitive_features) -> bool:
    if isinstance(sensitive_features, EncryptedMaskSet):
        if not _is_encrypted(y_pred):
            raise TypeError(
                "encrypted sensitive_features require an encrypted y_pred."
            )
        return True
    return _is_encrypted(y_pred)


# ---------------------------------------------------------------------------
# Demographic parity
# ---------------------------------------------------------------------------


def demographic_parity_difference(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
) -> float:
    if not _needs_encrypted_path(y_pred, sensitive_features):
        return _fl.demographic_parity_difference(
            y_true, y_pred,
            sensitive_features=sensitive_features,
            method=method, sample_weight=sample_weight,
        )
    _labels, masks = _resolve_masks(sensitive_features)
    sw = _sw(sample_weight)
    rates = selection_rate_per_group(y_pred, masks, sample_weight=sw)
    overall = next(iter(selection_rate_per_group(
        y_pred, _all_ones_mask(y_pred.n), sample_weight=sw,
    ).values()))
    return aggregate_difference(list(rates.values()), method=method, overall=overall)


def demographic_parity_ratio(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
) -> float:
    if not _needs_encrypted_path(y_pred, sensitive_features):
        return _fl.demographic_parity_ratio(
            y_true, y_pred,
            sensitive_features=sensitive_features,
            method=method, sample_weight=sample_weight,
        )
    _labels, masks = _resolve_masks(sensitive_features)
    sw = _sw(sample_weight)
    rates = selection_rate_per_group(y_pred, masks, sample_weight=sw)
    overall = next(iter(selection_rate_per_group(
        y_pred, _all_ones_mask(y_pred.n), sample_weight=sw,
    ).values()))
    return aggregate_ratio(list(rates.values()), method=method, overall=overall)


# ---------------------------------------------------------------------------
# Equalized odds (worst-case max(tpr_diff, fpr_diff) by default)
# ---------------------------------------------------------------------------


def equalized_odds_difference(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
    agg: Literal["worst_case", "mean"] = "worst_case",
) -> float:
    if agg not in ("worst_case", "mean"):
        raise ValueError(f"agg must be 'worst_case' or 'mean', got {agg!r}")
    if not _needs_encrypted_path(y_pred, sensitive_features):
        return _fl.equalized_odds_difference(
            y_true, y_pred,
            sensitive_features=sensitive_features,
            method=method, sample_weight=sample_weight, agg=agg,
        )
    _labels, masks = _resolve_masks(sensitive_features)
    sw = _sw(sample_weight)
    pos, neg = _pos_neg_counts(sensitive_features, y_true, sw)
    rates = confusion_rates_per_group(
        y_true, y_pred, masks, sample_weight=sw,
        positives_per_group=pos, negatives_per_group=neg,
    )
    overall = confusion_rates_per_group(
        y_true, y_pred, _all_ones_mask(y_pred.n), sample_weight=sw
    )
    o = next(iter(overall.values()))
    tpr_diff = aggregate_difference(
        [r["tpr"] for r in rates.values()], method=method, overall=o["tpr"]
    )
    fpr_diff = aggregate_difference(
        [r["fpr"] for r in rates.values()], method=method, overall=o["fpr"]
    )
    return max(tpr_diff, fpr_diff) if agg == "worst_case" else 0.5 * (tpr_diff + fpr_diff)


def equalized_odds_ratio(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
    agg: Literal["worst_case", "mean"] = "worst_case",
) -> float:
    if agg not in ("worst_case", "mean"):
        raise ValueError(f"agg must be 'worst_case' or 'mean', got {agg!r}")
    if not _needs_encrypted_path(y_pred, sensitive_features):
        return _fl.equalized_odds_ratio(
            y_true, y_pred,
            sensitive_features=sensitive_features,
            method=method, sample_weight=sample_weight, agg=agg,
        )
    _labels, masks = _resolve_masks(sensitive_features)
    sw = _sw(sample_weight)
    pos, neg = _pos_neg_counts(sensitive_features, y_true, sw)
    rates = confusion_rates_per_group(
        y_true, y_pred, masks, sample_weight=sw,
        positives_per_group=pos, negatives_per_group=neg,
    )
    overall = confusion_rates_per_group(y_true, y_pred, _all_ones_mask(y_pred.n), sample_weight=sw)
    o = next(iter(overall.values()))
    tpr_ratio = aggregate_ratio([r["tpr"] for r in rates.values()], method=method, overall=o["tpr"])
    fpr_ratio = aggregate_ratio([r["fpr"] for r in rates.values()], method=method, overall=o["fpr"])
    return min(tpr_ratio, fpr_ratio) if agg == "worst_case" else 0.5 * (tpr_ratio + fpr_ratio)


# ---------------------------------------------------------------------------
# Equal opportunity (TPR-only)
# ---------------------------------------------------------------------------


def equal_opportunity_difference(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
) -> float:
    if not _needs_encrypted_path(y_pred, sensitive_features):
        return _fl.equal_opportunity_difference(
            y_true, y_pred,
            sensitive_features=sensitive_features,
            method=method, sample_weight=sample_weight,
        )
    _labels, masks = _resolve_masks(sensitive_features)
    sw = _sw(sample_weight)
    pos, neg = _pos_neg_counts(sensitive_features, y_true, sw)
    rates = confusion_rates_per_group(
        y_true, y_pred, masks, sample_weight=sw,
        positives_per_group=pos, negatives_per_group=neg,
    )
    overall = confusion_rates_per_group(y_true, y_pred, _all_ones_mask(y_pred.n), sample_weight=sw)
    o = next(iter(overall.values()))
    return aggregate_difference(
        [r["tpr"] for r in rates.values()], method=method, overall=o["tpr"],
    )


def equal_opportunity_ratio(
    y_true,
    y_pred,
    *,
    sensitive_features,
    method: Literal["between_groups", "to_overall"] = "between_groups",
    sample_weight=None,
) -> float:
    if not _needs_encrypted_path(y_pred, sensitive_features):
        return _fl.equal_opportunity_ratio(
            y_true, y_pred,
            sensitive_features=sensitive_features,
            method=method, sample_weight=sample_weight,
        )
    _labels, masks = _resolve_masks(sensitive_features)
    sw = _sw(sample_weight)
    pos, neg = _pos_neg_counts(sensitive_features, y_true, sw)
    rates = confusion_rates_per_group(
        y_true, y_pred, masks, sample_weight=sw,
        positives_per_group=pos, negatives_per_group=neg,
    )
    overall = confusion_rates_per_group(y_true, y_pred, _all_ones_mask(y_pred.n), sample_weight=sw)
    o = next(iter(overall.values()))
    return aggregate_ratio(
        [r["tpr"] for r in rates.values()], method=method, overall=o["tpr"],
    )


def _sw(sample_weight):
    import numpy as np
    return None if sample_weight is None else np.asarray(sample_weight, dtype=float)
