"""Encrypted analogues of Fairlearn fairness metrics.

Per-group selection_rate / TPR / FPR computed under encryption; the
final aggregation (max-min, min/max, worst_case, mean) runs on the K
plaintext per-group rates after the audit boundary.
"""

from __future__ import annotations

import inspect
from typing import Any, Literal

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


def _call_fairlearn(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Call ``fn`` with only the kwargs it actually accepts.

    Some fairlearn versions accept ``agg`` on equalized-odds helpers and
    others don't. Drop unknown kwargs silently rather than crashing the
    plaintext passthrough — the encrypted path always honours ``agg``.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return fn(*args, **kwargs)
    params = sig.parameters
    accepts_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())
    if accepts_var_kw:
        return fn(*args, **kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **accepted)


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
        # Defensive: both attrs are dataclass fields so always present;
        # only fires for malformed manually-constructed instances.
        if not hasattr(sensitive_features, "positives") or not hasattr(  # pragma: no cover
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
        return _call_fairlearn(
            _fl.demographic_parity_difference,
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
        return _call_fairlearn(
            _fl.demographic_parity_ratio,
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
        return _call_fairlearn(
            _fl.equalized_odds_difference,
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
        return _call_fairlearn(
            _fl.equalized_odds_ratio,
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
        if hasattr(_fl, "equal_opportunity_difference"):
            return _call_fairlearn(
                _fl.equal_opportunity_difference,
                y_true, y_pred,
                sensitive_features=sensitive_features,
                method=method, sample_weight=sample_weight,
            )
        # Fallback for fairlearn versions that lack the helper.
        _labels, masks = group_masks(sensitive_features)
        sw = _sw(sample_weight)
        pos, neg = positive_negative_counts(y_true, masks, sample_weight=sw)
        # Plain numpy compute path (no ciphertext).
        rates = _plaintext_tpr_per_group(y_true, y_pred, masks, sw, pos)
        overall_rate = _plaintext_tpr_overall(y_true, y_pred, sw)
        return aggregate_difference(list(rates.values()), method=method, overall=overall_rate)
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
        if hasattr(_fl, "equal_opportunity_ratio"):
            return _call_fairlearn(
                _fl.equal_opportunity_ratio,
                y_true, y_pred,
                sensitive_features=sensitive_features,
                method=method, sample_weight=sample_weight,
            )
        _labels, masks = group_masks(sensitive_features)
        sw = _sw(sample_weight)
        pos, neg = positive_negative_counts(y_true, masks, sample_weight=sw)
        rates = _plaintext_tpr_per_group(y_true, y_pred, masks, sw, pos)
        overall_rate = _plaintext_tpr_overall(y_true, y_pred, sw)
        return aggregate_ratio(list(rates.values()), method=method, overall=overall_rate)
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
    return None if sample_weight is None else np.asarray(sample_weight, dtype=float)


def _plaintext_tpr_per_group(y_true, y_pred, masks, sw, positives):
    y = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    weights = np.ones_like(y) if sw is None else sw
    out: dict[object, float] = {}
    for label, mask in masks.items():
        m = mask * weights
        denom = positives[label]
        if denom <= 0:
            out[label] = 0.0
        else:
            out[label] = float((m * y * yp).sum() / denom)
    return out


def _plaintext_tpr_overall(y_true, y_pred, sw) -> float:
    y = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    weights = np.ones_like(y) if sw is None else sw
    denom = float((weights * y).sum())
    if denom <= 0:
        return 0.0
    return float((weights * y * yp).sum() / denom)
