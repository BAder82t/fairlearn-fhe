"""Encrypted circuits backing the metric ports.

Two mask flavors:

- ``Dict[label, np.ndarray]``: plaintext masks (depth 1 per metric;
  ct×pt + sum_all).
- :class:`EncryptedMaskSet`: encrypted masks (depth 2 per metric;
  ct×ct + sum_all). Group counts revealed as plaintext metadata.

Each circuit accepts either via :func:`_iter_masks`, so the public
metric functions branch only at the top level.
"""

from __future__ import annotations

import numpy as np

from ._groups import EncryptedMaskSet
from .encrypted import EncryptedVector

_EPS = 1e-12

MaskLike = dict[object, np.ndarray] | EncryptedMaskSet


def _safe_div(num: float, den: float) -> float:
    if den <= _EPS:
        return 0.0
    return num / den


def _iter_masks(masks: MaskLike, sample_weight: np.ndarray | None):
    """Yield ``(label, mask_obj, denominator, is_encrypted_mask)``.

    For plaintext masks the denominator is ``sum(weighted_mask)``; for
    encrypted masks the denominator is the auditor-public count
    (already weighted by sample_weight if applicable — encrypted masks
    are weighted by passing the weight as a plaintext multiplier on
    the underlying ciphertext via ``mul_pt`` upstream).
    """
    if isinstance(masks, EncryptedMaskSet):
        for lbl in masks.labels:
            yield lbl, masks.masks[lbl], masks.counts[lbl], True
        return
    for lbl, mask in masks.items():
        weighted = mask if sample_weight is None else mask * sample_weight
        yield lbl, weighted, float(weighted.sum()), False


def _sum_under_mask(
    y_pred_enc: EncryptedVector,
    mask_obj,
    is_encrypted_mask: bool,
    extra_pt: np.ndarray | None = None,
) -> float:
    """Compute decrypt(sum_all(y_pred * mask * extra_pt)).first_slot.

    ``extra_pt`` is an optional plaintext multiplier (e.g. y_true,
    sample_weight) folded into one ct×pt multiply.
    """
    if is_encrypted_mask:
        product = y_pred_enc.mul_ct(mask_obj)
        if extra_pt is not None:
            product = product.mul_pt(extra_pt)
        return float(product.sum_all().first_slot())
    pt = mask_obj if extra_pt is None else mask_obj * extra_pt
    return float(y_pred_enc.mul_pt(pt).sum_all().first_slot())


def selection_rate_per_group(
    y_pred_enc: EncryptedVector,
    masks: MaskLike,
    *,
    pos_label: float = 1.0,
    sample_weight: np.ndarray | None = None,
) -> dict[object, float]:
    if pos_label != 1 and pos_label != 1.0:
        raise NotImplementedError(
            "Encrypted selection_rate currently requires pos_label=1; "
            "encode predictions as {0, 1} or pre-translate before encrypting."
        )
    out: dict[object, float] = {}
    for label, mask_obj, denom, is_enc in _iter_masks(masks, sample_weight):
        extra = sample_weight if (is_enc and sample_weight is not None) else None
        numer = _sum_under_mask(y_pred_enc, mask_obj, is_enc, extra_pt=extra)
        out[label] = _safe_div(numer, denom)
    return out


def mean_prediction_per_group(
    y_pred_enc: EncryptedVector,
    masks: MaskLike,
    *,
    sample_weight: np.ndarray | None = None,
) -> dict[object, float]:
    out: dict[object, float] = {}
    for label, mask_obj, denom, is_enc in _iter_masks(masks, sample_weight):
        extra = sample_weight if (is_enc and sample_weight is not None) else None
        numer = _sum_under_mask(y_pred_enc, mask_obj, is_enc, extra_pt=extra)
        out[label] = _safe_div(numer, denom)
    return out


def confusion_rates_per_group(
    y_true: np.ndarray,
    y_pred_enc: EncryptedVector,
    masks: MaskLike,
    *,
    sample_weight: np.ndarray | None = None,
    positives_per_group: dict[object, float] | None = None,
    negatives_per_group: dict[object, float] | None = None,
) -> dict[object, dict[str, float]]:
    """Per-group {tpr, fpr, tnr, fnr}.

    With encrypted masks the auditor must pre-supply the per-group
    positive / negative counts (otherwise the denominators leak the
    label distribution under encryption). Pass them via
    ``positives_per_group`` / ``negatives_per_group``; if omitted and
    masks are encrypted, we assume the caller has decrypted them
    upstream.
    """
    y = np.asarray(y_true, dtype=float)
    sw = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)

    out: dict[object, dict[str, float]] = {}
    iter_masks = list(_iter_masks(masks, sw))
    for label, mask_obj, _denom, is_enc in iter_masks:
        if is_enc:
            n_pos = (positives_per_group or {}).get(label)
            n_neg = (negatives_per_group or {}).get(label)
            if n_pos is None or n_neg is None:
                raise ValueError(
                    "encrypted masks require positives_per_group + "
                    "negatives_per_group to be supplied (auditor-public "
                    "metadata)."
                )
            tp = _sum_under_mask(y_pred_enc, mask_obj, True, extra_pt=y * sw)
            fp = _sum_under_mask(y_pred_enc, mask_obj, True, extra_pt=(1.0 - y) * sw)
        else:
            m = mask_obj  # already weighted by sw inside _iter_masks
            m_pos = m * y
            m_neg = m * (1.0 - y)
            n_pos = float(m_pos.sum())
            n_neg = float(m_neg.sum())
            tp = float(y_pred_enc.mul_pt(m_pos).sum_all().first_slot())
            fp = float(y_pred_enc.mul_pt(m_neg).sum_all().first_slot())
        out[label] = {
            "tpr": _safe_div(tp, n_pos),
            "fnr": _safe_div(n_pos - tp, n_pos),
            "fpr": _safe_div(fp, n_neg),
            "tnr": _safe_div(n_neg - fp, n_neg),
        }
    return out


def positive_negative_counts(y_true, masks: MaskLike, sample_weight=None):
    """Return ``(positives, negatives)`` per group as plaintext dicts.

    Convenience for callers who hold plaintext y_true alongside
    encrypted masks: they can compute these counts and pass them into
    :func:`confusion_rates_per_group`.
    """
    y = np.asarray(y_true, dtype=float)
    sw = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    pos: dict[object, float] = {}
    neg: dict[object, float] = {}
    if isinstance(masks, EncryptedMaskSet):
        # Counts depend on the plaintext masks pre-encryption; the
        # caller is responsible for keeping that record. Without it we
        # raise — there is no safe fallback.
        raise ValueError(
            "positive_negative_counts cannot be derived from encrypted "
            "masks alone; pass plaintext masks or precompute upstream."
        )
    for lbl, mask in masks.items():
        m = mask * sw
        pos[lbl] = float((m * y).sum())
        neg[lbl] = float((m * (1.0 - y)).sum())
    return pos, neg


def aggregate_difference(
    values: list[float],
    *,
    method: str = "between_groups",
    overall: float | None = None,
) -> float:
    if not values:
        return 0.0
    if method == "between_groups":
        return float(max(values) - min(values))
    if method == "to_overall":
        ref = overall if overall is not None else sum(values) / len(values)
        return float(max(abs(v - ref) for v in values))
    raise ValueError(f"unknown method {method!r}")


def aggregate_ratio(
    values: list[float],
    *,
    method: str = "between_groups",
    overall: float | None = None,
) -> float:
    if not values:
        return 1.0
    if method == "between_groups":
        hi = max(values)
        lo = min(values)
        return float(_safe_div(lo, hi)) if hi > 0 else 0.0
    if method == "to_overall":
        ref = overall if overall is not None else sum(values) / len(values)
        if ref <= _EPS:
            return 0.0
        ratios = [min(v / ref, ref / v) if v > _EPS else 0.0 for v in values]
        return float(min(ratios))
    raise ValueError(f"unknown method {method!r}")
