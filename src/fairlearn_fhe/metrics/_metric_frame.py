"""Encrypted MetricFrame.

Two ways to use it:

1. ``MetricFrame(metrics=..., y_true=..., y_pred=..., sensitive_features=...)``
   — accepts encrypted ``y_pred``; if encrypted, returns an
   :class:`EncryptedMetricFrame`. The returned object exposes
   ``.overall``, ``.by_group``, ``.difference``, ``.ratio`` matching
   plaintext Fairlearn within CKKS noise tolerance.
2. ``MetricFrame.fhe(metrics=..., y_true=..., y_pred=..., sensitive_features=...)``
   — explicit alias that always uses the encrypted path; raises if
   ``y_pred`` is plaintext.

Only the canonical Fairlearn metrics — ``selection_rate``,
``mean_prediction``, ``true_positive_rate``, ``true_negative_rate``,
``false_positive_rate``, ``false_negative_rate``, ``count`` — are
recognised on the encrypted path. Custom callables fall back to
plaintext via decryption (a logged warning) if the user opts in via
``allow_decrypt=True``.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import fairlearn.metrics as _fl
import numpy as np
import pandas as pd

from .._circuits import (
    aggregate_difference,
    aggregate_ratio,
    confusion_rates_per_group,
    mean_prediction_per_group,
    positive_negative_counts,
    selection_rate_per_group,
)
from .._groups import EncryptedMaskSet, group_masks
from ..encrypted import EncryptedVector

_KNOWN_ENCRYPTED: dict[Any, str] = {
    _fl.selection_rate: "selection_rate",
    _fl.mean_prediction: "mean_prediction",
    _fl.true_positive_rate: "tpr",
    _fl.true_negative_rate: "tnr",
    _fl.false_positive_rate: "fpr",
    _fl.false_negative_rate: "fnr",
    _fl.count: "count",
}


def _is_encrypted(x) -> bool:
    return isinstance(x, EncryptedVector)


@dataclass
class EncryptedMetricFrame:
    overall: pd.Series
    by_group: pd.DataFrame
    metric_names: list[str]
    group_labels: list

    def difference(self, *, method: str = "between_groups"):
        out = {}
        for col in self.by_group.columns:
            values = list(self.by_group[col].values)
            ov = float(self.overall[col]) if col in self.overall.index else None
            out[col] = aggregate_difference(values, method=method, overall=ov)
        if len(out) == 1:
            return next(iter(out.values()))
        return pd.Series(out)

    def ratio(self, *, method: str = "between_groups"):
        out = {}
        for col in self.by_group.columns:
            values = list(self.by_group[col].values)
            ov = float(self.overall[col]) if col in self.overall.index else None
            out[col] = aggregate_ratio(values, method=method, overall=ov)
        if len(out) == 1:
            return next(iter(out.values()))
        return pd.Series(out)

    def group_min(self):
        out = {col: float(min(self.by_group[col].values)) for col in self.by_group.columns}
        if len(out) == 1:
            return next(iter(out.values()))
        return pd.Series(out)

    def group_max(self):
        out = {col: float(max(self.by_group[col].values)) for col in self.by_group.columns}
        if len(out) == 1:
            return next(iter(out.values()))
        return pd.Series(out)


def _resolve_metric(
    metric: Any,
    name: str,
    y_true: np.ndarray,
    y_pred_enc: EncryptedVector,
    masks,
    sample_weight: np.ndarray | None,
    allow_decrypt: bool,
) -> dict[object, float]:
    # Unwrap functools.partial so derived metrics built on top of the
    # canonical Fairlearn callables route through the encrypted path.
    base = metric
    while isinstance(base, functools.partial):
        base = base.func
    kind = _KNOWN_ENCRYPTED.get(base)
    is_enc_set = isinstance(masks, EncryptedMaskSet)
    if kind == "selection_rate":
        return selection_rate_per_group(y_pred_enc, masks, sample_weight=sample_weight)
    if kind == "mean_prediction":
        return mean_prediction_per_group(y_pred_enc, masks, sample_weight=sample_weight)
    if kind in ("tpr", "fpr", "tnr", "fnr"):
        if is_enc_set:
            pos, neg = masks.positives, masks.negatives
            if pos is None or neg is None:
                raise ValueError(
                    "EncryptedMaskSet missing positive/negative counts; "
                    "call encrypt_sensitive_features(ctx, sf, y_true=y_true)."
                )
        else:
            pos, neg = positive_negative_counts(y_true, masks, sample_weight=sample_weight)
        rates = confusion_rates_per_group(
            y_true, y_pred_enc, masks, sample_weight=sample_weight,
            positives_per_group=pos, negatives_per_group=neg,
        )
        return {label: rates[label][kind] for label in rates}
    if kind == "count":
        if is_enc_set:
            return dict(masks.counts)
        return {label: float(mask.sum()) for label, mask in masks.items()}

    if not allow_decrypt:
        raise ValueError(
            f"metric {name!r} is not in the encrypted catalogue. "
            "Pass ``allow_decrypt=True`` to fall back to plaintext via "
            "ciphertext decryption (defeats the privacy guarantee)."
        )
    if is_enc_set:
        raise ValueError(
            f"metric {name!r} is not in the encrypted catalogue and "
            "decrypt-fallback is not supported with encrypted "
            "sensitive_features."
        )
    # Decrypt-and-fallback path (plaintext masks only).
    y_p = y_pred_enc.decrypt()
    out: dict[object, float] = {}
    for label, mask in masks.items():
        sel = mask > 0
        kwargs = {}
        if sample_weight is not None:
            kwargs["sample_weight"] = sample_weight[sel]
        try:
            out[label] = float(metric(y_true[sel], y_p[sel], **kwargs))
        except TypeError:
            out[label] = float(metric(y_true[sel], y_p[sel]))
    return out


def MetricFrame(
    *,
    metrics: Any,
    y_true,
    y_pred,
    sensitive_features,
    sample_params: Mapping[str, Any] | None = None,
    allow_decrypt: bool = False,
):
    """Encrypted-aware MetricFrame factory.

    Mirrors :class:`fairlearn.metrics.MetricFrame` for plaintext inputs
    (returns the genuine fairlearn object so ``isinstance`` against
    :class:`fairlearn.metrics.MetricFrame` works). For encrypted
    ``y_pred`` returns an :class:`EncryptedMetricFrame` with the same
    accessors used by the fairness wrappers.

    This is a function rather than a class so that callers don't have
    to think about which concrete type they got back; introspect via
    ``isinstance(result, EncryptedMetricFrame)`` if you need to
    distinguish.
    """
    sf_encrypted = isinstance(sensitive_features, EncryptedMaskSet)
    if sf_encrypted and not _is_encrypted(y_pred):
        raise TypeError(
            "encrypted sensitive_features require an encrypted y_pred."
        )
    if not _is_encrypted(y_pred) and not sf_encrypted:
        return _fl.MetricFrame(
            metrics=metrics,
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=sensitive_features,
            sample_params=sample_params,
        )
    return _build_encrypted(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params=sample_params,
        allow_decrypt=allow_decrypt,
    )


def metric_frame_fhe(
    *,
    metrics: Any,
    y_true,
    y_pred,
    sensitive_features,
    sample_params: Mapping[str, Any] | None = None,
    allow_decrypt: bool = False,
) -> EncryptedMetricFrame:
    """Always-encrypted MetricFrame; raises if ``y_pred`` is plaintext."""
    if not _is_encrypted(y_pred):
        raise TypeError("metric_frame_fhe requires an encrypted y_pred (EncryptedVector).")
    return _build_encrypted(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features,
        sample_params=sample_params,
        allow_decrypt=allow_decrypt,
    )


# Backwards-compatible alias for ``MetricFrame.fhe(...)`` callers.
MetricFrame.fhe = metric_frame_fhe  # type: ignore[attr-defined]


def _build_encrypted(
    *,
    metrics,
    y_true,
    y_pred: EncryptedVector,
    sensitive_features,
    sample_params,
    allow_decrypt: bool,
) -> EncryptedMetricFrame:
    if isinstance(sensitive_features, EncryptedMaskSet):
        labels = list(sensitive_features.labels)
        masks: Any = sensitive_features
    else:
        labels, masks = group_masks(sensitive_features)
    y = np.asarray(y_true, dtype=float)

    if callable(metrics):
        metric_dict = {_metric_name(metrics): metrics}
    else:
        metric_dict = dict(metrics)  # mapping[name -> callable]

    sample_params = dict(sample_params or {})

    per_group_rows: dict[str, dict[object, float]] = {}
    overall: dict[str, float] = {}

    for name, metric in metric_dict.items():
        sw = None
        if metric_dict and isinstance(sample_params, dict):
            # Fairlearn supports either flat ``{sample_weight: ...}`` or
            # nested ``{metric_name: {sample_weight: ...}}``.
            if name in sample_params and isinstance(sample_params[name], Mapping):
                sw = sample_params[name].get("sample_weight")
            elif "sample_weight" in sample_params:
                sw = sample_params["sample_weight"]
        sw = None if sw is None else np.asarray(sw, dtype=float)

        per_group = _resolve_metric(metric, name, y, y_pred, masks, sw, allow_decrypt)
        per_group_rows[name] = per_group
        # Overall = same metric with a single all-ones mask.
        all_mask = {None: np.ones(len(y))}
        overall_dict = _resolve_metric(metric, name, y, y_pred, all_mask, sw, allow_decrypt)
        overall[name] = float(next(iter(overall_dict.values())))

    by_group = pd.DataFrame(
        {name: [per_group_rows[name][lbl] for lbl in labels] for name in metric_dict},
        index=pd.Index(labels, name="sensitive_feature_0"),
    )
    return EncryptedMetricFrame(
        overall=pd.Series(overall),
        by_group=by_group,
        metric_names=list(metric_dict.keys()),
        group_labels=list(labels),
    )


def _metric_name(metric: Callable) -> str:
    return getattr(metric, "__name__", repr(metric))
