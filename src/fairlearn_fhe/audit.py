"""High-level audit wrapper.

``audit_metric`` runs a metric under encryption, captures depth + op
counters, and returns a :class:`MetricEnvelope` ready for a regulator
to log alongside the parameter-set hash.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from typing import Any

import numpy as np

from . import metrics as fhe_metrics
from ._groups import EncryptedMaskSet, group_masks
from .context import CKKSContext, default_context
from .encrypted import EncryptedVector, encrypt, reset_op_counters, snapshot_op_counters
from .envelope import MetricEnvelope, parameter_set_from_context


class SmallGroupWarning(UserWarning):
    """Raised when an audited group has fewer than ``MIN_GROUP_SIZE`` samples.

    Aggregate metrics over very small groups can be re-identifying
    even when individual predictions are encrypted, because the metric
    value pins down a small number of possible label/prediction
    combinations. The threshold is conservative (10) and configurable
    by passing ``min_group_size`` to :func:`audit_metric`.
    """


DEFAULT_MIN_GROUP_SIZE = 10

_BASE_METRIC_FNS = {
    "selection_rate": fhe_metrics.selection_rate,
    "mean_prediction": fhe_metrics.mean_prediction,
    "true_positive_rate": fhe_metrics.true_positive_rate,
    "true_negative_rate": fhe_metrics.true_negative_rate,
    "false_positive_rate": fhe_metrics.false_positive_rate,
    "false_negative_rate": fhe_metrics.false_negative_rate,
    "demographic_parity_difference": fhe_metrics.demographic_parity_difference,
    "demographic_parity_ratio": fhe_metrics.demographic_parity_ratio,
    "equalized_odds_difference": fhe_metrics.equalized_odds_difference,
    "equalized_odds_ratio": fhe_metrics.equalized_odds_ratio,
    "equal_opportunity_difference": fhe_metrics.equal_opportunity_difference,
    "equal_opportunity_ratio": fhe_metrics.equal_opportunity_ratio,
}


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    return repr(value)


def _hash_values(values: Any) -> str:
    arr = np.asarray(values, dtype=object).ravel()
    payload = [_json_safe(v) for v in arr.tolist()]
    body = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(body).hexdigest()


def _audit_metadata(y_true: Any, sensitive_features: Any, kwargs: dict[str, Any]) -> tuple[
    str,
    dict[str, Any],
    dict[str, str],
]:
    if sensitive_features is None:
        trust_model = "no_sensitive_features"
    elif isinstance(sensitive_features, EncryptedMaskSet):
        trust_model = "encrypted_sensitive_features"
    else:
        trust_model = "plaintext_sensitive_features"

    metric_kwargs: dict[str, Any] = {}
    input_hashes = {"y_true": _hash_values(y_true)}

    for name, value in kwargs.items():
        if name == "sample_weight":
            input_hashes["sample_weight"] = _hash_values(value)
            metric_kwargs[name] = {
                "present": True,
                "n": int(np.asarray(value).size),
                "sha256": input_hashes["sample_weight"],
            }
        else:
            metric_kwargs[name] = _json_safe(value)

    if sensitive_features is not None and not isinstance(sensitive_features, EncryptedMaskSet):
        input_hashes["sensitive_features"] = _hash_values(sensitive_features)

    return trust_model, metric_kwargs, input_hashes


def _check_small_groups(
    sensitive_features: Any,
    min_group_size: int,
) -> None:
    if sensitive_features is None or min_group_size <= 0:
        return
    if isinstance(sensitive_features, EncryptedMaskSet):
        small = {
            lbl: count
            for lbl, count in sensitive_features.counts.items()
            if count < min_group_size
        }
    else:
        labels, masks = group_masks(sensitive_features)
        small = {lbl: float(masks[lbl].sum()) for lbl in labels
                 if float(masks[lbl].sum()) < min_group_size}
    if small:
        warnings.warn(
            f"audited groups with fewer than {min_group_size} samples: {small!r}; "
            "the decrypted metric scalar in the envelope can be "
            "re-identifying for small groups even though individual "
            "predictions remain encrypted.",
            SmallGroupWarning,
            stacklevel=3,
        )


def audit_metric(
    metric_name: str,
    y_true,
    y_pred,
    *,
    sensitive_features=None,
    ctx: CKKSContext | None = None,
    min_group_size: int = DEFAULT_MIN_GROUP_SIZE,
    **kwargs: Any,
) -> MetricEnvelope:
    """Run ``metric_name`` under encryption and return an audit envelope.

    ``y_pred`` may be plaintext (encrypted internally) or an
    :class:`EncryptedVector` (used as-is).

    A :class:`SmallGroupWarning` is emitted when any audited group has
    fewer than ``min_group_size`` samples (default 10) — set to ``0``
    to disable. The warning is informational; the envelope is still
    produced.
    """
    if metric_name not in _BASE_METRIC_FNS:
        raise KeyError(
            f"unknown metric {metric_name!r}; choose from {sorted(_BASE_METRIC_FNS)}"
        )
    fn = _BASE_METRIC_FNS[metric_name]
    ctx = ctx or default_context()

    _check_small_groups(sensitive_features, min_group_size)

    if not isinstance(y_pred, EncryptedVector):
        y_pred = encrypt(ctx, y_pred)

    reset_op_counters()
    if sensitive_features is None:
        value = float(fn(y_true, y_pred, **kwargs))
        n_groups = 1
    else:
        value = float(fn(y_true, y_pred, sensitive_features=sensitive_features, **kwargs))
        if isinstance(sensitive_features, EncryptedMaskSet):
            n_groups = len(sensitive_features.labels)
        else:
            labels, _ = group_masks(sensitive_features)
            n_groups = len(labels)

    counts = snapshot_op_counters()
    observed_depth = counts["ct_pt_muls"] + counts["ct_ct_muls"]
    trust_model, metric_kwargs, input_hashes = _audit_metadata(y_true, sensitive_features, kwargs)

    return MetricEnvelope(
        metric_name=metric_name,
        value=value,
        parameter_set=parameter_set_from_context(ctx, depth=6),
        observed_depth=observed_depth,
        op_counts=counts,
        n_samples=int(y_pred.n),
        n_groups=int(n_groups),
        metric_kwargs=metric_kwargs,
        trust_model=trust_model,
        input_hashes=input_hashes,
    )
