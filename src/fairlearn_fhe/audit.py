"""High-level audit wrapper.

``audit_metric`` runs a metric under encryption, captures depth + op
counters, and returns a :class:`MetricEnvelope` ready for a regulator
to log alongside the parameter-set hash.
"""

from __future__ import annotations

from typing import Any

from . import metrics as fhe_metrics
from ._groups import EncryptedMaskSet, group_masks
from .context import CKKSContext, default_context
from .encrypted import EncryptedVector, encrypt, reset_op_counters, snapshot_op_counters
from .envelope import MetricEnvelope, parameter_set_from_context

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


def audit_metric(
    metric_name: str,
    y_true,
    y_pred,
    *,
    sensitive_features=None,
    ctx: CKKSContext | None = None,
    **kwargs: Any,
) -> MetricEnvelope:
    """Run ``metric_name`` under encryption and return an audit envelope.

    ``y_pred`` may be plaintext (encrypted internally) or an
    :class:`EncryptedVector` (used as-is).
    """
    if metric_name not in _BASE_METRIC_FNS:
        raise KeyError(
            f"unknown metric {metric_name!r}; choose from {sorted(_BASE_METRIC_FNS)}"
        )
    fn = _BASE_METRIC_FNS[metric_name]
    ctx = ctx or default_context()

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

    return MetricEnvelope(
        metric_name=metric_name,
        value=value,
        parameter_set=parameter_set_from_context(ctx),
        observed_depth=observed_depth,
        op_counts=counts,
        n_samples=int(y_pred.n),
        n_groups=int(n_groups),
    )
