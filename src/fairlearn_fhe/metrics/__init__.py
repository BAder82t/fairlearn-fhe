"""``fairlearn_fhe.metrics`` — encrypted analogues of ``fairlearn.metrics``.

Imports mirror Fairlearn's public surface so a data scientist can swap

    from fairlearn.metrics import demographic_parity_difference

for

    from fairlearn_fhe.metrics import demographic_parity_difference

with no other code change. ``y_pred`` may be a plaintext array (the
function falls through to plain Fairlearn) or an
:class:`fairlearn_fhe.EncryptedVector`.
"""

from ._base_metrics import (
    selection_rate,
    mean_prediction,
    true_positive_rate,
    true_negative_rate,
    false_positive_rate,
    false_negative_rate,
    count,
)
from ._fairness_metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
    equal_opportunity_difference,
    equal_opportunity_ratio,
)
from ._metric_frame import MetricFrame, EncryptedMetricFrame
from ._make_derived_metric import make_derived_metric

__all__ = [
    "MetricFrame",
    "EncryptedMetricFrame",
    "make_derived_metric",
    "selection_rate",
    "mean_prediction",
    "true_positive_rate",
    "true_negative_rate",
    "false_positive_rate",
    "false_negative_rate",
    "count",
    "demographic_parity_difference",
    "demographic_parity_ratio",
    "equalized_odds_difference",
    "equalized_odds_ratio",
    "equal_opportunity_difference",
    "equal_opportunity_ratio",
]
