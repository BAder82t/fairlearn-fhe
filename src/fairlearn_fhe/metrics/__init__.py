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
    count,
    false_negative_rate,
    false_positive_rate,
    mean_prediction,
    selection_rate,
    true_negative_rate,
    true_positive_rate,
)
from ._fairness_metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equal_opportunity_ratio,
    equalized_odds_difference,
    equalized_odds_ratio,
)
from ._make_derived_metric import make_derived_metric
from ._metric_frame import EncryptedMetricFrame, MetricFrame
from ._per_rate_metrics import (
    false_negative_rate_difference,
    false_negative_rate_ratio,
    false_positive_rate_difference,
    false_positive_rate_ratio,
    selection_rate_difference,
    selection_rate_ratio,
    true_negative_rate_difference,
    true_negative_rate_ratio,
    true_positive_rate_difference,
    true_positive_rate_ratio,
)
from ._regression_metrics import (
    mean_absolute_error_group_max,
    mean_squared_error_group_max,
    r2_score_group_min,
)
from ._scoring_metrics import (
    accuracy_score_difference,
    accuracy_score_group_min,
    balanced_accuracy_score_group_min,
    f1_score_group_min,
    precision_score_group_min,
    recall_score_group_min,
    zero_one_loss_difference,
    zero_one_loss_group_max,
    zero_one_loss_ratio,
)

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
    # per-rate difference/ratio family
    "selection_rate_difference",
    "selection_rate_ratio",
    "true_positive_rate_difference",
    "true_positive_rate_ratio",
    "true_negative_rate_difference",
    "true_negative_rate_ratio",
    "false_positive_rate_difference",
    "false_positive_rate_ratio",
    "false_negative_rate_difference",
    "false_negative_rate_ratio",
    # scoring metrics
    "accuracy_score_difference",
    "accuracy_score_group_min",
    "balanced_accuracy_score_group_min",
    "precision_score_group_min",
    "recall_score_group_min",
    "f1_score_group_min",
    "zero_one_loss_difference",
    "zero_one_loss_group_max",
    "zero_one_loss_ratio",
    # regression metrics
    "mean_absolute_error_group_max",
    "mean_squared_error_group_max",
    "r2_score_group_min",
]
