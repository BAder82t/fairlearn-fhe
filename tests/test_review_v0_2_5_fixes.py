"""Regressions for the post-0.2.3 external review."""

import fairlearn.metrics as fl
import numpy as np
import pytest

from fairlearn_fhe import audit_metric, encrypt
from fairlearn_fhe.encrypted import op_session, session_max_depth
from fairlearn_fhe.metrics import (
    false_negative_rate,
    false_positive_rate,
    make_derived_metric,
    true_negative_rate,
    true_positive_rate,
)
from fairlearn_fhe.metrics._metric_frame import DecryptFallbackWarning, MetricFrame


def _custom(y_true, y_pred, sample_weight=None):
    return float(np.mean(y_pred))


def test_derived_metric_does_not_decrypt_by_default(small_dataset, encrypted_pred):
    y_true, _, sf = small_dataset
    m = make_derived_metric(metric=_custom, transform="difference")
    with pytest.raises(ValueError, match="allow_decrypt"):
        m(y_true, encrypted_pred, sensitive_features=sf)


def test_derived_metric_opt_in_decrypt_warns(small_dataset, encrypted_pred):
    y_true, _, sf = small_dataset
    m = make_derived_metric(metric=_custom, transform="difference", allow_decrypt=True)
    with pytest.warns(DecryptFallbackWarning):
        m(y_true, encrypted_pred, sensitive_features=sf)


@pytest.mark.parametrize(
    "fn", [true_positive_rate, true_negative_rate, false_positive_rate, false_negative_rate]
)
def test_encrypted_pos_label_other_than_one_rejected(fn, small_dataset, encrypted_pred):
    y_true, _, _ = small_dataset
    with pytest.raises(NotImplementedError):
        fn(y_true, encrypted_pred, pos_label=0)
    fn(y_true, encrypted_pred, pos_label=1)


def test_frame_partial_pos_label_rejected(small_dataset, encrypted_pred):
    import functools

    y_true, _, sf = small_dataset
    with pytest.raises(NotImplementedError):
        _ = MetricFrame(
            metrics=functools.partial(fl.true_positive_rate, pos_label=0),
            y_true=y_true, y_pred=encrypted_pred, sensitive_features=sf,
        ).overall


def test_audit_depth_is_chain_not_op_count(small_dataset):
    y_true, y_pred, sf = small_dataset
    env = audit_metric(
        "demographic_parity_difference", y_true, y_pred, sensitive_features=sf
    )
    assert env.observed_depth == 1
    assert env.op_counts["ct_pt_muls"] >= len(set(sf))


def test_sum_all_rotations_use_ceil(ctx):
    v = encrypt(ctx, [1.0, 2.0, 3.0])
    with op_session() as c:
        v.sum_all()
    assert c["rotations"] == 2
    assert session_max_depth(c) == 0
