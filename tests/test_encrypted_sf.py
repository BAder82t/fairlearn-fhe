"""Encrypted sensitive_features path: full triple-encrypted (y_pred + sf)
flow matching plaintext Fairlearn within CKKS noise tolerance.
"""

import fairlearn.metrics as fl
import pytest

from fairlearn_fhe import EncryptedMaskSet, encrypt_sensitive_features
from fairlearn_fhe.metrics import (
    MetricFrame,
    demographic_parity_difference,
    demographic_parity_ratio,
    equal_opportunity_difference,
    equalized_odds_difference,
)


@pytest.fixture
def enc_sf(ctx, small_dataset):
    y_true, _y_pred, sf = small_dataset
    return encrypt_sensitive_features(ctx, sf, y_true=y_true)


def test_encrypt_sensitive_features_shape(ctx, small_dataset):
    y_true, _y_pred, sf = small_dataset
    eset = encrypt_sensitive_features(ctx, sf, y_true=y_true)
    assert isinstance(eset, EncryptedMaskSet)
    assert sorted(eset.labels) == sorted(set(sf))
    assert eset.positives is not None
    assert eset.negatives is not None
    # Counts add up to n.
    assert abs(sum(eset.counts.values()) - len(y_true)) < 1e-9


def test_demographic_parity_diff_encrypted_sf(small_dataset, encrypted_pred, enc_sf, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.demographic_parity_difference(y_true, y_pred, sensitive_features=sf)
    fhe = demographic_parity_difference(y_true, encrypted_pred, sensitive_features=enc_sf)
    assert abs(plain - fhe) < tol


def test_demographic_parity_ratio_encrypted_sf(small_dataset, encrypted_pred, enc_sf, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.demographic_parity_ratio(y_true, y_pred, sensitive_features=sf)
    fhe = demographic_parity_ratio(y_true, encrypted_pred, sensitive_features=enc_sf)
    assert abs(plain - fhe) < tol


def test_equalized_odds_encrypted_sf(small_dataset, encrypted_pred, enc_sf, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.equalized_odds_difference(y_true, y_pred, sensitive_features=sf)
    fhe = equalized_odds_difference(y_true, encrypted_pred, sensitive_features=enc_sf)
    assert abs(plain - fhe) < tol


def test_equal_opportunity_encrypted_sf(small_dataset, encrypted_pred, enc_sf, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.equal_opportunity_difference(y_true, y_pred, sensitive_features=sf)
    fhe = equal_opportunity_difference(y_true, encrypted_pred, sensitive_features=enc_sf)
    assert abs(plain - fhe) < tol


def test_metric_frame_encrypted_sf(small_dataset, encrypted_pred, enc_sf, tol):
    y_true, y_pred, sf = small_dataset
    plain = fl.MetricFrame(
        metrics=fl.selection_rate, y_true=y_true, y_pred=y_pred, sensitive_features=sf,
    )
    enc = MetricFrame(
        metrics=fl.selection_rate, y_true=y_true, y_pred=encrypted_pred, sensitive_features=enc_sf,
    )
    for label in plain.by_group.index:
        p = float(plain.by_group.loc[label])
        e = float(enc.by_group.loc[label].iloc[0])
        assert abs(p - e) < tol


def test_metric_frame_encrypted_sf_confusion(small_dataset, encrypted_pred, enc_sf, tol):
    y_true, y_pred, sf = small_dataset
    fns = {"tpr": fl.true_positive_rate, "fpr": fl.false_positive_rate}
    plain = fl.MetricFrame(metrics=fns, y_true=y_true, y_pred=y_pred, sensitive_features=sf)
    enc = MetricFrame(metrics=fns, y_true=y_true, y_pred=encrypted_pred, sensitive_features=enc_sf)
    for col in ("tpr", "fpr"):
        for label in plain.by_group.index:
            p = float(plain.by_group.loc[label, col])
            e = float(enc.by_group.loc[label, col])
            assert abs(p - e) < tol


def test_encrypted_sf_requires_encrypted_y_pred(small_dataset, enc_sf):
    y_true, y_pred, _ = small_dataset
    with pytest.raises(TypeError):
        demographic_parity_difference(y_true, y_pred, sensitive_features=enc_sf)


def test_confusion_metric_without_label_counts_raises(ctx, small_dataset, encrypted_pred):
    y_true, y_pred, sf = small_dataset
    # Encrypt sf without supplying y_true → no label counts stamped.
    eset = encrypt_sensitive_features(ctx, sf)
    with pytest.raises(ValueError):
        equalized_odds_difference(y_true, encrypted_pred, sensitive_features=eset)


def test_envelope_with_encrypted_sf(small_dataset, encrypted_pred, enc_sf, ctx):
    """audit_metric records the higher depth observed when sf is encrypted."""
    from fairlearn_fhe import audit_metric
    y_true, _, _ = small_dataset
    env = audit_metric(
        "demographic_parity_difference",
        y_true, encrypted_pred,
        sensitive_features=enc_sf, ctx=ctx,
    )
    # Plaintext sf path: ct_pt only. Encrypted-sf path: at least one ct_ct mul.
    assert env.op_counts["ct_ct_muls"] >= 3
