"""Group-disaggregated regression metrics under encryption.

Mirrors the upstream Fairlearn helpers ``mean_squared_error_group_max``,
``mean_absolute_error_group_max``, and ``r2_score_group_min`` but
operating on encrypted CKKS predictions.

CKKS arithmetic is well-suited to MSE: the per-sample squared error
``(y - ŷ)²`` expands to ``y² - 2yŷ + ŷ²`` and only the ciphertext-
plaintext pieces (``-2yŷ`` and ``ŷ²``) need encrypted multiplies.

Multiplicative-depth budget:

- Plaintext masks: depth 2 — ``ŷ²`` is one ct×ct multiply (depth 1)
  and the per-group plaintext-mask multiply takes it to depth 2.
- Encrypted masks: depth 2 — we pre-fold ``mask · sw`` into a single
  ct×pt multiply (depth 1) and then ct×ct against ``ŷ²`` to land at
  depth 2 max. Note this matches the global depth-6 default with
  margin to spare; lower-depth contexts (e.g. depth-2 fixtures) will
  refuse the encrypted-mask path at the level-budget check.

MAE requires ``|y - ŷ|`` which is non-polynomial. We approximate via
``sqrt(MSE)`` evaluated on the **decrypted per-group sums** — the
ciphertext path produces the squared error sum and we take the
square root after decryption. This is exact for constant residuals
and an upper bound otherwise (Jensen's inequality); pass
``approximate=True`` (the default) to acknowledge the approximation,
or ``approximate=False`` to refuse it (raises ``NotImplementedError``).
The MSE ciphertext circuit is reused.

R² is ``1 - RSS / TSS`` where TSS is plaintext (computed from
``y_true`` only). RSS comes from the same MSE circuit.

All three return zero for groups with empty support, matching
Fairlearn semantics.
"""

from __future__ import annotations

import math
from typing import Any

import fairlearn.metrics as _fl
import numpy as np

from .._circuits import _safe_div
from .._groups import EncryptedMaskSet, group_masks
from ..encrypted import EncryptedVector


def _is_encrypted(x) -> bool:
    return isinstance(x, EncryptedVector)


def _sw(sample_weight):
    return None if sample_weight is None else np.asarray(sample_weight, dtype=float)


def _needs_encrypted(y_pred, sensitive_features) -> bool:
    if isinstance(sensitive_features, EncryptedMaskSet):
        if not _is_encrypted(y_pred):
            raise TypeError(
                "encrypted sensitive_features require an encrypted y_pred."
            )
        return True
    return _is_encrypted(y_pred)


def _per_group_mse_terms(
    y_true: np.ndarray,
    y_pred_enc: EncryptedVector,
    sensitive_features: Any,
    sample_weight: np.ndarray | None,
) -> dict[Any, dict[str, float]]:
    """Per-group ``{rss, n}`` raw counts.

    ``rss`` is the residual-sum-of-squares ``Σ_g w_i (y_i - ŷ_i)²``
    where ``g`` is the group filter, computed under encryption via:

        rss_g = Σ w_i mask_g_i y_i²        (plaintext)
              − 2 Σ w_i mask_g_i y_i ŷ_i   (ct×pt)
              + Σ w_i mask_g_i ŷ_i²        (ct×ct, depth 1)

    Group denominator ``n`` is the weight-normalised sample count.
    """
    y = np.asarray(y_true, dtype=float)
    sw = np.ones_like(y) if sample_weight is None else sample_weight

    if isinstance(sensitive_features, EncryptedMaskSet):
        labels = list(sensitive_features.labels)
        is_encrypted_mask = True
        get_mask = lambda lbl: sensitive_features.masks[lbl]  # noqa: E731
        # Encrypted-mask path: caller must pre-supply the mask×weight
        # plaintext metadata via ``positives + negatives`` (treated as
        # ``n`` here — the auditor needs the count regardless of label).
        n_per = {
            lbl: sensitive_features.positives[lbl] + sensitive_features.negatives[lbl]
            for lbl in labels
        }
    else:
        _labels, plain_masks = group_masks(sensitive_features)
        labels = list(plain_masks.keys())
        is_encrypted_mask = False
        get_mask = lambda lbl: plain_masks[lbl] * sw  # noqa: E731
        n_per = {lbl: float((plain_masks[lbl] * sw).sum()) for lbl in labels}

    yhat_sq = y_pred_enc.mul_ct(y_pred_enc)  # depth 1

    out: dict[Any, dict[str, float]] = {}
    for lbl in labels:
        if is_encrypted_mask:
            mask_ct = get_mask(lbl)
            # y is plaintext so y² is plaintext too; we can multiply the
            # encrypted group mask by the plaintext y²·sw vector.
            sum_y2 = float(mask_ct.mul_pt(y * y * sw).sum_all().first_slot())
            cross = float(
                y_pred_enc.mul_ct(mask_ct).mul_pt(2.0 * y * sw).sum_all().first_slot()
            )
            # Pre-fold mask · sw into a single ct×pt (depth 1), then
            # ct×ct against ŷ² (depth 1) — total depth 2. Doing
            # ``yhat_sq.mul_ct(mask_ct).mul_pt(sw)`` instead would land
            # at depth 3 and exceed any depth-2 fixture.
            weighted_mask = mask_ct.mul_pt(sw)
            yhat_sq_g = float(
                yhat_sq.mul_ct(weighted_mask).sum_all().first_slot()
            )
            rss = sum_y2 - cross + yhat_sq_g
        else:
            m = get_mask(lbl)  # mask × sw
            sum_y2 = float((m * y * y).sum())
            cross = float(y_pred_enc.mul_pt(2.0 * m * y).sum_all().first_slot())
            yhat_sq_g = float(yhat_sq.mul_pt(m).sum_all().first_slot())
            rss = sum_y2 - cross + yhat_sq_g
        out[lbl] = {"rss": max(rss, 0.0), "n": n_per[lbl]}
    return out


def _overall_mse(
    y_true: np.ndarray,
    y_pred_enc: EncryptedVector,
    sample_weight: np.ndarray | None,
) -> tuple[float, float]:
    """Overall ``(rss, n)`` matching the per-group circuit."""
    y = np.asarray(y_true, dtype=float)
    sw = np.ones_like(y) if sample_weight is None else sample_weight
    yhat_sq = y_pred_enc.mul_ct(y_pred_enc)
    sum_y2 = float((sw * y * y).sum())
    cross = float(y_pred_enc.mul_pt(2.0 * sw * y).sum_all().first_slot())
    yhat_sq_g = float(yhat_sq.mul_pt(sw).sum_all().first_slot())
    rss = sum_y2 - cross + yhat_sq_g
    n = float(sw.sum())
    return max(rss, 0.0), n


def _mse(rss: float, n: float) -> float:
    # ``rss`` is already clamped to >= 0 in _per_group_mse_terms; the
    # canonical _safe_div clips small CKKS noise above the bound away.
    return _safe_div(rss, n, clip_lower=0.0, clip_upper=None)


def _mae(rss: float, n: float) -> float:
    # MAE is approximated as sqrt(MSE) — exact for constant residuals,
    # an upper bound otherwise (Jensen's inequality). See module docstring.
    mse = _safe_div(rss, n, clip_lower=0.0, clip_upper=None)
    return math.sqrt(mse)


def _r2(rss: float, tss: float) -> float:
    # 1 - rss/tss; rss/tss is non-negative under our clamping. R² may
    # be arbitrarily negative for poor fits so we don't clip the result.
    if tss <= 0:
        return 0.0
    return 1.0 - _safe_div(rss, tss, clip_lower=0.0, clip_upper=None)


def mean_squared_error_group_max(
    y_true,
    y_pred,
    *,
    sensitive_features,
    sample_weight=None,
) -> float:
    if not _needs_encrypted(y_pred, sensitive_features):
        return _fl.mean_squared_error_group_max(
            y_true,
            y_pred,
            sensitive_features=sensitive_features,
            sample_weight=sample_weight,
        )
    sw = _sw(sample_weight)
    per_group = _per_group_mse_terms(y_true, y_pred, sensitive_features, sw)
    return float(max(_mse(c["rss"], c["n"]) for c in per_group.values()))


def mean_absolute_error_group_max(
    y_true,
    y_pred,
    *,
    sensitive_features,
    sample_weight=None,
    approximate: bool = True,
) -> float:
    """Encrypted-aware ``mean_absolute_error_group_max``.

    The encrypted path approximates MAE as ``sqrt(MSE)`` per group
    (exact for constant residuals, an upper bound otherwise; see
    module docstring). Pass ``approximate=False`` to refuse the
    approximation — the call then raises ``NotImplementedError``
    instead of silently returning an inflated value.
    """
    if not _needs_encrypted(y_pred, sensitive_features):
        return _fl.mean_absolute_error_group_max(
            y_true,
            y_pred,
            sensitive_features=sensitive_features,
            sample_weight=sample_weight,
        )
    if not approximate:
        raise NotImplementedError(
            "Encrypted MAE is only available as a sqrt(MSE) approximation; "
            "pass approximate=True to acknowledge this, or compute MAE "
            "from the decrypted predictions."
        )
    sw = _sw(sample_weight)
    per_group = _per_group_mse_terms(y_true, y_pred, sensitive_features, sw)
    return float(max(_mae(c["rss"], c["n"]) for c in per_group.values()))


def r2_score_group_min(
    y_true,
    y_pred,
    *,
    sensitive_features,
    sample_weight=None,
) -> float:
    if not _needs_encrypted(y_pred, sensitive_features):
        return _fl.r2_score_group_min(
            y_true,
            y_pred,
            sensitive_features=sensitive_features,
            sample_weight=sample_weight,
        )
    sw = _sw(sample_weight)
    per_group = _per_group_mse_terms(y_true, y_pred, sensitive_features, sw)
    y = np.asarray(y_true, dtype=float)
    sw_arr = np.ones_like(y) if sw is None else sw

    if isinstance(sensitive_features, EncryptedMaskSet):
        # We cannot derive per-group y means from encrypted masks; fall
        # back to the global mean for TSS, which is what scikit-learn
        # uses anyway when ``multioutput='uniform_average'``.
        y_mean = float((sw_arr * y).sum() / sw_arr.sum()) if sw_arr.sum() > 0 else 0.0
        tss_per: dict[Any, float] = {}
        for lbl in per_group:
            n = per_group[lbl]["n"]
            tss_per[lbl] = n * (
                float((sw_arr * (y - y_mean) ** 2).sum()) / float(sw_arr.sum())
                if sw_arr.sum() > 0
                else 0.0
            )
    else:
        _labels, plain_masks = group_masks(sensitive_features)
        tss_per = {}
        for lbl, mask in plain_masks.items():
            m = mask * sw_arr
            n_g = float(m.sum())
            if n_g <= 0:
                tss_per[lbl] = 0.0
                continue
            y_mean_g = float((m * y).sum() / n_g)
            tss_per[lbl] = float((m * (y - y_mean_g) ** 2).sum())

    return float(min(_r2(per_group[lbl]["rss"], tss_per[lbl]) for lbl in per_group))
