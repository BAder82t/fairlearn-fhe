"""Sensitive-feature → group-mask helpers.

Two trust modes:

- **Plaintext sf** (default): the auditor knows group membership; we
  build plaintext one-hot masks for ct×pt multiplies (depth 1 per
  metric).
- **Encrypted sf**: the vendor encrypts the per-row sensitive vector
  too; we hold it as an :class:`EncryptedMaskSet`. Per-group counts
  are revealed as auditor metadata (you can't compute group rates
  without the denominator anyway). Each metric costs one extra
  multiplicative level (ct×ct + ct×pt + sum_all → depth 2).

For multi-column sensitive features the cartesian-product of unique
values forms the group set, matching Fairlearn's `MetricFrame`.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .context import CKKSContext
from .encrypted import EncryptedVector


class MaskDecryptionWarning(UserWarning):
    """Raised when an API call decrypts encrypted group masks.

    Calling such an API undoes the Mode-B privacy guarantee for the
    sensitive features (per-row group membership becomes recoverable).
    The warning is emitted so callers cannot silently break the trust
    model they configured.
    """


@dataclass
class EncryptedMaskSet:
    """One-hot group masks under encryption + plaintext counts.

    ``masks[label]`` is an :class:`EncryptedVector`; ``counts[label]``
    is the plaintext sum of the mask (auditor-public metadata).

    For TPR/FPR-style metrics the auditor also needs per-group
    positive / negative counts. They can be attached via
    :meth:`attach_label_counts` once at encryption time when the
    plaintext y_true is known to the encrypting party.
    """
    labels: list
    masks: dict[object, EncryptedVector]
    counts: dict[object, float]
    n: int
    positives: dict[object, float] | None = None
    negatives: dict[object, float] | None = None

    def items(self):
        for lbl in self.labels:
            yield lbl, self.masks[lbl], self.counts[lbl]

    def attach_label_counts(
        self,
        y_true,
        sample_weight=None,
        *,
        plaintext_masks: dict[object, np.ndarray] | None = None,
    ) -> EncryptedMaskSet:
        """Stamp per-group positive/negative counts using plaintext y_true.

        ``plaintext_masks`` is the preferred input: pass the original
        ``{0,1}`` masks held by the encrypting party. When omitted this
        method falls back to **decrypting** every encrypted mask, which
        defeats Mode-B privacy for the sensitive features. A
        :class:`MaskDecryptionWarning` is raised in that case.

        Returns ``self`` for chaining. Prefer constructing the
        :class:`EncryptedMaskSet` via
        :func:`encrypt_sensitive_features` with ``y_true=`` so counts
        are stamped from plaintext at encryption time and this method
        is never needed.
        """
        y = np.asarray(y_true, dtype=float)
        sw = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        pos: dict[object, float] = {}
        neg: dict[object, float] = {}
        if plaintext_masks is None:
            warnings.warn(
                "attach_label_counts() decrypted the encrypted masks to "
                "compute per-group label counts; this discloses per-row "
                "group membership and breaks the Mode-B privacy guarantee. "
                "Pass plaintext_masks=... or use "
                "encrypt_sensitive_features(ctx, sf, y_true=y_true) at "
                "encryption time instead.",
                MaskDecryptionWarning,
                stacklevel=2,
            )
            for lbl in self.labels:
                m = np.asarray(self.masks[lbl].decrypt(), dtype=float).round()
                mw = m * sw
                pos[lbl] = float((mw * y).sum())
                neg[lbl] = float((mw * (1.0 - y)).sum())
        else:
            for lbl in self.labels:
                m = np.asarray(plaintext_masks[lbl], dtype=float)
                mw = m * sw
                pos[lbl] = float((mw * y).sum())
                neg[lbl] = float((mw * (1.0 - y)).sum())
        self.positives = pos
        self.negatives = neg
        return self


def encrypt_sensitive_features(
    ctx: CKKSContext,
    sensitive_features,
    *,
    y_true=None,
    sample_weight=None,
) -> EncryptedMaskSet:
    """Encode a plaintext sensitive-features array as an encrypted mask
    set. The vendor calls this before sending masks to the auditor.

    If ``y_true`` is supplied, per-group positive / negative counts
    are stamped on the result without re-decrypting (used by the
    confusion-matrix circuits).
    """
    labels, plaintext_masks = group_masks(sensitive_features)
    enc_masks: dict[object, EncryptedVector] = {}
    counts: dict[object, float] = {}
    pos: dict[object, float] | None = {} if y_true is not None else None
    neg: dict[object, float] | None = {} if y_true is not None else None
    sw = None
    if y_true is not None:
        y = np.asarray(y_true, dtype=float)
        sw = np.ones_like(y) if sample_weight is None else np.asarray(sample_weight, dtype=float)
    n = 0
    for lbl in labels:
        m = plaintext_masks[lbl]
        n = len(m)
        enc_masks[lbl] = EncryptedVector.encrypt(ctx, m)
        if sw is not None:
            counts[lbl] = float((m * sw).sum())
        else:
            counts[lbl] = float(m.sum())
        if pos is not None:
            mw = m * sw
            pos[lbl] = float((mw * y).sum())
            neg[lbl] = float((mw * (1.0 - y)).sum())
    return EncryptedMaskSet(
        labels=list(labels), masks=enc_masks, counts=counts, n=n,
        positives=pos, negatives=neg,
    )


def group_masks(sensitive_features) -> tuple[list, dict[object, np.ndarray]]:
    """Return ``(labels, masks)`` where ``masks[label]`` is a ``{0,1}``
    plaintext vector aligned with ``sensitive_features``.

    Group labels are ordered by Fairlearn's stable ordering: pandas
    factorization on the cartesian product of sensitive columns.
    """
    sf = _to_dataframe(sensitive_features)
    if sf.shape[1] == 1:
        keys = sf.iloc[:, 0].astype(object).to_numpy()
    else:
        keys = pd.MultiIndex.from_frame(sf).to_flat_index().to_numpy()

    labels: list = []
    seen = set()
    for k in keys:
        if k not in seen:
            seen.add(k)
            labels.append(k)
    labels.sort(key=_sort_key)

    masks = {
        lbl: np.fromiter((k == lbl for k in keys), dtype=float, count=len(keys))
        for lbl in labels
    }
    return labels, masks


def _to_dataframe(sensitive_features) -> pd.DataFrame:
    if isinstance(sensitive_features, pd.DataFrame):
        return sensitive_features
    if isinstance(sensitive_features, pd.Series):
        return sensitive_features.to_frame()
    arr = np.asarray(sensitive_features)
    if arr.ndim == 1:
        return pd.DataFrame({"sf0": arr})
    return pd.DataFrame(arr, columns=[f"sf{i}" for i in range(arr.shape[1])])


def _sort_key(value):
    """Stable sort key that tolerates mixed types (strings vs ints)."""
    if isinstance(value, tuple):
        return tuple(_sort_key(v) for v in value)
    return (str(type(value).__name__), value if isinstance(value, (int, float)) else str(value))
