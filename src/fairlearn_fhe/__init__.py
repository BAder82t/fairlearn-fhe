"""fairlearn-fhe — encrypted Fairlearn metrics.

Public surface mirrors :mod:`fairlearn.metrics`. The encrypted entry points
accept a :class:`EncryptedVector` (produced by :func:`encrypt`) anywhere
the plaintext API expects ``y_pred``; the rest of the call signature is
unchanged.

Trust model: ``y_pred`` is encrypted; ``y_true`` and ``sensitive_features``
are plaintext (auditor-known). Group-level rates are computed under
encryption and decrypted at the audit boundary.
"""

from __future__ import annotations

from .context import CKKSContext, build_context, default_context
from .encrypted import (
    EncryptedVector, encrypt, decrypt,
    OP_COUNTERS, reset_op_counters, snapshot_op_counters,
)
from .envelope import MetricEnvelope, ParameterSet, parameter_set_from_context
from .audit import audit_metric
from ._groups import EncryptedMaskSet, encrypt_sensitive_features

__all__ = [
    "CKKSContext",
    "build_context",
    "default_context",
    "EncryptedVector",
    "encrypt",
    "decrypt",
    "OP_COUNTERS",
    "reset_op_counters",
    "snapshot_op_counters",
    "MetricEnvelope",
    "ParameterSet",
    "parameter_set_from_context",
    "audit_metric",
    "EncryptedMaskSet",
    "encrypt_sensitive_features",
]

__version__ = "0.1.0"
