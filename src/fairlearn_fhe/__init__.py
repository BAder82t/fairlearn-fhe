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

from ._groups import EncryptedMaskSet, encrypt_sensitive_features
from .audit import audit_metric
from .context import CKKSContext, build_context, default_context
from .encrypted import (
    OP_COUNTERS,
    EncryptedVector,
    decrypt,
    encrypt,
    reset_op_counters,
    snapshot_op_counters,
)
from .envelope import (
    ENVELOPE_SCHEMA,
    MetricEnvelope,
    ParameterSet,
    parameter_set_from_context,
    validate_envelope,
)

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
    "ENVELOPE_SCHEMA",
    "parameter_set_from_context",
    "validate_envelope",
    "audit_metric",
    "EncryptedMaskSet",
    "encrypt_sensitive_features",
]

__version__ = "0.1.0"
