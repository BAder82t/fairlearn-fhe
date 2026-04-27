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

from ._groups import (
    EncryptedMaskSet,
    MaskDecryptionWarning,
    encrypt_sensitive_features,
)
from .audit import (
    DEFAULT_MIN_GROUP_SIZE,
    SmallGroupWarning,
    audit_metric,
)
from .context import (
    CKKSContext,
    build_context,
    default_context,
    reset_default_context,
    set_default_context,
)
from .encrypted import (
    OP_COUNTERS,
    EncryptedVector,
    decrypt,
    encrypt,
    reset_op_counters,
    snapshot_op_counters,
)
from .envelope import (
    ENVELOPE_JSON_SCHEMA,
    ENVELOPE_SCHEMA,
    MetricEnvelope,
    ParameterSet,
    canonical_envelope_payload,
    envelope_json_schema,
    estimate_security_bits,
    parameter_set_from_context,
    sign_envelope,
    validate_envelope,
    verify_envelope_signature,
)

__all__ = [
    "CKKSContext",
    "build_context",
    "default_context",
    "set_default_context",
    "reset_default_context",
    "EncryptedVector",
    "encrypt",
    "decrypt",
    "OP_COUNTERS",
    "reset_op_counters",
    "snapshot_op_counters",
    "MetricEnvelope",
    "ParameterSet",
    "ENVELOPE_SCHEMA",
    "ENVELOPE_JSON_SCHEMA",
    "envelope_json_schema",
    "canonical_envelope_payload",
    "estimate_security_bits",
    "parameter_set_from_context",
    "sign_envelope",
    "validate_envelope",
    "verify_envelope_signature",
    "audit_metric",
    "DEFAULT_MIN_GROUP_SIZE",
    "SmallGroupWarning",
    "EncryptedMaskSet",
    "MaskDecryptionWarning",
    "encrypt_sensitive_features",
]

__version__ = "0.2.0"
