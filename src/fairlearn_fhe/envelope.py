"""Canonical metric envelope.

Wraps a metric output with the FHE parameter set, depth observed, and
backend identifier so an auditor can verify the encrypted execution
matches the declared spec. Schema mirrors regaudit-fhe's
``ParameterSet`` so envelopes interoperate.
"""

from __future__ import annotations

import hashlib
import json
import time
from base64 import b64decode, b64encode
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .context import CKKSContext

ENVELOPE_SCHEMA = "fairlearn-fhe.metric-envelope.v1"
SIGNATURE_ALGORITHM = "Ed25519"


# JSON Schema (Draft 2020-12) describing the structure of a serialised
# :class:`MetricEnvelope`. ``ENVELOPE_SCHEMA`` (the version tag string)
# is included as the ``$id`` for stable referencing across releases.
ENVELOPE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": ENVELOPE_SCHEMA,
    "title": "fairlearn-fhe metric audit envelope",
    "description": (
        "JSON Schema describing the audit envelope produced by "
        "fairlearn_fhe.audit_metric and friends. Auditors validate "
        "envelopes against this schema before consuming the verdict."
    ),
    "type": "object",
    "required": [
        "schema_version",
        "metric_name",
        "value",
        "parameter_set",
        "parameter_set_hash",
        "observed_depth",
        "op_counts",
        "n_samples",
        "n_groups",
    ],
    "properties": {
        "schema_version": {"type": "string", "const": ENVELOPE_SCHEMA},
        "metric_name": {"type": "string", "minLength": 1},
        "value": {"type": "number"},
        "parameter_set": {
            "type": "object",
            "required": [
                "backend",
                "poly_modulus_degree",
                "security_bits",
                "multiplicative_depth",
                "coeff_mod_bit_sizes",
                "scaling_factor_bits",
            ],
            "properties": {
                "backend": {"type": "string", "minLength": 1},
                "poly_modulus_degree": {"type": "integer", "minimum": 1},
                "security_bits": {"type": "integer", "minimum": 0},
                "multiplicative_depth": {"type": "integer", "minimum": 0},
                "coeff_mod_bit_sizes": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1},
                },
                "scaling_factor_bits": {"type": "integer", "minimum": 1},
                "backend_version": {"type": "string"},
            },
        },
        "parameter_set_hash": {
            "type": "string",
            "pattern": "^[0-9a-f]{64}$",
            "description": "SHA-256 hex digest of the canonical parameter_set",
        },
        "observed_depth": {"type": "integer", "minimum": 0},
        "op_counts": {
            "type": "object",
            "additionalProperties": {"type": "integer", "minimum": 0},
        },
        "n_samples": {"type": "integer", "minimum": 0},
        "n_groups": {"type": "integer", "minimum": 0},
        "metric_kwargs": {"type": "object"},
        "trust_model": {"type": "string"},
        "input_hashes": {
            "type": "object",
            "additionalProperties": {
                "type": "string",
                "pattern": "^[0-9a-f]{64}$",
            },
        },
        "timestamp": {"type": "number", "minimum": 0},
        "signature": {
            "type": "object",
            "required": ["algorithm", "signature_b64"],
            "properties": {
                "algorithm": {"type": "string", "const": SIGNATURE_ALGORITHM},
                "signature_b64": {"type": "string", "minLength": 1},
                "public_key_id": {"type": "string"},
            },
        },
    },
    "additionalProperties": True,
}


def envelope_json_schema() -> dict[str, Any]:
    """Return a deep-copy of :data:`ENVELOPE_JSON_SCHEMA`.

    Convenience helper so callers can mutate the result without
    accidentally affecting the package-level default.
    """
    import copy

    return copy.deepcopy(ENVELOPE_JSON_SCHEMA)


@dataclass(frozen=True)
class ParameterSet:
    backend: str
    poly_modulus_degree: int
    security_bits: int
    multiplicative_depth: int
    coeff_mod_bit_sizes: tuple[int, ...]
    scaling_factor_bits: int
    backend_version: str = ""

    def __post_init__(self) -> None:
        # Enforce the modulus-chain cap on every construction path,
        # not just from_dict / build_context. Protects parameter_set_from_context
        # when the underlying backend context was built from
        # attacker-controlled serialised bytes.
        from .context import _MAX_COEFF_MODULI  # local import to avoid cycle

        if len(self.coeff_mod_bit_sizes) > _MAX_COEFF_MODULI:
            raise ValueError(
                f"coeff_mod_bit_sizes too long ({len(self.coeff_mod_bit_sizes)} "
                f"> {_MAX_COEFF_MODULI})."
            )

    def hash(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ParameterSet:
        coeff_raw = payload["coeff_mod_bit_sizes"]
        # Cap the chain length to the same bound build_context enforces;
        # an attacker-supplied envelope with a million-element chain
        # would otherwise allocate unbounded memory before any
        # downstream validator runs.
        from .context import _MAX_COEFF_MODULI  # local import to avoid cycle

        if hasattr(coeff_raw, "__len__") and len(coeff_raw) > _MAX_COEFF_MODULI:
            raise ValueError(
                f"coeff_mod_bit_sizes too long ({len(coeff_raw)} > "
                f"{_MAX_COEFF_MODULI}); refusing to deserialise."
            )
        coeff_tuple = tuple(int(v) for v in coeff_raw)
        if len(coeff_tuple) > _MAX_COEFF_MODULI:
            raise ValueError(
                f"coeff_mod_bit_sizes too long ({len(coeff_tuple)} > "
                f"{_MAX_COEFF_MODULI}); refusing to deserialise."
            )
        return cls(
            backend=str(payload["backend"]),
            poly_modulus_degree=int(payload["poly_modulus_degree"]),
            security_bits=int(payload["security_bits"]),
            multiplicative_depth=int(payload["multiplicative_depth"]),
            coeff_mod_bit_sizes=coeff_tuple,
            scaling_factor_bits=int(payload["scaling_factor_bits"]),
            backend_version=str(payload.get("backend_version", "")),
        )


@dataclass
class MetricEnvelope:
    metric_name: str
    value: float
    parameter_set: ParameterSet
    observed_depth: int
    op_counts: dict[str, int]
    n_samples: int
    n_groups: int
    metric_kwargs: dict[str, Any] = field(default_factory=dict)
    trust_model: str = ""
    input_hashes: dict[str, str] = field(default_factory=dict)
    signature: dict[str, str] | None = None
    schema_version: str = ENVELOPE_SCHEMA
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        body = {
            "schema_version": self.schema_version,
            "metric_name": self.metric_name,
            "value": float(self.value),
            "parameter_set": asdict(self.parameter_set),
            "parameter_set_hash": self.parameter_set.hash(),
            "observed_depth": int(self.observed_depth),
            "op_counts": dict(self.op_counts),
            "n_samples": int(self.n_samples),
            "n_groups": int(self.n_groups),
            "metric_kwargs": dict(self.metric_kwargs),
            "trust_model": self.trust_model,
            "input_hashes": dict(self.input_hashes),
            "timestamp": float(self.timestamp),
        }
        if self.signature is not None:
            body["signature"] = dict(self.signature)
        return body

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> MetricEnvelope:
        return cls(
            metric_name=str(payload["metric_name"]),
            value=float(payload["value"]),
            parameter_set=ParameterSet.from_dict(payload["parameter_set"]),
            observed_depth=int(payload["observed_depth"]),
            op_counts={str(k): int(v) for k, v in payload["op_counts"].items()},
            n_samples=int(payload["n_samples"]),
            n_groups=int(payload["n_groups"]),
            metric_kwargs=dict(payload.get("metric_kwargs", {})),
            trust_model=str(payload.get("trust_model", "")),
            input_hashes={str(k): str(v) for k, v in payload.get("input_hashes", {}).items()},
            signature=dict(payload["signature"]) if payload.get("signature") is not None else None,
            schema_version=str(payload.get("schema_version", ENVELOPE_SCHEMA)),
            timestamp=float(payload["timestamp"]),
        )

    @classmethod
    def from_json(cls, body: str) -> MetricEnvelope:
        return cls.from_dict(json.loads(body))


def canonical_envelope_payload(payload: Mapping[str, Any] | MetricEnvelope) -> bytes:
    """Return canonical JSON bytes for hashing or signing an envelope.

    The embedded ``signature`` field is excluded, so the same function can
    produce the payload before signing and during verification.
    """
    data = payload.to_dict() if isinstance(payload, MetricEnvelope) else dict(payload)
    data.pop("signature", None)
    return json.dumps(data, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sign_envelope(
    payload: Mapping[str, Any] | MetricEnvelope,
    private_key_pem: str | bytes,
) -> dict[str, Any]:
    """Return a signed envelope payload using an Ed25519 PEM private key.

    ``cryptography`` is imported lazily so unsigned envelope validation does
    not require the optional signing dependency.
    """
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("install fairlearn-fhe[signing] to sign envelopes") from exc

    key_bytes = (
        private_key_pem.encode("utf-8")
        if isinstance(private_key_pem, str)
        else private_key_pem
    )
    private_key = load_pem_private_key(key_bytes, password=None)
    if not isinstance(private_key, Ed25519PrivateKey):
        raise TypeError("private_key_pem must contain an Ed25519 private key")

    data = payload.to_dict() if isinstance(payload, MetricEnvelope) else dict(payload)
    data.pop("signature", None)
    signature = private_key.sign(canonical_envelope_payload(data))
    data["signature"] = {
        "algorithm": SIGNATURE_ALGORITHM,
        "signature_b64": b64encode(signature).decode("ascii"),
    }
    return data


def verify_envelope_signature(
    payload: Mapping[str, Any] | MetricEnvelope,
    public_key_pem: str | bytes,
) -> list[str]:
    """Return signature-verification errors for a signed envelope payload."""
    try:
        from cryptography.exceptions import InvalidSignature
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
        from cryptography.hazmat.primitives.serialization import load_pem_public_key
    except ImportError as exc:  # pragma: no cover - depends on optional dependency
        raise RuntimeError("install fairlearn-fhe[signing] to verify signatures") from exc

    data = payload.to_dict() if isinstance(payload, MetricEnvelope) else dict(payload)
    signature = data.get("signature")
    if not isinstance(signature, Mapping):
        return ["missing signature"]
    if signature.get("algorithm") != SIGNATURE_ALGORITHM:
        return [f"unsupported signature algorithm {signature.get('algorithm')!r}"]

    # Prefer the schema-canonical ``signature_b64`` field; fall back to
    # the legacy ``value`` key so envelopes signed with earlier
    # fairlearn-fhe releases still verify.
    sig_str = signature.get("signature_b64")
    if sig_str is None:
        sig_str = signature.get("value")
    if sig_str is None:
        return ["signature value must be base64"]
    try:
        signature_bytes = b64decode(str(sig_str), validate=True)
    except ValueError:
        return ["signature value must be base64"]

    key_bytes = (
        public_key_pem.encode("utf-8")
        if isinstance(public_key_pem, str)
        else public_key_pem
    )
    public_key = load_pem_public_key(key_bytes)
    if not isinstance(public_key, Ed25519PublicKey):
        raise TypeError("public_key_pem must contain an Ed25519 public key")

    try:
        public_key.verify(signature_bytes, canonical_envelope_payload(data))
    except InvalidSignature:
        return ["signature verification failed"]
    return []


_DEFAULT_MIN_SECURITY_BITS = 128
_MAX_OP_COUNT_KEY_LEN = 64


def validate_envelope(
    payload: Mapping[str, Any] | MetricEnvelope,
    *,
    allowed_metrics: Sequence[str] | None = None,
    max_observed_depth: int | None = None,
    max_age_seconds: float | None = None,
    now: float | None = None,
    min_security_bits: int | None = _DEFAULT_MIN_SECURITY_BITS,
) -> list[str]:
    """Return validation errors for an audit-envelope payload.

    An empty list means the envelope is structurally valid, the
    ``parameter_set_hash`` matches the embedded parameter set, and the
    observed depth is within the declared multiplicative-depth budget.
    This is intentionally dependency-free so it can run in lightweight
    verifier tooling.

    Optional verifier-side checks:

    - ``max_age_seconds`` rejects envelopes older than the given window
      (anti-replay). ``now`` defaults to ``time.time()``.
    - ``min_security_bits`` rejects parameter sets whose recorded
      security level is below the verifier's minimum. **Defaults to
      128**; pass ``min_security_bits=0`` to opt out (e.g. for testing
      benchmark contexts that intentionally use sub-128-bit
      parameters).
    """
    data = payload.to_dict() if isinstance(payload, MetricEnvelope) else dict(payload)
    errors: list[str] = []

    required = {
        "schema_version",
        "metric_name",
        "value",
        "parameter_set",
        "parameter_set_hash",
        "observed_depth",
        "op_counts",
        "n_samples",
        "n_groups",
        "metric_kwargs",
        "trust_model",
        "input_hashes",
        "timestamp",
    }
    missing = sorted(required - data.keys())
    if missing:
        return [f"missing required field: {name}" for name in missing]

    if data["schema_version"] != ENVELOPE_SCHEMA:
        errors.append(f"unsupported schema_version {data['schema_version']!r}")

    try:
        ps = ParameterSet.from_dict(data["parameter_set"])
    except (KeyError, TypeError, ValueError) as exc:
        errors.append(f"invalid parameter_set: {exc}")
        ps = None

    if ps is not None and data["parameter_set_hash"] != ps.hash():
        errors.append("parameter_set_hash does not match parameter_set")

    if allowed_metrics is not None and data["metric_name"] not in allowed_metrics:
        errors.append(f"metric_name {data['metric_name']!r} is not allowed")

    try:
        observed_depth = int(data["observed_depth"])
        if observed_depth < 0:
            errors.append("observed_depth must be non-negative")
        if ps is not None and observed_depth > ps.multiplicative_depth:
            errors.append("observed_depth exceeds parameter_set multiplicative_depth")
        if max_observed_depth is not None and observed_depth > max_observed_depth:
            errors.append("observed_depth exceeds verifier maximum")
    except (TypeError, ValueError):
        errors.append("observed_depth must be an integer")

    try:
        if int(data["n_samples"]) < 1:
            errors.append("n_samples must be positive")
    except (TypeError, ValueError):
        errors.append("n_samples must be an integer")

    try:
        if int(data["n_groups"]) < 0:
            errors.append("n_groups must be non-negative")
    except (TypeError, ValueError):
        errors.append("n_groups must be an integer")

    if not isinstance(data["op_counts"], Mapping):
        errors.append("op_counts must be a mapping")
    else:
        for name, value in data["op_counts"].items():
            # Cap key length so a hostile envelope can't blow up our
            # error strings (which feed terminal output and logs).
            display = (
                name if len(str(name)) <= _MAX_OP_COUNT_KEY_LEN
                else str(name)[:_MAX_OP_COUNT_KEY_LEN] + "…"
            )
            try:
                if int(value) < 0:
                    errors.append(f"op_counts[{display!r}] must be non-negative")
            except (TypeError, ValueError):
                errors.append(f"op_counts[{display!r}] must be an integer")

    if not isinstance(data["metric_kwargs"], Mapping):
        errors.append("metric_kwargs must be a mapping")
    if not isinstance(data["input_hashes"], Mapping):
        errors.append("input_hashes must be a mapping")
    if not isinstance(data["trust_model"], str):
        errors.append("trust_model must be a string")
    if "signature" in data:
        signature = data["signature"]
        if not isinstance(signature, Mapping):
            errors.append("signature must be a mapping")
        else:
            if signature.get("algorithm") != SIGNATURE_ALGORITHM:
                errors.append(
                    f"unsupported signature algorithm {signature.get('algorithm')!r}"
                )
            # Refuse the legacy ``value`` field at the schema layer:
            # otherwise a relay can strip ``signature_b64`` and
            # substitute a forged ``value`` to make a verifier that
            # only calls ``validate_envelope`` (without
            # ``verify_envelope_signature``) report "OK, signed" on a
            # forged envelope. ``verify_envelope_signature`` still
            # accepts the legacy field for back-compat reads of
            # already-signed v0.2.2 envelopes.
            if "signature_b64" not in signature:
                errors.append(
                    "signature must include the canonical 'signature_b64' "
                    "field (legacy 'value' field is no longer accepted by "
                    "validate_envelope; re-sign the envelope)."
                )

    try:
        float(data["value"])
    except (TypeError, ValueError):
        errors.append("value must be numeric")

    timestamp_value: float | None = None
    try:
        timestamp_value = float(data["timestamp"])
    except (TypeError, ValueError):
        errors.append("timestamp must be numeric")

    if max_age_seconds is not None and timestamp_value is not None:
        current = float(now) if now is not None else time.time()
        if current - timestamp_value > float(max_age_seconds):
            errors.append("envelope timestamp is older than max_age_seconds")
        if timestamp_value - current > float(max_age_seconds):
            errors.append("envelope timestamp is in the future")

    if min_security_bits is not None and ps is not None:
        if int(ps.security_bits) < int(min_security_bits):
            errors.append(
                f"security_bits {ps.security_bits} below verifier minimum "
                f"{int(min_security_bits)}"
            )

    return errors


def estimate_security_bits(
    poly_modulus_degree: int,
    coeff_mod_total_bits: int,
) -> int:
    """Conservative HE-standard lookup for CKKS classical security.

    Based on the HomomorphicEncryption.org standard (Table 1, classical
    security estimates for the RLWE distribution used by CKKS). Returns
    ``0`` when the parameter pair falls below the 128-bit table — the
    caller should treat that as "unknown / sub-128" and not record a
    misleading claim in the envelope.
    """
    n = int(poly_modulus_degree)
    q = int(coeff_mod_total_bits)
    # (N, max_logQ_for_128bit, max_logQ_for_192bit, max_logQ_for_256bit)
    table = [
        (1024, 27, 19, 14),
        (2048, 54, 37, 29),
        (4096, 109, 75, 58),
        (8192, 218, 152, 118),
        (16384, 438, 305, 237),
        (32768, 881, 611, 476),
    ]
    for ring, b128, b192, b256 in table:
        if n == ring:
            if q <= b256:
                return 256
            if q <= b192:
                return 192
            if q <= b128:
                return 128
            return 0
    return 0


def _openfhe_security_bits(ctx: CKKSContext) -> int:
    """Best-effort security estimate for an OpenFHE context.

    OpenFHE's ``HEStd`` enum maps directly onto the HE-standard
    classical security levels. When the binding is not available, fall
    back to the documented default (128-bit).
    """
    cc = getattr(ctx.raw, "crypto_context", None)
    if cc is None:
        return 128
    try:
        params = cc.GetCryptoParameters()
        level = getattr(params, "GetSecurityLevel", None)
        if level is None:
            return 128
        name = str(level()).rsplit(".", 1)[-1]
    except Exception:
        return 128
    if "192" in name:
        return 192
    if "256" in name:
        return 256
    return 128


def parameter_set_from_context(ctx: CKKSContext, *, depth: int | None = None) -> ParameterSet:
    """Build a :class:`ParameterSet` from an active CKKS context.

    ``depth`` is the *declared* multiplicative-depth budget. When omitted
    we infer it from the context (``len(coeff_mod_bit_sizes) - 2`` for
    TenSEAL; defaults to 6 for OpenFHE which sets the depth at build time).
    """
    backend_label = ctx.backend
    backend_version = ""
    if ctx.backend_name == "tenseal":
        try:
            import tenseal as _ts
            backend_version = getattr(_ts, "__version__", "")
        except Exception:
            pass
        coeff = tuple(getattr(ctx.raw, "coeff_mod_bit_sizes", ()) or ())
    else:  # openfhe
        try:
            import openfhe as _of
            backend_version = getattr(_of, "__version__", "")
        except Exception:
            pass
        # OpenFHE chooses the chain internally; we record the depth and
        # scaling_factor_bits, leaving the specific primes opaque.
        coeff = ()
    if coeff:
        sec_bits = estimate_security_bits(int(ctx.poly_modulus_degree), sum(coeff))
    elif ctx.backend_name == "openfhe":
        # OpenFHE picks the modulus chain itself to satisfy the
        # configured ``SetSecurityLevel`` (default ``HEStd_128_classic``).
        # Read the level off the context if exposed; otherwise record
        # the documented default of 128-bit. Recording 0 here would
        # cause every OpenFHE envelope to fail default validation even
        # though OpenFHE has actually built a 128-bit-secure context.
        sec_bits = _openfhe_security_bits(ctx)
    else:
        # Unknown backend with no coefficient chain — record 0 to
        # signal "unknown" rather than overclaim.
        sec_bits = 0
    if depth is None:
        depth = max(0, len(coeff) - 2) if coeff else 6
    # Defensive: refuse to claim an oversized chain in the envelope —
    # the dataclass __post_init__ also enforces this, but matching the
    # check here surfaces the failure with full context.
    from .context import _MAX_COEFF_MODULI  # local import to avoid cycle

    if len(coeff) > _MAX_COEFF_MODULI:
        raise ValueError(
            f"context.coeff_mod_bit_sizes has {len(coeff)} entries "
            f"(> {_MAX_COEFF_MODULI}); refusing to record."
        )
    # CKKS scales are always set to a power of two by build_context.
    # ``int(scale).bit_length() - 1`` is exact for that case; the
    # earlier ``round(scale).bit_length() - 1`` form was off-by-one
    # whenever ``scale`` came back as ``2^k - 1`` from rescaling
    # arithmetic.
    scale_int = int(ctx.scale)
    scaling_factor_bits = max(0, scale_int.bit_length() - 1)
    return ParameterSet(
        backend=backend_label,
        poly_modulus_degree=int(ctx.poly_modulus_degree),
        security_bits=sec_bits,
        multiplicative_depth=int(depth),
        coeff_mod_bit_sizes=coeff,
        scaling_factor_bits=scaling_factor_bits,
        backend_version=backend_version,
    )
