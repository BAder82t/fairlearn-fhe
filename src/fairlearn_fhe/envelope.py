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


@dataclass(frozen=True)
class ParameterSet:
    backend: str
    poly_modulus_degree: int
    security_bits: int
    multiplicative_depth: int
    coeff_mod_bit_sizes: tuple[int, ...]
    scaling_factor_bits: int
    backend_version: str = ""

    def hash(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(body).hexdigest()

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ParameterSet:
        return cls(
            backend=str(payload["backend"]),
            poly_modulus_degree=int(payload["poly_modulus_degree"]),
            security_bits=int(payload["security_bits"]),
            multiplicative_depth=int(payload["multiplicative_depth"]),
            coeff_mod_bit_sizes=tuple(int(v) for v in payload["coeff_mod_bit_sizes"]),
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
        "value": b64encode(signature).decode("ascii"),
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

    try:
        signature_bytes = b64decode(str(signature["value"]), validate=True)
    except (KeyError, ValueError):
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


def validate_envelope(
    payload: Mapping[str, Any] | MetricEnvelope,
    *,
    allowed_metrics: Sequence[str] | None = None,
    max_observed_depth: int | None = None,
) -> list[str]:
    """Return validation errors for an audit-envelope payload.

    An empty list means the envelope is structurally valid, the
    ``parameter_set_hash`` matches the embedded parameter set, and the
    observed depth is within the declared multiplicative-depth budget.
    This is intentionally dependency-free so it can run in lightweight
    verifier tooling.
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

    for field_name in ("n_samples", "n_groups"):
        try:
            if int(data[field_name]) < 1:
                errors.append(f"{field_name} must be positive")
        except (TypeError, ValueError):
            errors.append(f"{field_name} must be an integer")

    if not isinstance(data["op_counts"], Mapping):
        errors.append("op_counts must be a mapping")
    else:
        for name, value in data["op_counts"].items():
            try:
                if int(value) < 0:
                    errors.append(f"op_counts[{name!r}] must be non-negative")
            except (TypeError, ValueError):
                errors.append(f"op_counts[{name!r}] must be an integer")

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
        elif signature.get("algorithm") != SIGNATURE_ALGORITHM:
            errors.append(f"unsupported signature algorithm {signature.get('algorithm')!r}")

    try:
        float(data["value"])
    except (TypeError, ValueError):
        errors.append("value must be numeric")

    try:
        float(data["timestamp"])
    except (TypeError, ValueError):
        errors.append("timestamp must be numeric")

    return errors


def parameter_set_from_context(ctx: CKKSContext, *, depth: int = 6) -> ParameterSet:
    backend_label = ctx.backend
    backend_version = ""
    if ctx.backend_name == "tenseal":
        try:
            import tenseal as _ts
            backend_version = getattr(_ts, "__version__", "")
        except Exception:
            pass
        coeff: tuple[int, ...] = (60, 40, 40, 40, 40, 40, 40, 60)
    else:  # openfhe
        try:
            import openfhe as _of
            backend_version = getattr(_of, "__version__", "")
        except Exception:
            pass
        # OpenFHE chooses the chain internally; we record the depth and
        # scaling_factor_bits, leaving the specific primes opaque.
        coeff = ()
    return ParameterSet(
        backend=backend_label,
        poly_modulus_degree=int(ctx.poly_modulus_degree),
        security_bits=128,
        multiplicative_depth=int(depth),
        coeff_mod_bit_sizes=coeff,
        scaling_factor_bits=int(round(ctx.scale).bit_length() - 1),
        backend_version=backend_version,
    )
