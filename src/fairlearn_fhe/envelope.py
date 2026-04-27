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
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Tuple

from .context import CKKSContext


@dataclass(frozen=True)
class ParameterSet:
    backend: str
    poly_modulus_degree: int
    security_bits: int
    multiplicative_depth: int
    coeff_mod_bit_sizes: Tuple[int, ...]
    scaling_factor_bits: int
    backend_version: str = ""

    def hash(self) -> str:
        body = json.dumps(asdict(self), sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(body).hexdigest()


@dataclass
class MetricEnvelope:
    metric_name: str
    value: float
    parameter_set: ParameterSet
    observed_depth: int
    op_counts: Dict[str, int]
    n_samples: int
    n_groups: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "value": float(self.value),
            "parameter_set": asdict(self.parameter_set),
            "parameter_set_hash": self.parameter_set.hash(),
            "observed_depth": int(self.observed_depth),
            "op_counts": dict(self.op_counts),
            "n_samples": int(self.n_samples),
            "n_groups": int(self.n_groups),
            "timestamp": float(self.timestamp),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)


def parameter_set_from_context(ctx: CKKSContext, *, depth: int = 6) -> ParameterSet:
    backend_label = ctx.backend
    backend_version = ""
    if ctx.backend_name == "tenseal":
        try:
            import tenseal as _ts
            backend_version = getattr(_ts, "__version__", "")
        except Exception:
            pass
        coeff: Tuple[int, ...] = (60, 40, 40, 40, 40, 40, 40, 60)
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
