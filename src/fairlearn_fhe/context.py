"""CKKS context factory.

Two backends are supported:

- ``tenseal`` (default): pip-installable; CKKS via Microsoft SEAL.
- ``openfhe``: opt-in; native CKKS via OpenFHE-Python.

Use :func:`build_context` to construct one. The returned object is a
backend-specific :class:`CKKSContext` that the rest of the library
accepts uniformly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from ._backends import BackendName, get_backend, get_default_backend


@dataclass
class CKKSContext:
    backend_module: Any        # one of fairlearn_fhe._backends.*
    backend_name: BackendName
    raw: Any                   # backend-specific context
    scale: float
    poly_modulus_degree: int
    n_slots: int

    @property
    def backend(self) -> str:
        # Compatibility alias used by envelope.parameter_set_from_context.
        return getattr(self.raw, "backend", self.backend_name)

    def encrypt_vector(self, values: Sequence[float]):
        return self.backend_module.encrypt(self.raw, values)

    def decrypt_vector(self, ct, n: int) -> list[float]:
        if self.backend_name == "openfhe":
            return self.backend_module.decrypt(ct, n, self.raw)
        return self.backend_module.decrypt(ct, n)


_DEFAULT: CKKSContext | None = None


def default_context() -> CKKSContext:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build_context()
    return _DEFAULT


def build_context(
    *,
    backend: BackendName | None = None,
    poly_modulus_degree: int = 1 << 14,
    scale_bits: int = 40,
    coeff_mod_bit_sizes: Sequence[int] | None = None,
    batch_size: int | None = None,
) -> CKKSContext:
    """Build a CKKS context for the chosen backend.

    The defaults match a depth-6 fairness-metric circuit at 128-bit
    security. For ``backend="openfhe"`` the ``coeff_mod_bit_sizes``
    argument is ignored (OpenFHE chooses the chain itself); pass
    ``batch_size`` to control the number of plaintext slots.
    """
    backend_name: BackendName = backend or get_default_backend()
    mod = get_backend(backend_name)

    if backend_name == "tenseal":
        raw = mod.build_context(
            poly_modulus_degree=poly_modulus_degree,
            scale_bits=scale_bits,
            coeff_mod_bit_sizes=coeff_mod_bit_sizes,
        )
        return CKKSContext(
            backend_module=mod,
            backend_name=backend_name,
            raw=raw,
            scale=raw.scale,
            poly_modulus_degree=raw.poly_modulus_degree,
            n_slots=raw.n_slots,
        )
    if backend_name == "openfhe":
        bs = batch_size or (poly_modulus_degree // 2 if poly_modulus_degree else 1024)
        raw = mod.build_context(
            multiplicative_depth=6,
            scaling_mod_size=scale_bits,
            batch_size=bs,
        )
        return CKKSContext(
            backend_module=mod,
            backend_name=backend_name,
            raw=raw,
            scale=raw.scale,
            poly_modulus_degree=raw.poly_modulus_degree,
            n_slots=raw.n_slots,
        )
    raise ValueError(f"unknown backend {backend_name!r}")
