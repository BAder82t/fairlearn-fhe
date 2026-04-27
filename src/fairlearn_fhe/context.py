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
    has_secret_key: bool = True

    @property
    def backend(self) -> str:
        # Compatibility alias used by envelope.parameter_set_from_context.
        return getattr(self.raw, "backend", self.backend_name)

    def encrypt_vector(self, values: Sequence[float]):
        return self.backend_module.encrypt(self.raw, values)

    def decrypt_vector(self, ct, n: int) -> list[float]:
        if not self.has_secret_key:
            raise RuntimeError(
                "this context has no secret key (evaluator-only); "
                "ciphertexts must be decrypted by the keyholder."
            )
        if self.backend_name == "openfhe":
            return self.backend_module.decrypt(ct, n, self.raw)
        return self.backend_module.decrypt(ct, n)

    def make_evaluator_context(self) -> CKKSContext:
        """Return a copy of this context with the secret key removed.

        Use this when handing the context to an evaluator (e.g. the
        auditor in a Vendor→Auditor split): the evaluator can perform
        arithmetic and rotations but cannot decrypt. The keyholder
        retains the original secret-key context for the final decrypt.

        For TenSEAL this calls
        ``context.make_context_public()``; for OpenFHE this clones the
        :class:`OpenFHEContext` with ``keys.secretKey`` set to
        ``None``.
        """
        if self.backend_name == "tenseal":
            import copy
            new_raw = copy.copy(self.raw)
            inner = self.raw.context
            # TenSEAL exposes ``make_context_public`` to drop the secret key.
            try:
                pub = inner.copy()
                pub.make_context_public()
            except AttributeError:
                pub = inner
                pub.make_context_public()
            new_raw.context = pub
            return CKKSContext(
                backend_module=self.backend_module,
                backend_name=self.backend_name,
                raw=new_raw,
                scale=self.scale,
                poly_modulus_degree=self.poly_modulus_degree,
                n_slots=self.n_slots,
                has_secret_key=False,
            )
        if self.backend_name == "openfhe":
            import copy
            new_raw = copy.copy(self.raw)
            new_keys = copy.copy(self.raw.keys)
            new_keys.secretKey = None
            new_raw.keys = new_keys
            return CKKSContext(
                backend_module=self.backend_module,
                backend_name=self.backend_name,
                raw=new_raw,
                scale=self.scale,
                poly_modulus_degree=self.poly_modulus_degree,
                n_slots=self.n_slots,
                has_secret_key=False,
            )
        raise ValueError(f"unknown backend {self.backend_name!r}")


_DEFAULT: CKKSContext | None = None


def default_context() -> CKKSContext:
    """Return the lazily-initialised process-wide default context.

    The same object is returned on every call until
    :func:`reset_default_context` is invoked. Tests that want a clean
    context per case should pass an explicit ``ctx`` instead of relying
    on the default.
    """
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = build_context()
    return _DEFAULT


def set_default_context(ctx: CKKSContext) -> None:
    """Install ``ctx`` as the process-wide default context."""
    global _DEFAULT
    _DEFAULT = ctx


def reset_default_context() -> None:
    """Drop the cached default context (next call rebuilds)."""
    global _DEFAULT
    _DEFAULT = None


def build_context(
    *,
    backend: BackendName | None = None,
    poly_modulus_degree: int = 1 << 14,
    scale_bits: int = 40,
    coeff_mod_bit_sizes: Sequence[int] | None = None,
    batch_size: int | None = None,
    noise_flooding=None,
) -> CKKSContext:
    """Build a CKKS context for the chosen backend.

    The defaults match a depth-6 fairness-metric circuit at 128-bit
    security. For ``backend="openfhe"`` the ``coeff_mod_bit_sizes``
    argument is ignored (OpenFHE chooses the chain itself); pass
    ``batch_size`` to control the number of plaintext slots.

    ``noise_flooding`` enables decrypt-time noise flooding on backends
    that support it. Pass either ``True`` or one of the recognized
    strings (``"openfhe-NOISE_FLOODING_DECRYPT"``, ``"noise-flooding"``).
    Currently honoured by the OpenFHE backend only; TenSEAL ignores
    the flag because TenSEAL/SEAL has no NOISE_FLOODING_DECRYPT
    execution mode at the time of this release.
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
            noise_flooding=noise_flooding,
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
