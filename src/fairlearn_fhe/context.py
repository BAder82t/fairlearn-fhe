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

        For TenSEAL this serialises the inner context **without** the
        secret key and reloads it into a fully independent copy — the
        keyholder's original context is never mutated. For OpenFHE
        this clones the :class:`OpenFHEContext` with
        ``keys.secretKey`` set to ``None``.
        """
        if self.backend_name == "tenseal":
            import tenseal as ts

            from ._backends.tenseal_backend import TenSEALContext
            inner = self.raw.context
            # Serialise without the secret key, then reload — guarantees
            # an independent context object (the original is never
            # mutated, and the new one cannot decrypt).
            try:
                buf = inner.serialize(
                    save_public_key=True,
                    save_secret_key=False,
                    save_galois_keys=True,
                    save_relin_keys=True,
                )
            except (AttributeError, TypeError) as exc:
                raise RuntimeError(
                    "TenSEAL context does not support serialize(); cannot "
                    "produce an evaluator-only copy without risking the "
                    "secret key. Upgrade tenseal to >=0.3.14."
                ) from exc
            pub = ts.context_from(buf)
            # Build the wrapper from scratch with explicit fields so no
            # mutable state (including the original SEAL Context handle)
            # is shared between the keyholder's wrapper and the
            # evaluator's. ``copy.copy`` would have aliased every field.
            new_raw = TenSEALContext(
                context=pub,
                scale=self.raw.scale,
                poly_modulus_degree=self.raw.poly_modulus_degree,
                n_slots=self.raw.n_slots,
                coeff_mod_bit_sizes=tuple(self.raw.coeff_mod_bit_sizes),
                backend=self.raw.backend,
            )
            return CKKSContext(
                backend_module=self.backend_module,
                backend_name=self.backend_name,
                raw=new_raw,
                scale=self.scale,
                poly_modulus_degree=self.poly_modulus_degree,
                n_slots=self.n_slots,
                has_secret_key=False,
            )
        # OpenFHE branch is pinned as expected TypeError in tests because
        # the KeyPair binding is unpickleable; coverage skips it.
        if self.backend_name == "openfhe":  # pragma: no cover
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
        # Unreachable from public API: backend_name is constrained to
        # ``Literal["tenseal", "openfhe"]`` and the constructor goes
        # through ``set_default_backend`` which validates. Kept as a
        # final defensive raise.
        raise ValueError(f"unknown backend {self.backend_name!r}")  # pragma: no cover


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


# Conservative cap on the size of a CKKS coefficient-modulus chain. The
# HE-standard table tops out around 8 prime moduli for any ring we
# support; values much above this either exceed the ring's bound (no
# valid security level) or indicate a malformed parameter set.
_MAX_COEFF_MODULI = 64


class InsecureCKKSParametersWarning(UserWarning):
    """Emitted when build_context runs with sub-128-bit security explicitly enabled."""


def _validate_tenseal_params(
    poly_modulus_degree: int,
    coeff_mod_bit_sizes: Sequence[int] | None,
    *,
    insecure_allow_low_security: bool,
) -> None:
    """Reject CKKS parameter combinations that fall below 128-bit security.

    Pass ``insecure_allow_low_security=True`` to bypass — this only
    exists for testing the security-bits accounting itself; production
    callers should never set it. When the bypass is taken we emit an
    :class:`InsecureCKKSParametersWarning` so that a misconfigured
    pipeline that picked up the flag from a config file (rather than a
    deliberate caller decision) at least leaves a trace in logs.
    """
    import warnings

    from .envelope import estimate_security_bits  # local import to avoid cycle

    if coeff_mod_bit_sizes is None:
        return  # backend default is known-safe at the supported ring sizes
    coeff = list(coeff_mod_bit_sizes)
    if len(coeff) < 2:
        raise ValueError(
            "coeff_mod_bit_sizes must have at least two primes "
            "(special + scaling); got "
            f"{len(coeff)}."
        )
    if len(coeff) > _MAX_COEFF_MODULI:
        raise ValueError(
            f"coeff_mod_bit_sizes too long ({len(coeff)} > {_MAX_COEFF_MODULI}); "
            "this is well past any sensible HE-standard chain."
        )
    if any(int(b) <= 0 for b in coeff):
        raise ValueError("coeff_mod_bit_sizes entries must be positive integers")
    sec = estimate_security_bits(int(poly_modulus_degree), sum(int(b) for b in coeff))
    if sec < 128:
        if not insecure_allow_low_security:
            raise ValueError(
                f"build_context refused to construct a context at <128-bit "
                f"security (poly_modulus_degree={poly_modulus_degree}, "
                f"sum(coeff_mod_bit_sizes)={sum(int(b) for b in coeff)} → "
                f"estimated security_bits={sec}). Either reduce the modulus "
                "chain or increase poly_modulus_degree. To run anyway (e.g. "
                "for benchmarking) pass insecure_allow_low_security=True."
            )
        warnings.warn(
            f"build_context running at estimated security_bits={sec} "
            f"(<128) because insecure_allow_low_security=True was set. "
            "Audit envelopes produced under this context will record "
            "the true (low) security level.",
            InsecureCKKSParametersWarning,
            stacklevel=3,
        )


def build_context(
    *,
    backend: BackendName | None = None,
    poly_modulus_degree: int = 1 << 14,
    scale_bits: int = 40,
    coeff_mod_bit_sizes: Sequence[int] | None = None,
    batch_size: int | None = None,
    noise_flooding=None,
    insecure_allow_low_security: bool = False,
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

    Caller-supplied ``coeff_mod_bit_sizes`` are validated against the
    HE-standard 128-bit security table; combinations that fall below
    are rejected unless ``insecure_allow_low_security=True`` is set.
    """
    backend_name: BackendName = backend or get_default_backend()
    mod = get_backend(backend_name)

    if backend_name == "tenseal":
        _validate_tenseal_params(
            poly_modulus_degree,
            coeff_mod_bit_sizes,
            insecure_allow_low_security=insecure_allow_low_security,
        )
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
    # Unreachable: ``get_backend`` above already raises for unknown
    # backend names. Kept as a defensive guard.
    raise ValueError(f"unknown backend {backend_name!r}")  # pragma: no cover
