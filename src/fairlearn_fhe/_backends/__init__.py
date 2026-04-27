"""Backend dispatch.

Two backends share an identical surface:

- ``tenseal`` (default) — pip-installable; CKKS via Microsoft SEAL.
- ``openfhe`` — opt-in; native CKKS via OpenFHE-Python.

A backend exposes:

    build_context(...) -> _BackendContext
    encrypt(ctx, values) -> _BackendCiphertext
    decrypt(ct) -> List[float]
    add / sub / neg / mul_pt / mul_scalar / mul_ct / sum_all

The :class:`fairlearn_fhe.EncryptedVector` wraps the active backend's
ciphertext and dispatches to the matching backend functions.
"""

from __future__ import annotations

from typing import Literal

BackendName = Literal["tenseal", "openfhe"]

_DEFAULT_BACKEND: BackendName = "tenseal"


def get_default_backend() -> BackendName:
    return _DEFAULT_BACKEND


def set_default_backend(name: BackendName) -> None:
    global _DEFAULT_BACKEND
    if name not in ("tenseal", "openfhe"):
        raise ValueError(f"unknown backend {name!r}; choose tenseal or openfhe")
    _DEFAULT_BACKEND = name


def get_backend(name: BackendName | None = None):
    name = name or _DEFAULT_BACKEND
    if name == "tenseal":
        from . import tenseal_backend as mod
        return mod
    if name == "openfhe":
        from . import openfhe_backend as mod
        return mod
    raise ValueError(f"unknown backend {name!r}")


def list_backends() -> list[str]:
    """Return every backend name the dispatcher knows about."""
    return ["tenseal", "openfhe"]
