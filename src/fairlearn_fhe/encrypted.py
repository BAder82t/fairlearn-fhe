"""EncryptedVector — backend-agnostic CKKS ciphertext wrapper.

Operations dispatch to the active backend (tenseal | openfhe). The
arithmetic surface (add / sub / neg / mul_pt / mul_scalar / mul_ct /
sum_all / decrypt) is identical regardless of backend.

Operation counters track depth and per-op tallies for the metric
envelope. Counters are global; reset between calls with
:func:`reset_op_counters`.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .context import CKKSContext

_COUNTER_KEYS = (
    "ct_ct_muls",
    "ct_pt_muls",
    "ct_scalar_muls",
    "rotations",
    "additions",
    "subtractions",
)

OP_COUNTERS: dict[str, int] = {k: 0 for k in _COUNTER_KEYS}
_COUNTER_LOCK = threading.Lock()


def _inc(key: str, by: int = 1) -> None:
    with _COUNTER_LOCK:
        OP_COUNTERS[key] += by


def reset_op_counters() -> None:
    with _COUNTER_LOCK:
        for k in OP_COUNTERS:
            OP_COUNTERS[k] = 0


def snapshot_op_counters() -> dict[str, int]:
    with _COUNTER_LOCK:
        return dict(OP_COUNTERS)


@dataclass
class EncryptedVector:
    ciphertext: object
    n: int
    ctx: CKKSContext
    depth: int = 0

    # ------------------------------------------------------------------
    # Construction / IO
    # ------------------------------------------------------------------

    @classmethod
    def encrypt(cls, ctx: CKKSContext, values: Sequence[float]) -> EncryptedVector:
        vals = [float(v) for v in np.asarray(values).ravel()]
        ct = ctx.encrypt_vector(vals)
        return cls(ciphertext=ct, n=len(vals), ctx=ctx, depth=0)

    def decrypt(self) -> np.ndarray:
        vals = self.ctx.decrypt_vector(self.ciphertext, self.n)
        return np.asarray(vals, dtype=float)[: self.n]

    def first_slot(self) -> float:
        return float(self.decrypt()[0])

    # ------------------------------------------------------------------
    # Arithmetic — backend dispatch
    # ------------------------------------------------------------------

    def _be(self):
        return self.ctx.backend_module

    def _is_openfhe(self) -> bool:
        return self.ctx.backend_name == "openfhe"

    def __add__(self, other) -> EncryptedVector:
        _inc("additions")
        be = self._be()
        if isinstance(other, EncryptedVector):
            if self._is_openfhe():
                new_ct = be.add(self.ciphertext, other.ciphertext, self.ctx.raw)
            else:
                new_ct = be.add(self.ciphertext, other.ciphertext)
            return EncryptedVector(new_ct, self.n, self.ctx, depth=max(self.depth, other.depth))
        addend = _as_list(other, self.n)
        # Plaintext add: route via mul_pt's plaintext encoder when possible;
        # both backends accept ciphertext + Python list directly.
        if self._is_openfhe():
            cc = self.ctx.raw.crypto_context
            pt = cc.MakeCKKSPackedPlaintext(addend)
            new_ct = cc.EvalAdd(self.ciphertext, pt)
        else:
            new_ct = self.ciphertext + addend
        return EncryptedVector(new_ct, self.n, self.ctx, depth=self.depth)

    __radd__ = __add__

    def __sub__(self, other) -> EncryptedVector:
        _inc("subtractions")
        be = self._be()
        if isinstance(other, EncryptedVector):
            if self._is_openfhe():
                new_ct = be.sub(self.ciphertext, other.ciphertext, self.ctx.raw)
            else:
                new_ct = be.sub(self.ciphertext, other.ciphertext)
            return EncryptedVector(new_ct, self.n, self.ctx, depth=max(self.depth, other.depth))
        subv = _as_list(other, self.n)
        if self._is_openfhe():
            cc = self.ctx.raw.crypto_context
            pt = cc.MakeCKKSPackedPlaintext(subv)
            new_ct = cc.EvalSub(self.ciphertext, pt)
        else:
            new_ct = self.ciphertext - subv
        return EncryptedVector(new_ct, self.n, self.ctx, depth=self.depth)

    def __neg__(self) -> EncryptedVector:
        be = self._be()
        if self._is_openfhe():
            new_ct = be.neg(self.ciphertext, self.ctx.raw)
        else:
            new_ct = be.neg(self.ciphertext)
        return EncryptedVector(new_ct, self.n, self.ctx, depth=self.depth)

    def mul_pt(self, plaintext) -> EncryptedVector:
        _inc("ct_pt_muls")
        pt = _as_list(plaintext, self.n)
        be = self._be()
        if self._is_openfhe():
            new_ct = be.mul_pt(self.ciphertext, pt, self.ctx.raw)
        else:
            new_ct = be.mul_pt(self.ciphertext, pt)
        return EncryptedVector(new_ct, self.n, self.ctx, depth=self.depth + 1)

    def mul_scalar(self, s: float) -> EncryptedVector:
        _inc("ct_scalar_muls")
        be = self._be()
        if self._is_openfhe():
            new_ct = be.mul_scalar(self.ciphertext, float(s), self.ctx.raw)
        else:
            new_ct = be.mul_scalar(self.ciphertext, float(s))
        # CKKS scalar mul consumes one multiplicative level on both backends.
        return EncryptedVector(new_ct, self.n, self.ctx, depth=self.depth + 1)

    def mul_ct(self, other: EncryptedVector) -> EncryptedVector:
        _inc("ct_ct_muls")
        be = self._be()
        if self._is_openfhe():
            new_ct = be.mul_ct(self.ciphertext, other.ciphertext, self.ctx.raw)
        else:
            new_ct = be.mul_ct(self.ciphertext, other.ciphertext)
        return EncryptedVector(new_ct, self.n, self.ctx, depth=max(self.depth, other.depth) + 1)

    def sum_all(self) -> EncryptedVector:
        if self.n > 1:
            _inc("rotations", int(np.log2(self.n)))
        be = self._be()
        if self._is_openfhe():
            new_ct = be.sum_all(self.ciphertext, self.n, self.ctx.raw)
        else:
            new_ct = be.sum_all(self.ciphertext, self.n)
        return EncryptedVector(new_ct, self.n, self.ctx, depth=self.depth)


def encrypt(ctx: CKKSContext, values: Sequence[float]) -> EncryptedVector:
    return EncryptedVector.encrypt(ctx, values)


def decrypt(ev: EncryptedVector) -> np.ndarray:
    return ev.decrypt()


def _as_list(value, target_len: int) -> list[float]:
    if isinstance(value, np.ndarray):
        return [float(v) for v in value.ravel().tolist()]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)] * target_len
