"""OpenFHE-Python CKKS backend.

Provides the same operations as the TenSEAL backend (add / sub / neg /
mul_pt / mul_scalar / mul_ct / sum_all) so the encrypted-vector
abstraction over them is backend-agnostic.

OpenFHE owns the secret/public/eval keys; we keep them on the context
object since OpenFHE-Python ties decryption to a specific keypair.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

NAME = "openfhe-ckks"


@dataclass
class OpenFHEContext:
    crypto_context: object
    keys: object
    scale: float
    poly_modulus_degree: int
    n_slots: int
    backend: str = NAME


def _features():
    import openfhe as fhe
    return [
        fhe.PKESchemeFeature.PKE,
        fhe.PKESchemeFeature.KEYSWITCH,
        fhe.PKESchemeFeature.LEVELEDSHE,
        fhe.PKESchemeFeature.ADVANCEDSHE,
    ]


def build_context(
    *,
    multiplicative_depth: int = 6,
    scaling_mod_size: int = 40,
    batch_size: int = 1024,
) -> OpenFHEContext:
    import openfhe as fhe

    params = fhe.CCParamsCKKSRNS()
    params.SetMultiplicativeDepth(multiplicative_depth)
    params.SetScalingModSize(scaling_mod_size)
    params.SetBatchSize(batch_size)

    cc = fhe.GenCryptoContext(params)
    for f in _features():
        cc.Enable(f)

    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)

    # Halevi-Shoup rotation ladder for sum_all + general access.
    rot_steps: list[int] = []
    step = 1
    while step < batch_size:
        rot_steps.append(step)
        rot_steps.append(-step)
        step *= 2
    cc.EvalRotateKeyGen(keys.secretKey, rot_steps)
    cc.EvalSumKeyGen(keys.secretKey)

    return OpenFHEContext(
        crypto_context=cc,
        keys=keys,
        scale=float(2 ** scaling_mod_size),
        poly_modulus_degree=int(cc.GetRingDimension()),
        n_slots=int(batch_size),
    )


def encrypt(ctx: OpenFHEContext, values: Sequence[float]):
    cc = ctx.crypto_context
    vals = [float(v) for v in np.asarray(values).ravel()]
    pt = cc.MakeCKKSPackedPlaintext(vals)
    return cc.Encrypt(ctx.keys.publicKey, pt)


def decrypt(ct, n: int, ctx: OpenFHEContext) -> list[float]:
    cc = ctx.crypto_context
    pt = cc.Decrypt(ctx.keys.secretKey, ct)
    pt.SetLength(n)
    return list(pt.GetRealPackedValue())[:n]


def add(a, b, ctx: OpenFHEContext):
    return ctx.crypto_context.EvalAdd(a, b)


def sub(a, b, ctx: OpenFHEContext):
    return ctx.crypto_context.EvalSub(a, b)


def neg(a, ctx: OpenFHEContext):
    # EvalNegate is free of multiplicative-level cost; EvalMult(-1.0) would
    # consume a level on OpenFHE.
    return ctx.crypto_context.EvalNegate(a)


def mul_pt(ct, plaintext_list, ctx: OpenFHEContext):
    pt = ctx.crypto_context.MakeCKKSPackedPlaintext(list(plaintext_list))
    return ctx.crypto_context.EvalMult(ct, pt)


def mul_scalar(ct, s: float, ctx: OpenFHEContext):
    return ctx.crypto_context.EvalMult(ct, float(s))


def mul_ct(a, b, ctx: OpenFHEContext):
    return ctx.crypto_context.EvalMult(a, b)


def sum_all(ct, n: int, ctx: OpenFHEContext):
    # Sum exactly the populated slots. EvalSum requires a power-of-two
    # window ≤ the configured batch size; round up to the smallest such
    # window that covers ``n``. Padding slots are zero in our
    # encrypt() path so summing them in is benign.
    if n <= 0:
        return ct
    window = 1
    while window < int(n):
        window <<= 1
    if window > ctx.n_slots:
        window = ctx.n_slots
    return ctx.crypto_context.EvalSum(ct, window)
