"""TenSEAL CKKS backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

NAME = "tenseal-ckks"


@dataclass
class TenSEALContext:
    context: object
    scale: float
    poly_modulus_degree: int
    n_slots: int
    coeff_mod_bit_sizes: tuple[int, ...] = ()
    backend: str = NAME


def build_context(
    *,
    poly_modulus_degree: int = 1 << 14,
    scale_bits: int = 40,
    coeff_mod_bit_sizes: Sequence[int] | None = None,
) -> TenSEALContext:
    import tenseal as ts
    if coeff_mod_bit_sizes is None:
        coeff_mod_bit_sizes = [60, 40, 40, 40, 40, 40, 40, 60]
    coeff = tuple(int(b) for b in coeff_mod_bit_sizes)
    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=list(coeff),
    )
    ctx.global_scale = float(2 ** scale_bits)
    ctx.generate_galois_keys()
    ctx.generate_relin_keys()
    return TenSEALContext(
        context=ctx,
        scale=float(2 ** scale_bits),
        poly_modulus_degree=poly_modulus_degree,
        n_slots=poly_modulus_degree // 2,
        coeff_mod_bit_sizes=coeff,
    )


def encrypt(ctx: TenSEALContext, values: Sequence[float]):
    import tenseal as ts
    vals = [float(v) for v in np.asarray(values).ravel()]
    return ts.ckks_vector(ctx.context, vals)


def decrypt(ct, n: int) -> list[float]:
    return list(ct.decrypt())[:n]


def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def neg(a):
    return a * -1.0


def mul_pt(ct, plaintext_list):
    return ct * plaintext_list


def mul_scalar(ct, s: float):
    return ct * float(s)


def mul_ct(a, b):
    return a * b


def sum_all(ct, n: int):
    return ct.sum()
