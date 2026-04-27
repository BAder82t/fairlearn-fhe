"""Encrypted-vs-plaintext timing for the canonical metric set.

Run::

    python benchmarks/bench_metrics.py --n 1024

Output is a table ready to drop into the README.
"""

from __future__ import annotations

import argparse
import time
from contextlib import contextmanager

import numpy as np

import fairlearn.metrics as fl
from fairlearn_fhe import build_context, encrypt
from fairlearn_fhe.metrics import (
    demographic_parity_difference,
    demographic_parity_ratio,
    equalized_odds_difference,
    equal_opportunity_difference,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
)


@contextmanager
def timer():
    t0 = time.perf_counter()
    out = {}
    try:
        yield out
    finally:
        out["elapsed"] = time.perf_counter() - t0


def main(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sf = rng.choice(["A", "B", "C"], size=n)

    print(f"Building CKKS context (poly_modulus_degree=2^14, depth=6)...")
    with timer() as t_ctx:
        ctx = build_context()
    with timer() as t_enc:
        y_pred_enc = encrypt(ctx, y_pred)
    print(f"  context: {t_ctx['elapsed']*1000:.1f} ms")
    print(f"  encrypt n={n}: {t_enc['elapsed']*1000:.1f} ms")

    cases = [
        ("selection_rate", lambda: fl.selection_rate(y_true, y_pred),
         lambda: selection_rate(y_true, y_pred_enc)),
        ("true_positive_rate", lambda: fl.true_positive_rate(y_true, y_pred),
         lambda: true_positive_rate(y_true, y_pred_enc)),
        ("false_positive_rate", lambda: fl.false_positive_rate(y_true, y_pred),
         lambda: false_positive_rate(y_true, y_pred_enc)),
        ("demographic_parity_difference",
         lambda: fl.demographic_parity_difference(y_true, y_pred, sensitive_features=sf),
         lambda: demographic_parity_difference(y_true, y_pred_enc, sensitive_features=sf)),
        ("demographic_parity_ratio",
         lambda: fl.demographic_parity_ratio(y_true, y_pred, sensitive_features=sf),
         lambda: demographic_parity_ratio(y_true, y_pred_enc, sensitive_features=sf)),
        ("equalized_odds_difference",
         lambda: fl.equalized_odds_difference(y_true, y_pred, sensitive_features=sf),
         lambda: equalized_odds_difference(y_true, y_pred_enc, sensitive_features=sf)),
        ("equal_opportunity_difference",
         lambda: fl.equal_opportunity_difference(y_true, y_pred, sensitive_features=sf),
         lambda: equal_opportunity_difference(y_true, y_pred_enc, sensitive_features=sf)),
    ]

    print()
    print(f"{'metric':<35} {'plain (ms)':>11} {'fhe (ms)':>11} {'abs_err':>10}")
    print("-" * 70)
    for name, plain_fn, fhe_fn in cases:
        with timer() as tp:
            p = plain_fn()
        with timer() as tf:
            f = fhe_fn()
        print(f"{name:<35} {tp['elapsed']*1000:>10.2f} {tf['elapsed']*1000:>10.2f} {abs(p-f):>10.2e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1024)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    main(args.n, args.seed)
