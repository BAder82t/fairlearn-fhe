import numpy as np
import pytest

from fairlearn_fhe import build_context, encrypt


@pytest.fixture(scope="session")
def ctx():
    return build_context(poly_modulus_degree=1 << 13, scale_bits=40,
                          coeff_mod_bit_sizes=[60, 40, 40, 60])


@pytest.fixture
def small_dataset():
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n).astype(float)
    y_pred = rng.integers(0, 2, size=n).astype(float)
    sensitive = rng.choice(["A", "B", "C"], size=n)
    return y_true, y_pred, sensitive


@pytest.fixture
def encrypted_pred(ctx, small_dataset):
    _, y_pred, _ = small_dataset
    return encrypt(ctx, y_pred)


_TOLERANCE = 1e-3


@pytest.fixture
def tol():
    return _TOLERANCE
