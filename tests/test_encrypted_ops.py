"""EncryptedVector arithmetic wrapper behavior."""

import numpy as np

from fairlearn_fhe import decrypt, encrypt, reset_op_counters, snapshot_op_counters


def test_encrypted_vector_arithmetic(ctx, tol):
    a = encrypt(ctx, [1.0, 2.0, 3.0, 4.0])
    b = encrypt(ctx, [0.5, 1.0, 1.5, 2.0])

    reset_op_counters()
    assert np.allclose(decrypt(a + b), [1.5, 3.0, 4.5, 6.0], atol=tol)
    assert np.allclose(decrypt(a + [1.0, 1.0, 1.0, 1.0]), [2.0, 3.0, 4.0, 5.0], atol=tol)
    assert np.allclose(decrypt(a - b), [0.5, 1.0, 1.5, 2.0], atol=tol)
    assert np.allclose(decrypt(a - 1.0), [0.0, 1.0, 2.0, 3.0], atol=tol)
    assert np.allclose(decrypt(-a), [-1.0, -2.0, -3.0, -4.0], atol=tol)
    assert np.allclose(decrypt(a.mul_scalar(2.0)), [2.0, 4.0, 6.0, 8.0], atol=tol)
    assert np.allclose(decrypt(a.mul_ct(b)), [0.5, 2.0, 4.5, 8.0], atol=tol)
    assert abs(a.sum_all().first_slot() - 10.0) < tol

    counts = snapshot_op_counters()
    assert counts["additions"] >= 2
    assert counts["subtractions"] >= 2
    assert counts["ct_scalar_muls"] == 1
    assert counts["ct_ct_muls"] == 1
    assert counts["rotations"] >= 2


def test_plaintext_conversion_shapes(ctx, tol):
    ev = encrypt(ctx, np.array([[1.0, 2.0], [3.0, 4.0]]))
    assert ev.n == 4
    assert np.allclose(decrypt(ev + (1.0, 1.0, 1.0, 1.0)), [2.0, 3.0, 4.0, 5.0], atol=tol)
