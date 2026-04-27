"""Smallest end-to-end demo from the spec.

Vendor encrypts predictions, auditor holds y_true and sensitive_features
in the clear, computes demographic parity disparity on ciphertext.
"""

import numpy as np

from fairlearn_fhe import build_context, encrypt
from fairlearn_fhe.metrics import demographic_parity_difference

rng = np.random.default_rng(0)
n = 200
y_true = rng.integers(0, 2, size=n).astype(float)
y_pred = rng.integers(0, 2, size=n).astype(float)
sensitive = rng.choice(["A", "B"], size=n)

ctx = build_context()
y_pred_enc = encrypt(ctx, y_pred)

import fairlearn.metrics as fl
plain = fl.demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive)
fhe   = demographic_parity_difference(y_true, y_pred_enc, sensitive_features=sensitive)
print(f"plaintext: {plain:.6f}")
print(f"encrypted: {fhe:.6f}")
print(f"abs error: {abs(plain - fhe):.2e}")
