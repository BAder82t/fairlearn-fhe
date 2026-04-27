# Quickstart

The smallest end-to-end example.

```python
import numpy as np

from fairlearn_fhe import build_context, encrypt
from fairlearn_fhe.metrics import demographic_parity_difference

rng = np.random.default_rng(0)
n = 200
y_true     = rng.integers(0, 2, size=n).astype(float)
y_pred     = rng.integers(0, 2, size=n).astype(float)
sensitive  = rng.choice(["A", "B"], size=n)

ctx = build_context()                     # CKKS, 128-bit security, depth 6
y_pred_enc = encrypt(ctx, y_pred)         # ciphertext

disp = demographic_parity_difference(
    y_true, y_pred_enc, sensitive_features=sensitive,
)
print(f"disparity: {disp:.6f}")
```

Output:

```
plaintext: 0.089009
encrypted: 0.089009
abs error: 1.26e-07
```

## With a MetricFrame

```python
from fairlearn_fhe.metrics import MetricFrame
import fairlearn.metrics as fl

mf = MetricFrame(
    metrics={"tpr": fl.true_positive_rate, "fpr": fl.false_positive_rate},
    y_true=y_true,
    y_pred=y_pred_enc,
    sensitive_features=sensitive,
)

print(mf.by_group)
print("difference:", mf.difference())
print("ratio:     ", mf.ratio())
```

## Audit envelope

`audit_metric` wraps the call and returns an audit envelope with parameter-set hash, observed depth, and op counts. The envelope is structured for logging and verifier tooling; it is not signed by default.

```python
from fairlearn_fhe import audit_metric

env = audit_metric(
    "demographic_parity_difference",
    y_true, y_pred,
    sensitive_features=sensitive,
)
print(env.to_json())
```

```json
{
  "schema_version": "fairlearn-fhe.metric-envelope.v1",
  "metric_name": "demographic_parity_difference",
  "value": 0.089009,
  "parameter_set": {"backend": "tenseal-ckks", "poly_modulus_degree": 16384, ...},
  "parameter_set_hash": "9c3a7b...",
  "observed_depth": 2,
  "op_counts": {"ct_pt_muls": 4, "ct_ct_muls": 0, "rotations": 56, ...},
  "n_samples": 200,
  "n_groups": 2,
  "metric_kwargs": {},
  "trust_model": "plaintext_sensitive_features",
  "input_hashes": {"y_true": "...", "sensitive_features": "..."}
}
```

Verifier-side code can reject malformed or tampered envelopes without importing an FHE backend:

```python
from fairlearn_fhe import validate_envelope

errors = validate_envelope(env.to_dict())
assert errors == []
```

The same verifier is available as a CLI:

```bash
fairlearn-fhe-verify envelope.json
```

Signed envelopes are optional:

```python
from fairlearn_fhe import sign_envelope, verify_envelope_signature

signed = sign_envelope(env, private_key_pem)
assert verify_envelope_signature(signed, public_key_pem) == []
```

## Encrypted sensitive features (Mode B)

```python
from fairlearn_fhe import encrypt_sensitive_features

sf_enc = encrypt_sensitive_features(ctx, sensitive, y_true=y_true)
disp = demographic_parity_difference(y_true, y_pred_enc, sensitive_features=sf_enc)
```

See [Trust models](trust-models.md) for the cost / privacy tradeoff.
