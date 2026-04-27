# fairlearn-fhe

**Drop-in encrypted Fairlearn metrics.** Identical API surface; ciphertext arithmetic over CKKS via TenSEAL or OpenFHE.

```python
# plaintext (existing Fairlearn workflow)
from fairlearn.metrics import demographic_parity_difference
disp = demographic_parity_difference(y_true, y_pred, sensitive_features=A)

# encrypted (one import change)
from fairlearn_fhe import build_context, encrypt
from fairlearn_fhe.metrics import demographic_parity_difference

ctx = build_context()
y_pred_enc = encrypt(ctx, y_pred)
disp = demographic_parity_difference(y_true, y_pred_enc, sensitive_features=A)
```

`disp` is numerically equivalent to the plaintext result within CKKS noise tolerance (`<1e-4` abs error in default settings; typically `~1e-7` on TenSEAL, `~1e-10` on OpenFHE).

---

## Why

Vendors auditing fairness on protected attributes cannot share raw labels, predictions, or sensitive features with auditors. Fairlearn (1.5K+ stars, the JMLR'23 standard fairness library) has no FHE plugin. fairlearn-fhe ports the canonical metric set to CKKS without changing the call signature.

## What is shipped

- 12 canonical metrics (selection_rate, mean_prediction, true/false positive/negative rate, demographic parity diff/ratio, equalized odds diff/ratio, equal opportunity diff/ratio).
- `MetricFrame` and `make_derived_metric` with the same `.by_group`, `.overall`, `.difference()`, `.ratio()`, `.group_min()`, `.group_max()` accessors.
- Two trust models: encrypted `y_pred` only (Mode A); encrypted `y_pred` + sensitive features (Mode B).
- Two CKKS backends: TenSEAL (pip) and OpenFHE (production-grade, opt-in).
- A canonical `MetricEnvelope` with parameter-set hash, observed depth, and op counts for regulator logging.

## Trust posture

- `y_true` is plaintext — it is the auditor's ground truth.
- `y_pred` is encrypted (always).
- `sensitive_features` are plaintext (Mode A) or encrypted with revealed group counts (Mode B).
- Group-size denominators (and TPR/FPR positive/negative counts) are auditor-public metadata. There is no fairness signal without them.

See [Trust models](trust-models.md) for the full posture.

## Install

```bash
pip install fairlearn-fhe          # tenseal backend
pip install fairlearn-fhe[openfhe] # add openfhe backend (requires C++ build)
```

## License

Apache-2.0. Compatible with Fairlearn (MIT).
