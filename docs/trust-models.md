# Trust models

fairlearn-fhe supports two postures. Both keep `y_pred` encrypted; they differ in how `sensitive_features` are handled.

## Mode A — encrypted predictions, plaintext sensitive features (default)

| Input | Form |
| --- | --- |
| `y_true` | plaintext (auditor) |
| `y_pred` | ciphertext |
| `sensitive_features` | plaintext (auditor) |
| Group counts | plaintext |

**Cost per metric:** ct×pt + slot-sum → multiplicative depth 1.

**When to use:** the auditor knows demographic membership for each row (most regulatory audits) but the vendor wants to keep model outputs confidential. This matches the regaudit-fhe `audit_fairness` convention.

```python
from fairlearn_fhe import build_context, encrypt
from fairlearn_fhe.metrics import demographic_parity_difference

ctx = build_context()
y_pred_enc = encrypt(ctx, y_pred)
disp = demographic_parity_difference(
    y_true, y_pred_enc, sensitive_features=sensitive,
)
```

## Mode B — encrypted predictions and sensitive features

| Input | Form |
| --- | --- |
| `y_true` | plaintext (auditor) |
| `y_pred` | ciphertext |
| `sensitive_features` | ciphertext (one-hot per group) |
| Group counts | plaintext (revealed metadata) |
| Per-group positive/negative counts | plaintext (revealed metadata) |

**Cost per metric:** ct×ct + ct×pt + slot-sum → multiplicative depth 2. About 2× the Mode A latency.

**When to use:** the vendor and the auditor are in different trust zones, and the auditor should not learn per-row demographic labels. The marginal counts (group sizes, per-group positive/negative counts) are unavoidably revealed; without them no fairness disparity can be computed.

```python
from fairlearn_fhe import build_context, encrypt, encrypt_sensitive_features
from fairlearn_fhe.metrics import demographic_parity_difference, equalized_odds_difference

ctx = build_context()
y_pred_enc = encrypt(ctx, y_pred)
sf_enc     = encrypt_sensitive_features(ctx, sensitive, y_true=y_true)

dp = demographic_parity_difference(y_true, y_pred_enc, sensitive_features=sf_enc)
eo = equalized_odds_difference(y_true, y_pred_enc, sensitive_features=sf_enc)
```

The `y_true=` argument at encryption time stamps the per-group positive/negative counts the auditor will need for confusion-matrix-based metrics.

## What stays plaintext in both modes

- **`y_true`** — the auditor's ground truth labels.
- **Group counts** — needed for any rate denominator.
- **Per-group positive/negative counts** — needed for TPR/FPR/TNR/FNR denominators.

This is not a regression vs the encrypted ideal; it is a property of the metric definitions. A "TPR per group" formula is `sum(y_pred · y_true · mask) / sum(y_true · mask)`. The denominator carries information about the joint distribution of label and group, and any mechanism that produces a usable rate must reveal it.

## What is never revealed

- Individual `y_pred` values.
- In Mode B, individual `sensitive_features` row labels.
- The values of intermediate ciphertexts at any point in the circuit.
