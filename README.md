# fairlearn-fhe

Drop-in encrypted Fairlearn metrics. Identical API surface; ciphertext arithmetic over CKKS via TenSEAL.

`fairlearn-fhe` is an early-stage project maintained at
<https://github.com/BAder82t/fairlearn-fhe>.

```python
# plaintext
from fairlearn.metrics import demographic_parity_difference
disp = demographic_parity_difference(y_true, y_pred, sensitive_features=A)

# encrypted (one import change)
from fairlearn_fhe.metrics import demographic_parity_difference
from fairlearn_fhe import build_context, encrypt
ctx = build_context()
y_p_enc = encrypt(ctx, y_pred)
disp = demographic_parity_difference(y_true, y_p_enc, sensitive_features=A)
```

`disp` is numerically equivalent to the plaintext result within CKKS noise tolerance (`< 1e-4` abs error in default settings).

## Trust models

Two modes are supported. The default ports the regaudit-fhe convention; the second goes further.

### Mode A — encrypted predictions, plaintext sensitive features (default)

- **Encrypted:** `y_pred`.
- **Plaintext:** `y_true`, `sensitive_features`, group counts.
- **Cost:** one ct×pt multiply + slot-sum per group → depth 1.

### Mode B — fully-encrypted predictions and sensitive features

```python
from fairlearn_fhe import build_context, encrypt, encrypt_sensitive_features
from fairlearn_fhe.metrics import demographic_parity_difference

ctx = build_context()
y_pred_enc = encrypt(ctx, y_pred)
sf_enc     = encrypt_sensitive_features(ctx, sensitive_features, y_true=y_true)

disp = demographic_parity_difference(y_true, y_pred_enc, sensitive_features=sf_enc)
```

- **Encrypted:** `y_pred`, the per-row group-membership masks.
- **Plaintext (auditor metadata):** group counts, per-group positive/negative counts (passed via `y_true=` at encryption time).
- **Cost:** ct×ct + ct×pt + slot-sum per group → depth 2.

`y_true` remains plaintext in both modes (it is the auditor's ground truth). The denominators of TPR/FPR-style metrics — per-group positive/negative counts — are always revealed: there is no fairness signal without them.

Per-group rates are decrypted at the audit boundary; final aggregation (`max`, `min`, ratio, difference) runs on those K plaintext scalars.

## Supported metrics

| Plaintext name | Encrypted? | Mechanism |
| --- | --- | --- |
| `selection_rate` | yes | sum(y_pred·mask)/n_g |
| `true_positive_rate` | yes | sum(y_pred·y_true·mask)/sum(y_true·mask) |
| `true_negative_rate` | yes | sum((1-y_pred)·(1-y_true)·mask)/sum((1-y_true)·mask) |
| `false_positive_rate` | yes | sum(y_pred·(1-y_true)·mask)/sum((1-y_true)·mask) |
| `false_negative_rate` | yes | sum((1-y_pred)·y_true·mask)/sum(y_true·mask) |
| `mean_prediction` | yes | sum(y_pred·mask)/n_g |
| `demographic_parity_difference` | yes | max-min selection_rate over groups |
| `demographic_parity_ratio` | yes | min/max selection_rate over groups |
| `equalized_odds_difference` | yes | max(tpr_diff, fpr_diff) |
| `equalized_odds_ratio` | yes | min(tpr_ratio, fpr_ratio) |
| `equal_opportunity_difference` | yes | tpr max-min |
| `equal_opportunity_ratio` | yes | tpr min/max |

Plus `MetricFrame.fhe()` returning an `EncryptedMetricFrame`.

## Backends

Two CKKS backends share a single API:

```python
from fairlearn_fhe import build_context

ctx_tenseal = build_context(backend="tenseal")  # default; pip-installable
ctx_openfhe = build_context(backend="openfhe")  # native OpenFHE backend, opt-in
```

Benchmarked on n=1024, 3 sensitive groups, depth-6 circuit:

| backend | ctx build | encrypt | dp_diff | dp abs err | eo_diff | eo abs err |
|---|---|---|---|---|---|---|
| tenseal | 888 ms | 7.5 ms | 284 ms | 1e-7 | 562 ms | 2e-7 |
| openfhe | 321 ms | 13.5 ms | 505 ms | 2e-10 | 1015 ms | 4e-11 |

On the included benchmark, OpenFHE gives lower numeric error; TenSEAL is faster
per metric and ships via pip on every supported platform.

## Install

```bash
pip install fairlearn-fhe          # tenseal backend
pip install fairlearn-fhe[openfhe] # add openfhe backend (requires C++ build)
```

## License

Apache-2.0. Compatible with Fairlearn (MIT).
