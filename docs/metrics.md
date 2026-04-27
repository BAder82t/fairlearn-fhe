# Metrics catalogue

All twelve canonical Fairlearn metrics are ported. Each row maps the plaintext function to its encrypted analogue and lists the depth budget consumed.

## Base metrics

| Plaintext | Encrypted | Mode A depth | Mode B depth |
| --- | --- | --- | --- |
| `selection_rate` | `fairlearn_fhe.metrics.selection_rate` | 1 | 2 |
| `mean_prediction` | `fairlearn_fhe.metrics.mean_prediction` | 1 | 2 |
| `true_positive_rate` | `fairlearn_fhe.metrics.true_positive_rate` | 1 | 2 |
| `true_negative_rate` | `fairlearn_fhe.metrics.true_negative_rate` | 1 | 2 |
| `false_positive_rate` | `fairlearn_fhe.metrics.false_positive_rate` | 1 | 2 |
| `false_negative_rate` | `fairlearn_fhe.metrics.false_negative_rate` | 1 | 2 |
| `count` | `fairlearn_fhe.metrics.count` | 0 | 0 |

## Disaggregated fairness metrics

| Plaintext | Encrypted | Aggregation |
| --- | --- | --- |
| `demographic_parity_difference` | `fairlearn_fhe.metrics.demographic_parity_difference` | max-min selection_rate |
| `demographic_parity_ratio` | `fairlearn_fhe.metrics.demographic_parity_ratio` | min/max selection_rate |
| `equalized_odds_difference` | `fairlearn_fhe.metrics.equalized_odds_difference` | `worst_case` or `mean` of {tpr_diff, fpr_diff} |
| `equalized_odds_ratio` | `fairlearn_fhe.metrics.equalized_odds_ratio` | `worst_case` or `mean` of {tpr_ratio, fpr_ratio} |
| `equal_opportunity_difference` | `fairlearn_fhe.metrics.equal_opportunity_difference` | tpr max-min |
| `equal_opportunity_ratio` | `fairlearn_fhe.metrics.equal_opportunity_ratio` | tpr min/max |

Both `method="between_groups"` and `method="to_overall"` are supported, matching plaintext Fairlearn.

## MetricFrame

```python
from fairlearn_fhe.metrics import MetricFrame
import fairlearn.metrics as fl

mf = MetricFrame(
    metrics={"tpr": fl.true_positive_rate, "fpr": fl.false_positive_rate},
    y_true=y_true, y_pred=y_pred_enc,
    sensitive_features=sensitive,
)
mf.by_group       # DataFrame of per-group rates
mf.overall        # Series of overall rates
mf.difference()   # max-min per metric column
mf.ratio()        # min/max per metric column
mf.group_min()
mf.group_max()
```

## make_derived_metric

```python
from fairlearn_fhe.metrics import make_derived_metric
import fairlearn.metrics as fl

dpdiff = make_derived_metric(metric=fl.selection_rate, transform="difference")
disp = dpdiff(y_true, y_pred_enc, sensitive_features=sensitive)
```

Plaintext path delegates to fairlearn (exact equality); encrypted path routes through our `MetricFrame`.

## Caveats

- `selection_rate(pos_label=L)` for `L != 1` requires pre-encoding predictions to `{0, 1}` before encryption (CKKS does not natively support encrypted equality testing for arbitrary labels). The library raises `NotImplementedError` for this case.
- Weighted aggregations (`sample_weight`) are supported throughout, both as flat and nested `sample_params` dicts.
- Custom user metrics fall through `MetricFrame` only with `allow_decrypt=True`, which decrypts ciphertexts and defeats the privacy guarantee. The library refuses to do this silently.
