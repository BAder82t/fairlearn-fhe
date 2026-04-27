# Benchmarks

All numbers measured on n=1024 rows, 3 sensitive groups, depth-6 CKKS circuit, default parameters.

## TenSEAL backend, n=1024

| metric | plaintext (ms) | fhe (ms) | abs error |
| --- | ---: | ---: | ---: |
| selection_rate | 0.04 | 74.67 | 7.15e-7 |
| true_positive_rate | 1.00 | 142.96 | 7.04e-7 |
| false_positive_rate | 0.65 | 140.94 | 7.10e-7 |
| demographic_parity_difference | 16.85 | 278.48 | 9.98e-8 |
| demographic_parity_ratio | 8.13 | 277.60 | 8.40e-9 |
| equalized_odds_difference | 12.73 | 556.94 | 2.10e-7 |
| equal_opportunity_difference | 8.53 | 560.21 | 8.09e-8 |

Context build: ~888 ms. Encryption of 1024 rows: ~7.5 ms.

## OpenFHE backend, n=1024

| metric | fhe (ms) | abs error |
| --- | ---: | ---: |
| demographic_parity_difference | 505 | 2.1e-10 |
| equalized_odds_difference | 1015 | 4.4e-11 |

Context build: ~321 ms. Encryption of 1024 rows: ~13.5 ms.

## Mode A vs Mode B (encrypted sensitive features)

| | Mode A (plaintext sf) | Mode B (encrypted sf) |
| --- | --- | --- |
| Multiplicative depth | 1 | 2 |
| ct×ct mults / metric | 0 | ≥3 (one per group) |
| Latency multiplier | 1× | ≈2× |
| Abs error | typical CKKS noise | same order of magnitude |

## Reproducing

```bash
python benchmarks/bench_metrics.py --n 1024
```
