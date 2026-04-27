# Backends

Two CKKS backends share an identical API. Pick at context construction:

```python
from fairlearn_fhe import build_context

ctx_tenseal = build_context(backend="tenseal")  # default
ctx_openfhe = build_context(backend="openfhe")  # opt-in
```

Or set the process-wide default:

```python
from fairlearn_fhe._backends import set_default_backend
set_default_backend("openfhe")
```

## Comparison

| | TenSEAL | OpenFHE |
| --- | --- | --- |
| Install | `pip install fairlearn-fhe` | `pip install fairlearn-fhe[openfhe]` (C++ build) |
| Underlying lib | Microsoft SEAL | OpenFHE 1.x |
| Default ring | 2¹⁴ | tunable, 2¹⁵ default at depth 6 |
| Latency | faster per metric | ~2× TenSEAL |
| Precision | ~1e-7 abs err | ~1e-10 abs err |
| Bootstrapping | no | yes (not used by fairness metrics) |
| Packaging | pip-installable default | native dependency, opt-in |

## Benchmark (n=1024, 3 sensitive groups, depth-6 circuit)

| backend | ctx build | encrypt | dp_diff | dp abs err | eo_diff | eo abs err |
| --- | --- | --- | --- | --- | --- | --- |
| tenseal | 888 ms | 7.5 ms | 284 ms | 1e-7 | 562 ms | 2e-7 |
| openfhe | 321 ms | 13.5 ms | 505 ms | 2e-10 | 1015 ms | 4e-11 |

On this benchmark, OpenFHE gives lower numeric error; TenSEAL is faster and
ships via pip on every supported platform.

## Backend-specific notes

### TenSEAL

- Default `coeff_mod_bit_sizes`: `[60, 40, 40, 40, 40, 40, 40, 60]` — 6 multiplicative levels at 128-bit security.
- `poly_modulus_degree=2**14` by default; pass `2**15` for higher precision and a longer rotation-key set.

### OpenFHE

- OpenFHE chooses the modulus chain internally given `multiplicative_depth=6`, `scaling_mod_size=40`, `batch_size=N`.
- Pass `batch_size` to `build_context` to override the default slot count (default = `poly_modulus_degree // 2`).
- Galois keys are pregenerated for the Halevi-Shoup ladder; `EvalSum` uses them for `sum_all`.

## Dispatch internals

`fairlearn_fhe._backends` exposes a tiny registry:

```python
from fairlearn_fhe._backends import get_backend, get_default_backend, set_default_backend
```

The `EncryptedVector` wrapper threads the active backend through every operation (`add`, `sub`, `mul_pt`, `mul_ct`, `sum_all`, `decrypt`), so every metric circuit is backend-agnostic.
