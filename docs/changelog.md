# Changelog

## 0.2.1

Patch release. Test-coverage closeout, no API changes.

### Tests

- 232 tests pass; line coverage rises from 86% to **94%**.
- New `tests/test_v0_2_1_coverage.py` covers Mode B (encrypted-mask)
  paths for the v0.2 metric ports (per-rate diff/ratio family,
  scoring disaggregations, regression disaggregations), the plaintext
  fallback when upstream Fairlearn is missing the new helpers
  (`equal_opportunity_difference` / `_ratio` and friends), context
  lifecycle (`set_default_context`, `reset_default_context`,
  `make_evaluator_context` for TenSEAL), CLI residuals (stdin,
  oversize, missing public key, legacy `metric` envelope key), audit
  edge paths (small-group warning, no-sensitive-features label,
  encrypted-sensitive-features label), and every
  `validate_envelope` negative branch.
- Pinned the OpenFHE `make_evaluator_context` brokenness as an
  expected `TypeError` (`KeyPair` is unpickleable); a future fix that
  introduces a wrapper around the OpenFHE `KeyPair` should flip the
  test from passing to failing.

### Notes

- Remaining 6% coverage gap is scattered defensive branches and
  OpenFHE backend paths that need the `KeyPair` workaround above.
- No production code changed; existing imports continue to work.

## 0.2.0

Feature release: extended metric coverage, OpenFHE noise-flooding
wiring, multi-subcommand CLI, and a formal envelope JSON Schema.

### Added — metric coverage

- **Per-rate `_difference` / `_ratio` family**:
  `selection_rate_difference` / `_ratio`,
  `true_positive_rate_difference` / `_ratio`,
  `true_negative_rate_difference` / `_ratio`,
  `false_positive_rate_difference` / `_ratio`,
  `false_negative_rate_difference` / `_ratio`. Each delegates to the
  upstream Fairlearn helper for plaintext input and to the
  encrypted circuit + `aggregate_difference` / `aggregate_ratio`
  helpers for encrypted input.
- **Scoring disaggregations**: `accuracy_score_difference`,
  `accuracy_score_group_min`, `balanced_accuracy_score_group_min`,
  `precision_score_group_min`, `recall_score_group_min`,
  `f1_score_group_min`, `zero_one_loss_difference`,
  `zero_one_loss_group_max`, `zero_one_loss_ratio`. All reduce to
  the existing TP/FP/TN/FN ciphertext sums; the per-group score is
  computed in plaintext post-decrypt.
- **Regression disaggregations**: `mean_squared_error_group_max`,
  `mean_absolute_error_group_max`, `r2_score_group_min`. CKKS
  computes per-group RSS via one ct×ct multiply (`ŷ²`); MAE is
  approximated as `sqrt(MSE)` (exact for constant residuals, an
  upper bound otherwise).
- **`selection_rate(pos_label=0)`** is now supported under
  encryption (computed as `1 - selection_rate(pos_label=1)`).
  Other `pos_label` values still raise `NotImplementedError` with
  a clear message — equality against an arbitrary label requires a
  non-polynomial comparator that CKKS cannot evaluate at depth.

### Added — backend / context

- **OpenFHE `noise_flooding`** is now wired through `build_context`.
  Pass `noise_flooding="openfhe-NOISE_FLOODING_DECRYPT"` (or
  `"noise-flooding"`, `True`) to enable OpenFHE's
  `EXEC_NOISE_FLOODING` execution mode where the linked
  openfhe-python build supports it. Underscore / hyphen / case
  variants of the label are normalised. TenSEAL ignores the flag
  with no behaviour change.

### Added — CLI

- **New `fairlearn-fhe` entry point** with subcommands:
  - `verify` — validate an envelope (the original behaviour);
  - `inspect` — pretty-print or JSON-dump an envelope summary
    (`--json`);
  - `schema` — emit the envelope JSON Schema (`--pretty` to indent);
  - `doctor` — show backend availability.
- The legacy `fairlearn-fhe-verify` entry point is preserved and
  maps directly to `fairlearn-fhe verify` so existing CI scripts
  keep working.

### Added — envelope schema

- **`ENVELOPE_JSON_SCHEMA`** (draft-2020-12) and
  `envelope_json_schema()` factory function exposed at the package
  level. Validators can now check envelopes against a formal schema
  rather than the version-tag string.

### Added — docs

- New [Threat model](threat-model.md) page formalising what an
  auditor learns vs does not learn from a verdict, what the
  encrypted-mask Mode B buys, and recommended deployments.

### Notes

- All v0.1.0 imports continue to work unchanged.
- Tests: 154 passed, ruff clean. Coverage 86% (the remaining gap is
  in encrypted-mask Mode B paths of the new metrics — covered for
  the existing demographic-parity / equalized-odds metrics, pending
  for the new regression and scoring helpers).

## 0.1.0

Initial release.

- 12 canonical Fairlearn metrics ported to CKKS.
- `MetricFrame`, `EncryptedMetricFrame`, `make_derived_metric`.
- TenSEAL and OpenFHE backends behind a single dispatch.
- Mode A (encrypted `y_pred`, plaintext sensitive features).
- Mode B (encrypted `y_pred` and sensitive features) via `encrypt_sensitive_features()`.
- `MetricEnvelope` with parameter-set hash, observed depth, op counts.
- `validate_envelope()` for dependency-light audit-envelope verification.
- `fairlearn-fhe-verify` CLI for regulator-side envelope checks.
- Optional Ed25519 envelope signing and signature verification.
- Replay metadata for metric kwargs, trust model, and input hashes.
- `audit_metric()` one-call wrapper for audit envelope output.
- Tests cover edge cases, envelope validation, backend dispatch, encrypted arithmetic, CLI verification, and optional signature verification.
