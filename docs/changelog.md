# Changelog

## 0.1.0

Initial release.

- 12 canonical Fairlearn metrics ported to CKKS.
- `MetricFrame`, `EncryptedMetricFrame`, `make_derived_metric`.
- TenSEAL and OpenFHE backends behind a single dispatch.
- Mode A (encrypted `y_pred`, plaintext sensitive features).
- Mode B (encrypted `y_pred` and sensitive features) via `encrypt_sensitive_features()`.
- `MetricEnvelope` with parameter-set hash, observed depth, op counts.
- `validate_envelope()` for dependency-light audit-envelope verification.
- `audit_metric()` one-call wrapper for audit envelope output.
- 57 passing tests plus 1 OpenFHE-gated test, including edge cases (multi-column sensitive features, sample weights, large n, single group, all-positive class, soft predictions, envelope validation).
