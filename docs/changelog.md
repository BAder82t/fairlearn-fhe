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
- `fairlearn-fhe-verify` CLI for regulator-side envelope checks.
- Optional Ed25519 envelope signing and signature verification.
- Replay metadata for metric kwargs, trust model, and input hashes.
- `audit_metric()` one-call wrapper for audit envelope output.
- Tests cover edge cases, envelope validation, backend dispatch, encrypted arithmetic, CLI verification, and optional signature verification.
