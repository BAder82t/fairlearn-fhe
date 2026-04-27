# Threat model

This document specifies what an auditor learns from a `fairlearn-fhe`
verdict and what they don't. Read alongside [trust-models.md](trust-models.md),
which describes which inputs are encrypted in each mode.

## Roles

- **Data steward** holds plaintext `y_true`, the per-row sensitive
  features, and the ML model that produced `y_pred`. Encrypts `y_pred`
  (Mode A) and optionally the sensitive-feature mask matrix (Mode B)
  before sharing them with the auditor.
- **Auditor** holds plaintext `y_true`, plaintext sensitive features
  (Mode A) or only the auditor-public group counts (Mode B), and runs
  the encrypted metric. Has the secret key in the default deployment;
  in a separated-trust deployment the secret key is held by a third
  party that decrypts only the K per-group rates.
- **Adversary** is anyone outside the (data steward, auditor, key
  holder) circle.

## What the auditor learns from a verdict

In **both modes** the auditor sees:

1. The K plaintext per-group rates (selection rate, TPR, FPR, …) at the
   audit boundary. This is unavoidable: a fairness audit is meaningless
   without the per-group disaggregation.
2. The aggregated verdict (e.g. `demographic_parity_difference`,
   `equalized_odds_ratio`).
3. Group counts (denominators of the rates). Either supplied by the
   auditor (Mode A) or stamped onto `EncryptedMaskSet` as plaintext
   metadata (Mode B).
4. The `MetricEnvelope` — backend, parameter set, observed depth, op
   counts, input hashes, and timestamp. None of those leak the
   underlying `y_pred` values.

In **Mode A** (default) the auditor *also* holds plaintext
sensitive features, because the auditor passes them in.

In **Mode B** the auditor sees:

5. The encrypted per-row group-membership masks. Decrypting any of
   them recovers per-row membership; the harness assumes the auditor
   does not have the secret key in this mode (or only uses it to
   decrypt the K aggregate rates).

## What the auditor does NOT learn

- **Individual `y_pred` values** — encrypted under the steward's key.
  The audit boundary decrypts only the K aggregate rates per metric,
  not the n-vector of predictions.
- **Individual sensitive feature values** in Mode B (subject to the
  caveat above about secret-key custody).
- **`y_true` values from the steward's perspective** — `y_true` is an
  auditor input and stays in plaintext on the auditor side. If the
  steward also wanted to keep `y_true` private, that requires a
  different protocol entirely (joint encryption + comparison
  primitives), out of scope for this release.

## What the verdict implies for the model owner

A `VULNERABLE`-style fairness verdict (e.g. `demographic_parity_difference
> 0.1`) reveals **only that the gap exists at the level the auditor
configured**. It does not reveal:

- The direction of the gap (which group is favoured) — though this is
  often inferable from the per-group rates the auditor sees.
- Individual prediction errors.
- Any information about samples from groups that did not pass the
  `min_group_size` floor (those are warned and SKIPPED before the
  ciphertext circuit runs).

The audit envelope is the artefact intended to be shared **with third
parties** (regulators, model-owner internal review, downstream
consumers). It carries hashes and counts, never raw vectors.

## CKKS-specific caveats

CKKS is an *approximate* scheme. Per-group rates carry CKKS noise on the
order of the configured scale. Default settings yield `< 1e-4` absolute
error on the included benchmark. If a fairness gap is smaller than this
floor, the verdict is in the noise band and should not be trusted as a
hard pass/fail signal.

`build_context(noise_flooding=...)` enables OpenFHE's
`NOISE_FLOODING_DECRYPT` execution mode where supported, which raises
the noise-floor of decrypted values to mask the LWE error and harden
the configuration against the IND-CPA-D class of attacks (Cheon et al.
2024/127). The mitigation costs depth budget; the default is off
because most fairness audits do not need it. Enable it when:

- the auditor cannot be trusted to handle the encrypted rate vector
  responsibly (e.g. they share the secret key with a wider circle), or
- the deployment is a multi-party threshold setup where the
  decryption oracle is exposed to non-trusted parties.

## Out of scope

- Side-channel attacks on the underlying SEAL/OpenFHE binary
  (timing, power, EM). `fairlearn-fhe` does not measure these; for
  side-channel hardening see the upstream library's documentation
  and consider running the
  [fhe-attack-replay](https://github.com/BAder82t/fhe-attack-replay)
  harness against your build.
- Fault injection on the accelerator (GlitchFHE-style attacks). Same
  pointer.
- Plaintext steganography in `y_true` or sensitive features —
  protecting against a colluding data steward is out of scope.

## Recommended deployment

For a routine internal fairness audit:

1. Use **Mode A** with TenSEAL at the default 128-bit parameters.
2. Set `min_group_size=30` (the default) or higher for any group whose
   rate the auditor will share publicly.
3. Sign envelopes with `sign_envelope` if the verdict will be sent to a
   regulator.
4. Verify envelopes with `fairlearn-fhe verify --public-key …
   --max-age 86400 --min-security-bits 128`.

For an external regulator-facing audit:

1. Switch to **Mode B** so the auditor never holds the per-row
   sensitive features.
2. Have a third party hold the secret key and decrypt only the K
   per-group rates.
3. Use `build_context(noise_flooding="openfhe-NOISE_FLOODING_DECRYPT")`
   when the OpenFHE backend is available.
