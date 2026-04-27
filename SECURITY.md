# Security Policy

`fairlearn-fhe` is an early-stage startup project for encrypted fairness-metric
audits. It is not a certified cryptographic product.

## Reporting

Report security issues privately to `b@vaultbytes.com`.

Please include:

- affected version or commit
- backend used (`tenseal` or `openfhe`)
- minimal reproduction, if available
- expected and observed behavior

Do not open public issues for vulnerabilities until a fix or disclosure plan is
agreed.

## Scope

In scope:

- incorrect encrypted metric results
- parameter-set or audit-envelope integrity issues
- accidental plaintext leakage beyond the documented trust model
- dependency or packaging issues that affect users

Out of scope:

- unsupported parameter sets
- attacks requiring access to secret keys or local developer machines
- compliance, legal, or certification claims

