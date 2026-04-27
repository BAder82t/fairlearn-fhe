# Contributing

Thanks for considering a contribution to `fairlearn-fhe`.

This project is intentionally small and focused: encrypted Fairlearn-compatible
metrics, audit envelopes, and regulator-facing verification primitives above
existing FHE backends.

## Development

Install in editable mode:

```bash
python -m pip install -e ".[dev,docs]"
```

Run checks before opening a pull request:

```bash
python -m pytest
python -m ruff check .
python -m mkdocs build --strict
python -m build
python -m twine check dist/*
```

The GitHub Actions release workflow can publish to TestPyPI manually, or to
PyPI from a GitHub release, once trusted publishing is configured for the
`testpypi` and `pypi` environments.

## Contribution Guidelines

- Keep API changes compatible with Fairlearn where practical.
- Document trust-model or leakage changes in `docs/trust-models.md`.
- Add tests for every metric, backend, or envelope behavior change.
- Avoid cryptographic-core changes; this package builds application, audit, and
  developer-experience layers on top of TenSEAL and OpenFHE.
