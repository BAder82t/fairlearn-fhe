"""Command-line tools for fairlearn-fhe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .envelope import validate_envelope, verify_envelope_signature


def _read_json(path: str) -> dict[str, Any]:
    body = sys.stdin.read() if path == "-" else Path(path).read_text(encoding="utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("envelope JSON must be an object")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="fairlearn-fhe-verify",
        description="Validate a fairlearn-fhe audit envelope.",
    )
    parser.add_argument("envelope", help="Path to envelope JSON, or '-' for stdin")
    parser.add_argument(
        "--allowed-metric",
        action="append",
        default=None,
        help="Allowed metric name. Repeat to allow multiple metrics.",
    )
    parser.add_argument("--max-depth", type=int, default=None, help="Maximum observed depth.")
    parser.add_argument("--public-key", help="PEM Ed25519 public key for signature verification.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    args = parser.parse_args(argv)

    try:
        payload = _read_json(args.envelope)
        errors = validate_envelope(
            payload,
            allowed_metrics=args.allowed_metric,
            max_observed_depth=args.max_depth,
        )
        if args.public_key:
            key = Path(args.public_key).read_bytes()
            errors.extend(verify_envelope_signature(payload, key))
    except Exception as exc:  # noqa: BLE001 - CLI should turn all failures into exit code 2.
        errors = [str(exc)]

    if args.json:
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
    elif errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
    else:
        print("OK")

    return 0 if not errors else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
