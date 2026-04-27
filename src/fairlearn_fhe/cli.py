"""Command-line tools for fairlearn-fhe."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .envelope import validate_envelope, verify_envelope_signature

# Envelopes are small JSON documents; cap input to guard CLI users from
# buffering an attacker-controlled multi-megabyte file.
_MAX_ENVELOPE_BYTES = 1 * 1024 * 1024


def _read_json(path: str, *, max_bytes: int = _MAX_ENVELOPE_BYTES) -> dict[str, Any]:
    if path == "-":
        body = sys.stdin.read(max_bytes + 1)
        if len(body) > max_bytes:
            raise ValueError(f"envelope JSON exceeds {max_bytes} bytes")
    else:
        p = Path(path)
        size = p.stat().st_size
        if size > max_bytes:
            raise ValueError(f"envelope JSON exceeds {max_bytes} bytes (got {size})")
        body = p.read_text(encoding="utf-8")
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
    parser.add_argument(
        "--max-age", type=float, default=None,
        help="Reject envelopes older than this many seconds (anti-replay).",
    )
    parser.add_argument(
        "--min-security-bits", type=int, default=None,
        help="Reject envelopes whose recorded security level is below this value.",
    )
    parser.add_argument("--public-key", help="PEM Ed25519 public key for signature verification.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON output.")
    args = parser.parse_args(argv)

    payload: dict[str, Any] | None = None
    errors: list[str] = []
    try:
        payload = _read_json(args.envelope)
    except (OSError, ValueError) as exc:
        errors = [f"failed to read envelope: {exc}"]

    if payload is not None:
        errors = validate_envelope(
            payload,
            allowed_metrics=args.allowed_metric,
            max_observed_depth=args.max_depth,
            max_age_seconds=args.max_age,
            min_security_bits=args.min_security_bits,
        )
        if args.public_key:
            try:
                key = Path(args.public_key).read_bytes()
            except OSError as exc:
                errors.append(f"failed to read public key: {exc}")
            else:
                try:
                    errors.extend(verify_envelope_signature(payload, key))
                except (TypeError, ValueError) as exc:
                    errors.append(f"invalid public key: {exc}")

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
