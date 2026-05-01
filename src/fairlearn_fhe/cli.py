"""Command-line tools for fairlearn-fhe.

Two entry points are installed:

- ``fairlearn-fhe`` — multi-subcommand CLI (``verify``, ``inspect``,
  ``schema``, ``doctor``). The default subcommand is ``verify``.
- ``fairlearn-fhe-verify`` — kept as a legacy alias that maps directly
  to ``fairlearn-fhe verify`` so existing CI invocations don't break.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .envelope import (
    ENVELOPE_JSON_SCHEMA,
    validate_envelope,
    verify_envelope_signature,
)

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


# ---------------------------------------------------------------------------
# verify (the original CLI behaviour, now a subcommand)
# ---------------------------------------------------------------------------


def _add_verify_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("envelope", help="Path to envelope JSON, or '-' for stdin")
    p.add_argument(
        "--allowed-metric",
        action="append",
        default=None,
        help="Allowed metric name. Repeat to allow multiple metrics.",
    )
    p.add_argument("--max-depth", type=int, default=None, help="Maximum observed depth.")
    p.add_argument(
        "--max-age",
        type=float,
        default=None,
        help="Reject envelopes older than this many seconds (anti-replay).",
    )
    p.add_argument(
        "--min-security-bits",
        type=int,
        default=128,
        help=(
            "Reject envelopes whose recorded security level is below "
            "this value (default: 128). Pass 0 to disable the check."
        ),
    )
    p.add_argument(
        "--public-key", help="PEM Ed25519 public key for signature verification."
    )
    p.add_argument(
        "--require-signature",
        action="store_true",
        help=(
            "Fail with an error if the envelope is unsigned or no "
            "--public-key was supplied. Without this flag an unsigned "
            "envelope still passes structural validation."
        ),
    )
    p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON output."
    )


def _cmd_verify(args: argparse.Namespace) -> int:
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
        # Run cryptographic verification when a public key is supplied.
        # The boolean ``verified`` is True iff the signature block was
        # present, the key was readable, and ``verify_envelope_signature``
        # returned no errors. ``--require-signature`` then forces a
        # single hard error if verification did not succeed for any
        # reason — missing key, key read error, missing signature block,
        # or signature mismatch.
        verified = False
        if args.public_key:
            try:
                key = Path(args.public_key).read_bytes()
            except OSError as exc:
                errors.append(f"failed to read public key: {exc}")
            else:
                try:
                    sig_errors = verify_envelope_signature(payload, key)
                except (TypeError, ValueError) as exc:
                    errors.append(f"invalid public key: {exc}")
                else:
                    errors.extend(sig_errors)
                    verified = (not sig_errors) and ("signature" in payload)
        if args.require_signature and not verified:
            errors.append(
                "--require-signature was set but the envelope was not "
                "cryptographically verified (supply --public-key and "
                "ensure the envelope is signed)."
            )

    if args.json:
        print(json.dumps({"valid": not errors, "errors": errors}, sort_keys=True))
    elif errors:
        print("INVALID")
        for error in errors:
            print(f"- {error}")
    else:
        print("OK")
    return 0 if not errors else 1


# ---------------------------------------------------------------------------
# inspect — pretty-print summary of envelope contents
# ---------------------------------------------------------------------------


import re as _re

# ANSI CSI sequences (`\x1b[...<letter>`) — strip the whole sequence,
# not just the leading ESC, so that a partial escape arriving on a
# terminal that re-injects ESC can't still be interpreted.
_ANSI_CSI_RE = _re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
_ANSI_OSC_RE = _re.compile(r"\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")

# Unicode characters that can re-order or hide rendered text on a
# bidi-aware terminal. We strip them so a crafted ``metric_name``
# can't visually flip "INVALID" into "OK" inside ``inspect`` output.
_UNICODE_HOSTILE = frozenset(
    "​‌‍‎‏"  # ZWSP, ZWNJ, ZWJ, LRM, RLM
    "‪‫‬‭‮"  # bidi embedding/override
    "⁦⁧⁨⁩"        # bidi isolates
    "  "                    # line/paragraph separators
    "﻿"                          # BOM / ZWNBSP
)


def _safe_str(value: Any, *, max_len: int = 128) -> str:
    """Sanitise an untrusted value for printing to a terminal.

    Strips ANSI escape sequences, ASCII control characters, and the
    common Unicode bidi / zero-width characters that can be used to
    visually misrepresent inspect output. Caps length to ``max_len``.
    """
    s = str(value)
    s = _ANSI_OSC_RE.sub("", s)
    s = _ANSI_CSI_RE.sub("", s)
    cleaned = "".join(
        ch for ch in s
        if ch not in _UNICODE_HOSTILE
        and (ch == "\t" or (ord(ch) >= 32 and ord(ch) != 127))
    )
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len] + "…"
    return cleaned


def _cmd_inspect(args: argparse.Namespace) -> int:
    try:
        payload = _read_json(args.envelope)
    except (OSError, ValueError) as exc:
        print(f"failed to read envelope: {exc}", file=sys.stderr)
        return 2

    metric = payload.get("metric_name") or payload.get("metric") or "<unknown>"
    backend = (payload.get("parameter_set") or {}).get("backend") or "<unknown>"
    depth = payload.get("observed_depth")
    sec_bits = (payload.get("parameter_set") or {}).get("security_bits")
    trust_model = payload.get("trust_model") or "<unknown>"
    op_counts = payload.get("op_counts") or {}
    signed = "signature" in payload
    timestamp = payload.get("timestamp_unix") or payload.get("timestamp")

    if args.json:
        out = {
            "metric": metric,
            "backend": backend,
            "trust_model": trust_model,
            "observed_depth": depth,
            "security_bits": sec_bits,
            "op_counts": op_counts,
            "signed": signed,
            "timestamp": timestamp,
        }
        print(json.dumps(out, sort_keys=True))
    else:
        print(f"metric:         {_safe_str(metric)}")
        print(f"backend:        {_safe_str(backend)}")
        print(f"trust_model:    {_safe_str(trust_model)}")
        print(f"observed_depth: {_safe_str(depth)}")
        print(f"security_bits:  {_safe_str(sec_bits)}")
        print(f"signed:         {'yes' if signed else 'no'}")
        if timestamp is not None:
            print(f"timestamp:      {_safe_str(timestamp)}")
        if op_counts:
            print("op_counts:")
            for k in sorted(op_counts, key=str):
                print(f"  - {_safe_str(k, max_len=64)}: {_safe_str(op_counts[k])}")
    return 0


# ---------------------------------------------------------------------------
# schema — print the envelope JSON Schema
# ---------------------------------------------------------------------------


def _cmd_schema(args: argparse.Namespace) -> int:
    print(json.dumps(ENVELOPE_JSON_SCHEMA, indent=2 if args.pretty else None, sort_keys=True))
    return 0


# ---------------------------------------------------------------------------
# doctor — show backend availability
# ---------------------------------------------------------------------------


def _cmd_doctor(_args: argparse.Namespace) -> int:
    from ._backends import get_backend, list_backends

    print("backends:")
    for name in list_backends():
        try:
            mod = get_backend(name)
            mod_module = mod.__name__
            available, note = _probe_backend(name)
            status = "available" if available else "missing"
            print(f"  - {name}: {status} ({mod_module})" + (f" — {note}" if note else ""))
        except Exception as exc:  # pragma: no cover - defensive
            print(f"  - {name}: error ({type(exc).__name__}: {exc})")
    return 0


def _probe_backend(name: str) -> tuple[bool, str]:
    if name == "tenseal":
        try:
            import tenseal  # noqa: F401
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return True, ""
    if name == "openfhe":
        try:
            import openfhe  # noqa: F401
        except (ImportError, ModuleNotFoundError, OSError) as exc:
            return False, f"{type(exc).__name__}: {exc}"
        return True, ""
    return False, "unknown backend"  # pragma: no cover


# ---------------------------------------------------------------------------
# Parser construction + dispatch
# ---------------------------------------------------------------------------


def _build_main_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fairlearn-fhe",
        description=(
            "fairlearn-fhe — encrypted Fairlearn metrics. Default "
            "subcommand is 'verify' for backwards compatibility with the "
            "fairlearn-fhe-verify entry point."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=False)

    verify_p = sub.add_parser("verify", help="Validate an audit envelope.")
    _add_verify_args(verify_p)

    inspect_p = sub.add_parser("inspect", help="Print a human-readable summary of an envelope.")
    inspect_p.add_argument("envelope", help="Path to envelope JSON, or '-' for stdin")
    inspect_p.add_argument(
        "--json", action="store_true", help="Emit machine-readable JSON output."
    )

    schema_p = sub.add_parser("schema", help="Print the envelope JSON Schema.")
    schema_p.add_argument(
        "--pretty", action="store_true", help="Indent the JSON output for readability."
    )

    sub.add_parser("doctor", help="Show backend availability.")
    return parser


_SUBCOMMANDS = {"verify", "inspect", "schema", "doctor"}


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    # Backward-compat: if the first non-flag arg isn't a known subcommand
    # the caller is using the legacy ``fairlearn-fhe-verify <path>`` form.
    # Inject ``verify`` so existing scripts and tests keep working.
    first_positional = next((a for a in raw_argv if not a.startswith("-")), None)
    if first_positional and first_positional not in _SUBCOMMANDS:
        raw_argv = ["verify", *raw_argv]

    parser = _build_main_parser()
    args = parser.parse_args(raw_argv)
    if args.command == "verify":
        return _cmd_verify(args)
    if args.command == "inspect":
        return _cmd_inspect(args)
    if args.command == "schema":
        return _cmd_schema(args)
    if args.command == "doctor":
        return _cmd_doctor(args)
    parser.print_help(sys.stderr)
    return 2


def main_verify_legacy(argv: list[str] | None = None) -> int:
    """Legacy entry point for ``fairlearn-fhe-verify`` (no subcommand).

    Maps the historical positional/keyword surface directly to the
    ``verify`` subcommand so existing CI scripts keep working.
    """
    parser = argparse.ArgumentParser(
        prog="fairlearn-fhe-verify",
        description="Validate a fairlearn-fhe audit envelope.",
    )
    _add_verify_args(parser)
    args = parser.parse_args(argv)
    return _cmd_verify(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
