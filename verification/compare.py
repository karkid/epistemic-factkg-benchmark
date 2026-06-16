"""Shared JSON comparator — used by all verification areas.

Reads reference JSON from argv[1], current snapshot from stdin.
Recursively compares: floats with tolerance, ints/strings exact.

Usage:
    python verification/split_design/generate.py | \\
        python verification/compare.py verification/split_design/reference.json [label]

Exit codes: 0 = all match, 1 = mismatch found.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

PASS = "✓"
FAIL = "✗"
TOL  = 0.001


def _compare(current, reference, path: str, failures: list) -> None:
    if isinstance(reference, dict):
        for k, v in reference.items():
            _compare(current.get(k) if isinstance(current, dict) else None,
                     v, f"{path}.{k}" if path else k, failures)
    elif isinstance(reference, list):
        ok = current == reference
        _report(ok, path, current, reference, failures)
    elif isinstance(reference, float):
        actual = float(current) if current is not None else float("nan")
        ok = abs(actual - reference) <= TOL
        _report(ok, path, actual, reference, failures)
    else:
        ok = current == reference
        _report(ok, path, current, reference, failures)


def _report(ok: bool, path: str, actual, expected, failures: list) -> None:
    if ok:
        print(f"  {PASS} {path}: {actual}")
    else:
        msg = f"  {FAIL} {path}: {actual}  (expected {expected})"
        print(msg)
        failures.append(msg)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: generate.py | compare.py <reference.json> [label]", file=sys.stderr)
        sys.exit(2)

    ref_path = Path(sys.argv[1])
    label    = sys.argv[2] if len(sys.argv) > 2 else ref_path.parent.name

    if not ref_path.exists():
        print(f"  {FAIL} reference not found: {ref_path}", file=sys.stderr)
        print("  Run the generate.py --save to create it.", file=sys.stderr)
        sys.exit(1)

    try:
        current   = json.load(sys.stdin)
        reference = json.loads(ref_path.read_text())
    except json.JSONDecodeError as e:
        print(f"  {FAIL} JSON parse error: {e}", file=sys.stderr)
        sys.exit(1)

    if "__skip__" in current:
        print(f"  SKIP {label}: {current['__skip__']}")
        sys.exit(0)

    failures: list = []
    _compare(current, reference, "", failures)

    if failures:
        print(f"\n{FAIL} {label}: {len(failures)} mismatch(es) — update MD and run generate-refs.")
        sys.exit(1)
    else:
        print(f"\n{PASS} {label}: all checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
