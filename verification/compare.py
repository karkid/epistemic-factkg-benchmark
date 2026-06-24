"""Shared JSON comparator — used by all verification areas.

Reads reference JSON from argv[1], current snapshot from stdin.
Recursively compares: floats with tolerance, ints/strings exact.

Output: one summary line per suite (N passed), plus a diff table for failures only.

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


def _collect(current, reference, path: str, rows: list) -> None:
    if isinstance(reference, dict):
        for k, v in reference.items():
            _collect(current.get(k) if isinstance(current, dict) else None,
                     v, f"{path}.{k}" if path else k, rows)
    elif isinstance(reference, list):
        rows.append({"path": path, "actual": current, "expected": reference,
                     "ok": current == reference, "diff": None})
    elif isinstance(reference, float):
        actual = float(current) if current is not None else float("nan")
        ok = abs(actual - reference) <= TOL
        rows.append({"path": path, "actual": actual, "expected": reference,
                     "ok": ok, "diff": round(actual - reference, 4)})
    else:
        rows.append({"path": path, "actual": current, "expected": reference,
                     "ok": current == reference, "diff": None})


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: generate.py | compare.py <reference.json> [label]", file=sys.stderr)
        sys.exit(2)

    ref_path = Path(sys.argv[1])
    label    = sys.argv[2] if len(sys.argv) > 2 else ref_path.parent.name

    if not ref_path.exists():
        print(f"  {FAIL} reference not found: {ref_path}", file=sys.stderr)
        print("  Run generate.py --save to create it.", file=sys.stderr)
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

    rows: list = []
    _collect(current, reference, "", rows)

    passed = [r for r in rows if r["ok"]]
    failed = [r for r in rows if not r["ok"]]

    if failed:
        col_w = max((len(r["path"]) for r in failed), default=8)
        col_w = max(col_w, 8)
        print(f"  {'Check':<{col_w}}  {'Yours':>8}  {'Paper':>8}  {'Diff':>8}")
        print(f"  {'─' * col_w}  {'─'*8}  {'─'*8}  {'─'*8}")
        for r in failed:
            diff_str = f"{r['diff']:+.4f}" if r["diff"] is not None else "—"
            print(f"  {r['path']:<{col_w}}  {_fmt(r['actual']):>8}  {_fmt(r['expected']):>8}  {diff_str:>8}")
        print()
        print(f"  {PASS} {len(passed)} passed   {FAIL} {len(failed)} failed")
        print(f"\n{FAIL} {label}: {len(failed)} mismatch(es)")
        sys.exit(1)
    else:
        print(f"  {PASS} {len(passed)} checks passed")
        print(f"\n{PASS} {label}: all checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
