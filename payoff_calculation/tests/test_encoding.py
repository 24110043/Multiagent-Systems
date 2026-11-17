#!/usr/bin/env python3
"""
tests/test_encoding.py
Encoding & Mojibake Integrity Test.

CS 329 Multi-Agent Systems (Autumn 2026)
Empirical Payoff Function Estimation

Scans all active .md, .csv, .json, and .py files in payoff_calculation/
and asserts that zero files contain mojibake sequences such as 'â€', 'â', or 'Ã'.
"""

import sys
from pathlib import Path

PAYOFF_DIR = Path(__file__).resolve().parent.parent

FORBIDDEN_SUBSTRINGS = ["â€", "â€”", "â€“", "â", "Ã"]


def test_encoding_integrity():
    print("\n" + "=" * 70)
    print(" EXECUTING ENCODING & MOJIBAKE INTEGRITY TEST")
    print("=" * 70)

    violations = []
    scanned_count = 0

    for ext in ["*.md", "*.csv", "*.json", "*.py"]:
        for f in PAYOFF_DIR.rglob(ext):
            if "archive_old_runs" in str(f) or ".git" in str(f) or f.name in ["test_encoding.py", "fix_mojibake.py"]:
                continue
            scanned_count += 1
            try:
                content = f.read_text(encoding="utf-8")
            except Exception as e:
                violations.append((str(f.relative_to(PAYOFF_DIR)), f"Failed to read as UTF-8: {e}"))
                continue

            found_bad = []
            for sub in FORBIDDEN_SUBSTRINGS:
                if sub in content:
                    found_bad.append(sub)

            if found_bad:
                violations.append((str(f.relative_to(PAYOFF_DIR)), f"Contains forbidden mojibake: {found_bad}"))

    if violations:
        print(f"\n[FAIL] Found {len(violations)} files with encoding / mojibake violations:")
        for path, err in violations:
            print(f"  - {path}: {err}")
        sys.exit(1)
    else:
        print(f"\n[PASS] All {scanned_count} files verified clean of mojibake and properly UTF-8 encoded!")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    test_encoding_integrity()
