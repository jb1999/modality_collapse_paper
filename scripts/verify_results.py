#!/usr/bin/env python3
"""Verify experiment results against expected values.

Loads results from results/exp1/ and checks key metrics against expected
values stored in tests/expected_values.json.  Reports PASS/FAIL per check
with an overall summary.

Usage:
    uv run python scripts/verify_results.py
    uv run python scripts/verify_results.py --results-dir results/exp1/
    uv run python scripts/verify_results.py --expected tests/expected_values.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def load_csv_rows(path: Path) -> list[dict]:
    """Load a CSV file and return list of row dicts."""
    with open(path) as f:
        return list(csv.DictReader(f))


def find_csv_row(rows: list[dict], filters: dict) -> dict | None:
    """Find the first row matching all key=value filters."""
    for row in rows:
        if all(row.get(k) == v for k, v in filters.items()):
            return row
    return None


def check_value(name: str, actual: float, expected: float, tolerance: float) -> tuple[bool, str]:
    """Check if actual is within tolerance of expected.  Returns (passed, message)."""
    diff = abs(actual - expected)
    passed = diff <= tolerance
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}: expected={expected}, actual={actual:.6f}, diff={diff:.6f}, tol={tolerance}"
    return passed, msg


def run_probe_checks(section: dict, proj_root: Path) -> list[tuple[bool, str]]:
    """Run all probe-related checks."""
    results = []
    for _key, group in section.items():
        csv_path = proj_root / group["file"]
        if not csv_path.exists():
            for check in group["checks"]:
                results.append((False, f"  [FAIL] {check['name']}: missing file {csv_path}"))
            continue

        rows = load_csv_rows(csv_path)

        for check in group["checks"]:
            name = check["name"]

            if check.get("computed"):
                # Computed retention: numerator / denominator * 100
                num_spec = check["numerator"]
                den_spec = check["denominator"]

                num_row = find_csv_row(rows, {
                    "hook_point": num_spec["hook_point"],
                    "info_type": num_spec["info_type"],
                })
                den_row = find_csv_row(rows, {
                    "hook_point": den_spec["hook_point"],
                    "info_type": den_spec["info_type"],
                })

                if num_row is None or den_row is None:
                    results.append((False, f"  [FAIL] {name}: could not find matching rows"))
                    continue

                num_val = float(num_row[num_spec["field"]]) - num_spec.get("minus", 0)
                den_val = float(den_row[den_spec["field"]]) - den_spec.get("minus", 0)

                if abs(den_val) < 1e-9:
                    results.append((False, f"  [FAIL] {name}: denominator is zero"))
                    continue

                actual = (num_val / den_val) * 100
            else:
                row = find_csv_row(rows, {
                    "hook_point": check["hook_point"],
                    "info_type": check["info_type"],
                })
                if row is None:
                    results.append((False, f"  [FAIL] {name}: row not found"))
                    continue
                actual = float(row[check["field"]])

            results.append(check_value(name, actual, check["expected"], check["tolerance"]))

    return results


def run_json_checks(section: dict, proj_root: Path) -> list[tuple[bool, str]]:
    """Run checks against JSON result files."""
    results = []
    for _key, group in section.items():
        json_path = proj_root / group["file"]
        if not json_path.exists():
            for check in group["checks"]:
                results.append((False, f"  [FAIL] {check['name']}: missing file {json_path}"))
            continue

        with open(json_path) as f:
            data = json.load(f)

        for check in group["checks"]:
            name = check["name"]
            actual = float(data[check["field"]])
            results.append(check_value(name, actual, check["expected"], check["tolerance"]))

    return results


def run_mode_alignment_checks(section: dict, proj_root: Path) -> list[tuple[bool, str]]:
    """Run mode alignment checks (both summary and per-mode results CSVs)."""
    results = []
    for _key, group in section.items():
        for check in group["checks"]:
            name = check["name"]
            source = check["source"]

            if source == "summary":
                csv_path = proj_root / group["summary_file"]
                if not csv_path.exists():
                    results.append((False, f"  [FAIL] {name}: missing file {csv_path}"))
                    continue
                rows = load_csv_rows(csv_path)
                if not rows:
                    results.append((False, f"  [FAIL] {name}: empty summary CSV"))
                    continue
                actual = float(rows[0][check["field"]])

            elif source == "results":
                csv_path = proj_root / group["results_file"]
                if not csv_path.exists():
                    results.append((False, f"  [FAIL] {name}: missing file {csv_path}"))
                    continue
                rows = load_csv_rows(csv_path)
                target_idx = str(check["mode_index"])
                row = find_csv_row(rows, {"mode_index": target_idx})
                if row is None:
                    results.append((False, f"  [FAIL] {name}: mode_index {target_idx} not found"))
                    continue
                actual = float(row[check["field"]])
            else:
                results.append((False, f"  [FAIL] {name}: unknown source '{source}'"))
                continue

            results.append(check_value(name, actual, check["expected"], check["tolerance"]))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify experiment results")
    parser.add_argument(
        "--expected", default="tests/expected_values.json",
        help="Path to expected values JSON",
    )
    parser.add_argument(
        "--results-dir", default="results/exp1/",
        help="Results directory (used for display only; paths come from expected_values.json)",
    )
    args = parser.parse_args()

    proj_root = Path(__file__).resolve().parent.parent

    expected_path = proj_root / args.expected
    if not expected_path.exists():
        print(f"ERROR: Expected values file not found: {expected_path}")
        sys.exit(1)

    with open(expected_path) as f:
        expected = json.load(f)

    all_results: list[tuple[bool, str]] = []

    sections = [
        ("Probe Checks", "probes", run_probe_checks),
        ("Lipschitz Checks", "lipschitz", run_json_checks),
        ("Causal Ablation Checks", "causal_ablation", run_json_checks),
        ("Mode Alignment Checks", "mode_alignment", run_mode_alignment_checks),
        ("MS Swap Checks", "ms_swap", run_json_checks),
        ("Gradient Projection Checks", "gradient_projection", run_json_checks),
    ]

    for title, key, checker in sections:
        if key in expected:
            print(f"\n=== {title} ===")
            section_results = checker(expected[key], proj_root)
            for _, msg in section_results:
                print(msg)
            all_results.extend(section_results)

    # --- Summary ---
    n_total = len(all_results)
    n_pass = sum(1 for passed, _ in all_results if passed)
    n_fail = n_total - n_pass

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {n_pass}/{n_total} passed, {n_fail} failed")
    print(f"{'=' * 60}")

    if n_fail > 0:
        print("\nFailed checks:")
        for passed, msg in all_results:
            if not passed:
                print(msg)
        sys.exit(1)
    else:
        print("\nAll checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
