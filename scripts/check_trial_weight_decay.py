"""Audit weight-decay settings recorded by trial run_config.json files."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Check weight_decay in every trial run_config.json under a log directory."
    )
    parser.add_argument("old_logs", nargs="?", type=Path, default=Path("old_logs"))
    parser.add_argument(
        "--show-all", action="store_true", help="Print valid trials too (default: only problems)."
    )
    args = parser.parse_args()

    configs = sorted(
        path for path in args.logs.rglob("run_config.json") if any(p.startswith("trial_") for p in path.parts)
    )
    if not configs:
        print(f"No trial run_config.json files found below {args.logs}")
        return 2

    counts: Counter[str] = Counter()
    for path in configs:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            trial_args = data.get("args", {})
            value = trial_args.get("weight_decay")
            tuned_value = trial_args.get("hparam_params", {}).get("weight_decay")

            if value is None:
                status = "MISSING"
            elif not isinstance(value, (int, float)):
                status = "INVALID"
            elif value <= 0:
                status = "NOT_APPLIED"
            elif tuned_value is not None and value != tuned_value:
                status = "MISMATCH"
            else:
                # A run config proves what was requested, not what an optimizer used.
                status = "CONFIGURED"
        except (OSError, json.JSONDecodeError) as exc:
            status, value, tuned_value = "UNREADABLE", None, None
            error = str(exc)
        else:
            error = ""

        counts[status] += 1
        if args.show_all or status != "CONFIGURED":
            relative = path.relative_to(args.logs)
            detail = f"weight_decay={value!r} hparam={tuned_value!r}"
            if error:
                detail = error
            print(f"{status:11} {relative}  {detail}")

    print("\nSummary: " + ", ".join(f"{key}={value}" for key, value in sorted(counts.items())))
    print(
        "Note: CONFIGURED means the value was recorded in run_config.json; "
        "the config alone cannot prove that the optimizer applied it."
    )
    return 1 if any(key != "CONFIGURED" for key in counts) else 0


if __name__ == "__main__":
    raise SystemExit(main())
