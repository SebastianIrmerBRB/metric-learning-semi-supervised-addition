"""Delete debug.log files written by initialize_logger under a log tree.

The DEBUG sink is verbose: a single SSL/HPO fold can emit hundreds of megabytes
because per-batch records embed full index and reference lists. Nothing reads
debug.log programmatically, so old ones are pure disk cost.

Examples:
    python scripts/remove_debug_logs.py
    python scripts/remove_debug_logs.py --yes
    python scripts/remove_debug_logs.py logs/ssl --min-size 10MB --yes
    python scripts/remove_debug_logs.py --older-than 14 --yes
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

DEFAULT_ROOTS = ("logs",)
DEFAULT_NAME = "debug.log"
SIZE_SUFFIXES = {"": 1, "B": 1, "K": 1024, "KB": 1024, "M": 1024**2, "MB": 1024**2, "G": 1024**3, "GB": 1024**3}


def parse_size(value: str) -> int:
    """Parse a byte count that may carry a K/M/G suffix, e.g. '10MB'."""

    text = value.strip().upper()
    for suffix in sorted(SIZE_SUFFIXES, key=len, reverse=True):
        if suffix and text.endswith(suffix):
            number, unit = text[: -len(suffix)], suffix
            break
    else:
        number, unit = text, ""
    try:
        scaled = float(number) * SIZE_SUFFIXES[unit]
    except ValueError:
        raise argparse.ArgumentTypeError(f"Could not parse a size from {value!r}.") from None
    if scaled < 0:
        raise argparse.ArgumentTypeError(f"Size must not be negative: {value!r}")
    return int(scaled)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively find and delete debug.log files. Without --yes the script "
            "only reports what it would remove."
        )
    )
    parser.add_argument(
        "roots",
        nargs="*",
        type=Path,
        default=[Path(root) for root in DEFAULT_ROOTS],
        help=f"Directories to walk. Defaults to {' '.join(DEFAULT_ROOTS)}.",
    )
    parser.add_argument(
        "--name",
        default=DEFAULT_NAME,
        help=f"File name to match. Defaults to {DEFAULT_NAME}.",
    )
    parser.add_argument(
        "--min-size",
        type=parse_size,
        default=0,
        help="Only delete files at least this large, for example 10MB.",
    )
    parser.add_argument(
        "--older-than",
        type=float,
        default=None,
        metavar="DAYS",
        help="Only delete files whose last modification is older than DAYS days.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print every matched file instead of only the largest ones.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Actually delete. Without this flag the script only prints what would be removed.",
    )
    return parser.parse_args()


def human_bytes(count: int) -> str:
    size = float(count)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def find_debug_logs(
    roots: list[Path],
    name: str,
    min_size: int,
    older_than_days: float | None,
) -> list[tuple[Path, int]]:
    cutoff = None if older_than_days is None else time.time() - older_than_days * 86400
    matches: list[tuple[Path, int]] = []
    seen: set[Path] = set()

    for root in roots:
        if not root.exists():
            raise SystemExit(f"Directory does not exist: {root}")
        if not root.is_dir():
            raise SystemExit(f"Not a directory: {root}")
        for path in root.rglob(name):
            # rglob descends into symlinked directories, so the same file can be
            # reached twice; deleting a symlink target outside the tree is worse.
            if path.is_symlink() or not path.is_file():
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            stat = path.stat()
            if stat.st_size < min_size:
                continue
            if cutoff is not None and stat.st_mtime >= cutoff:
                continue
            seen.add(resolved)
            matches.append((path, stat.st_size))

    matches.sort(key=lambda item: item[1], reverse=True)
    return matches


def remove_files(matches: list[tuple[Path, int]], dry_run: bool) -> tuple[int, list[tuple[Path, str]]]:
    if dry_run:
        return sum(size for _, size in matches), []

    freed = 0
    failures: list[tuple[Path, str]] = []
    for path, size in matches:
        try:
            path.unlink()
        except OSError as error:
            failures.append((path, str(error)))
            continue
        freed += size
    return freed, failures


def report(
    matches: list[tuple[Path, int]],
    freed: int,
    failures: list[tuple[Path, str]],
    dry_run: bool,
    list_all: bool,
) -> None:
    if not matches:
        print("No matching files found.")
        return

    shown = matches if list_all else matches[:10]
    heading = "Would remove" if dry_run else "Removed"
    print(f"{heading} {len(matches)} file(s), {human_bytes(freed)}:")
    for path, size in shown:
        print(f"  {human_bytes(size):>10}  {path}")
    if len(shown) < len(matches):
        print(f"  ... and {len(matches) - len(shown)} more (use --list to see all)")

    if failures:
        print(f"Failed to delete {len(failures)} file(s):")
        for path, error in failures:
            print(f"  {path}: {error}")
    if dry_run:
        print("Dry run only. Re-run with --yes to delete these files.")


def main() -> None:
    args = parse_args()
    matches = find_debug_logs(
        roots=[Path(root) for root in args.roots],
        name=args.name,
        min_size=args.min_size,
        older_than_days=args.older_than,
    )
    dry_run = not args.yes
    freed, failures = remove_files(matches, dry_run)
    report(matches, freed, failures, dry_run, args.list)


if __name__ == "__main__":
    main()
