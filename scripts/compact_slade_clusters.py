#!/usr/bin/env python3
"""Drop the cluster centers from saved SLADE refreshes, except the first and last.

Runs before 2026-10-01 kept ``centers`` (k x d float16) in every
``slade_clusters/refresh_*.npz``. They are ~99% of each file and nothing reads
them: the embedding viewer uses positions, clusters, num_clusters, epoch and
center_similarity, which stay bit-identical. A run that re-clusters every epoch
at In-Shop's k=5996 saved ~8 MB per epoch, up to 4.5 GB per fold. The training
code now keeps centers on the first and the newest refresh only; this brings
older runs to the same layout.

Dry run by default: lists what would be freed, measured from the zip directory
without decompressing. ``--apply`` rewrites the files atomically. A file that
cannot be read (e.g. still being written by a live run) is reported and skipped.

    venv/bin/python scripts/compact_slade_clusters.py /mnt/f/Studien/Phase_2/geometry
    venv/bin/python scripts/compact_slade_clusters.py /mnt/f/Studien/Phase_2/geometry --apply
"""

import argparse
import os
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.ssl.slade import SLADE_CLUSTER_DIR, drop_refresh_centers

REFRESH_NAME = re.compile(r"^refresh_(\d+)_epoch_\d+\.npz$")


def cluster_directories(roots):
    for root in roots:
        for directory, subdirectories, _ in os.walk(root, followlinks=True):
            if Path(directory).name == SLADE_CLUSTER_DIR:
                subdirectories[:] = []
                yield Path(directory)


def droppable_refreshes(directory):
    """Every refresh but the first and the last, in refresh order."""

    refreshes = sorted(
        (int(match.group(1)), directory / name)
        for name in os.listdir(directory)
        if (match := REFRESH_NAME.match(name))
    )
    return [path for _, path in refreshes[1:-1]]


def centers_bytes(path):
    with zipfile.ZipFile(path) as archive:
        return sum(info.compress_size for info in archive.infolist() if info.filename == "centers.npy")


def human(count):
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(count) < 1024 or unit == "TB":
            return f"{count:.1f} {unit}" if unit != "B" else f"{count} B"
        count /= 1024


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path, help="directories searched for slade_clusters/")
    parser.add_argument("--apply", action="store_true", help="rewrite the files (default: dry run)")
    args = parser.parse_args()

    total = files = runs = 0
    failed = []
    for directory in cluster_directories(args.roots):
        freed = count = 0
        for path in droppable_refreshes(directory):
            try:
                size = drop_refresh_centers(path) if args.apply else centers_bytes(path)
            except (OSError, ValueError, zipfile.BadZipFile) as error:
                failed.append((path, error))
                continue
            if size:
                freed += size
                count += 1
        if count:
            runs += 1
            files += count
            total += freed
            print(f"{human(freed):>10}  {count:4d} files  {directory}")
    verb = "freed" if args.apply else "would free"
    print(f"\n{verb} {human(total)} from {files} files in {runs} runs")
    for path, error in failed:
        print(f"skipped {path}: {error}", file=sys.stderr)
    if not args.apply and total:
        print("dry run; pass --apply to rewrite")


if __name__ == "__main__":
    main()
