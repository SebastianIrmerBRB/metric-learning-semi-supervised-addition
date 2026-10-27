#!/usr/bin/env python3
"""Pack the per-epoch heads of finished runs into one file per run, or unpack them.

``--save_epoch_head_interval`` writes ``epoch_heads/epoch_<NNNN>.pt`` and the run
packs them into ``epoch_heads.npz`` itself when its fold ends. This catches the
runs that did not get there (killed, or trained before packing existed). The
packing is lossless: every epoch is read back bit-identical before its file is
deleted (see training/epoch_head_pack.py; ~1.5x smaller).

Dry run by default: lists the directories and their size. ``--apply`` packs them.
A directory whose run is still writing heads must not be packed: pass only
finished roots. ``--unpack`` writes ``epoch_heads/*.pt`` back from every
``epoch_heads.npz`` found, for tools that want one file per epoch.

    venv/bin/python scripts/pack_epoch_heads.py logs/phase_2/geometry
    venv/bin/python scripts/pack_epoch_heads.py logs/phase_2/geometry --apply
    venv/bin/python scripts/pack_epoch_heads.py logs/phase_2/geometry --unpack
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.epoch_head_pack import pack_epoch_heads, unpack_epoch_heads
from training.resume import EPOCH_HEADS_DIRNAME, PACKED_EPOCH_HEADS_FILENAME


def run_directories(roots, marker):
    """Run directories under ``roots`` holding ``marker`` (a dir or a file name)."""

    for root in roots:
        for directory, subdirectories, files in os.walk(root, followlinks=True):
            if marker in subdirectories or marker in files:
                yield Path(directory)
            subdirectories[:] = [name for name in subdirectories if name != EPOCH_HEADS_DIRNAME]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("roots", nargs="+", type=Path)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--apply", action="store_true", help="pack (default: list only)")
    action.add_argument("--unpack", action="store_true", help="write epoch_heads/*.pt back")
    args = parser.parse_args(argv)

    if args.unpack:
        for run_dir in run_directories(args.roots, PACKED_EPOCH_HEADS_FILENAME):
            print(f"unpacked {unpack_epoch_heads(run_dir / PACKED_EPOCH_HEADS_FILENAME)}")
        return 0

    total_raw = total_packed = 0
    failures = 0
    for run_dir in run_directories(args.roots, EPOCH_HEADS_DIRNAME):
        heads = sorted((run_dir / EPOCH_HEADS_DIRNAME).glob("epoch_*.pt"))
        size = sum(path.stat().st_size for path in heads)
        if not args.apply:
            print(f"{len(heads):5d} heads {size / 1e6:9.1f} MB  {run_dir}")
            total_raw += size
            continue
        try:
            sizes = pack_epoch_heads(run_dir)
        except Exception as exc:
            failures += 1
            print(f"SKIPPED {run_dir}: {exc}")
            continue
        if sizes is None:
            continue
        raw_bytes, packed_bytes = sizes
        total_raw += raw_bytes
        total_packed += packed_bytes
        print(f"{raw_bytes / 1e6:9.1f} MB -> {packed_bytes / 1e6:9.1f} MB  {run_dir}")
    if args.apply:
        print(f"total {total_raw / 1e6:.1f} MB -> {total_packed / 1e6:.1f} MB, {failures} skipped")
    else:
        print(f"total {total_raw / 1e6:.1f} MB unpacked; --apply packs them")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
