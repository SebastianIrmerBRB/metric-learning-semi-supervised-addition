"""Report the storage dtype and the *effective* precision of backbone feature caches.

The storage dtype is float32 by construction, so the interesting question is
whether the stored values were produced at lower precision and then upcast.
A value that went through float16/bfloat16 and was upcast to float32 has its
low mantissa bits zero and round-trips through the half dtype unchanged; a
genuine float32 computation fills all 23 mantissa bits.

Limits of what the bits can show:

* Only a half-precision tensor *at the output* leaves a trace. Under CUDA bf16
  autocast DINOv2 still ends in a LayerNorm, which autocast runs in float32, so
  an ``--eval_amp`` build looks like full float32 here. TF32 convolutions
  are likewise washed out by the float32 layers after them. Detecting those
  needs a float32 re-extraction of a few rows to compare against.
* Unfilled rows of a v3 ``.features.npy`` matrix are NaN and are skipped.

Usage:
    python scripts/check_cache_precision.py                  # every cache under data/
    python scripts/check_cache_precision.py path/to/x.features.npy data/CUB/backbone_cache
    python scripts/check_cache_precision.py --sample-rows 5000 --max-pt-files 500
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HALF_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def mantissa_trailing_zeros(values):
    """Trailing zero bits of each float32 mantissa (23 for exact powers of two)."""

    bits = values.astype(np.float32).view(np.uint32) & np.uint32(0x7FFFFF)
    zeros = np.full(bits.shape, 23, dtype=np.int64)
    nonzero = bits != 0
    lowest_set = bits[nonzero] & (~bits[nonzero] + np.uint32(1))
    zeros[nonzero] = np.log2(lowest_set.astype(np.float64)).astype(np.int64)
    return zeros


def analyze(values):
    values = np.asarray(values, dtype=np.float32).ravel()
    values = values[np.isfinite(values) & (values != 0)]
    if values.size == 0:
        return None
    tensor = torch.from_numpy(values)
    report = {
        "n": values.size,
        "min": float(values.min()),
        "max": float(values.max()),
        "abs_max": float(np.abs(values).max()),
        "mean_used_mantissa_bits": float(23 - mantissa_trailing_zeros(values).mean()),
    }
    for name, dtype in HALF_DTYPES.items():
        roundtrip = tensor.to(dtype).to(torch.float32)
        report[f"exact_in_{name}"] = float((roundtrip == tensor).double().mean())
    return report


def verdict(report):
    # bfloat16 first: small-magnitude bf16 values are also exact in float16.
    if report["exact_in_bfloat16"] > 0.99:
        return "values were upcast from bfloat16"
    if report["exact_in_float16"] > 0.99:
        return "values were upcast from float16"
    if report["mean_used_mantissa_bits"] > 20:
        return "full float32 at the output (autocast/TF32 upstream not detectable from bits)"
    return "reduced mantissa usage, but not an exact half-precision cast -- inspect manually"


def load_npy(path, sample_rows, rng):
    matrix = np.load(path, mmap_mode="r")
    storage = f"{matrix.dtype} {matrix.shape}"
    if matrix.ndim != 2:
        return storage, np.asarray(matrix), None
    filled = np.flatnonzero(np.isfinite(matrix[:, 0]))
    fill_note = f"{len(filled)}/{matrix.shape[0]} rows filled"
    if len(filled) > sample_rows:
        filled = np.sort(rng.choice(filled, size=sample_rows, replace=False))
    return storage, np.asarray(matrix[filled]), fill_note


def load_pt_dir(paths, max_files, rng):
    if len(paths) > max_files:
        paths = [paths[i] for i in np.sort(rng.choice(len(paths), size=max_files, replace=False))]
    dtypes, arrays = set(), []
    for path in paths:
        obj = torch.load(path, map_location="cpu", weights_only=True)
        tensors = obj.values() if isinstance(obj, dict) else [obj]
        for tensor in tensors:
            if torch.is_tensor(tensor) and tensor.is_floating_point():
                dtypes.add(str(tensor.dtype).removeprefix("torch."))
                arrays.append(tensor.float().numpy().ravel())
    storage = ", ".join(sorted(dtypes)) or "no float tensors"
    return storage, (np.concatenate(arrays) if arrays else np.empty(0)), f"{len(paths)} .pt files sampled"


def collect_targets(inputs):
    npy_files, pt_dirs = [], {}
    for root in inputs:
        root = Path(root)
        candidates = [root] if root.is_file() else root.rglob("*")
        for path in candidates:
            if path.name.endswith(".features.npy") or (path.suffix == ".npy" and path.is_file()):
                npy_files.append(path)
            elif path.suffix == ".pt" and path.is_file():
                pt_dirs.setdefault(path.parent, []).append(path)
    return sorted(set(npy_files)), dict(sorted(pt_dirs.items()))


def print_report(label, storage, note, report):
    print(f"\n{label}")
    print(f"  storage dtype : {storage}" + (f"   [{note}]" if note else ""))
    if report is None:
        print("  no finite non-zero values to analyze")
        return
    print(
        f"  values        : n={report['n']:,}  range=[{report['min']:.4g}, {report['max']:.4g}]"
        + ("  (exceeds float16 max 65504)" if report["abs_max"] > 65504 else "")
    )
    print(f"  mantissa bits : {report['mean_used_mantissa_bits']:.2f} of 23 used on average")
    print(
        f"  exact in fp16 : {report['exact_in_float16']:.2%}   "
        f"exact in bf16: {report['exact_in_bfloat16']:.2%}"
    )
    print(f"  verdict       : {verdict(report)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="*", default=["data"], help="cache files or directories (default: data)")
    parser.add_argument("--sample-rows", type=int, default=2000, help="filled rows sampled per .npy matrix")
    parser.add_argument("--max-pt-files", type=int, default=200, help=".pt files sampled per directory")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    npy_files, pt_dirs = collect_targets(args.paths)
    if not npy_files and not pt_dirs:
        print(f"no .npy or .pt cache files found under {args.paths}", file=sys.stderr)
        return 1
    for path in npy_files:
        storage, values, note = load_npy(path, args.sample_rows, rng)
        print_report(path, storage, note, analyze(values))
    for directory, paths in pt_dirs.items():
        storage, values, note = load_pt_dir(paths, args.max_pt_files, rng)
        print_report(f"{directory}/*.pt", storage, note, analyze(values))
    return 0


if __name__ == "__main__":
    sys.exit(main())
