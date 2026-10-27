"""Recompute cached backbone rows at known precisions and report which one the cache matches.

``check_cache_precision.py`` can only see a half-precision cast at the output.
CUDA bf16 autocast (``--eval_amp``) and TF32 convolutions are hidden behind
DINOv2's final float32 LayerNorm, so this script re-extracts a sample of filled
cache rows three ways -- pure float32, bf16 autocast, float32 with TF32 -- and
compares each against the stored values.

Cache files are named by a hash of the dataset/transform/backbone identity, so
each file is located by rebuilding the dataset bundles the engine builds (every
candidate protocol x resize mode) and computing their index spec with the real
key code. Files no candidate reproduces are listed as unmatched.

Usage:
    python scripts/recompute_cache_precision.py
    python scripts/recompute_cache_precision.py --datasets Cars196 CUB --rows 128
"""

import argparse
import contextlib
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import utils  # noqa: E402
from models.retrieval_model import (  # noqa: E402
    BACKBONE_CACHE_VERSION,
    DinoWrapper,
    _indexed_feature_matrix_path,
)
from training.frozen_feature_cache import make_frozen_feature_index_spec  # noqa: E402
from utils.dataset_protocols import is_dataset_ready  # noqa: E402

CANDIDATE_PROTOCOLS = {
    "Cars196": ["official"],
    "CUB": ["official", "cub_alternating_100_100"],
    "DeepFashionInShop": ["official"],
    "StanfordOnlineProducts": ["official"],
    "iNat": ["semi_inat_oracle_50_50", "semi_inat_known_810", "official"],
}


@contextlib.contextmanager
def precision(mode, device):
    """float32 = everything full precision; bf16 = eval autocast; tf32 = TF32 conv+matmul."""

    saved = (torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32)
    tf32 = mode == "tf32"
    torch.backends.cudnn.allow_tf32 = tf32
    torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        with torch.no_grad(), torch.autocast(
            device_type=torch.device(device).type,
            dtype=torch.bfloat16,
            enabled=mode == "bf16",
        ):
            yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = saved


def find_cache_matches(dataset_name, dino_sizes, model):
    """Yield (path, feature_dataset, local_positions, rows, description) for existing cache files."""

    # Relative, exactly as make_run_model passes it: str(cache_dir) is part of the key.
    cache_root = Path("data") / dataset_name / "backbone_cache"
    for protocol in CANDIDATE_PROTOCOLS.get(dataset_name, ["official"]):
        for resize_mode in utils.IMAGE_RESIZE_MODES:
            try:
                bundle = utils.setup_dataset_bundle(
                    dataset_name,
                    seed=0,
                    dataset_protocol=protocol,
                    image_resize_mode=resize_mode,
                )
            except Exception as exc:  # protocol not applicable to this dataset
                print(f"  skip {dataset_name}/{protocol}/{resize_mode}: {type(exc).__name__}: {exc}")
                continue
            splits = {
                "train": bundle.train_dataset,
                "valid": bundle.valid_dataset,
                "test": bundle.test_dataset,
            }
            for dino_size in dino_sizes:
                model.dino_size = dino_size
                model.cache_dir = cache_root / f"v{BACKBONE_CACHE_VERSION}" / f"dinov2_vit{dino_size}14"
                for split_name, dataset in splits.items():
                    if dataset is None or len(dataset) == 0:
                        continue
                    # Mirror _prepare_datasets + precompute: training runs on the feature transform.
                    if split_name == "train":
                        dataset = utils.make_feature_transform_dataset(dataset, require_feature_transform=True)
                    spec = make_frozen_feature_index_spec(
                        None,
                        model,
                        dataset,
                        require_feature_transform=split_name == "train",
                        use_feature_transform=True,
                        num_views=1,
                    )
                    if spec is None:
                        continue
                    path = _indexed_feature_matrix_path(model.cache_dir, spec.key)
                    # Non-existing paths are kept as capacity-matched fallback candidates
                    # for cache files written under an older key derivation.
                    yield path, path.exists(), spec.capacity, dataset, spec.row_indices, dino_size, (
                        f"{dataset_name} protocol={protocol} resize={resize_mode} split={split_name}"
                    )


def compare_rows(model, device, path, feature_dataset, row_indices, num_rows, batch_size, rng):
    matrix = np.load(path, mmap_mode="r")
    filled_local = np.flatnonzero(np.isfinite(matrix[row_indices, 0]))
    if len(filled_local) == 0:
        return None
    if len(filled_local) > num_rows:
        filled_local = np.sort(rng.choice(filled_local, size=num_rows, replace=False))
    cached = torch.from_numpy(np.asarray(matrix[row_indices[filled_local]], dtype=np.float32))

    recomputed = defaultdict(list)
    for start in range(0, len(filled_local), batch_size):
        batch_positions = filled_local[start:start + batch_size]
        images = torch.stack([feature_dataset[int(i)][0] for i in batch_positions]).to(device)
        for mode in ("float32", "bf16", "tf32"):
            with precision(mode, device):
                recomputed[mode].append(model.forward_backbone(images).float().cpu())

    result = {"rows": len(filled_local)}
    for mode, chunks in recomputed.items():
        features = torch.cat(chunks)
        diff = features - cached
        result[mode] = {
            "exact_rows": float((diff == 0).all(dim=1).double().mean()),
            "max_abs": float(diff.abs().max()),
            "rel_l2": float(diff.norm() / cached.norm()),
        }
    return result


def print_result(path, description, result):
    print(f"\n{path}\n  {description}")
    if result is None:
        print("  no filled rows in this view")
        return
    print(f"  {result['rows']} rows recomputed")
    for mode in ("float32", "bf16", "tf32"):
        r = result[mode]
        print(f"  vs {mode:7s}: exact rows {r['exact_rows']:6.1%}  max|diff| {r['max_abs']:.3e}  rel L2 {r['rel_l2']:.3e}")
    best = min(("float32", "bf16", "tf32"), key=lambda m: result[m]["rel_l2"])
    print(f"  closest     : {best}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--datasets", nargs="*", default=list(CANDIDATE_PROTOCOLS))
    parser.add_argument("--rows", type=int, default=64, help="filled rows recomputed per cache file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import os
    os.chdir(REPO_ROOT)  # dataset roots are relative to the repo
    rng = np.random.default_rng(args.seed)

    all_files = {
        p.resolve()
        for name in args.datasets
        for p in (REPO_ROOT / "data" / name / "backbone_cache").glob(f"v{BACKBONE_CACHE_VERSION}/*/*.features.npy")
    }
    matched = set()
    models = {}

    for dataset_name in args.datasets:
        cache_dirs = sorted((REPO_ROOT / "data" / dataset_name / "backbone_cache" / f"v{BACKBONE_CACHE_VERSION}").glob("dinov2_vit*14"))
        dino_sizes = [d.name.removeprefix("dinov2_vit").removesuffix("14") for d in cache_dirs]
        if not dino_sizes:
            continue
        if not is_dataset_ready(dataset_name, Path("data") / dataset_name):
            print(f"\n{dataset_name}: dataset not ready locally, skipped (would download)")
            continue
        print(f"\n=== {dataset_name} (backbones: {', '.join(dino_sizes)}) ===")
        for dino_size in dino_sizes:
            if dino_size not in models:
                models[dino_size] = DinoWrapper(dino_size=dino_size, feat_dim=None, backbone_tuning="frozen").to(args.device).eval()
        # Key identity reads dino_size/cache_dir from the model; one probe object serves every size.
        probe = next(iter(models.values()))
        candidates = list(find_cache_matches(dataset_name, dino_sizes, probe))
        seen_paths = set()
        for path, exists, _, dataset, row_indices, dino_size, description in candidates:
            if not exists:
                continue
            key = (path.resolve(), tuple(row_indices[:8]), len(row_indices))
            if key in seen_paths:
                continue
            seen_paths.add(key)
            matched.add(path.resolve())
            feature_dataset = utils.make_feature_transform_dataset(dataset)
            result = compare_rows(models[dino_size], args.device, path, feature_dataset, row_indices, args.rows, args.batch_size, rng)
            print_result(path, f"{description}  dinov2_vit{dino_size}14", result)

        # Fallback: files whose key no candidate reproduces. Try every candidate view with
        # the same capacity and backbone, and accept one only if the recomputed features
        # actually agree -- a wrong image mapping differs by O(1), bf16 by ~2e-2.
        dataset_files = sorted(
            p.resolve() for p in (Path("data") / dataset_name / "backbone_cache").glob(f"v{BACKBONE_CACHE_VERSION}/*/*.features.npy")
        )
        for file_path in dataset_files:
            if file_path in matched:
                continue
            num_rows = np.load(file_path, mmap_mode="r").shape[0]
            file_size = file_path.parent.name.removeprefix("dinov2_vit").removesuffix("14")
            tried = set()
            for _, _, capacity, dataset, row_indices, dino_size, description in candidates:
                signature = (description.rsplit(" split=", 1)[0], capacity, tuple(row_indices[:8]), len(row_indices))
                if dino_size != file_size or capacity != num_rows or signature in tried:
                    continue
                tried.add(signature)
                feature_dataset = utils.make_feature_transform_dataset(dataset)
                probe_result = compare_rows(models[dino_size], args.device, file_path, feature_dataset, row_indices, 8, args.batch_size, rng)
                if probe_result is None:
                    continue
                if min(probe_result[m]["rel_l2"] for m in ("float32", "bf16", "tf32")) > 0.1:
                    continue
                matched.add(file_path)
                result = compare_rows(models[dino_size], args.device, file_path, feature_dataset, row_indices, args.rows, args.batch_size, rng)
                print_result(
                    file_path.relative_to(REPO_ROOT),
                    f"{description}  dinov2_vit{dino_size}14  [matched by content, not key]",
                    result,
                )
                break

    unmatched = sorted(all_files - matched)
    if unmatched:
        print("\nCache files no candidate configuration reproduced (not checked):")
        for path in unmatched:
            rows = np.load(path, mmap_mode="r").shape
            print(f"  {path.relative_to(REPO_ROOT)}  {rows}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
