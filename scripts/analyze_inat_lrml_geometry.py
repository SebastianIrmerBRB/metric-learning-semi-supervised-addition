"""Read-only retrieval diagnostics on the projections from diagnose_inat_lrml.

Centering uses the training pool's mean only. It is an inference-time diagnostic,
not an implemented training fix. The initial-score check verifies cache alignment.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    options = parser.parse_args()
    import main as entry
    import numpy as np
    import pandas as pd
    import torch
    import torch.nn.functional as F
    from sklearn.model_selection import GroupKFold
    from utils.local_datasets import SemiINaturalist2021
    from utils.dataset_protocols import get_dataset_class
    import utils
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="WARNING")

    torch.set_num_threads(4)
    entry.configure_float32_precision()
    metadata = json.loads((options.run_dir / "run_config.json").read_text())["args"]
    name = metadata["dataset"]
    data_root = ROOT / "data" / name
    if name == "iNat":
        dataset = SemiINaturalist2021(data_root, split="train", mode="known")
    else:
        dataset = get_dataset_class(name)(root=str(data_root), split="train", download=False)
    labels_np = np.asarray(dataset.labels)
    caches = list(data_root.glob("backbone_cache/v3/dinov2_vitb14/*.features.npy"))
    matching = [p for p in caches if np.load(p, mmap_mode="r").shape[0] == len(labels_np)]
    # Cars can contain multiple resize/transform caches with the same row count.
    # The score assertion below protects against using the wrong one.
    cache = matching[0]
    folds = list(GroupKFold(n_splits=4, shuffle=True, random_state=7).split(labels_np, groups=labels_np))
    train, valid = folds[int(metadata["cv_fold"])]
    features = torch.from_numpy(np.load(cache)).float().cuda()
    labels = torch.from_numpy(labels_np).cuda()
    output = {"run_dir": str(options.run_dir), "cache": str(cache), "variants": []}
    for stage in ("initial", "final"):
        state = torch.load(options.run_dir / f"diagnostic_{stage}_fc.pt", map_location="cuda", weights_only=True)
        with torch.no_grad():
            pre = F.linear(features, state["weight"], state["bias"])
            raw = F.normalize(pre, dim=1)
            geometry_rows = np.random.default_rng(17).choice(len(raw), min(4096, len(raw)), replace=False)
            geometry_z = raw[geometry_rows]
            singular = torch.linalg.svdvals((geometry_z - geometry_z.mean(0))[:2048])
            probabilities = singular / singular.sum()
            output[stage + "_centered_effective_rank"] = float(torch.exp(
                -(probabilities * probabilities.clamp_min(1e-12).log()).sum()))
            pre_mean = pre[train].mean(0)
            unit_mean = raw[train].mean(0)
            variants = {"ordinary": raw,
                        "center_before_normalize": F.normalize(pre - pre_mean, dim=1),
                        "center_after_normalize": F.normalize(raw - unit_mean, dim=1)}
            for variant, z in variants.items():
                p1, mapr, per_class = utils.evaluate_embeddings(
                    z[valid], labels[valid], device="cuda", return_per_class=True,
                    name=f"{stage}/{variant}")
                record = {"stage": stage, "variant": variant,
                          "precision_at_1": p1, "map_at_r": mapr,
                          "validation_centroid_norm": float(z[valid].mean(0).norm())}
                output["variants"].append(record)
                if variant == "ordinary":
                    output[stage + "_validation_per_class"] = per_class
                print(json.dumps(record), flush=True)
                if stage == "initial" and variant == "ordinary":
                    curve = pd.read_csv(options.run_dir / "metrics.csv")
                    expected = curve.loc[curve.split == "valid", "mean_average_precision_at_r"].iloc[0]
                    assert abs(mapr - expected) < 1e-5, ("Feature cache alignment failed", mapr, expected)
            train_p1, train_map = utils.evaluate_embeddings(
                raw[train], labels[train], device="cuda", name=f"{stage}/all_training_images")
            output[stage + "_training_retrieval"] = {"precision_at_1": train_p1, "map_at_r": train_map}
            print(json.dumps({"stage": stage, "training_map_at_r": train_map}), flush=True)
            support = np.load(options.run_dir / "split/labeled_indices.npy")
            support_p1, support_map = utils.evaluate_embeddings(
                raw[support], labels[support], device="cuda", name=f"{stage}/labeled_support")
            output[stage + "_support_retrieval"] = {"precision_at_1": support_p1, "map_at_r": support_map}
            print(json.dumps({"stage": stage, "support_map_at_r": support_map}), flush=True)
            rng = np.random.default_rng(43)
            # Species-balanced random positive pairs versus uniformly random
            # different-species pairs quantify global class separation.
            positive_left, positive_right = [], []
            for label in np.unique(labels_np[train]):
                indices = train[labels_np[train] == label]
                pairs = rng.choice(indices, (100, 2), replace=True)
                keep = pairs[:, 0] != pairs[:, 1]
                positive_left.extend(pairs[keep, 0].tolist())
                positive_right.extend(pairs[keep, 1].tolist())
            negative = rng.choice(train, (50000, 2), replace=True)
            negative = negative[labels_np[negative[:, 0]] != labels_np[negative[:, 1]]]
            pos = (raw[positive_left] * raw[positive_right]).sum(1)
            neg = (raw[negative[:, 0]] * raw[negative[:, 1]]).sum(1)
            output[stage + "_geometry"] = {
                "train_mean_positive_cosine": float(pos.mean()),
                "train_mean_negative_cosine": float(neg.mean()),
                "train_centroid_norm": float(raw[train].mean(0).norm()),
                "raw_feature_centroid_norm": float(F.normalize(features[train], dim=1).mean(0).norm()),
                "fc_bias_norm": float(state["bias"].norm()),
                "fc_weight_frobenius": float(state["weight"].norm()),
            }
    (options.run_dir / "posthoc_geometry.json").write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
