"""Controlled single-fold LRML runs using the production training engine.

Only this diagnostic driver is modified; source experiment/method files stay
intact. Each arm uses fold 0 of the configured four-fold split. Geometry and
graph purity are observations only; the oracle arm intentionally uses hidden
training species labels to test the effect of erroneous graph edges.
"""

import argparse
import copy
import gc
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["baseline", "supervised", "weak", "oracle"])
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--dataset", default="iNat")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--loss", default="NTXentLoss")
    parser.add_argument("--tag", default="initial")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lr", type=float)
    parser.add_argument("--k-shot", type=int, default=8)
    parser.add_argument("--length", type=int)
    options = parser.parse_args()

    # main publishes the native thread budget before loading numerical modules.
    sys.argv = [str(ROOT / "main.py"), "--num_threads", "4"]
    import main as entry
    import numpy as np
    import torch
    from scipy import sparse
    from training import engine, semi_supervised
    from training.ssl.config import SemiSupervisedConfig

    base = entry.parse_args_with_experiment_config([
        "--experiment-config", "configs/phase_1/i_nat_2021/lrml.json",
        "--num_threads", "4", "--device", options.device,
    ])
    base.dataset = options.dataset
    base.epochs = options.epochs
    base.patience = options.epochs + 1
    base.seed = options.seed
    base.cv_fold = options.fold
    base.loss = options.loss
    base.miner = "no_miner"
    base.loss_miner_grid = None
    base.label_budget_grid = None
    base.k_shot_grid = None
    base.ssl_label_sampling_modes = None
    base.hparam_config = None
    base.final_test_after_hpo = False
    base.evaluate_test = False
    base.save_head_state = True
    base.log_epoch_diagnostics = True
    base.log_batch_diagnostics = True
    base.ssl_gradient_contribution_log_interval = 10
    base.num_workers = 0
    base.tensorboard = False
    base.resume_interrupted_runs = False
    if options.lr is not None:
        base.lr = options.lr
    if options.length is not None:
        base.length_before_new_iter_override = options.length
    base.frozen_feature_residency = "gpu" if options.device == "cuda" else "cpu"
    if options.dataset != "iNat":
        base.dataset_protocol = "official"
    entry.configure_threads(base)
    entry.configure_float32_precision()
    entry.utils.set_eval_autocast_enabled(base.eval_amp)
    entry.apply_low_memory_mode(base)

    template = json.loads((ROOT / "configs/ssl_methods/lrml.json").read_text())
    template.update(label_sampling_mode="class_subset_k_shot", labeled_fraction=1.0,
                    labeled_per_class=options.k_shot, seed=7, support_seed=7)
    original_epoch = engine._EpochTrainer.train_epoch
    observations = []

    def measured_epoch(trainer, loader, epoch, *args, **kwargs):
        if epoch == 0:
            torch.save(trainer.model.fc.state_dict(), Path(trainer.args.log_dir) / "diagnostic_initial_fc.pt")
        result = original_epoch(trainer, loader, epoch, *args, **kwargs)
        if epoch == options.epochs - 1:
            torch.save(trainer.model.fc.state_dict(), Path(trainer.args.log_dir) / "diagnostic_final_fc.pt")
        regularizer = trainer.regularizer
        record = {"epoch": int(epoch), "step": trainer.global_step}
        if regularizer is not None:
            record["regularizer_weight"] = float(regularizer.regularizer_weight)
            adjacency = getattr(regularizer, "adjacency", None)
            if adjacency is not None:
                edges = sparse.triu(adjacency, k=1, format="coo")
                labels = regularizer._train_dataset_labels[regularizer.graph_positions]
                same = labels[edges.row] == labels[edges.col]
                record.update(graph_nodes=len(labels), graph_edges=len(same),
                              graph_purity=float(same.mean()),
                              graph_components=int(sparse.csgraph.connected_components(adjacency)[0]))
        # Read a deterministic feature subsample for geometry only. Cache rows
        # are shared by the production fold datasets; no labels enter training.
        cache_root = ROOT / "data" / options.dataset / "backbone_cache/v3/dinov2_vitb14"
        from models.retrieval_model import _INDEXED_FEATURE_MATRICES
        candidates = [m.path for m in _INDEXED_FEATURE_MATRICES.values()
                      if m.path.parent.resolve() == cache_root.resolve()]
        if candidates:
            cache_path = max(candidates, key=lambda p: p.stat().st_size)
            record["geometry_cache"] = str(cache_path)
            matrix = np.load(cache_path, mmap_mode="r")
            rows = np.random.default_rng(17).choice(len(matrix), min(4096, len(matrix)), replace=False)
            features = torch.from_numpy(np.array(matrix[rows], copy=True)).to(options.device).float()
            with torch.no_grad():
                pre = trainer.model.fc(features)
                z = torch.nn.functional.normalize(pre, dim=1)
                centroid = z.mean(0).norm()
                centered = z - z.mean(0)
                singular = torch.linalg.svdvals(centered[:2048])
                distribution = singular / singular.sum()
                rank = torch.exp(-(distribution * distribution.clamp_min(1e-12).log()).sum())
                record.update(centroid_norm=float(centroid), centered_effective_rank=float(rank),
                              pre_norm=float(pre.norm(dim=1).mean()),
                              rms_pair_distance=float((2 * (1 - centroid.square())).clamp_min(0).sqrt()))
        observations.append(record)
        path = Path(trainer.args.log_dir) / "lrml_geometry.json"
        path.write_text(json.dumps(observations, indent=2) + "\n")
        print("LRML_DIAGNOSTIC " + json.dumps(record), flush=True)
        return result

    engine._EpochTrainer.train_epoch = measured_epoch
    for arm in options.arms:
        observations.clear()
        args = copy.deepcopy(base)
        args.save_dir = f"diagnostics/inat_lrml/{options.tag}/{options.dataset}/{options.loss}/seed_{options.seed}/fold_{options.fold}/{arm}"
        values = copy.deepcopy(template)
        params = values["method_params"]
        graph = params["regularizer_params"]
        if arm == "supervised":
            params = {k: v for k, v in params.items() if not k.startswith("regularizer_target_ratio")}
            params["regularizer_weight"] = 0.0
            values["method_params"] = params
        elif arm == "weak":
            params["regularizer_target_ratio"] = 0.1
        elif arm == "oracle":
            graph["oracle_same_class_edges"] = True
        elif arm == "static":
            values["update_mode"] = "once"
        elif arm == "l2":
            params["regularizer_target_ratio_statistic"] = "l2"
        elif arm == "contrastive":
            graph["contrastive"] = True
        elif arm == "unnormalized":
            graph["normalized_laplacian"] = False
        elif arm != "baseline":
            raise ValueError(f"Unknown arm: {arm}")
        config = SemiSupervisedConfig(**values)
        print(f"START_ARM {arm}", flush=True)
        engine.run_training(args, config, cv_fold=options.fold)
        print(f"FINISHED_ARM {arm}", flush=True)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
