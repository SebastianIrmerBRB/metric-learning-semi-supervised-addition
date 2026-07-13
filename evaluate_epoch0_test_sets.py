from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
from pytorch_metric_learning.utils.accuracy_calculator import (
    AccuracyCalculator,
    maybe_get_avg_of_avgs,
    nan_accuracy,
    try_getting_not_lone_labels,
)

torch.multiprocessing.set_sharing_strategy("file_system")

REPO_ROOT = Path(__file__).resolve().parents[0]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.chdir(REPO_ROOT)

import main as experiment_main  # noqa: E402
import utils  # noqa: E402
from models.retrieval_model import DinoWrapper  # noqa: E402
from training import semi_supervised  # noqa: E402


DEFAULT_PRECISION_KS = (2, 4, 8)


def precision_rank(value: str) -> int:
    parsed = int(value)
    if parsed <= 1:
        raise argparse.ArgumentTypeError(
            "additional precision k values must be at least 2; Precision@1 is already reported"
        )
    return parsed


def normalize_precision_ks(precision_ks) -> tuple[int, ...]:
    normalized = tuple(sorted(set(int(k) for k in precision_ks)))
    if not normalized:
        raise ValueError("at least one additional precision k value is required")
    if normalized[0] <= 1:
        raise ValueError(
            "additional precision k values must be at least 2; Precision@1 is already reported"
        )
    return normalized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate test-set retrieval metrics before any optimization steps. "
            "This is the same model state used for the pre-training validation check in normal runs, "
            "but evaluated directly on D_test."
        )
    )
    parser.add_argument(
        "--experiment-config",
        "--experiment_config",
        dest="experiment_configs",
        action="append",
        type=Path,
        required=True,
        help="Experiment JSON config to evaluate. Repeat to evaluate multiple configured test sets.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/epoch0_test_sets"),
        help="Directory for summary.csv and summary.json.",
    )
    parser.add_argument("--device", default=None, help="Override config device. Use 'auto' for cuda-if-available.")
    parser.add_argument("--batch-size", type=int, default=None, help="Override evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=None, help="Override DataLoader workers.")
    parser.add_argument("--dataloader-start-method", default=None, help="Override DataLoader start method.")
    parser.add_argument(
        "--model-seed",
        type=int,
        default=None,
        help=(
            "Override the model initialization seed. If omitted, the experiment config seed is used. "
            "This only changes metrics when the epoch-0 model has random heads, such as feat_dim projection."
        ),
    )
    parser.add_argument(
        "--per-class",
        action="store_true",
        help="Write per-class retrieval metrics for each evaluated test set.",
    )
    parser.add_argument(
        "--precision-k",
        dest="precision_ks",
        type=precision_rank,
        nargs="+",
        default=DEFAULT_PRECISION_KS,
        metavar="K",
        help=(
            "Additional ranks used for Precision@k (default: 2 4 8). Precision@1 is "
            "already reported separately. For example, use "
            "'--precision-k 10 20 30 50' for In-Shop-style reporting."
        ),
    )
    parser.add_argument(
        "--pacmap",
        action="store_true",
        help="Write PacMAP 2D coordinates and a group-colored PNG scatter plot for each evaluated test set.",
    )
    return parser.parse_args()


def resolve_config_path(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def parse_experiment_config(path: Path) -> argparse.Namespace:
    # parse_args_with_experiment_config currently prints the resolved split seed;
    # keep this reporting script's stdout focused on its own summary.
    with contextlib.redirect_stdout(io.StringIO()):
        return experiment_main.parse_args_with_experiment_config(["--experiment-config", str(path)])


def apply_cli_overrides(args: argparse.Namespace, cli_args: argparse.Namespace) -> argparse.Namespace:
    if cli_args.device is not None:
        args.device = resolve_device(cli_args.device)
    if cli_args.batch_size is not None:
        args.batch_size = int(cli_args.batch_size)
    if cli_args.num_workers is not None:
        args.num_workers = int(cli_args.num_workers)
    if cli_args.dataloader_start_method is not None:
        args.dataloader_start_method = cli_args.dataloader_start_method
    if cli_args.model_seed is not None:
        args.seed = int(cli_args.model_seed)
    return args


def resolve_device(value: str) -> str:
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return utils.normalize_device_name(value)


def make_epoch0_model(args: argparse.Namespace, ssl_config: semi_supervised.SemiSupervisedConfig) -> DinoWrapper:
    loss_driven_ssl = ssl_config.method in semi_supervised.LOSS_DRIVEN_METHODS
    ssl_method = semi_supervised.get_method(ssl_config)
    regularized_ssl = ssl_method is not None and ssl_method.is_regularization_method
    regularizer = ssl_method.make_regularizer(ssl_config) if regularized_ssl else None

    if loss_driven_ssl:
        stml_params = dict(getattr(args, "loss_params", {}))
        model_kwargs = {
            "stml": True,
            "stml_g_dim": getattr(args, "stml_g_dim", None),
            "stml_normalize_student": bool(stml_params.get("normalize_student", False)),
        }
    else:
        model_kwargs = {} if regularizer is None else regularizer.model_kwargs(args)

    model = DinoWrapper(
        dino_size=args.dino_size,
        feat_dim=args.feat_dim,
        backbone_tuning=args.backbone_tuning,
        use_cache=args.use_cache,
        cache_dir=Path("data") / args.dataset / "backbone_cache",
        **model_kwargs,
    )
    return model.to(args.device)


def load_test_bundle(args: argparse.Namespace) -> utils.DatasetBundle:
    return utils.setup_dataset_bundle(
        args.dataset,
        seed=args.seed,
        data_split_seed=args.data_split_seed,
        cv_k=1,
        cv_fold=None,
        cv_mode=args.cv_mode,
        val_mode=args.val_mode,
        dataset_protocol=args.dataset_protocol,
        cifar_imbalance_factor=args.cifar_imbalance_factor,
        cifar_train_fraction=args.cifar_train_fraction,
        cifar_test_fraction=args.cifar_test_fraction,
        full_train=False,
    )


def label_summary(dataset) -> tuple[int | str, int | str]:
    labels = getattr(dataset, "labels", None)
    if labels is None:
        return "", ""
    labels = np.asarray(labels, dtype=np.int64)
    return int(labels.size), int(len(np.unique(labels)))


def safe_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def relative_config_label(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


class PrecisionAtKAccuracyCalculator(AccuracyCalculator):
    """Extend pytorch-metric-learning's calculator with top-k retrieval precision."""

    def __init__(self, precision_ks: tuple[int, ...], **kwargs):
        self.precision_ks = normalize_precision_ks(precision_ks)
        super().__init__(**kwargs)

    def requires_knn(self):
        return [*super().requires_knn(), "precision_at_k"]

    def determine_k(self, bin_counts, num_reference_embeddings, ref_includes_query):
        metric_k = super().determine_k(bin_counts, num_reference_embeddings, ref_includes_query)
        available_neighbors = num_reference_embeddings - int(ref_includes_query)
        precision_k = max(self.precision_ks, default=0)
        return min(max(metric_k, precision_k), available_neighbors)

    def calculate_precision_at_k(
        self,
        knn_labels,
        query_labels,
        not_lone_query_mask,
        label_counts,
        **kwargs,
    ):
        knn_labels, query_labels = try_getting_not_lone_labels(
            knn_labels,
            query_labels,
            not_lone_query_mask,
        )
        if knn_labels is None:
            return {
                k: nan_accuracy(label_counts[0], self.return_per_class)
                for k in self.precision_ks
            }

        ground_truth = query_labels[:, None]
        precisions = {}
        for k in self.precision_ks:
            effective_k = min(k, knn_labels.shape[1])
            matches = self.label_comparison_fn(ground_truth, knn_labels[:, :effective_k])
            hits = torch.any(matches, dim=1).to(torch.float64)
            precisions[k] = maybe_get_avg_of_avgs(
                hits,
                ground_truth,
                self.avg_of_avgs,
                self.return_per_class,
            )
        return precisions


def make_report_per_class_metrics(
    query_labels: np.ndarray,
    accuracy: dict,
    reference_labels: np.ndarray | None,
    ref_includes_query: bool,
) -> dict:
    per_class_metrics = utils.make_per_class_retrieval_metrics(
        query_labels,
        accuracy,
        reference_labels=reference_labels,
        ref_includes_query=ref_includes_query,
    )
    additional_values = {
        "r_precision": accuracy["r_precision"],
        **{
            f"precision_at_{k}": values
            for k, values in accuracy["precision_at_k"].items()
        },
    }
    for metric_name, values in additional_values.items():
        if len(values) != len(per_class_metrics):
            raise ValueError(
                f"Per-class {metric_name} count does not match eligible evaluation classes"
            )
        for class_metrics, value in zip(per_class_metrics.values(), values):
            class_metrics[metric_name] = float(value)
    return per_class_metrics


def evaluate_report_metrics(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    name: str,
    dataset,
    precision_ks: tuple[int, ...],
    return_per_class: bool,
) -> tuple[dict, dict | None]:
    """Calculate all retrieval metrics needed by the epoch-0 report in one KNN pass."""

    precision_ks = normalize_precision_ks(precision_ks)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels).reshape(-1)
    utils.validate_finite_embeddings(embeddings, name)
    evaluation_sets = utils.make_evaluation_embedding_sets(embeddings, labels, dataset=dataset)
    query_embeddings = evaluation_sets["query_embeddings"]
    query_labels = evaluation_sets["query_labels"]
    reference_embeddings = evaluation_sets["reference_embeddings"]
    reference_labels = evaluation_sets["reference_labels"]
    ref_includes_query = evaluation_sets["ref_includes_query"]

    calculator = PrecisionAtKAccuracyCalculator(
        precision_ks,
        include=(
            "precision_at_1",
            "mean_average_precision_at_r",
            "r_precision",
            "precision_at_k",
        ),
        return_per_class=return_per_class,
        k="max_bin_count",
        device=torch.device("cpu"),
    )
    accuracy = calculator.get_accuracy(
        query_embeddings,
        query_labels,
        reference=reference_embeddings,
        reference_labels=reference_labels,
        ref_includes_query=ref_includes_query,
    )

    if return_per_class:
        per_class_metrics = make_report_per_class_metrics(
            query_labels,
            accuracy,
            reference_labels,
            ref_includes_query,
        )
        metrics = {
            metric_name: utils.weighted_per_class_metric(per_class_metrics, metric_name)
            for metric_name in (
                "precision_at_1",
                "mean_average_precision_at_r",
                "r_precision",
                *(f"precision_at_{k}" for k in precision_ks),
            )
        }
    else:
        per_class_metrics = None
        metrics = {
            "precision_at_1": float(accuracy["precision_at_1"]),
            "mean_average_precision_at_r": float(accuracy["mean_average_precision_at_r"]),
            "r_precision": float(accuracy["r_precision"]),
            **{
                f"precision_at_{k}": float(value)
                for k, value in accuracy["precision_at_k"].items()
            },
        }

    if evaluation_sets["mode"] == utils.QUERY_GALLERY_EVALUATION:
        utils.logger.info(
            f"{name}: query-gallery retrieval with {len(query_labels)} queries and "
            f"{len(reference_labels)} gallery images"
        )
    precision_log = ", ".join(
        f"Precision@{k} = {metrics[f'precision_at_{k}'] * 100:.1f}"
        for k in precision_ks
    )
    utils.logger.info(
        f"{name}: Precision@1 = {metrics['precision_at_1'] * 100:.1f}, "
        f"MAP@R = {metrics['mean_average_precision_at_r'] * 100:.1f}, "
        f"R-Precision = {metrics['r_precision'] * 100:.1f}, {precision_log}"
    )
    return metrics, per_class_metrics


def write_pacmap_artifacts(
    config_path: Path,
    dataset,
    embeddings: np.ndarray,
    labels: np.ndarray,
    args: argparse.Namespace,
    cli_args: argparse.Namespace,
) -> dict:
    config_token = safe_token(relative_config_label(config_path))
    artifacts = utils.write_pacmap_visualization(
        embeddings,
        labels,
        output_dir=cli_args.output_dir,
        stem=f"{config_token}_pacmap",
        title=f"{args.dataset} epoch-0 test embeddings - PacMAP",
        dataset=dataset,
        dataset_name=args.dataset,
    )
    return {
        "pacmap_coordinates": str(artifacts["coordinates"]),
        "pacmap_plot": str(artifacts["plot"]),
        "pacmap_sample_count": int(artifacts["sample_count"]),
        "pacmap_total_count": int(len(labels)),
        "pacmap_color_basis": str(artifacts["color_basis"]),
    }


def evaluate_config(config_path: Path, cli_args: argparse.Namespace) -> dict:
    args = parse_experiment_config(config_path)
    args = apply_cli_overrides(args, cli_args)
    utils.seed_everything(args.seed, device=args.device)

    ssl_config = semi_supervised.load_ssl_config(args.ssl_config, default_seed=args.seed)
    ssl_config = experiment_main.resolve_mode_ssl_config(args, ssl_config)
    utils.validate_dataloader_settings(
        device=args.device,
        num_workers=args.num_workers,
        ssl_embedding_num_workers=0,
        start_method=args.dataloader_start_method,
    )

    dataset_bundle = load_test_bundle(args)
    test_loader = utils.make_eval_loader(
        dataset_bundle.test_dataset,
        batch_size=args.batch_size,
        seed=args.seed,
        num_workers=args.num_workers,
        start_method=args.dataloader_start_method,
    )
    model = make_epoch0_model(args, ssl_config)

    embeddings, labels = utils.extract_eval_embeddings(
        model,
        test_loader,
        name=f"{args.dataset} epoch0 test",
        device=args.device,
    )
    precision_ks = normalize_precision_ks(cli_args.precision_ks)
    metrics, per_class_metrics = evaluate_report_metrics(
        embeddings,
        labels,
        name=f"{args.dataset} epoch0 test",
        dataset=dataset_bundle.test_dataset,
        precision_ks=precision_ks,
        return_per_class=cli_args.per_class,
    )

    test_size, test_num_classes = label_summary(dataset_bundle.test_dataset)
    row = {
        "experiment_config": relative_config_label(config_path),
        "dataset": args.dataset,
        "dataset_protocol": args.dataset_protocol,
        "data_split_seed": args.data_split_seed,
        "model_seed": args.seed,
        "dino_size": args.dino_size,
        "feat_dim": "" if args.feat_dim is None else args.feat_dim,
        "backbone_tuning": args.backbone_tuning,
        "use_cache": bool(args.use_cache),
        "test_size": test_size,
        "test_num_classes": test_num_classes,
        "optimization_steps": 0,
        "training_epochs": 0,
        "test_precision_at_1": metrics["precision_at_1"],
        "test_mean_average_precision_at_r": metrics["mean_average_precision_at_r"],
        "test_r_precision": metrics["r_precision"],
        **{
            f"test_precision_at_{k}": metrics[f"precision_at_{k}"]
            for k in precision_ks
        },
    }
    if cli_args.pacmap:
        row.update(
            write_pacmap_artifacts(
                config_path,
                dataset_bundle.test_dataset,
                embeddings,
                labels,
                args,
                cli_args,
            )
        )
    return {
        "row": row,
        "split_info": dataset_bundle.split_info,
        "per_class_metrics": per_class_metrics,
    }


def write_outputs(results: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [result["row"] for result in results]
    csv_path = output_dir / "summary.csv"
    json_path = output_dir / "summary.json"

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = [
        {
            **result["row"],
            "split_info": result["split_info"],
            "per_class_metrics": result["per_class_metrics"],
        }
        for result in results
    ]
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    for result in results:
        per_class_metrics = result["per_class_metrics"]
        if per_class_metrics is None:
            continue
        config_token = safe_token(result["row"]["experiment_config"])
        per_class_path = output_dir / f"{config_token}_per_class.json"
        per_class_path.write_text(json.dumps(per_class_metrics, indent=2, sort_keys=True), encoding="utf-8")

    print(f"Wrote epoch-0 test summary: {csv_path}")
    print(f"Wrote epoch-0 test details: {json_path}")
    for row in rows:
        precision_fields = sorted(
            (
                (int(key.removeprefix("test_precision_at_")), value)
                for key, value in row.items()
                if key.startswith("test_precision_at_")
                and key != "test_precision_at_1"
            ),
            key=lambda item: item[0],
        )
        metric_summary = [
            f"P@1={row['test_precision_at_1']:.6f}",
            f"MAP@R={row['test_mean_average_precision_at_r']:.6f}",
            f"R-Precision={row['test_r_precision']:.6f}",
            *(f"P@{k}={value:.6f}" for k, value in precision_fields),
        ]
        print(
            f"{row['experiment_config']}: "
            f"{', '.join(metric_summary)}, "
            f"test_size={row['test_size']}, classes={row['test_num_classes']}"
        )




def main() -> None:
    cli_args = parse_args()
    config_paths = [resolve_config_path(path) for path in cli_args.experiment_configs]
    results = [evaluate_config(path, cli_args) for path in config_paths]
    write_outputs(results, cli_args.output_dir)


if __name__ == "__main__":
    main()
