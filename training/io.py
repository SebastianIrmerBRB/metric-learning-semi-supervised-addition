"""Serialization helpers shared by experiment modules."""

import json
from pathlib import Path

from .types import TrainingResult


def recall_at_k_to_json(mapping):
    """JSON object keys must be strings, so K is stored as a decimal string."""

    if not mapping:
        return None
    return {str(int(k)): float(value) for k, value in mapping.items()}


def recall_at_k_from_json(values):
    if not values:
        return None
    return {int(k): float(value) for k, value in values.items()}


def measurements_to_json(mapping):
    if not mapping:
        return None
    return {str(name): float(value) for name, value in mapping.items()}


def measurements_from_json(values):
    if not values:
        return None
    return {str(name): float(value) for name, value in values.items()}


def result_to_dict(result):
    return {
        "log_dir": str(result.log_dir),
        "metrics_csv": str(result.metrics_csv),
        "best_valid_precision_at_1": result.best_valid_precision_at_1,
        "best_valid_mean_average_precision_at_r": result.best_valid_mean_average_precision_at_r,
        "test_precision_at_1": result.test_precision_at_1,
        "test_mean_average_precision_at_r": result.test_mean_average_precision_at_r,
        "final_train_loss": result.final_train_loss,
        "last_epoch": result.last_epoch,
        "selected_epoch": result.selected_epoch,
        "global_step": result.global_step,
        "epoch0_test_precision_at_1": result.epoch0_test_precision_at_1,
        "epoch0_test_mean_average_precision_at_r": result.epoch0_test_mean_average_precision_at_r,
        "cv_k": result.cv_k,
        "cv_mode": result.cv_mode,
        "cv_fold": result.cv_fold,
        "fold_results": result.fold_results,
        "validation_retrieval_backend": getattr(
            result,
            "validation_retrieval_backend",
            None,
        ),
        "test_retrieval_backend": getattr(result, "test_retrieval_backend", None),
        "test_pacmap_coordinates": None
        if result.test_pacmap_coordinates is None
        else str(result.test_pacmap_coordinates),
        "test_pacmap_plot": None if result.test_pacmap_plot is None else str(result.test_pacmap_plot),
        "test_tsne_coordinates": None
        if result.test_tsne_coordinates is None
        else str(result.test_tsne_coordinates),
        "test_tsne_plot": None if result.test_tsne_plot is None else str(result.test_tsne_plot),
        "test_embeddings_path": None
        if getattr(result, "test_embeddings_path", None) is None
        else str(result.test_embeddings_path),
        "head_state_path": None
        if getattr(result, "head_state_path", None) is None
        else str(result.head_state_path),
        "warmup_best_valid_precision_at_1": getattr(
            result,
            "warmup_best_valid_precision_at_1",
            None,
        ),
        "warmup_best_valid_mean_average_precision_at_r": getattr(
            result,
            "warmup_best_valid_mean_average_precision_at_r",
            None,
        ),
        "warmup_selected_epoch": getattr(result, "warmup_selected_epoch", None),
        "warmup_checkpoint_mode": getattr(result, "warmup_checkpoint_mode", None),
        "warmup_restored_epoch": getattr(result, "warmup_restored_epoch", None),
        "test_precision_at_1_std": getattr(result, "test_precision_at_1_std", None),
        "test_mean_average_precision_at_r_std": getattr(
            result,
            "test_mean_average_precision_at_r_std",
            None,
        ),
        "concatenated_test_precision_at_1": getattr(
            result,
            "concatenated_test_precision_at_1",
            None,
        ),
        "concatenated_test_mean_average_precision_at_r": getattr(
            result,
            "concatenated_test_mean_average_precision_at_r",
            None,
        ),
        "concatenated_test_embedding_dim": getattr(
            result,
            "concatenated_test_embedding_dim",
            None,
        ),
        "best_valid_recall_at_k": recall_at_k_to_json(
            getattr(result, "best_valid_recall_at_k", None)
        ),
        "test_recall_at_k": recall_at_k_to_json(getattr(result, "test_recall_at_k", None)),
        "epoch0_test_recall_at_k": recall_at_k_to_json(
            getattr(result, "epoch0_test_recall_at_k", None)
        ),
        "concatenated_test_recall_at_k": recall_at_k_to_json(
            getattr(result, "concatenated_test_recall_at_k", None)
        ),
        "best_valid_measurements": measurements_to_json(
            getattr(result, "best_valid_measurements", None)
        ),
        "test_measurements": measurements_to_json(
            getattr(result, "test_measurements", None)
        ),
        "epoch0_test_measurements": measurements_to_json(
            getattr(result, "epoch0_test_measurements", None)
        ),
        "test_measurements_std": measurements_to_json(
            getattr(result, "test_measurements_std", None)
        ),
        "concatenated_test_measurements": measurements_to_json(
            getattr(result, "concatenated_test_measurements", None)
        ),
    }

def result_from_dict(values):
    """Rebuild a :class:`TrainingResult` recorded by :func:`result_to_dict`.

    Records written by older revisions may lack fields added later, so every
    optional entry falls back to the dataclass default instead of failing.
    """

    return TrainingResult(
        log_dir=Path(values["log_dir"]),
        metrics_csv=Path(values["metrics_csv"]),
        best_valid_precision_at_1=values.get("best_valid_precision_at_1"),
        best_valid_mean_average_precision_at_r=values.get("best_valid_mean_average_precision_at_r"),
        test_precision_at_1=values.get("test_precision_at_1"),
        test_mean_average_precision_at_r=values.get("test_mean_average_precision_at_r"),
        final_train_loss=values.get("final_train_loss"),
        last_epoch=values.get("last_epoch", 0),
        selected_epoch=values.get("selected_epoch", 0),
        global_step=values.get("global_step", 0),
        epoch0_test_precision_at_1=values.get("epoch0_test_precision_at_1"),
        epoch0_test_mean_average_precision_at_r=values.get("epoch0_test_mean_average_precision_at_r"),
        cv_k=values.get("cv_k", 1),
        cv_mode=values.get("cv_mode"),
        cv_fold=values.get("cv_fold"),
        fold_results=values.get("fold_results"),
        validation_retrieval_backend=values.get("validation_retrieval_backend"),
        test_retrieval_backend=values.get("test_retrieval_backend"),
        test_pacmap_coordinates=as_optional_path(values.get("test_pacmap_coordinates")),
        test_pacmap_plot=as_optional_path(values.get("test_pacmap_plot")),
        test_tsne_coordinates=as_optional_path(values.get("test_tsne_coordinates")),
        test_tsne_plot=as_optional_path(values.get("test_tsne_plot")),
        test_embeddings_path=as_optional_path(values.get("test_embeddings_path")),
        head_state_path=as_optional_path(values.get("head_state_path")),
        warmup_best_valid_precision_at_1=values.get(
            "warmup_best_valid_precision_at_1"
        ),
        warmup_best_valid_mean_average_precision_at_r=values.get(
            "warmup_best_valid_mean_average_precision_at_r"
        ),
        warmup_selected_epoch=values.get("warmup_selected_epoch"),
        warmup_checkpoint_mode=values.get("warmup_checkpoint_mode"),
        warmup_restored_epoch=values.get("warmup_restored_epoch"),
        test_precision_at_1_std=values.get("test_precision_at_1_std"),
        test_mean_average_precision_at_r_std=values.get(
            "test_mean_average_precision_at_r_std"
        ),
        concatenated_test_precision_at_1=values.get("concatenated_test_precision_at_1"),
        concatenated_test_mean_average_precision_at_r=values.get(
            "concatenated_test_mean_average_precision_at_r"
        ),
        concatenated_test_embedding_dim=values.get("concatenated_test_embedding_dim"),
        best_valid_recall_at_k=recall_at_k_from_json(values.get("best_valid_recall_at_k")),
        test_recall_at_k=recall_at_k_from_json(values.get("test_recall_at_k")),
        epoch0_test_recall_at_k=recall_at_k_from_json(values.get("epoch0_test_recall_at_k")),
        concatenated_test_recall_at_k=recall_at_k_from_json(
            values.get("concatenated_test_recall_at_k")
        ),
        best_valid_measurements=measurements_from_json(
            values.get("best_valid_measurements")
        ),
        test_measurements=measurements_from_json(values.get("test_measurements")),
        epoch0_test_measurements=measurements_from_json(
            values.get("epoch0_test_measurements")
        ),
        test_measurements_std=measurements_from_json(
            values.get("test_measurements_std")
        ),
        concatenated_test_measurements=measurements_from_json(
            values.get("concatenated_test_measurements")
        ),
    )

def as_optional_path(value):
    return None if value is None else Path(value)

def namespace_to_dict(args):
    return {key: to_jsonable(value) for key, value in vars(args).items()}

def to_jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)

def is_scalar(value):
    return isinstance(value, (str, int, float, bool)) or value is None

def write_json(path, data):
    with Path(path).open("w") as json_file:
        json.dump(to_jsonable(data), json_file, indent=2, sort_keys=True)

def read_json(path):
    with Path(path).open(encoding="utf-8") as json_file:
        return json.load(json_file)
