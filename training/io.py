"""Serialization helpers shared by experiment modules."""

import json
from pathlib import Path

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
        "test_pacmap_coordinates": None
        if result.test_pacmap_coordinates is None
        else str(result.test_pacmap_coordinates),
        "test_pacmap_plot": None if result.test_pacmap_plot is None else str(result.test_pacmap_plot),
        "test_tsne_coordinates": None
        if result.test_tsne_coordinates is None
        else str(result.test_tsne_coordinates),
        "test_tsne_plot": None if result.test_tsne_plot is None else str(result.test_tsne_plot),
    }

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
