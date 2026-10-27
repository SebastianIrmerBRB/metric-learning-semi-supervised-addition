"""Supervised loader and batch helpers shared by scenario runners."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import utils as project_utils
from scripts.ssl_loss_debugger.training.context import DebugContext
from training import semi_supervised


def make_labeled_loader(
    context: DebugContext,
    source_dataset: Dataset | None = None,
    return_indices: bool = False,
) -> tuple[Dataset, DataLoader]:
    source_dataset = context.dataset if source_dataset is None else source_dataset
    labeled_dataset = semi_supervised.build_labeled_training_dataset(
        train_dataset=source_dataset,
        train_labels_mapper=context.label_mapper,
        split=context.split,
        return_indices=return_indices,
    )
    loader = project_utils.make_train_loader(
        labeled_dataset,
        batch_size=context.args.batch_size,
        sampler_m=context.args.sampler_m,
        seed=context.args.seed,
        length_before_new_iter=context.args.batch_size * context.args.steps,
        num_workers=0,
        start_method="default",
        persistent_workers=False,
        pin_memory=False,
    )
    return labeled_dataset, loader


def unpack_training_batch(batch: tuple[Any, ...]):
    if len(batch) == 2:
        images, labels = batch
        return images, labels, torch.ones(len(labels), dtype=torch.float32), None
    if len(batch) in {3, 4}:
        images, labels, sample_weights, *optional_index = batch
        return images, labels, sample_weights.float(), optional_index[0] if optional_index else None
    raise ValueError(f"Expected a 2- to 4-field supervised batch, got {len(batch)}")


def map_labels(labels: Iterable[int] | torch.Tensor, context: DebugContext) -> torch.Tensor:
    values = labels.detach().cpu().tolist() if torch.is_tensor(labels) else list(labels)
    return torch.as_tensor(
        [context.label_mapper[int(label)] for label in values],
        dtype=torch.long,
        device=context.device,
    )


def trainable_parameters(*modules: nn.Module) -> list[nn.Parameter]:
    seen: set[int] = set()
    result = []
    for module in modules:
        for parameter in module.parameters():
            if parameter.requires_grad and id(parameter) not in seen:
                result.append(parameter)
                seen.add(id(parameter))
    return result
