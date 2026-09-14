"""Deterministic feature-dataset and embedding extraction helpers."""

import copy
from contextlib import contextmanager

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import utils

from .algorithms import ssl_algorithm_device


@contextmanager
def ssl_compute_device(device):
    """Select the device used by an entire out-of-batch SSL pipeline."""

    device = torch.device(utils.normalize_device_name(device))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                f"SSL computation requested {device}, but CUDA is not available"
            )
        # FAISS, CuPy, and Torch helpers that do not receive an explicit device
        # all follow the process's active CUDA device inside this context.
        with torch.cuda.device(device), ssl_algorithm_device(device):
            yield device
        return
    with ssl_algorithm_device(device):
        yield device


def _module_device(model):
    """Return the first parameter/buffer device, or ``None`` for a stateless module."""

    for tensor in model.parameters():
        return tensor.device
    for tensor in model.buffers():
        return tensor.device
    return None


def extract_embeddings(
    model,
    dataset,
    positions,
    device,
    batch_size,
    num_workers,
    seed,
    start_method,
    desc,
    embedding_kind="default",
    loader=None
):
    """Extract deterministic embeddings on ``device`` and restore model placement."""

    # Work on a copy using deterministic feature transforms; training
    # augmentation would make pseudo-labels depend on random image distortions.
    # Subset preserves the positions order, which all pseudo-label methods rely
    # on when splitting the resulting embedding matrix.
    if loader is None:
        loader = make_embedding_loader(
        dataset, positions=positions,
        batch_size=batch_size, num_workers=num_workers, seed=seed, start_method=start_method
    )

    # Pseudo-labels should use stable evaluation behavior. A CPU training run
    # may still reserve a CUDA device for this phase, so temporarily migrate
    # the same module and put it back before optimizer-driven training resumes.
    was_training = model.training
    original_device = _module_device(model)
    all_embeddings = []
    with ssl_compute_device(device) as compute_device:
        should_move_model = (
            original_device is not None and original_device != compute_device
        )
        try:
            if should_move_model:
                model.to(compute_device)
            model.eval()
            with torch.no_grad():
                for images, _ in tqdm(loader, desc=desc):
                    # Labels are deliberately ignored: pseudo-label generation
                    # must use only images/embeddings for the unlabeled pool.
                    if embedding_kind == "default":
                        forward_cached = getattr(model, "forward_cached", None)
                        embeddings = utils.forward_model_inputs(
                            model,
                            images,
                            compute_device,
                            use_cache=forward_cached is not None,
                        )
                    elif embedding_kind == "stml_g":
                        forward_stml_cached = getattr(model, "forward_stml_cached", None)
                        if forward_stml_cached is None:
                            raise AttributeError("Model does not expose forward_stml_cached")
                        embeddings, _ = forward_stml_cached(images, compute_device)
                    else:
                        raise ValueError(f"Unknown embedding_kind: {embedding_kind}")
                    # .float() before .numpy(): numpy has no bfloat16, so an
                    # embedding produced under autocast must leave torch as
                    # float32 rather than raising on the conversion.
                    all_embeddings.append(
                        embeddings.detach().float().cpu().numpy().astype(np.float32)
                    )
        finally:
            if should_move_model:
                model.to(original_device)
                if compute_device.type == "cuda":
                    torch.cuda.empty_cache()
            model.train(was_training)

    # Concatenation restores one [num_positions, embedding_dim] matrix in loader
    # order.
    return np.concatenate(all_embeddings)


def make_feature_dataset(dataset):
    """Copy a dataset and replace augmentation with its feature transform."""

    if utils.dataset_has_precomputed_backbone_features(dataset):
        # Raw backbone feature tensors are already deterministic and have no
        # image transform to replace. Reuse the read-only dataset so every
        # pseudo-label refresh does not duplicate the full in-memory matrix.
        return dataset

    # Copy before changing transforms so the real training dataset continues to
    # use stochastic augmentation.
    feature_dataset = copy.deepcopy(dataset)
    feature_transform = utils.get_nested_feature_transform(dataset)
    if feature_transform is not None:
        set_nested_transform(feature_dataset, feature_transform)
    return feature_dataset


def set_nested_transform(dataset, transform):
    """Set the transform on the base dataset beneath any Subset wrappers."""

    utils.set_nested_transform(dataset, transform)


def make_embedding_loader(dataset, positions, batch_size, num_workers, seed, start_method):
    feature_dataset = make_feature_dataset(dataset)   # deepcopy now happens once
    kwargs = utils.make_dataloader_kwargs(num_workers, seed, start_method)
    if kwargs.get("num_workers", 0) > 0:
        kwargs["persistent_workers"] = True           # invalid with num_workers=0
    return DataLoader(
        Subset(feature_dataset, [int(p) for p in positions]),
        batch_size=batch_size, shuffle=False, **kwargs,
    )
