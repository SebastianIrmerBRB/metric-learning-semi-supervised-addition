"""Deterministic feature-dataset and embedding extraction helpers."""

import copy

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import utils


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
    """Extract deterministic evaluation-transform embeddings for given positions."""

    # Work on a copy using deterministic feature transforms; training
    # augmentation would make pseudo-labels depend on random image distortions.
    # Subset preserves the positions order, which all pseudo-label methods rely
    # on when splitting the resulting embedding matrix.
    if loader is None:
        loader = make_embedding_loader(
        dataset, positions=positions,
        batch_size=batch_size, num_workers=num_workers, seed=seed, start_method=start_method
    )

    # Pseudo-labels should use stable evaluation behavior, but restore the
    # caller's previous mode after extraction.
    was_training = model.training
    model.eval()
    all_embeddings = []
    with torch.no_grad():
        for images, _ in tqdm(loader, desc=desc):
            # Labels are deliberately ignored: pseudo-label generation must use
            # only images/embeddings for the unlabeled candidate pool.
            if embedding_kind == "default":
                forward_cached = getattr(model, "forward_cached", None)
                embeddings = utils.forward_model_inputs(
                    model,
                    images,
                    device,
                    use_cache=forward_cached is not None,
                )
            elif embedding_kind == "stml_g":
                forward_stml_cached = getattr(model, "forward_stml_cached", None)
                if forward_stml_cached is None:
                    raise AttributeError("Model does not expose forward_stml_cached")
                embeddings, _ = forward_stml_cached(images, device)
            else:
                raise ValueError(f"Unknown embedding_kind: {embedding_kind}")
            all_embeddings.append(embeddings.cpu().numpy().astype(np.float32))
    if was_training:
        model.train()

    # Concatenation restores one [num_positions, embedding_dim] matrix in loader
    # order.
    return np.concatenate(all_embeddings)


def make_feature_dataset(dataset):
    """Copy a dataset and replace augmentation with its feature transform."""

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
