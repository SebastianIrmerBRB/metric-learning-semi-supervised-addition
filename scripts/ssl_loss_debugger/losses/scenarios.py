"""Scenario runners that exercise loss modules directly."""

from __future__ import annotations

import utils as project_utils
from scripts.ssl_loss_debugger.losses.factory import (
    compute_supervised_loss,
    make_supervised_criterion,
)
from scripts.ssl_loss_debugger.training.batches import (
    make_labeled_loader,
    map_labels,
    unpack_training_batch,
)
from scripts.ssl_loss_debugger.training.context import DebugContext
from scripts.ssl_loss_debugger.utils.debugging import (
    describe_tree,
    maybe_break,
    print_gradients,
    print_split,
    section,
)
from scripts.ssl_loss_debugger.utils.visualization import visualize_model_embedding_spaces


def run_direct_proxy_loss(context: DebugContext, loss_name: str) -> None:
    section(loss_name)
    print_split(context)
    criterion = make_supervised_criterion(
        loss_name,
        context.num_classes,
        context.args.embedding_dim,
        context.device,
    )
    _, loader = make_labeled_loader(context)
    batch = next(iter(loader))
    describe_tree("supervised_batch", batch)
    inputs, original_labels, sample_weights, _ = unpack_training_batch(batch)
    embeddings = context.model(inputs.to(context.device))
    labels = map_labels(original_labels, context)
    maybe_break(
        context.args,
        "before-loss",
        criterion=criterion,
        embeddings=embeddings,
        labels=labels,
        sample_weights=sample_weights,
    )
    loss = compute_supervised_loss(criterion, embeddings, labels, sample_weights)
    logits = criterion.get_logits(embeddings)
    print(f"loss={float(loss.detach()):.8f}")
    describe_tree("proxy_logits", logits)
    describe_tree("effective_sample_weights", sample_weights)
    maybe_break(context.args, "after-loss", loss=loss, logits=logits)
    maybe_break(context.args, "before-backward", loss=loss)
    loss.backward()
    print_gradients(("model", context.model), ("criterion", criterion))
    visualize_model_embedding_spaces(
        context,
        loss_name,
        criterion=criterion,
    )
    project_utils.shutdown_dataloaders(loader)
