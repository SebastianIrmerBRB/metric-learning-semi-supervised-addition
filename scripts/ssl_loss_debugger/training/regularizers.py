"""Training loop for regularized SSL debugger scenarios."""

from __future__ import annotations

from typing import Any

import torch

import utils as project_utils
from scripts.ssl_loss_debugger.losses.factory import (
    compute_supervised_loss,
    make_supervised_criterion,
)
from scripts.ssl_loss_debugger.training.batches import (
    make_labeled_loader,
    map_labels,
    trainable_parameters,
    unpack_training_batch,
)
from scripts.ssl_loss_debugger.training.configuration import make_regularizer_config
from scripts.ssl_loss_debugger.training.context import DebugContext
from scripts.ssl_loss_debugger.training.seraph_trace import (
    build_seraph_comparison,
    debug_true_labels,
)
from scripts.ssl_loss_debugger.utils.debugging import (
    GraphEpochTracker,
    RegularizerEpochTracker,
    describe_tree,
    maybe_break,
    print_gradients,
    print_graph,
    print_regularizer_batch_details,
    print_seraph_comparison_diagnostics,
    print_split,
    section,
)
from scripts.ssl_loss_debugger.utils.visualization import (
    save_seraph_comparison_artifacts,
    visualization_scenario_dir,
    visualize_model_embedding_spaces,
)
from training import semi_supervised


def reports_this_epoch(context: DebugContext, epoch: int) -> bool:
    """Return whether ``epoch`` prints per-step detail and writes artifacts.

    The first and last epoch always report so a throttled run still shows the
    state the run started from and the state it ended in.
    """

    if epoch in (0, context.args.epochs - 1):
        return True
    return epoch % context.args.report_every == 0


def run_regularizer(context: DebugContext, name: str) -> None:
    section(f"{context.args.supervised_loss} + {name}")
    print_split(context)
    config = make_regularizer_config(context, name)
    method = semi_supervised.get_method(config)
    if method is None:
        raise RuntimeError(f"Could not resolve regularization method for {name}")
    regularizer = method.make_regularizer(config)
    regularizer.set_batch_diagnostics_enabled(True)
    print(f"method={config.method}, method_params={config.method_params}")

    maybe_break(
        context.args,
        "before-build",
        regularizer=regularizer,
        config=config,
        dataset=context.dataset,
        split=context.split,
        model=context.model,
    )
    regularizer.configure_model(
        student_model=context.model,
        train_dataset=context.dataset,
        split=context.split,
        train_labels_mapper=context.label_mapper,
        device=context.device,
    )
    source_dataset = regularizer.make_supervised_source_dataset(context.dataset)
    labeled_dataset, supervised_loader = make_labeled_loader(
        context,
        source_dataset=source_dataset,
        # SERAPH itself does not need the indices; the debugger does, to name the
        # labeled endpoint of every scored labeled-unlabeled pair.
        return_indices=(regularizer.requires_labeled_indices or name == "seraph"),
    )
    regularizer_dataset = regularizer.build_dataset(
        context.dataset,
        context.split,
        use_cache=False,
    )
    print(
        f"supervised dataset={type(labeled_dataset).__name__}(len={len(labeled_dataset)}), "
        f"regularizer dataset={type(regularizer_dataset).__name__}"
        f"(len={0 if regularizer_dataset is None else len(regularizer_dataset)})"
    )
    maybe_break(
        context.args,
        "after-build",
        regularizer=regularizer,
        regularizer_dataset=regularizer_dataset,
        labeled_dataset=labeled_dataset,
    )
    criterion = make_supervised_criterion(
        context.args.supervised_loss,
        context.num_classes,
        context.args.embedding_dim,
        context.device,
    )
    optimizer = torch.optim.SGD(
        trainable_parameters(context.model, criterion),
        lr=1e-2,
    )
    state = regularizer.initialize_state(context.model, context.device)
    context.model.train()

    graph_tracker = GraphEpochTracker()
    epoch_tracker = RegularizerEpochTracker()
    combined_loader = None
    try:
        for epoch in range(context.args.epochs):
            reporting = reports_this_epoch(context, epoch)
            if reporting:
                print(f"\nepoch {epoch}: rebuilding/refreshing regularizer loader")
            maybe_break(
                context.args,
                "before-loader",
                epoch=epoch,
                regularizer=regularizer,
                model=context.model,
                config=config,
                supervised_loader=supervised_loader,
            )
            combined_loader = regularizer.make_loader(
                model=context.model,
                train_dataset=context.dataset,
                supervised_loader=supervised_loader,
                device=context.device,
                config=config,
                batch_size=context.args.batch_size,
                seed=context.args.seed + epoch,
                num_workers=0,
                start_method="default",
                epoch=epoch,
                log_dir=(
                    visualization_scenario_dir(context, name)
                    if context.args.visualize and reporting
                    else None
                ),
            )
            if reporting:
                print_graph(regularizer)
                graph_tracker.record(epoch, regularizer)
            maybe_break(
                context.args,
                "after-loader",
                epoch=epoch,
                regularizer=regularizer,
                combined_loader=combined_loader,
            )
            _run_regularizer_epoch(
                context=context,
                name=name,
                regularizer=regularizer,
                combined_loader=combined_loader,
                labeled_dataset=labeled_dataset,
                criterion=criterion,
                optimizer=optimizer,
                state=state,
                epoch=epoch,
                reporting=reporting,
                epoch_tracker=epoch_tracker,
            )
    finally:
        if combined_loader is not None:
            combined_loader.shutdown(include_persistent=True)
        project_utils.shutdown_dataloaders(supervised_loader)

    graph_tracker.print_summary()
    epoch_tracker.print_summary()
    visualize_model_embedding_spaces(
        context,
        name,
        criterion=criterion,
        regularizer=regularizer,
        state=state,
    )


def _run_regularizer_epoch(
    *,
    context: DebugContext,
    name: str,
    regularizer: Any,
    combined_loader: Any,
    labeled_dataset: Any,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: Any,
    epoch: int,
    reporting: bool,
    epoch_tracker: Any,
) -> None:
    for step, (supervised_batch, regularizer_batch) in enumerate(combined_loader):
        if step >= context.args.steps:
            break
        artifact_step = epoch * context.args.steps + step
        if reporting:
            print(f"\nepoch {epoch}, step {step}")
            describe_tree("supervised_batch", supervised_batch)
            describe_tree("regularizer_batch", regularizer_batch)
            print_regularizer_batch_details(context, regularizer, regularizer_batch)

        optimizer.zero_grad(set_to_none=True)
        inputs, original_labels, sample_weights, supervised_indices = (
            unpack_training_batch(supervised_batch)
        )
        supervised_inputs = inputs
        labels = map_labels(original_labels, context)
        sample_weights = sample_weights.to(context.device)
        supervised_batch_size = len(labels)

        regularizer_embeddings = None
        if regularizer.uses_joint_forward:
            joint_inputs = torch.cat([inputs, regularizer_batch[0]], dim=0)
            joint_embeddings = context.model(joint_inputs.to(context.device))
            embeddings = joint_embeddings[:supervised_batch_size]
            regularizer_embeddings = joint_embeddings[supervised_batch_size:]
        else:
            embeddings = context.model(inputs.to(context.device))

        supervised_loss = compute_supervised_loss(
            criterion,
            embeddings,
            labels,
            sample_weights,
        )
        regularizer_kwargs: dict[str, Any] = {}
        if regularizer.uses_joint_forward:
            regularizer_kwargs.update(
                supervised_embeddings=embeddings,
                supervised_labels=labels,
                regularizer_embeddings=regularizer_embeddings,
            )
        supervised_positions = None
        if name == "seraph":
            if supervised_indices is None:
                raise RuntimeError(
                    "SERAPH debugger needs labeled subset indices to decode source samples"
                )
            # Debugger-side bookkeeping only: SERAPH's compute_loss takes no
            # positions, so this stays out of regularizer_kwargs.
            labeled_positions = torch.as_tensor(
                labeled_dataset.positions,
                dtype=torch.long,
            )
            supervised_positions = labeled_positions[supervised_indices.long()]
        if regularizer.requires_supervised_objective:
            regularizer_kwargs.update(
                supervised_inputs=supervised_inputs,
                supervised_indices=supervised_indices,
                supervised_criterion=criterion,
                supervised_miner=None,
                supervised_is_classification=True,
            )

        timings: dict[str, float] = {}
        maybe_break(
            context.args,
            "before-loss",
            regularizer=regularizer,
            criterion=criterion,
            state=state,
            supervised_batch=supervised_batch,
            regularizer_batch=regularizer_batch,
            supervised_embeddings=embeddings,
            regularizer_embeddings=regularizer_embeddings,
            supervised_loss=supervised_loss,
            regularizer_kwargs=regularizer_kwargs,
        )
        regularizer_call_kwargs = {
            "student_model": context.model,
            "state": state,
            "batch": regularizer_batch,
            "device": context.device,
            "timings": timings,
            **regularizer_kwargs,
        }
        regularizer_loss = regularizer.compute_loss(**regularizer_call_kwargs)
        total_loss = regularizer.combine_losses(supervised_loss, regularizer_loss)
        diagnostics = regularizer.batch_diagnostics()
        comparison_diagnostics = {}
        if name == "seraph" and reporting:
            comparison_diagnostics = build_seraph_comparison(
                regularizer=regularizer,
                head=context.model.seraph_posterior,
                diagnostics=diagnostics,
                supervised_embeddings=embeddings,
                regularizer_embeddings=regularizer_embeddings,
                supervised_positions=supervised_positions,
                supervised_labels=labels,
                unlabeled_positions=regularizer_batch[2],
                max_pairs_per_block=context.args.seraph_trace_pairs,
                true_labels={
                    "unlabeled": debug_true_labels(
                        context.dataset.labels,
                        regularizer_batch[2],
                    ),
                    "labeled": debug_true_labels(
                        context.dataset.labels,
                        supervised_positions,
                    ),
                },
            )
            print_seraph_comparison_diagnostics(context, comparison_diagnostics)
            save_seraph_comparison_artifacts(
                context=context,
                comparisons=comparison_diagnostics,
                supervised_inputs=supervised_inputs,
                regularizer_inputs=regularizer_batch[0],
                step=artifact_step,
            )
        if reporting:
            print(
                f"supervised_loss={float(supervised_loss.detach()):.8f}, "
                f"regularizer_loss={float(regularizer_loss.detach()):.8f}, "
                f"combined_loss={float(total_loss.detach()):.8f}"
            )
            if timings:
                print(f"timings: {timings}")
            if diagnostics:
                print(f"diagnostics: {diagnostics}")
        maybe_break(
            context.args,
            "after-loss",
            supervised_loss=supervised_loss,
            regularizer_loss=regularizer_loss,
            total_loss=total_loss,
            diagnostics=diagnostics,
            comparison_diagnostics=comparison_diagnostics,
            state=state,
        )
        maybe_break(context.args, "before-backward", total_loss=total_loss)

        named_gradient_modules = (("model", context.model), ("criterion", criterion))
        weighted_supervised_loss = regularizer.supervised_weight * supervised_loss
        weighted_regularizer_loss = regularizer.regularizer_weight * regularizer_loss

        # Retain the shared forward graph while measuring each weighted term in
        # isolation. Clear .grad between passes so the reports are contributions,
        # not accumulated gradients. Only the final combined pass reaches step().
        weighted_supervised_loss.backward(retain_graph=True)
        supervised_gradient_norm = print_gradients(
            *named_gradient_modules,
            heading=(
                "gradient norms (weighted supervised term, "
                f"weight={regularizer.supervised_weight:g})"
            ),
            show_missing=True,
            quiet=not reporting,
        )

        optimizer.zero_grad(set_to_none=True)
        weighted_regularizer_loss.backward(retain_graph=True)
        regularizer_gradient_norm = print_gradients(
            *named_gradient_modules,
            heading=(
                "gradient norms (weighted regularizer term, "
                f"weight={regularizer.regularizer_weight:g})"
            ),
            show_missing=True,
            quiet=not reporting,
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        combined_gradient_norm = print_gradients(
            *named_gradient_modules,
            heading="gradient norms (combined loss used by optimizer)",
            show_missing=True,
            quiet=not reporting,
        )
        if reporting:
            if supervised_gradient_norm == 0.0:
                ratio = "undefined (supervised gradient norm is zero)"
            else:
                ratio = f"{regularizer_gradient_norm / supervised_gradient_norm:.6g}"
            print(
                "global gradient norm summary: "
                f"supervised={supervised_gradient_norm:.6g}, "
                f"regularizer={regularizer_gradient_norm:.6g}, "
                f"regularizer/supervised={ratio}, "
                f"combined={combined_gradient_norm:.6g}"
            )
        row = epoch_tracker.record(
            epoch=epoch,
            step=step,
            supervised_loss=float(supervised_loss.detach()),
            regularizer_loss=float(regularizer_loss.detach()),
            combined_loss=float(total_loss.detach()),
            supervised_gradient_norm=supervised_gradient_norm,
            regularizer_gradient_norm=regularizer_gradient_norm,
            extra=_tracked_diagnostics(name, diagnostics),
        )
        if not reporting:
            # The epochs that print nothing else still contribute their line to
            # the series a long run is watched through.
            epoch_tracker.print_row(row)
        optimizer.step()
        regularizer.after_optimizer_step(context.model, state)


# Per-scenario diagnostics worth following step by step, alongside the losses and
# gradient norms every scenario reports.
TRACKED_DIAGNOSTICS = {
    "seraph": (
        # docs/seraph.md: the gradient peaks near mean |logit| = 2.4 and vanishes
        # at both ends, and eta and kappa are what move it there.
        "train/seraph/uu/logit_abs_mean",
        "train/seraph/uu/pair_entropy",
        "train/seraph/uu/threshold",
        "train/seraph/uu/kappa",
        "train/seraph/lu/logit_abs_mean",
        "train/seraph/lu/threshold",
    ),
}


def _tracked_diagnostics(name: str, diagnostics: dict[str, float]) -> dict[str, float]:
    return {
        key.rsplit("train/", 1)[-1]: diagnostics[key]
        for key in TRACKED_DIAGNOSTICS.get(name, ())
        if key in diagnostics
    }
