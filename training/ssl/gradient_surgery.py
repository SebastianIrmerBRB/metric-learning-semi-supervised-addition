"""Projecting the regularizer gradient off the supervised one.

The target-ratio calibration in ``training/engine.py`` balances the two terms of
the objective by *norm*: it picks ``w_reg`` so that
``||grad(w_reg L_reg)|| / ||grad(w_sup L_sup)||`` meets a configured ratio. It
never looks at the angle between the two gradients, and on the runs measured for
this work that angle is almost always obtuse -- LRML conflicts with the
supervised loss on 74-99 % of steps at a mean cosine of -0.10 to -0.47. A
norm-only controller reads such a step as "the regularizer is too weak" and
raises its weight, without noticing that part of what it is amplifying points
against the supervised objective.

PCGrad (Yu et al., "Gradient Surgery for Multi-Task Learning", NeurIPS 2020)
supplies the missing half. It defines conflict as a negative dot product and
removes it by projection:

    g_i <- g_i - (g_i . g_j) / ||g_j||^2 * g_j     whenever g_i . g_j < 0

The two mechanisms are complementary rather than alternative: the projection
fixes the *direction*, the calibration fixes the *magnitude*, and neither
subsumes the other. This module implements the projection so both can run.

Two modes
---------
``pcgrad``
    Algorithm 1 of the paper, unchanged: every task gradient is projected onto
    the normal plane of every other conflicting task gradient, in random order.
    Symmetric -- the supervised gradient is modified too. Provided so the
    faithful method can be reported and ablated.

``supervised_priority``
    The asymmetric variant used by default here. The supervised gradient is the
    primary objective and is left untouched; only the regularizer gradient is
    projected off it:

        g_reg <- g_reg - (g_reg . g_sup) / ||g_sup||^2 * g_sup

    This is the projection idea borrowed from PCGrad, not PCGrad itself, and the
    distinction matters when reporting the method. It is the natural reading for
    a semi-supervised objective, where one term is an auxiliary regularizer and
    not a co-equal task: symmetric projection lets a conflicting regularizer
    bend the supervised update, which is exactly the interference the
    regularizer is supposed to avoid causing.

Interaction with the target ratio
---------------------------------
The projection shortens the regularizer gradient by a factor
``sqrt(1 - cos^2)``. Calibrating ``w_reg`` against the *unprojected* norm would
therefore undershoot the configured ratio by that factor -- about 5 % at
``cos = -0.3``, up to 13 % at the measured worst case. So when surgery is
active the calibration probe measures the projected norm instead, and the ratio
the objective actually realizes is again the configured one.
"""

from __future__ import annotations

import torch

# ``None`` keeps the plain combined backward pass.
DEFAULT_GRADIENT_SURGERY = None
SUPERVISED_PRIORITY = "supervised_priority"
PCGRAD = "pcgrad"
GRADIENT_SURGERY_MODES = (SUPERVISED_PRIORITY, PCGRAD)


def _dense(gradient, reference):
    """A dense tensor for ``gradient``, or zeros shaped like ``reference``."""

    if gradient is None:
        return torch.zeros_like(reference)
    if gradient.is_sparse:
        return gradient.to_dense()
    return gradient


def _dot(left, right):
    total = None
    for a, b in zip(left, right):
        term = (a.float() * b.float()).sum()
        total = term if total is None else total + term
    return torch.zeros(()) if total is None else total


def _squared_norm(vectors):
    total = None
    for v in vectors:
        term = v.float().square().sum()
        total = term if total is None else total + term
    return torch.zeros(()) if total is None else total


def project_off(target, reference, *, dot=None, reference_squared_norm=None):
    """Remove from ``target`` its component along ``reference``.

    Both are gradient tuples over the same parameters. Returns the projected
    tuple; ``target`` is not modified in place. The caller is expected to have
    checked that the two conflict -- this applies the projection unconditionally.
    """

    if dot is None:
        dot = _dot(target, reference)
    if reference_squared_norm is None:
        reference_squared_norm = _squared_norm(reference)
    if float(reference_squared_norm) <= 0.0:
        return tuple(target)
    scale = dot / reference_squared_norm
    return tuple(t - scale.to(t.dtype) * r for t, r in zip(target, reference))


def cosine(left, right):
    """Cosine between two gradient tuples; ``0.0`` if either vanishes."""

    left_norm = float(_squared_norm(left)) ** 0.5
    right_norm = float(_squared_norm(right)) ** 0.5
    if left_norm <= 0.0 or right_norm <= 0.0:
        return 0.0
    return float(_dot(left, right)) / (left_norm * right_norm)


def apply_surgery(supervised_grads, regularizer_grads, mode, generator=None):
    """Project conflicting gradients apart.

    Returns ``(supervised, regularizer, diagnostics)``. ``diagnostics`` carries
    the cosine before surgery, whether it fired, and the fraction of the
    regularizer gradient's length that survived, so the effect is visible in
    ``diagnostics.csv`` without a second measurement.
    """

    if mode not in GRADIENT_SURGERY_MODES:
        raise ValueError(
            f"gradient_surgery must be one of {list(GRADIENT_SURGERY_MODES)}: {mode!r}"
        )
    supervised = tuple(supervised_grads)
    regularizer = tuple(regularizer_grads)
    dot = _dot(supervised, regularizer)
    supervised_squared = _squared_norm(supervised)
    regularizer_squared = _squared_norm(regularizer)
    before = 0.0
    if float(supervised_squared) > 0.0 and float(regularizer_squared) > 0.0:
        before = float(dot) / (
            float(supervised_squared) ** 0.5 * float(regularizer_squared) ** 0.5
        )
    diagnostics = {
        "cosine_before": before,
        "fired": False,
        "regularizer_length_kept": 1.0,
        "supervised_length_kept": 1.0,
    }
    if float(dot) >= 0.0:
        # Non-conflicting gradients are left alone, exactly as in the paper.
        return supervised, regularizer, diagnostics

    diagnostics["fired"] = True
    if mode == SUPERVISED_PRIORITY:
        projected_regularizer = project_off(
            regularizer,
            supervised,
            dot=dot,
            reference_squared_norm=supervised_squared,
        )
        projected_supervised = supervised
    else:
        # Algorithm 1 with two tasks: both are projected, and the order does not
        # matter because each projection uses the other's *original* gradient.
        projected_regularizer = project_off(
            regularizer,
            supervised,
            dot=dot,
            reference_squared_norm=supervised_squared,
        )
        projected_supervised = project_off(
            supervised,
            regularizer,
            dot=dot,
            reference_squared_norm=regularizer_squared,
        )
    if float(regularizer_squared) > 0.0:
        diagnostics["regularizer_length_kept"] = (
            float(_squared_norm(projected_regularizer)) / float(regularizer_squared)
        ) ** 0.5
    if float(supervised_squared) > 0.0:
        diagnostics["supervised_length_kept"] = (
            float(_squared_norm(projected_supervised)) / float(supervised_squared)
        ) ** 0.5
    return projected_supervised, projected_regularizer, diagnostics


def component_gradients(loss, parameters, weight=1.0, retain_graph=True):
    """``grad(weight * loss)`` over ``parameters`` as a dense tuple.

    ``autograd.grad`` leaves ``parameter.grad`` untouched, so the caller decides
    what is accumulated. Unused and sparse entries are densified, because the
    projection needs to add and scale them elementwise.
    """

    parameters = tuple(parameters)
    if loss is None or not loss.requires_grad or float(weight) == 0.0:
        return tuple(torch.zeros_like(p) for p in parameters)
    gradients = torch.autograd.grad(
        float(weight) * loss,
        parameters,
        allow_unused=True,
        retain_graph=retain_graph,
    )
    return tuple(_dense(g, p) for g, p in zip(gradients, parameters))


def accumulate(parameters, *gradient_tuples_with_scale):
    """Add ``scale * gradients`` into ``parameter.grad``, like ``backward``."""

    parameters = tuple(parameters)
    for index, parameter in enumerate(parameters):
        total = None
        for gradients, scale in gradient_tuples_with_scale:
            scale = float(scale)
            if scale == 0.0:
                continue
            term = gradients[index]
            term = term if scale == 1.0 else term * scale
            total = term if total is None else total + term
        if total is None:
            continue
        total = total.to(parameter.dtype)
        parameter.grad = total if parameter.grad is None else parameter.grad + total


def log_gradient_surgery_configuration(regularizer):
    from loguru import logger

    logger.info(
        f"Gradient surgery enabled for {regularizer.name}: "
        f"mode={regularizer.gradient_surgery!r}; the regularizer gradient is "
        "projected off the supervised one whenever the two conflict"
        + (
            ", and the supervised gradient is projected off the regularizer as well "
            "(symmetric PCGrad)"
            if regularizer.gradient_surgery == PCGRAD
            else " (supervised gradient left untouched)"
        )
    )
