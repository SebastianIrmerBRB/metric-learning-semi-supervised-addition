"""GradNorm weighting for every weighted term in the training objective.

Implements Chen et al., "GradNorm: Gradient Normalization for Adaptive Loss
Balancing in Deep Multitask Networks" (arXiv:1711.02257) for the objective this
trainer optimizes, ``w_sup * L_sup + w_reg * L_reg + sum_k w_k * L_k``, where the
extra terms are the ones a regularizer adds itself in ``combine_losses``.

Every weight becomes learnable and is trained, at every update, against

    L_grad = sum_i | G_i - Gbar * r_i ** alpha |

where ``G_i = ||grad_W(w_i * L_i)||`` over the shared parameters ``W``, ``Gbar``
is the mean over the terms, and ``r_i = (L_i / L_i(0)) / mean_j(L_j / L_j(0))``
is the relative inverse training rate. The target is treated as a constant, so
the only thing the update moves is the weights themselves. ``alpha`` sets how
hard a term that trains slowly is pulled back: ``0`` asks only for equal
gradient norms, larger values increasingly favor whichever term has made least
progress.

The term count is whatever the objective has. A plain regularizer gives the two
terms ``(supervised, regularizer)``; SLADE's Eq 9 gives three, since its basis
loss rides alongside the ranking term under its own ``lambda_2``, and the
paper's own words for choosing ``lambda_1`` and ``lambda_2`` -- "empirically
chosen to make the magnitudes of the losses similar in scale" -- are GradNorm's
problem statement.

``W`` must be *shared*: the paper takes it over weights every term's gradient
reaches. Regularizers that attach their own module to the student model declare
it in ``private_model_module_names`` so the trainer can leave those parameters
out. Counting them would let a term's private head set the balance of the shared
trunk, and for SLADE it would also hide its basis warmup, during which the
embedding is detached and the basis gradient reaches ``W_a`` alone.

``W`` is taken over *all* shared parameters rather than the last shared layer.
The paper defines ``W`` as any subset of the shared weights and picks the last
layer only "to save on compute costs", so this is the more exact reading of the
definition. It is also the more expensive one: the paper measured ~5% added
training time for its choice, while a full-trunk measurement costs one
``autograd.grad`` over the trunk per term per update.

Section 5.1 fixes the update rule: "Updating w_i(t) is performed at a learning
rate of 0.025 ... All optimizers are Adam, although we find that GradNorm is
insensitive to the optimizer chosen." So the weights get their own Adam at
``DEFAULT_GRAD_NORM_LR``, separate from the model optimizer, whose learning rate
is unrelated (2e-5 in the paper's own runs).

Three deviations from the reference implementation
(github.com/LucasBoTang/GradNorm). All three follow the paper where the
reference does not, so none of them is an approximation:

* ``G_i`` is computed as ``w_i * ||grad_W(L_i)||`` with the norm detached rather
  than through ``create_graph=True``. ``grad_W(w * L) == w * grad_W(L)`` exactly,
  so the derivative with respect to ``w_i`` is identical -- verified
  bit-identical on a shared-trunk two-task problem -- and this avoids building a
  double-backward graph on every update.
* Renormalization writes the weights in place and keeps one optimizer. The
  reference rewraps the tensor and rebuilds Adam every step, which discards the
  moments and leaves Adam permanently taking its bias-corrected first step, i.e.
  sign-SGD at a fixed stride of ``lr``. Algorithm 1 says to update ``w_i`` "using
  grad_wi L_grad" under standard update rules, so the moments belong.
* ``L_grad`` is differentiated only with respect to the weights. The reference
  backpropagates it into the network as well, and its model optimizer then
  applies that gradient; the paper is explicit that "L_grad is then
  differentiated only with respect to the w_i".

One deliberate deviation from the *paper*, in the renormalization step: the
default pins the supervised weight rather than holding ``sum_i w_i`` at ``T``,
so GradNorm adapts the SSL terms against a metric loss whose scale is already
tuned. ``grad_norm_renormalize="sum"`` restores Algorithm 1 exactly. See
``RENORMALIZE_MODES`` below for what each gauge costs.
"""

from __future__ import annotations

import math

import torch
from loguru import logger


DEFAULT_GRAD_NORM_LR = 0.025
DEFAULT_GRAD_NORM_UPDATE_INTERVAL = 1
# ``sum`` is Algorithm 1's constraint -- "we also renormalize the weights w_i(t)
# so that sum_i w_i(t) = T in order to decouple gradient normalization from the
# global learning rate" -- taking T from the configured weights rather than
# assuming the term count. It is the faithful mode, and the one to use when the
# question is what GradNorm does to this objective.
#
# ``supervised`` is the default instead, because the question here is usually
# narrower: what weight should the SSL term carry against a metric loss whose
# scale is already tuned. It pins the supervised weight at its configured value
# and rescales the rest around it, so GradNorm adapts the regularizer's weights
# and leaves the supervised term alone. That is also what SLADE's Eq 9 does by
# hand -- the labeled ranking loss carries a bare coefficient of 1 and lambda_1,
# lambda_2 are chosen relative to it.
#
# Both are gauge choices: scaling every weight by c scales every G_i and Gbar by
# c, so the balance GradNorm converges to is the same either way, and only the
# scale of the total objective differs. Two consequences of that scale are worth
# knowing. Under ``sum`` the total is fixed, which is the decoupling the paper
# asks for; on terms whose gradients differ by orders of magnitude it reaches the
# balance by driving the supervised weight down, measured on Cars196 with lrml
# hitting the 1e-8 floor and training a regularizer-only model. Under
# ``supervised`` the total is not fixed: the same imbalance is met by lifting the
# other weights, which multiplies the total loss -- and with it the model's
# effective learning rate -- by that factor, unseen by the LR schedule. If the
# learned regularizer weight ends up orders above its starting value, read it as
# that, not just as a balance.
# ``none`` leaves the raw Adam updates alone.
RENORMALIZE_MODES = ("sum", "supervised", "none")
DEFAULT_GRAD_NORM_RENORMALIZE = "supervised"

# Fraction of its starting value the supervised weight may fall to before the
# run is worth warning about: below this the metric loss is effectively off.
SUPERVISED_COLLAPSE_FRACTION = 0.01

# Weights are kept strictly positive: a negative weight would flip a term into
# rewarding what it is meant to penalize.
MINIMUM_WEIGHT = 1e-8

# How the weights themselves are stored while Adam updates them.
#
# ``linear`` is the paper: w_i is the parameter, so an Adam step moves it by an
# absolute amount of order ``lr``. That is fine while the balanced weights sit
# near 1, which is where the paper's tasks start (w_i(0) = 1, losses of similar
# scale). It fails once a term's gradient is orders above another's, because the
# balanced weight then falls below ``lr`` itself and Adam cannot land on it: it
# overshoots into the MINIMUM_WEIGHT clamp, the clamp puts w_i * G_i under the
# target so the L_grad gradient flips sign and pushes back up, and the weight
# limit-cycles between the floor and O(lr) instead of settling. Measured on
# Cars196 with lrml + CircleLoss, whose balanced supervised weight is ~3e-3
# against an ``lr`` of 0.025: the weight sat at the floor on 11% of steps, and on
# those the supervised term contributed nothing at all.
#
# ``log`` stores v_i = log(w_i) and lets Adam work there, which makes the step
# relative rather than absolute -- roughly a factor of exp(lr) per update, ~2.5%
# at the default. The weight can then reach 1e-3 or 1e-8 and hold, because the
# step shrinks with it. It changes the update rule, not the objective: L_grad,
# the targets and the renormalization are untouched, and dG_i/dv_i = w_i * G_i
# keeps the same sign as dG_i/dw_i, so it moves toward the same balance.
# Use it when the balanced weights are far from 1; ``linear`` reproduces the
# paper and stays the default.
PARAMETERIZATIONS = ("linear", "log")
DEFAULT_GRAD_NORM_PARAMETERIZATION = "linear"

SUPERVISED_TERM = "supervised"
REGULARIZER_TERM = "regularizer"


class GradNormWeights:
    """Learnable objective weights trained by GradNorm.

    The first two terms are always ``(supervised, regularizer)``; ``extra_weights``
    adds one more per term the regularizer contributes in ``combine_losses``,
    keyed by the same name it reports in ``extra_loss_components``.
    """

    def __init__(
        self,
        alpha,
        supervised_weight=1.0,
        regularizer_weight=1.0,
        extra_weights=None,
        lr=DEFAULT_GRAD_NORM_LR,
        update_interval=DEFAULT_GRAD_NORM_UPDATE_INTERVAL,
        renormalize=DEFAULT_GRAD_NORM_RENORMALIZE,
        parameterization=DEFAULT_GRAD_NORM_PARAMETERIZATION,
        device=None,
    ):
        self.alpha = float(alpha)
        self.lr = float(lr)
        self.update_interval = int(update_interval)
        self.renormalize = str(renormalize)
        self.parameterization = str(parameterization)
        validate_grad_norm_settings(
            self.alpha,
            self.lr,
            self.update_interval,
            self.renormalize,
            self.parameterization,
        )
        extra_weights = dict(extra_weights or {})
        for reserved in (SUPERVISED_TERM, REGULARIZER_TERM):
            if reserved in extra_weights:
                raise ValueError(
                    f"GradNorm extra component name {reserved!r} collides with a built-in term"
                )
        self.extra_names = tuple(extra_weights)
        self.names = (SUPERVISED_TERM, REGULARIZER_TERM, *self.extra_names)
        initial = torch.tensor(
            [
                float(supervised_weight),
                float(regularizer_weight),
                *(float(weight) for weight in extra_weights.values()),
            ],
            dtype=torch.float32,
            device=device,
        ).clamp_min(MINIMUM_WEIGHT)
        # The weight sum the paper preserves, taken from the configured weights
        # rather than assumed to be the term count.
        self.weight_sum = float(initial.sum().item())
        self.supervised_weight_anchor = float(initial[0].item())
        self._raw = self._to_raw(initial).requires_grad_(True)
        self.optimizer = torch.optim.Adam([self._raw], lr=self.lr)
        self.initial_losses = None
        self.updates = 0
        self._last_diagnostics = {}
        self._warned_supervised_collapse = False
        self._warned_undefined_training_rate = False

    def _to_raw(self, weights):
        """Map weights into the space Adam updates them in."""

        if self.parameterization == "log":
            return weights.clamp_min(MINIMUM_WEIGHT).log()
        return weights

    @property
    def weights(self):
        """The live weights, in ``self.names`` order, differentiable.

        Under ``log`` this is ``exp(v)`` rather than the stored parameter, so the
        chain rule carries dG_i/dv_i = w_i * G_i into the Adam step for free and
        the rest of ``update`` is written against weights either way.
        """

        if self.parameterization == "log":
            return self._raw.exp()
        return self._raw

    @property
    def supervised_weight(self):
        return float(self.weights.detach()[0].item())

    @property
    def regularizer_weight(self):
        return float(self.weights.detach()[1].item())

    def extra_weight(self, name):
        return float(self.weights.detach()[self.names.index(name)].item())

    def extra_weights(self):
        """The learned weights for the regularizer's own objective terms."""

        return {name: self.extra_weight(name) for name in self.extra_names}

    def is_due(self, step):
        return step % self.update_interval == 0

    def update(
        self,
        component_norms,
        supervised_loss,
        regularization_loss,
        extra_losses=None,
    ):
        """Run one GradNorm step from the unweighted component gradient norms.

        ``component_norms`` carries ``||grad_W(L_i)||`` for every term, i.e. the
        norms measured with all weights set to one, with the regularizer's own
        terms in ``component_norms.extra``. Returns whether the weights changed,
        so the caller knows the objective moved.
        """

        extra_losses = dict(extra_losses or {})
        measured_norms = dict(component_norms.extra)
        missing = [
            name
            for name in self.extra_names
            if name not in measured_norms or name not in extra_losses
        ]
        if missing:
            # A term the objective declares but this batch does not carry would
            # leave Gbar averaging over a different set of terms every step.
            return False

        device = self._raw.device
        raw_norms = torch.tensor(
            [
                float(component_norms.supervised),
                float(component_norms.regularizer),
                *(float(measured_norms[name]) for name in self.extra_names),
            ],
            dtype=torch.float32,
            device=device,
        )
        losses = torch.tensor(
            [
                float(supervised_loss),
                float(regularization_loss),
                *(float(extra_losses[name]) for name in self.extra_names),
            ],
            dtype=torch.float32,
            device=device,
        )
        if not torch.isfinite(raw_norms).all() or not torch.isfinite(losses).all():
            return False
        if bool((raw_norms <= 0).any()):
            # Neither the gradient targets nor the training rates mean anything
            # when a term contributed no gradient to this batch.
            return False

        if self.initial_losses is None:
            if bool((losses <= 0).any()):
                # L_i(0) is a denominator; a non-positive initial loss would make
                # every later training rate meaningless, so wait for a usable batch.
                return False
            self.initial_losses = losses

        # ||grad(w * L)|| == w * ||grad(L)||, so the weighted norm stays a
        # differentiable function of the weights without a second backward.
        weighted_norms = self.weights * raw_norms
        mean_norm = weighted_norms.mean().detach()
        loss_ratio = losses / self.initial_losses
        if bool((loss_ratio < 0).any()) or float(loss_ratio.mean().item()) <= 0.0:
            # r_i is a ratio of loss ratios, so the paper's training rates are
            # only defined while every L_i keeps the sign of its L_i(0). A
            # negative ratio has no meaning to carry into ``** alpha``, and a
            # mean at zero makes every r_i diverge. Skip rather than substitute:
            # clamping a negative ratio to zero would silently report the term as
            # having trained infinitely fast and drive its weight to the floor.
            self._warn_on_undefined_training_rate()
            return False
        relative_rate = loss_ratio / loss_ratio.mean()
        target = (mean_norm * relative_rate ** self.alpha).detach()
        grad_norm_loss = torch.abs(weighted_norms - target).sum()

        self.optimizer.zero_grad(set_to_none=True)
        grad_norm_loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            self._apply_constraints()
        self.updates += 1
        self._last_diagnostics = {
            "train/grad_norm/loss": float(grad_norm_loss.detach().item()),
            "train/grad_norm/updates": float(self.updates),
        }
        weights = self.weights.detach()
        for index, name in enumerate(self.names):
            self._last_diagnostics[f"train/grad_norm/{name}_weight"] = float(
                weights[index].item()
            )
            if name == SUPERVISED_TERM:
                # r and the target are read against the supervised term, so
                # reporting them for it as well would only restate the mean.
                continue
            self._last_diagnostics[f"train/grad_norm/relative_rate_{name}"] = float(
                relative_rate[index].item()
            )
            self._last_diagnostics[f"train/grad_norm/target_{name}"] = float(
                target[index].item()
            )
        return True

    def _apply_constraints(self):
        """Clamp to positive weights and apply the renormalization mode."""

        weights = self.weights.detach().clamp_min(MINIMUM_WEIGHT)
        if self.renormalize == "sum":
            total = float(weights.sum().item())
            if total > 0:
                weights = weights * (self.weight_sum / total)
        elif self.renormalize == "supervised":
            supervised = float(weights[0].item())
            if supervised > 0:
                weights = weights * (self.supervised_weight_anchor / supervised)
        self._raw.copy_(self._to_raw(weights.clamp_min(MINIMUM_WEIGHT)))
        self._warn_on_supervised_collapse()

    def _warn_on_undefined_training_rate(self):
        """Say so once when the loss ratios stop defining a training rate."""

        if self._warned_undefined_training_rate:
            return
        self._warned_undefined_training_rate = True
        logger.warning(
            "GradNorm skipped an update: a loss changed sign relative to its "
            "initial value, or the mean loss ratio reached zero, so the relative "
            "inverse training rates r_i are undefined. Balancing resumes on the "
            "next batch where every term keeps the sign of its initial loss."
        )

    def _warn_on_supervised_collapse(self):
        """Say so once when balancing has effectively switched off the metric loss."""

        if self._warned_supervised_collapse or self.supervised_weight_anchor <= 0:
            return
        fraction = self.supervised_weight / self.supervised_weight_anchor
        if fraction >= SUPERVISED_COLLAPSE_FRACTION:
            return
        self._warned_supervised_collapse = True
        logger.warning(
            f"GradNorm has driven the supervised weight to {self.supervised_weight:g}, "
            f"{fraction:.2%} of its starting value. A small weight is not by itself a "
            "collapse: GradNorm equalizes w_i * ||grad L_i||, so a term whose gradient "
            "is orders above the others is supposed to end up with a weight orders "
            "below them, and its share of the gradient can still be at parity. Read "
            "train/gradient_contribution/regularizer_to_supervised_ratio to tell the "
            "two apart. If that ratio sits near 1, this is balance. If the weight is "
            f"instead bouncing off the {MINIMUM_WEIGHT:g} floor, the balanced weight is "
            "below grad_norm_lr and the linear parameterization cannot land on it -- "
            "grad_norm_parameterization='log' makes the step relative so it can."
        )

    def diagnostics(self):
        return dict(self._last_diagnostics)

    def describe(self):
        weights = ", ".join(
            f"{name}={self.weights.detach()[index].item():g}"
            for index, name in enumerate(self.names)
        )
        return (
            f"alpha={self.alpha:g}, lr={self.lr:g}, "
            f"update_interval={self.update_interval}, renormalize={self.renormalize!r}, "
            f"parameterization={self.parameterization!r}, initial weights=({weights})"
        )


def validate_grad_norm_settings(
    alpha,
    lr,
    update_interval,
    renormalize,
    parameterization=DEFAULT_GRAD_NORM_PARAMETERIZATION,
):
    """Reject settings that cannot produce a usable GradNorm update."""

    if not math.isfinite(alpha) or alpha < 0:
        raise ValueError("grad_norm_alpha must be finite and non-negative")
    if not math.isfinite(lr) or lr <= 0:
        raise ValueError("grad_norm_lr must be finite and positive")
    if update_interval < 1:
        raise ValueError("grad_norm_update_interval must be at least 1")
    if renormalize not in RENORMALIZE_MODES:
        raise ValueError(
            f"grad_norm_renormalize must be one of {list(RENORMALIZE_MODES)}: {renormalize!r}"
        )
    if parameterization not in PARAMETERIZATIONS:
        raise ValueError(
            "grad_norm_parameterization must be one of "
            f"{list(PARAMETERIZATIONS)}: {parameterization!r}"
        )


def log_grad_norm_configuration(regularizer):
    """Announce the active GradNorm settings once per run."""

    extra = regularizer.grad_norm_balanced_components()
    balanced = ", ".join(("supervised", regularizer.name, *extra))
    excluded = tuple(regularizer.grad_norm_exclude_components)
    logger.info(
        f"GradNorm loss balancing enabled for {regularizer.name}: "
        f"alpha={regularizer.grad_norm_alpha:g}, lr={regularizer.grad_norm_lr:g}, "
        f"update_interval={regularizer.grad_norm_update_interval}, "
        f"renormalize={regularizer.grad_norm_renormalize!r}, "
        f"parameterization={regularizer.grad_norm_parameterization!r}, "
        f"balancing {len(extra) + 2} terms ({balanced})"
        + (f", excluding {', '.join(excluded)}" if excluded else "")
    )
