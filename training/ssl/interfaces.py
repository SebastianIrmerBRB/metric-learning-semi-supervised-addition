"""Extension interfaces for pseudo-label methods and regularizers."""

import math

from loguru import logger

import utils

from . import grad_norm
from . import gradient_surgery as gradient_surgery_module


# Batches whose supervised/regularizer gradient-norm ratio is measured in each
# calibration window. A handful is enough for a scale estimate and keeps the
# extra backward passes negligible against a full run.
DEFAULT_TARGET_RATIO_BATCHES = 20
DEFAULT_TARGET_RATIO_RECALIBRATION_INTERVAL_SAMPLES = 10_000

# Statistic used to summarize the raw supervised/regularizer gradient-norm
# ratios in a finite calibration buffer. The median remains the robust default;
# the arithmetic mean is exposed for ablations and for regimes whose ratios are
# demonstrably light-tailed. EMA is a separate, windowless estimator below.
TARGET_RATIO_AGGREGATIONS = frozenset({"mean", "median"})
DEFAULT_TARGET_RATIO_AGGREGATION = "median"
# How each gradient is summarized before the ratio is formed. "l2" is the
# Euclidean norm. "max_mean" is the statistic Wang et al. use for learning-rate
# annealing in physics-informed networks (doi:10.1137/20M1318043): the largest
# absolute coordinate of the supervised gradient over the mean absolute
# coordinate of the regularizer gradient.
DEFAULT_TARGET_RATIO_STATISTIC = "l2"
# A non-L2 statistic is anchored to the Euclidean one at the first probe so that
# regularizer_target_ratio keeps one meaning across statistics. Setting this to
# False restores the raw behaviour, where the statistic itself sets the operating
# point -- which is what Wang et al. do, since they have no target ratio.
DEFAULT_TARGET_RATIO_STATISTIC_ANCHOR = True
TARGET_RATIO_STATISTICS = ("l2", "max_mean")

# Measurements kept by the sliding-window schedule. Instead of accumulating
# probes until a window boundary clears them, the configured statistic runs
# over the last ``probe_memory`` of them, so the sample size and the lookback
# are both constant and there is no stretch at the end of a window with no
# measurement at all. Opt in per config; unset keeps the windowed schedule.
DEFAULT_TARGET_RATIO_PROBE_MEMORY = None

# Smoothing factor for the exponential-moving-average schedule, the third way
# to summarize probes. Like the sliding finite-buffer statistic it has no reset,
# but it keeps
# no buffer either: each probe folds into the estimate with weight ``alpha``.
# ``1 / alpha`` is a useful response-horizon heuristic, not a hard lookback:
# every older probe retains exponentially decaying, non-zero weight. Unlike a
# median it is mean-like, so one probe that lands far off moves it by ``alpha``
# times that distance -- the outlier protection a median gives for free is gone.
DEFAULT_TARGET_RATIO_EMA_ALPHA = None


class BaseSemiSupervisedMethod:
    """Interface implemented by each pseudo-label generation strategy."""

    name = None
    generates_pseudo_labels = True
    is_regularization_method = False
    # Whether update_mode='every_n_samples' may drive this method. It asks for
    # rebuilds inside an epoch, which is only sound for a method whose whole
    # refresh is one call the engine can repeat; see should_rebuild_on_epoch.
    supports_sample_scoped_refresh = False

    def validate_config(self, config, source=""):
        return None

    def generate_pseudo_labels(
        self,
        model,
        train_dataset,
        split,
        device,
        config,
        epoch=None,
        start_method="spawn",
        log_dir=None,
    ):
        raise NotImplementedError

    def prepare_pseudo_labels_for_training(self, pseudo_labels, config):
        """Convert accepted predictions into the weights consumed by training.

        Filtering and sampler-capacity rescue must see the method's original
        confidence scores. Methods that use a different training weight can
        transform those scores only after selection through this hook.
        """

        return pseudo_labels

    def pseudo_label_filter_threshold(self, config):
        """Return the normal confidence cutoff for shared pseudo-label filtering.

        Most methods use the shared cutoff directly. Methods whose prediction
        procedure has an earlier confidence boundary can override this hook so
        they may expose lower-confidence *rescue candidates* without changing
        which predictions are accepted during normal filtering.
        """

        return float(config.confidence_threshold)


class BaseTrainingRegularizer:
    """Pluggable unlabeled regularization term combined with a supervised loss."""

    name = None
    # Opt in explicitly: regularizers needing stochastic/paired image views
    # cannot safely train from one deterministic frozen-backbone feature row.
    supports_frozen_feature_precompute = False
    provides_trainable_projection_without_feat_dim = False
    uses_joint_forward = False
    # Most regularizers add their scalar to the supervised loss and share one
    # backward/optimizer step. A regularizer that mirrors an algorithm with
    # ordered updates can instead yield fresh losses one at a time through
    # ``separate_optimizer_step_losses``. The trainer steps the model optimizer
    # after every yield, so later losses are evaluated at the updated weights.
    uses_separate_optimizer_steps = False
    requires_labeled_indices = False
    requires_supervised_objective = False
    # See BaseSemiSupervisedMethod.supports_sample_scoped_refresh. A regularizer
    # that opts in must honor request_refresh in its own make_loader.
    supports_sample_scoped_refresh = False

    # Set from method_params.regularizer_target_ratio. When present, the trainer
    # replaces regularizer_weight with a value calibrated on sampled regularized
    # batches so that ||grad(w * L_reg)|| / ||grad(w_sup * L_sup)|| meets it. The
    # engine updates it on a sample-scoped probe schedule and summarizes the
    # measured ratios with the configured aggregation. This makes the searched
    # knob scale-free: the same range then means the same thing for every loss,
    # whose gradient scale otherwise varies by orders of magnitude.
    regularizer_target_ratio = None
    regularizer_target_ratio_batches = DEFAULT_TARGET_RATIO_BATCHES
    regularizer_target_ratio_recalibration_interval_samples = (
        DEFAULT_TARGET_RATIO_RECALIBRATION_INTERVAL_SAMPLES
    )
    regularizer_target_ratio_update_interval_samples = None
    regularizer_target_ratio_probe_memory = DEFAULT_TARGET_RATIO_PROBE_MEMORY
    regularizer_target_ratio_ema_alpha = DEFAULT_TARGET_RATIO_EMA_ALPHA
    regularizer_target_ratio_aggregation = DEFAULT_TARGET_RATIO_AGGREGATION
    regularizer_target_ratio_statistic = DEFAULT_TARGET_RATIO_STATISTIC
    regularizer_target_ratio_statistic_anchor = DEFAULT_TARGET_RATIO_STATISTIC_ANCHOR
    regularizer_target_ratio_diagnostics = False

    # Set from method_params.gradient_surgery. When present, the trainer projects
    # the regularizer gradient off the supervised one on every step where the two
    # conflict, before the weighted sum is formed. It is orthogonal to the weight
    # mechanisms above: those set the regularizer's magnitude, this sets its
    # direction, and the calibration probe measures the projected gradient so the
    # configured ratio is the one the objective realizes.
    gradient_surgery = gradient_surgery_module.DEFAULT_GRADIENT_SURGERY

    # Set from method_params.grad_norm_alpha. When present, both weights are
    # learned during training by GradNorm (arXiv:1711.02257) instead of being
    # fixed or periodically calibrated, and it takes precedence over both.
    grad_norm_alpha = None
    grad_norm_lr = grad_norm.DEFAULT_GRAD_NORM_LR
    grad_norm_update_interval = grad_norm.DEFAULT_GRAD_NORM_UPDATE_INTERVAL
    grad_norm_renormalize = grad_norm.DEFAULT_GRAD_NORM_RENORMALIZE
    grad_norm_parameterization = grad_norm.DEFAULT_GRAD_NORM_PARAMETERIZATION
    # Set from method_params.grad_norm_exclude_components. Names listed here stay
    # out of the balancing and keep the weight the config or the calibration gave
    # them; see set_grad_norm.
    grad_norm_exclude_components = ()

    # Every name this regularizer can report in extra_loss_components, whether or
    # not the current configuration produces the term. It exists so an exclusion
    # list can tell a typo from a term that is merely switched off.
    extra_component_names = ()

    def __init__(self, regularizer_weight=1.0, supervised_weight=1.0):
        self.regularizer_weight = float(regularizer_weight)
        self.supervised_weight = float(supervised_weight)
        # Global graph/sampling refreshes may use a dedicated accelerator even
        # when optimizer-driven training remains on the CPU.
        self.ssl_device = None
        self.use_cache = False
        self.collect_batch_diagnostics = False
        # One-shot flag set by the engine under update_mode='every_n_samples',
        # where the refresh schedule is counted in consumed samples and so is
        # not expressible through the epoch number make_loader receives.
        self._refresh_requested = False
        if not math.isfinite(self.regularizer_weight) or self.regularizer_weight < 0:
            raise ValueError("regularizer_weight must be finite and non-negative")
        if not math.isfinite(self.supervised_weight) or self.supervised_weight < 0:
            raise ValueError("supervised_weight must be finite and non-negative")
        if self.regularizer_weight == 0 and self.supervised_weight == 0:
            raise ValueError("regularizer_weight and supervised_weight cannot both be zero")

    def request_refresh(self):
        """Ask the next ``make_loader`` call to rebuild, whatever the epoch says."""

        if not self.supports_sample_scoped_refresh:
            raise NotImplementedError(
                f"{self.name} does not support sample-scoped graph refreshes"
            )
        self._refresh_requested = True

    def consume_refresh_request(self):
        """Return and clear a pending :meth:`request_refresh`."""

        requested = self._refresh_requested
        self._refresh_requested = False
        return requested

    def set_target_ratio(
        self,
        target_ratio,
        calibration_batches=None,
        recalibration_interval_samples=None,
        update_interval_samples=None,
        probe_memory=None,
        ema_alpha=None,
        aggregation=None,
        statistic=None,
        statistic_anchor=None,
        log_diagnostics=False,
    ):
        """Ask the trainer to calibrate ``regularizer_weight`` to a gradient ratio.

        ``None`` keeps the configured weight. Any other value overrides it, so the
        weight the config carries becomes only the pre-calibration probe.

        ``recalibration_interval_samples`` is the size of the calibration window.
        ``update_interval_samples`` directly controls the sample cadence of
        gradient probes; zero means every regularized optimizer step. When it is
        omitted, ``calibration_batches`` probes are distributed across the window
        for backward compatibility. ``aggregation`` selects ``median`` (the robust
        default) or the arithmetic ``mean`` for finite windows and sliding
        buffers. The first eligible batch is always measured, so a probe weight
        that is orders away from the calibrated one never trains for more than a
        single step.

        ``probe_memory`` selects the sliding schedule instead: probes keep
        arriving at ``update_interval_samples`` forever and the configured
        statistic runs over the last ``probe_memory`` of them, with no window to
        reset and no frozen stretch once a window's probes are spent. It replaces
        the window, so it cannot be combined with
        ``recalibration_interval_samples`` or ``calibration_batches``.

        ``ema_alpha`` selects the exponential moving average instead: probes
        arrive on the same cadence but fold into a running estimate rather than
        a buffer. The reciprocal ``1 / alpha`` is its conventional response-
        horizon heuristic; unlike a sliding buffer, the EMA has no finite cutoff.
        It is an alternative to ``probe_memory`` and to the window, and it trades
        the median's robustness for a smooth response -- a single wild probe
        moves the weight by ``alpha`` times its distance.
        """

        if target_ratio is None:
            self.regularizer_target_ratio = None
            return
        target_ratio = float(target_ratio)
        if not math.isfinite(target_ratio) or target_ratio <= 0:
            raise ValueError("regularizer_target_ratio must be finite and positive")
        if calibration_batches is None:
            calibration_batches = DEFAULT_TARGET_RATIO_BATCHES
        calibration_batches = int(calibration_batches)
        if calibration_batches < 1:
            raise ValueError("regularizer_target_ratio_batches must be at least 1")
        if recalibration_interval_samples is None:
            recalibration_interval_samples = (
                DEFAULT_TARGET_RATIO_RECALIBRATION_INTERVAL_SAMPLES
            )
        recalibration_interval_samples = int(recalibration_interval_samples)
        if recalibration_interval_samples < 1:
            raise ValueError(
                "regularizer_target_ratio_recalibration_interval_samples "
                "must be at least 1"
            )
        if update_interval_samples is not None:
            update_interval_samples = int(update_interval_samples)
            if update_interval_samples < 0:
                raise ValueError(
                    "regularizer_target_ratio_update_interval_samples "
                    "must be non-negative"
                )
        if statistic is None:
            statistic = DEFAULT_TARGET_RATIO_STATISTIC
        statistic = str(statistic).strip().lower()
        if statistic not in TARGET_RATIO_STATISTICS:
            raise ValueError(
                "regularizer_target_ratio_statistic must be one of "
                f"{sorted(TARGET_RATIO_STATISTICS)}, got {statistic!r}"
            )
        if aggregation is None:
            aggregation = DEFAULT_TARGET_RATIO_AGGREGATION
        aggregation = str(aggregation).strip().lower()
        if aggregation not in TARGET_RATIO_AGGREGATIONS:
            raise ValueError(
                "regularizer_target_ratio_aggregation must be one of "
                f"{sorted(TARGET_RATIO_AGGREGATIONS)}, got {aggregation!r}"
            )
        if ema_alpha is not None:
            ema_alpha = float(ema_alpha)
            if not math.isfinite(ema_alpha) or not 0.0 < ema_alpha <= 1.0:
                raise ValueError(
                    "regularizer_target_ratio_ema_alpha must be in (0, 1]"
                )
            if probe_memory is not None:
                raise ValueError(
                    "regularizer_target_ratio_ema_alpha and "
                    "regularizer_target_ratio_probe_memory are alternative ways "
                    "to summarize probes without a window; configure only one"
                )
            if aggregation != DEFAULT_TARGET_RATIO_AGGREGATION:
                raise ValueError(
                    "regularizer_target_ratio_ema_alpha is itself an aggregation "
                    "rule and cannot be combined with "
                    "regularizer_target_ratio_aggregation"
                )
            if update_interval_samples is None:
                raise ValueError(
                    "regularizer_target_ratio_ema_alpha needs "
                    "regularizer_target_ratio_update_interval_samples to set the "
                    "probe cadence; the moving average has no window to spread "
                    "probes over"
                )
        if probe_memory is not None:
            probe_memory = int(probe_memory)
            if probe_memory < 1:
                raise ValueError(
                    "regularizer_target_ratio_probe_memory must be at least 1"
                )
            if update_interval_samples is None:
                raise ValueError(
                    "regularizer_target_ratio_probe_memory needs "
                    "regularizer_target_ratio_update_interval_samples to set the "
                    "probe cadence; the sliding schedule has no window to spread "
                    "probes over"
                )
        self.regularizer_target_ratio = target_ratio
        self.regularizer_target_ratio_batches = calibration_batches
        self.regularizer_target_ratio_recalibration_interval_samples = (
            recalibration_interval_samples
        )
        self.regularizer_target_ratio_update_interval_samples = (
            update_interval_samples
        )
        self.regularizer_target_ratio_probe_memory = probe_memory
        self.regularizer_target_ratio_ema_alpha = ema_alpha
        self.regularizer_target_ratio_aggregation = aggregation
        self.regularizer_target_ratio_statistic = statistic
        self.regularizer_target_ratio_statistic_anchor = (
            DEFAULT_TARGET_RATIO_STATISTIC_ANCHOR
            if statistic_anchor is None
            else bool(statistic_anchor)
        )
        self.regularizer_target_ratio_diagnostics = bool(log_diagnostics)
        # The probe weight only has to be non-zero: it keeps the phase regularized
        # until the first batch replaces it with the calibrated value.
        self.regularizer_weight = 1.0

    def extra_target_ratios(self):
        """Target gradient ratios for terms ``combine_losses`` adds itself.

        Maps a name to a target ratio against the supervised gradient, for
        objective terms that ride alongside ``regularizer_weight * L_reg``
        rather than inside it. A term with its own parameters and its own scale
        has no reason to inherit the regularizer's weight, and folding it in
        would also make ``regularizer_weight`` calibrate against a gradient that
        is partly not the regularizer's.

        The names must match ``extra_loss_components`` so one term is reported
        and calibrated under a single label.
        """

        return {}

    def calibratable_component_loss(self, name):
        """Return this step's unweighted loss for ``name``, or ``None``.

        ``None`` means the term is not present in this batch, which is not the
        same as it being zero: the trainer skips the batch instead of recording
        a ratio it cannot measure.
        """

        return None

    def apply_calibrated_component_weight(self, name, weight):
        """Adopt the calibrated weight the trainer measured for ``name``."""

        raise NotImplementedError(
            f"{self.name} declared a target ratio for {name!r} but cannot store its weight"
        )

    def steady_state_active(self):
        """Whether the objective this regularizer exists to optimize now reaches ``W``.

        A regularizer whose objective starts up in stages -- terms that switch on
        only after a warmup, a threshold, or enough accumulated statistics -- is
        not yet itself while those stages run, even though it is already
        contributing a gradient. The engine's own guard only skips batches with a
        *zero* gradient component, which a partially active objective does not
        produce, so a regularizer that starts up in stages has to say so here.

        Two consumers ask this same question. Each target-ratio calibration
        window must avoid a start-up phase. ``restart_selection_after_warmup``
        rebaselines model selection on the SSL phase, so it must not rebaseline
        on an epoch the steady-state terms never touched -- otherwise the stage
        boundary becomes the new warm-up boundary and, when its length is a
        searched hyperparameter, the new confound.

        Callers must treat a ``True`` as latching. The condition can be data
        dependent and need not be monotone: SLADE's Eq 7 mining gate switches
        back off when the two Gaussians stop separating, and a selection
        baseline that tracked the flag would restart repeatedly mid-run.
        """

        return True

    def ready_for_target_ratio_calibration(self):
        """Whether this batch's regularizer gradient is worth calibrating on.

        A regularizer whose objective starts up in stages -- terms that switch
        on only after a warmup, a threshold, or enough accumulated statistics --
        must not be measured before its steady-state terms are present. That is
        exactly ``steady_state_active``, which is where a regularizer declares
        its start-up stages once for both callers. The engine asks again in each
        periodic calibration window.
        """

        return self.steady_state_active()

    def grad_norm_extra_components(self):
        """Objective terms beyond ``regularizer_weight * L_reg`` GradNorm should balance.

        Maps a name from ``extra_loss_components`` to its starting weight. A
        regularizer that adds terms in ``combine_losses`` reports them here so
        GradNorm learns their weights too instead of leaving them fixed beside
        the two it does learn.

        This is the declaration: what the regularizer offers. What GradNorm ends
        up balancing is ``grad_norm_balanced_components``, which drops whatever
        the configuration excluded.
        """

        return {}

    def grad_norm_balanced_components(self):
        """The extra terms GradNorm actually balances, after exclusions.

        A term GradNorm can measure is not always a term it should weight. Its
        signal is the relative inverse training rate ``L(t) / L(0)``, which reads
        a falling loss as "training fast, downweight" -- right for a task, and
        backwards for a structural prior whose whole purpose is to drive its own
        loss down. Excluding such a term leaves it to a fixed weight or to
        ``extra_target_ratios``, and leaves the rest of the objective balanced.
        """

        excluded = frozenset(self.grad_norm_exclude_components)
        return {
            name: weight
            for name, weight in self.grad_norm_extra_components().items()
            if name not in excluded
        }

    def private_model_module_names(self):
        """Modules ``configure_model`` attaches to the student model.

        Their parameters belong to this regularizer alone, so no other objective
        term reaches them. Every comparison of one term's gradient against
        another's -- GradNorm, target-ratio calibration, and the gradient
        contribution diagnostics -- is taken over *shared* weights, so the
        trainer excludes these from all three.
        """

        return ()

    def clear_extra_target_ratios(self, keep=()):
        """Drop per-component target ratios when GradNorm takes over their weights.

        ``keep`` names the components GradNorm is not balancing, whose ratios are
        therefore not superseded and must survive.
        """

        return None

    def set_gradient_surgery(self, mode):
        """Project the regularizer gradient off the supervised one when they conflict.

        ``None`` keeps the plain combined backward pass.
        ``"supervised_priority"`` leaves the supervised gradient untouched and
        removes only the regularizer's opposing component -- the asymmetric
        reading of PCGrad used by default here, appropriate when one term is an
        auxiliary regularizer rather than a co-equal task. ``"pcgrad"`` is
        Algorithm 1 of Yu et al. unchanged, which projects both.

        This is independent of how the weights are set. It can be combined with
        a target ratio or with GradNorm, and changes only the direction that the
        weighted regularizer gradient points in.
        """

        if mode is None:
            self.gradient_surgery = None
            return
        mode = str(mode).strip().lower()
        if mode not in gradient_surgery_module.GRADIENT_SURGERY_MODES:
            raise ValueError(
                "gradient_surgery must be one of "
                f"{list(gradient_surgery_module.GRADIENT_SURGERY_MODES)}, got {mode!r}"
            )
        self.gradient_surgery = mode

    def set_grad_norm(
        self,
        alpha,
        lr=None,
        update_interval=None,
        renormalize=None,
        parameterization=None,
        exclude_components=None,
    ):
        """Hand both loss weights to GradNorm for the duration of training.

        ``None`` leaves the weighting to ``regularizer_target_ratio`` or the
        configured weight. Any other value takes precedence over both: the
        configured weights become GradNorm's starting point, and the weight sum
        they define is what the paper's renormalization preserves.

        ``exclude_components`` names extra terms to leave out of the balancing.
        Precedence is per weight, not per run: an excluded term is the one weight
        GradNorm does not take over, so its ``extra_target_ratios`` entry stands
        and is calibrated as it would be without GradNorm.
        """

        if alpha is None:
            self.grad_norm_alpha = None
            self.grad_norm_exclude_components = ()
            return
        alpha = float(alpha)
        lr = grad_norm.DEFAULT_GRAD_NORM_LR if lr is None else float(lr)
        update_interval = (
            grad_norm.DEFAULT_GRAD_NORM_UPDATE_INTERVAL
            if update_interval is None
            else int(update_interval)
        )
        renormalize = (
            grad_norm.DEFAULT_GRAD_NORM_RENORMALIZE
            if renormalize is None
            else str(renormalize)
        )
        parameterization = (
            grad_norm.DEFAULT_GRAD_NORM_PARAMETERIZATION
            if parameterization is None
            else str(parameterization)
        )
        grad_norm.validate_grad_norm_settings(
            alpha,
            lr,
            update_interval,
            renormalize,
            parameterization,
        )
        self.grad_norm_exclude_components = self._resolve_grad_norm_exclusions(
            exclude_components
        )
        self.grad_norm_alpha = alpha
        self.grad_norm_lr = lr
        self.grad_norm_update_interval = update_interval
        self.grad_norm_renormalize = renormalize
        self.grad_norm_parameterization = parameterization
        # GradNorm owns every weight from the first regularized batch, so a
        # calibrated starting point would only be overwritten -- and a component
        # still calibrating would be fighting GradNorm over the same number. An
        # excluded component is exactly the case where neither is true.
        self.regularizer_target_ratio = None
        self.clear_extra_target_ratios(keep=self.grad_norm_exclude_components)
        if self.regularizer_weight <= 0:
            # A zero weight has no gradient for GradNorm to rescale.
            self.regularizer_weight = 1.0
        for name, weight in self.grad_norm_balanced_components().items():
            if weight <= 0:
                self.apply_calibrated_component_weight(name, 1.0)

    def _resolve_grad_norm_exclusions(self, exclude_components):
        """Normalize and check the configured exclusion list.

        An unknown name is rejected rather than ignored: it is a typo whose only
        symptom would be a term still being balanced. A name the regularizer
        knows but this configuration does not produce is only warned about, so a
        study that switches such a term off in some trials keeps one config.
        """

        if exclude_components is None:
            return ()
        if isinstance(exclude_components, str):
            raise ValueError(
                f"{self.name} grad_norm_exclude_components must be a list of "
                f"component names, not the string {exclude_components!r}"
            )
        names = tuple(dict.fromkeys(str(name) for name in exclude_components))
        known = frozenset(self.extra_component_names)
        unknown = sorted(name for name in names if name not in known)
        if unknown:
            raise ValueError(
                f"{self.name} grad_norm_exclude_components {unknown} are not objective "
                f"terms of this regularizer. Available: {sorted(known)}"
            )
        declared = self.grad_norm_extra_components()
        inactive = sorted(name for name in names if name not in declared)
        if inactive:
            logger.warning(
                f"{self.name} grad_norm_exclude_components {inactive} are not balanced "
                "by GradNorm in this configuration, so excluding them changes nothing"
            )
        return names

    def set_ssl_device(self, device):
        """Select the out-of-batch SSL device, defaulting to the training device."""

        self.ssl_device = None if device is None else utils.normalize_device_name(device)

    def get_ssl_device(self, training_device):
        return training_device if self.ssl_device is None else self.ssl_device

    def model_kwargs(self, args):
        return {}

    def configure_graph_batching(self, config):
        """Apply the shared graph-construction mode, if this regularizer supports it."""

        if config.graph_batch_mode != "global":
            raise ValueError(
                f"{self.name} does not support graph_batch_mode={config.graph_batch_mode!r}"
            )

    def configure_model(self, student_model, train_dataset, split, train_labels_mapper, device):
        """Attach trainable method-specific modules before optimizer creation."""

        return None

    def set_steps_per_epoch(self, steps_per_epoch):
        """Resolve any epoch-denominated schedule now that the epoch length is known.

        The sampler epoch is the fold training pool over ``batch_size``, so it is
        settled per fold rather than per run. Methods with no epoch-denominated
        setting ignore this.
        """

        return None

    def make_supervised_source_dataset(self, train_dataset):
        """Optionally replace the labeled stream's augmentation source."""

        return train_dataset

    def validate_run_args(self, args):
        return None

    def build_dataset(self, train_dataset, split, use_cache=False):
        raise NotImplementedError

    def make_regularizer_source_dataset(self, train_dataset, use_cache=False):
        self.use_cache = bool(use_cache)
        if self.use_cache and not utils.dataset_has_precomputed_backbone_features(train_dataset):
            return utils.make_feature_transform_dataset(
                train_dataset,
                require_feature_transform=True,
            )
        return train_dataset

    def make_loader(
        self,
        model,
        train_dataset,
        supervised_loader,
        device,
        config,
        batch_size,
        seed,
        num_workers,
        start_method,
        epoch,
        log_dir=None,
    ):
        raise NotImplementedError

    def initialize_state(self, student_model, device):
        return None

    def combine_losses(self, supervised_loss, regularization_loss):
        return (
            self.supervised_weight * supervised_loss
            + self.regularizer_weight * regularization_loss
        )

    def extra_loss_components(self):
        """Objective terms ``combine_losses`` adds beyond the two weighted losses.

        Maps a diagnostic name to ``(loss, weight)`` for the most recent batch.
        Gradient-contribution logging scores each one separately; adding them to
        the training loss remains ``combine_losses``' job.
        """

        return {}

    def compute_loss(self, student_model, state, batch, device, timings=None):
        raise NotImplementedError

    def separate_optimizer_step_losses(
        self,
        student_model,
        state,
        batch,
        device,
        timings=None,
    ):
        """Yield ``(name, raw_loss)`` terms that each receive an optimizer step.

        This hook is consumed only when ``uses_separate_optimizer_steps`` is
        true. Every yielded loss is multiplied by ``regularizer_weight`` by the
        trainer. Implementations may be generators: execution resumes only
        after the preceding loss has been backpropagated and stepped, which is
        what lets the next forward observe the newly updated model.
        """

        raise NotImplementedError

    def after_optimizer_step(self, student_model, state):
        return None

    def consume_pending_optimizer_state_resets(self):
        """Return parameters this regularizer re-initialized since the last call.

        A regularizer that re-initializes one of its own modules mid-run leaves
        the optimizer holding moments fitted to the parameter's *previous*
        values, so the "fresh" module would take its first steps under a stale
        second moment. Returning the parameters here lets the trainer drop those
        optimizer state entries; the queue is cleared by the call.
        """

        return ()

    def set_batch_diagnostics_enabled(self, enabled):
        """Enable optional per-batch diagnostic computation for this run."""

        self.collect_batch_diagnostics = bool(enabled)

    def batch_diagnostics(self):
        """Return detached scalar diagnostics for the most recent batch."""

        return {}
