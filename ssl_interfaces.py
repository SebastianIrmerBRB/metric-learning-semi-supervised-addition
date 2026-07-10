"""Extension interfaces for pseudo-label methods and regularizers."""

import math

import utils


class BaseSemiSupervisedMethod:
    """Interface implemented by each pseudo-label generation strategy."""

    name = None
    generates_pseudo_labels = True
    is_regularization_method = False

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


class BaseTrainingRegularizer:
    """Pluggable unlabeled regularization term combined with a supervised loss."""

    name = None
    provides_trainable_projection_without_feat_dim = False

    def __init__(self, regularizer_weight=1.0, supervised_weight=1.0):
        self.regularizer_weight = float(regularizer_weight)
        self.supervised_weight = float(supervised_weight)
        self.use_cache = False
        if not math.isfinite(self.regularizer_weight) or self.regularizer_weight < 0:
            raise ValueError("regularizer_weight must be finite and non-negative")
        if not math.isfinite(self.supervised_weight) or self.supervised_weight < 0:
            raise ValueError("supervised_weight must be finite and non-negative")
        if self.regularizer_weight == 0 and self.supervised_weight == 0:
            raise ValueError("regularizer_weight and supervised_weight cannot both be zero")

    def model_kwargs(self, args):
        return {}

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

    def compute_loss(self, student_model, state, batch, device, timings=None):
        raise NotImplementedError

    def after_optimizer_step(self, student_model, state):
        return None
