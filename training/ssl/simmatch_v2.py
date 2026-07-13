"""Single-GPU SimMatchV2 regularization for deep metric learning.

The graph-consistency objective and memory-bank flow follow the official
SimMatchV2 implementation while the student retrieval embedding remains the
embedding consumed by the project's normal PML loss/miner and evaluation code.
"""

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfm
import torchvision.transforms.v2 as v2
from loguru import logger
from torch.utils.data import DataLoader, Dataset

import utils
from .data import CombinedTrainingLoader
from .interfaces import BaseTrainingRegularizer


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def make_simmatch_v2_weak_transform(image_size=224):
    """Return the weak view used by the official SimMatchV2 training recipe."""

    return tfm.Compose(
        [
            v2.RGB(),
            tfm.RandomResizedCrop(int(image_size), scale=(0.2, 1.0)),
            tfm.RandomHorizontalFlip(),
            tfm.ToTensor(),
            tfm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def make_simmatch_v2_strong_transform(image_size=224):
    """Return the MoCo-style strong view used by upstream SimMatchV2."""

    return tfm.Compose(
        [
            v2.RGB(),
            tfm.RandomResizedCrop(int(image_size), scale=(0.2, 1.0)),
            tfm.RandomHorizontalFlip(),
            tfm.RandomApply([tfm.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            tfm.RandomGrayscale(p=0.2),
            tfm.RandomApply(
                [tfm.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))],
                p=0.5,
            ),
            tfm.ToTensor(),
            tfm.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _copy_with_transform(dataset, transform):
    transformed = copy.deepcopy(dataset)
    utils.set_nested_transform(transformed, transform)
    return transformed


class SimMatchV2UnlabeledDataset(Dataset):
    """Expose strong/weak views while keeping unlabeled targets inaccessible.

    The strong view is returned first because the shared training loop includes
    the first regularizer tensor in the student's joint labeled/strong forward.
    """

    def __init__(self, dataset, positions, weak_transform=None, strong_transform=None):
        self.positions = np.asarray(positions, dtype=np.int64)
        if len(self.positions) == 0:
            raise ValueError("simmatch_v2 requires at least one unlabeled sample")
        weak_transform = weak_transform or make_simmatch_v2_weak_transform()
        strong_transform = strong_transform or make_simmatch_v2_strong_transform()
        self.weak_dataset = _copy_with_transform(dataset, weak_transform)
        self.strong_dataset = _copy_with_transform(dataset, strong_transform)

    def __len__(self):
        return len(self.positions)

    def __getitem__(self, index):
        position = int(self.positions[index])
        strong = self.strong_dataset[position][0]
        weak = self.weak_dataset[position][0]
        return strong, weak, position


class SimMatchV2Heads(nn.Module):
    """Training-only class and graph heads placed over retrieval embedding ``f``."""

    GRAPH_SPACES = {"embedding", "projection"}

    def __init__(
        self,
        feat_dim,
        num_classes,
        graph_space="projection",
        projection_dim=128,
        projection_hidden_dim=None,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_classes = int(num_classes)
        self.graph_space = str(graph_space)
        if self.feat_dim <= 0:
            raise ValueError("simmatch_v2 feat_dim must be positive")
        if self.num_classes <= 1:
            raise ValueError("simmatch_v2 requires at least two training classes")
        if self.graph_space not in self.GRAPH_SPACES:
            raise ValueError(
                f"simmatch_v2 graph_space must be one of {sorted(self.GRAPH_SPACES)}"
            )

        self.classifier = nn.Linear(self.feat_dim, self.num_classes)
        if self.graph_space == "embedding":
            self.graph_dim = self.feat_dim
            self.projector = nn.Identity()
        else:
            projection_dim = int(projection_dim)
            hidden_dim = self.feat_dim if projection_hidden_dim is None else int(projection_hidden_dim)
            if projection_dim <= 0:
                raise ValueError("simmatch_v2 projection_dim must be positive")
            if hidden_dim <= 0:
                raise ValueError("simmatch_v2 projection_hidden_dim must be positive")
            self.graph_dim = projection_dim
            self.projector = nn.Sequential(
                nn.Linear(self.feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, projection_dim),
            )

    def forward(self, retrieval_embeddings):
        logits = self.classifier(retrieval_embeddings)
        graph_embeddings = F.normalize(self.projector(retrieval_embeddings), p=2.0, dim=1)
        return logits, graph_embeddings


class SimMatchV2State(nn.Module):
    """EMA teacher, labeled memory, unlabeled FIFO memory, and DA state."""

    def __init__(
        self,
        student_model,
        labeled_bank_labels,
        num_classes,
        graph_dim,
        queue_size,
        device,
    ):
        super().__init__()
        labeled_bank_labels = torch.as_tensor(labeled_bank_labels, dtype=torch.long)
        if labeled_bank_labels.ndim != 1 or len(labeled_bank_labels) == 0:
            raise ValueError("simmatch_v2 labeled bank labels must be a nonempty vector")
        if torch.any((labeled_bank_labels < 0) | (labeled_bank_labels >= int(num_classes))):
            raise ValueError("simmatch_v2 labeled bank labels must be dense training-class IDs")
        if int(graph_dim) <= 0:
            raise ValueError("simmatch_v2 graph_dim must be positive")
        if int(queue_size) <= 0:
            raise ValueError("simmatch_v2 queue_size must be positive")

        self.ema = copy.deepcopy(student_model)
        self.ema.requires_grad_(False)
        self.ema.eval()

        labeled_bank = F.normalize(
            torch.randn(len(labeled_bank_labels), int(graph_dim), dtype=torch.float32),
            p=2.0,
            dim=1,
        )
        unlabeled_bank = F.normalize(
            torch.randn(int(queue_size), int(graph_dim), dtype=torch.float32),
            p=2.0,
            dim=1,
        )
        self.register_buffer("l_bank", labeled_bank)
        self.register_buffer(
            "l_labels",
            F.one_hot(labeled_bank_labels, num_classes=int(num_classes)).to(torch.float32),
        )
        self.register_buffer("u_bank", unlabeled_bank)
        # Upstream initializes an empty class memory. A uniform distribution is
        # the neutral, finite equivalent until real teacher predictions arrive.
        self.register_buffer(
            "u_labels",
            torch.full(
                (int(queue_size), int(num_classes)),
                1.0 / float(num_classes),
                dtype=torch.float32,
            ),
        )
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
        self.register_buffer(
            "da",
            torch.full((1, int(num_classes)), 1.0 / float(num_classes), dtype=torch.float32),
        )
        self.to(device)

    @torch.no_grad()
    def distribution_alignment(self, probabilities, momentum, eps=1e-7):
        probabilities = probabilities.float()
        aligned = probabilities / self.da.clamp_min(float(eps))
        aligned = aligned / aligned.sum(dim=1, keepdim=True).clamp_min(float(eps))
        batch_probability = probabilities.mean(dim=0, keepdim=True)
        self.da.mul_(float(momentum)).add_(batch_probability, alpha=1.0 - float(momentum))
        return aligned

    @torch.no_grad()
    def propagate(self, graph_embeddings, initial_probabilities, top_n, temperature, alpha, eps=1e-7):
        """Propagate one weak-view label through its top-n labeled anchors."""

        graph_embeddings = graph_embeddings.float()
        initial_probabilities = initial_probabilities.float()
        top_n = int(top_n)
        if top_n <= 0 or top_n > len(self.l_bank):
            raise ValueError(
                f"simmatch_v2 top_n must be in [1, {len(self.l_bank)}], got {top_n}"
            )

        neighbor_indices = torch.topk(
            graph_embeddings @ self.l_bank.T,
            k=top_n,
            largest=True,
            sorted=False,
            dim=1,
        ).indices
        neighbor_features = self.l_bank[neighbor_indices]
        neighbor_labels = self.l_labels[neighbor_indices]
        features = torch.cat([graph_embeddings.unsqueeze(1), neighbor_features], dim=1)
        labels = torch.cat([initial_probabilities.unsqueeze(1), neighbor_labels], dim=1)

        node_count = top_n + 1
        identity = torch.eye(node_count, device=features.device, dtype=features.dtype)
        transition_logits = features @ features.transpose(1, 2) / float(temperature)
        transition_logits = transition_logits.masked_fill(
            identity.bool().unsqueeze(0),
            -torch.finfo(transition_logits.dtype).max,
        )
        transition = F.softmax(transition_logits, dim=-1)
        system = identity.unsqueeze(0) - float(alpha) * transition
        propagated = (1.0 - float(alpha)) * torch.linalg.solve(system, labels)
        pseudo_labels = propagated[:, 0].clamp_min(0.0)
        return pseudo_labels / pseudo_labels.sum(dim=1, keepdim=True).clamp_min(float(eps))

    @torch.no_grad()
    def update_labeled_bank(self, graph_embeddings, bank_indices):
        bank_indices = torch.as_tensor(bank_indices, device=self.l_bank.device, dtype=torch.long)
        if bank_indices.ndim != 1 or len(bank_indices) != len(graph_embeddings):
            raise ValueError("simmatch_v2 labeled bank indices must align with labeled embeddings")
        if torch.any((bank_indices < 0) | (bank_indices >= len(self.l_bank))):
            raise IndexError("simmatch_v2 labeled bank index is out of range")
        self.l_bank[bank_indices] = graph_embeddings.detach().to(
            device=self.l_bank.device,
            dtype=self.l_bank.dtype,
        )

    @torch.no_grad()
    def update_unlabeled_bank(self, graph_embeddings, probabilities):
        graph_embeddings = graph_embeddings.detach().to(
            device=self.u_bank.device,
            dtype=self.u_bank.dtype,
        )
        probabilities = probabilities.detach().to(
            device=self.u_labels.device,
            dtype=self.u_labels.dtype,
        )
        if len(graph_embeddings) != len(probabilities):
            raise ValueError("simmatch_v2 queue features and labels must have equal length")
        if len(graph_embeddings) == 0:
            return
        if len(graph_embeddings) > len(self.u_bank):
            graph_embeddings = graph_embeddings[-len(self.u_bank) :]
            probabilities = probabilities[-len(self.u_bank) :]

        pointer = int(self.ptr.item())
        first_count = min(len(graph_embeddings), len(self.u_bank) - pointer)
        self.u_bank[pointer : pointer + first_count] = graph_embeddings[:first_count]
        self.u_labels[pointer : pointer + first_count] = probabilities[:first_count]
        remaining = len(graph_embeddings) - first_count
        if remaining:
            self.u_bank[:remaining] = graph_embeddings[first_count:]
            self.u_labels[:remaining] = probabilities[first_count:]
        self.ptr[0] = (pointer + len(graph_embeddings)) % len(self.u_bank)


def _validate_finite_nonnegative(name, value):
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"simmatch_v2 {name} must be finite and non-negative")
    return value


class SimMatchV2Regularizer(BaseTrainingRegularizer):
    """Compose SimMatchV2 graph consistency with any configured PML objective."""

    name = "simmatch_v2"
    uses_joint_forward = True
    requires_labeled_indices = True
    requires_supervised_objective = True

    def __init__(
        self,
        regularizer_weight=1.0,
        supervised_weight=1.0,
        graph_space="projection",
        projection_dim=128,
        projection_hidden_dim=None,
        queue_size=4096,
        temperature=0.1,
        propagation_alpha=0.1,
        top_n=128,
        ema_momentum=0.999,
        distribution_alignment=True,
        da_momentum=0.9,
        unlabeled_ratio=1.0,
        unlabeled_batch_size=None,
        lambda_x=1.0,
        lambda_nn=10.0,
        lambda_ee=5.0,
        lambda_ne=5.0,
        lambda_dml=1.0,
        image_size=224,
        eps=1e-7,
    ):
        super().__init__(
            regularizer_weight=regularizer_weight,
            supervised_weight=supervised_weight,
        )
        self.graph_space = str(graph_space)
        self.projection_dim = int(projection_dim)
        self.projection_hidden_dim = (
            None if projection_hidden_dim is None else int(projection_hidden_dim)
        )
        self.queue_size = int(queue_size)
        self.temperature = float(temperature)
        self.propagation_alpha = float(propagation_alpha)
        self.top_n = int(top_n)
        self.ema_momentum = float(ema_momentum)
        self.distribution_alignment_enabled = bool(distribution_alignment)
        self.da_momentum = float(da_momentum)
        self.unlabeled_ratio = float(unlabeled_ratio)
        self.unlabeled_batch_size = (
            None if unlabeled_batch_size is None else int(unlabeled_batch_size)
        )
        self.lambda_x = _validate_finite_nonnegative("lambda_x", lambda_x)
        self.lambda_nn = _validate_finite_nonnegative("lambda_nn", lambda_nn)
        self.lambda_ee = _validate_finite_nonnegative("lambda_ee", lambda_ee)
        self.lambda_ne = _validate_finite_nonnegative("lambda_ne", lambda_ne)
        self.lambda_dml = _validate_finite_nonnegative("lambda_dml", lambda_dml)
        self.image_size = int(image_size)
        self.eps = float(eps)

        if self.graph_space not in SimMatchV2Heads.GRAPH_SPACES:
            raise ValueError(
                f"simmatch_v2 graph_space must be one of {sorted(SimMatchV2Heads.GRAPH_SPACES)}"
            )
        if self.projection_dim <= 0:
            raise ValueError("simmatch_v2 projection_dim must be positive")
        if self.projection_hidden_dim is not None and self.projection_hidden_dim <= 0:
            raise ValueError("simmatch_v2 projection_hidden_dim must be positive")
        if self.queue_size <= 0:
            raise ValueError("simmatch_v2 queue_size must be positive")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise ValueError("simmatch_v2 temperature must be finite and positive")
        if not math.isfinite(self.propagation_alpha) or not (0 <= self.propagation_alpha < 1):
            raise ValueError("simmatch_v2 propagation_alpha must be in [0, 1)")
        if self.top_n <= 0:
            raise ValueError("simmatch_v2 top_n must be positive")
        if not math.isfinite(self.ema_momentum) or not (0 <= self.ema_momentum < 1):
            raise ValueError("simmatch_v2 ema_momentum must be in [0, 1)")
        if not math.isfinite(self.da_momentum) or not (0 <= self.da_momentum < 1):
            raise ValueError("simmatch_v2 da_momentum must be in [0, 1)")
        if not math.isfinite(self.unlabeled_ratio) or self.unlabeled_ratio <= 0:
            raise ValueError("simmatch_v2 unlabeled_ratio must be finite and positive")
        if self.unlabeled_batch_size is not None and self.unlabeled_batch_size <= 0:
            raise ValueError("simmatch_v2 unlabeled_batch_size must be positive")
        if self.image_size <= 0:
            raise ValueError("simmatch_v2 image_size must be positive")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("simmatch_v2 eps must be finite and positive")

        self.dataset = None
        self.labeled_bank_labels = None
        self.num_classes = None
        self.graph_dim = None
        self.confidence_threshold = 0.0
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        self._last_diagnostics = {}

    def validate_run_args(self, args):
        if bool(getattr(args, "use_cache", False)):
            raise ValueError(
                "simmatch_v2 requires stochastic weak/strong image views and cannot use backbone caching"
            )

    def configure_model(self, student_model, train_dataset, split, train_labels_mapper, device):
        if self.regularizer_weight == 0:
            return
        num_classes = len(train_labels_mapper)
        labeled_positions = np.asarray(split.labeled_positions, dtype=np.int64)
        labeled_bank_labels = np.asarray(train_dataset.labels, dtype=np.int64)[labeled_positions]
        present_classes = set(int(label) for label in np.unique(labeled_bank_labels))
        missing_classes = sorted(set(range(num_classes)) - present_classes)
        if missing_classes:
            raise ValueError(
                "simmatch_v2 assumes closed-set SSL and needs at least one labeled sample from every "
                f"training class; {len(missing_classes)} classes have no labeled support. "
                "Use label_sampling_mode='per_class_min' or class_subset_k_shot with labeled_fraction=1."
            )
        if self.top_n > len(labeled_bank_labels):
            raise ValueError(
                f"simmatch_v2 top_n={self.top_n} exceeds the labeled bank size "
                f"{len(labeled_bank_labels)}"
            )
        if hasattr(student_model, "simmatch_v2_heads"):
            raise RuntimeError("simmatch_v2 heads are already configured on this model")

        heads = SimMatchV2Heads(
            feat_dim=student_model.feat_dim,
            num_classes=num_classes,
            graph_space=self.graph_space,
            projection_dim=self.projection_dim,
            projection_hidden_dim=self.projection_hidden_dim,
        ).to(device)
        student_model.add_module("simmatch_v2_heads", heads)
        self.labeled_bank_labels = labeled_bank_labels
        self.num_classes = int(num_classes)
        self.graph_dim = int(heads.graph_dim)
        logger.info(
            "Configured SimMatchV2 training heads: "
            f"classes={self.num_classes}, labeled_bank={len(labeled_bank_labels)}, "
            f"graph_space={self.graph_space}, graph_dim={self.graph_dim}, queue={self.queue_size}"
        )

    def make_supervised_source_dataset(self, train_dataset):
        if self.regularizer_weight == 0:
            return train_dataset
        return _copy_with_transform(
            train_dataset,
            make_simmatch_v2_weak_transform(self.image_size),
        )

    def build_dataset(self, train_dataset, split, use_cache=False):
        self.use_cache = bool(use_cache)
        if self.use_cache:
            raise ValueError("simmatch_v2 does not support cached backbone features")
        utils.shutdown_dataloaders(self._regularizer_loader)
        self._regularizer_loader = None
        self._regularizer_loader_cache_key = None
        if self.regularizer_weight == 0:
            self.dataset = None
            return None
        self.dataset = SimMatchV2UnlabeledDataset(
            dataset=train_dataset,
            positions=split.unlabeled_positions,
            weak_transform=make_simmatch_v2_weak_transform(self.image_size),
            strong_transform=make_simmatch_v2_strong_transform(self.image_size),
        )
        return self.dataset

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
        if self.dataset is None:
            raise RuntimeError("simmatch_v2 build_dataset must run before make_loader")
        self.confidence_threshold = float(config.confidence_threshold)
        requested_batch_size = (
            self.unlabeled_batch_size
            if self.unlabeled_batch_size is not None
            else max(1, int(round(float(batch_size) * self.unlabeled_ratio)))
        )
        regularizer_batch_size = min(requested_batch_size, len(self.dataset))
        worker_count = utils.dataloader_num_workers_for_dataset(self.dataset, num_workers)
        cache_key = (
            id(self.dataset),
            regularizer_batch_size,
            int(worker_count),
            str(start_method),
        )
        if self._regularizer_loader is None or self._regularizer_loader_cache_key != cache_key:
            utils.shutdown_dataloaders(self._regularizer_loader)
            self._regularizer_loader = DataLoader(
                self.dataset,
                batch_size=regularizer_batch_size,
                shuffle=True,
                drop_last=False,
                **utils.make_dataloader_kwargs(
                    worker_count,
                    seed,
                    start_method,
                    persistent_workers=True,
                    pin_memory=True,
                ),
            )
            self._regularizer_loader_cache_key = cache_key
            logger.info(
                "SimMatchV2 unlabeled loader: "
                f"pool={len(self.dataset)}, batch_size={regularizer_batch_size}, "
                f"ratio={self.unlabeled_ratio}, steps={len(supervised_loader)}"
            )
        return CombinedTrainingLoader(supervised_loader, self._regularizer_loader)

    def initialize_state(self, student_model, device):
        if self.labeled_bank_labels is None or not hasattr(student_model, "simmatch_v2_heads"):
            raise RuntimeError("simmatch_v2 model must be configured before state initialization")
        return SimMatchV2State(
            student_model=student_model,
            labeled_bank_labels=self.labeled_bank_labels,
            num_classes=self.num_classes,
            graph_dim=self.graph_dim,
            queue_size=self.queue_size,
            device=device,
        )

    def _pseudo_labeled_dml_loss(
        self,
        criterion,
        miner,
        is_classification,
        labeled_embeddings,
        labeled_labels,
        unlabeled_embeddings,
        pseudo_labels,
        confidence,
        mask,
    ):
        if self.lambda_dml == 0 or not bool(mask.any()):
            return labeled_embeddings.sum() * 0.0
        embeddings = torch.cat([labeled_embeddings, unlabeled_embeddings[mask]], dim=0)
        labels = torch.cat([labeled_labels, pseudo_labels[mask]], dim=0)
        if getattr(criterion, "supports_sample_weights", False):
            sample_weights = torch.cat(
                [
                    torch.ones(len(labeled_labels), device=embeddings.device, dtype=torch.float32),
                    confidence[mask].to(device=embeddings.device, dtype=torch.float32),
                ]
            )
            return criterion(embeddings, labels, sample_weights=sample_weights)
        if not is_classification and miner is not None:
            return criterion(embeddings, labels, miner(embeddings, labels))
        return criterion(embeddings, labels)

    def compute_loss(
        self,
        student_model,
        state,
        batch,
        device,
        timings=None,
        supervised_embeddings=None,
        supervised_labels=None,
        regularizer_embeddings=None,
        supervised_inputs=None,
        supervised_indices=None,
        supervised_criterion=None,
        supervised_miner=None,
        supervised_is_classification=False,
    ):
        if state is None:
            raise ValueError("simmatch_v2 requires initialized EMA and memory-bank state")
        if any(
            value is None
            for value in (
                supervised_embeddings,
                supervised_labels,
                regularizer_embeddings,
                supervised_inputs,
                supervised_indices,
                supervised_criterion,
            )
        ):
            raise ValueError("simmatch_v2 requires the labeled, objective, and joint-forward context")
        if batch is None or len(batch) < 2:
            raise ValueError("simmatch_v2 requires strong and weak unlabeled views")
        strong_inputs, weak_inputs = batch[:2]
        if len(regularizer_embeddings) != len(strong_inputs):
            raise ValueError("simmatch_v2 strong embeddings must align with the strong-view batch")

        student_logits_x, _ = student_model.simmatch_v2_heads(supervised_embeddings)
        student_logits_u, student_graph_u = student_model.simmatch_v2_heads(
            regularizer_embeddings
        )

        with torch.no_grad():
            teacher_inputs = torch.cat(
                [
                    supervised_inputs.to(device, non_blocking=True),
                    weak_inputs.to(device, non_blocking=True),
                ],
                dim=0,
            )
            teacher_embeddings = state.ema(teacher_inputs)
            teacher_logits, teacher_graph = state.ema.simmatch_v2_heads(teacher_embeddings)
            labeled_batch_size = len(supervised_embeddings)
            teacher_graph_x = teacher_graph[:labeled_batch_size].float()
            teacher_graph_u = teacher_graph[labeled_batch_size:].float()
            teacher_probabilities = F.softmax(
                teacher_logits[labeled_batch_size:].float(),
                dim=1,
            )
            if self.distribution_alignment_enabled:
                teacher_probabilities = state.distribution_alignment(
                    teacher_probabilities,
                    momentum=self.da_momentum,
                    eps=self.eps,
                )
            teacher_relations = F.softmax(
                teacher_graph_u @ state.u_bank.T / self.temperature,
                dim=1,
            )
            propagated_probabilities = state.propagate(
                teacher_graph_u,
                teacher_probabilities,
                top_n=self.top_n,
                temperature=self.temperature,
                alpha=self.propagation_alpha,
                eps=self.eps,
            )
            confidence = teacher_probabilities.max(dim=1).values
            hard_pseudo_labels = propagated_probabilities.argmax(dim=1)
            mask = confidence.ge(self.confidence_threshold)

        student_relations = F.softmax(
            student_graph_u.float() @ state.u_bank.detach().clone().T / self.temperature,
            dim=1,
        )
        loss_x = F.cross_entropy(student_logits_x.float(), supervised_labels)
        node_node_per_sample = -(
            propagated_probabilities * F.log_softmax(student_logits_u.float(), dim=1)
        ).sum(dim=1)
        loss_nn = (node_node_per_sample * mask.to(node_node_per_sample.dtype)).mean()
        loss_ee = -(
            teacher_relations * student_relations.clamp_min(self.eps).log()
        ).sum(dim=1).mean()
        class_from_edges = student_relations @ state.u_labels.detach().clone()
        loss_ne = -(
            teacher_probabilities * class_from_edges.clamp_min(self.eps).log()
        ).sum(dim=1).mean()

        dml_embeddings = torch.cat(
            [supervised_embeddings, regularizer_embeddings[mask]],
            dim=0,
        )
        dml_labels = torch.cat(
            [supervised_labels, hard_pseudo_labels[mask]],
            dim=0,
        )
        if supervised_is_classification or supervised_miner is None:
            loss_dml = supervised_criterion(dml_embeddings, dml_labels)
        else:
            miner_outputs = supervised_miner(dml_embeddings, dml_labels)
            loss_dml = supervised_criterion(dml_embeddings, dml_labels, miner_outputs)

        total = (
            self.lambda_x * loss_x
            + self.lambda_nn * loss_nn
            + self.lambda_ee * loss_ee
            + self.lambda_ne * loss_ne
           #  + self.lambda_dml * loss_dml
        )
        if not torch.isfinite(total):
            raise FloatingPointError("simmatch_v2 produced a non-finite regularization loss")

        state.update_labeled_bank(teacher_graph_x, supervised_indices)
        state.update_unlabeled_bank(teacher_graph_u, teacher_probabilities)
        if self.collect_batch_diagnostics:
            self._last_diagnostics = {
                "train/simmatch_v2/classification_loss": loss_x.detach(),
                "train/simmatch_v2/node_node_loss": loss_nn.detach(),
                "train/simmatch_v2/edge_edge_loss": loss_ee.detach(),
                "train/simmatch_v2/node_edge_loss": loss_ne.detach(),
                "train/simmatch_v2/pseudo_dml_loss": loss_dml.detach(),
                "train/simmatch_v2/confident_fraction": mask.float().mean().detach(),
                "train/simmatch_v2/teacher_confidence": confidence.mean().detach(),
                "train/simmatch_v2/propagated_confidence": (
                    propagated_probabilities.max(dim=1).values.mean().detach()
                ),
                "train/simmatch_v2/queue_pointer": float(state.ptr.item()),
            }
        else:
            self._last_diagnostics = {}
        return total

    @torch.no_grad()
    def after_optimizer_step(self, student_model, state):
        if state is None:
            return
        momentum = self.ema_momentum
        student_parameters = dict(student_model.named_parameters())
        for name, ema_parameter in state.ema.named_parameters():
            ema_parameter.mul_(momentum).add_(student_parameters[name], alpha=1.0 - momentum)
        student_buffers = dict(student_model.named_buffers())
        for name, ema_buffer in state.ema.named_buffers():
            student_buffer = student_buffers[name]
            if torch.is_floating_point(ema_buffer):
                ema_buffer.mul_(momentum).add_(student_buffer, alpha=1.0 - momentum)
            else:
                ema_buffer.copy_(student_buffer)
        state.ema.eval()

    def batch_diagnostics(self):
        result = {}
        for name, value in self._last_diagnostics.items():
            result[name] = float(value.detach().item()) if torch.is_tensor(value) else float(value)
        return result
