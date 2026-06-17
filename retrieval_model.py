import hashlib
import os
import time
import uuid
import warnings
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfm

DINOV2_REPO = "facebookresearch/dinov2:main"
DINOV2_HUB_MAX_ATTEMPTS = 4
DINOV2_HUB_RETRY_DELAY_SECONDS = 5
BACKBONE_CACHE_VERSION = 2

DINOV2_ARCHS = {
    "s": 384,
    "b": 768,
    "l": 1024,
    "g": 1536,
}
BACKBONE_TUNING_FULL = "full"
BACKBONE_TUNING_FROZEN = "frozen"
BACKBONE_TUNING_LAST_BLOCKS_PREFIX = "last_"


def normalize_backbone_tuning(value):
    """Normalize a DINO fine-tuning policy."""

    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {BACKBONE_TUNING_FULL, BACKBONE_TUNING_FROZEN}:
        return normalized
    if normalized.startswith(BACKBONE_TUNING_LAST_BLOCKS_PREFIX):
        suffix = normalized[len(BACKBONE_TUNING_LAST_BLOCKS_PREFIX):]
        for ending in ("_blocks", "_block"):
            if suffix.endswith(ending):
                suffix = suffix[:-len(ending)]
                break
        try:
            num_blocks = int(suffix)
        except ValueError as exc:
            raise ValueError(
                "backbone_tuning must be 'full', 'frozen', or 'last_N_blocks'"
            ) from exc
        if num_blocks <= 0:
            raise ValueError("last_N_blocks requires N to be positive")
        return f"last_{num_blocks}_blocks"
    raise ValueError("backbone_tuning must be 'full', 'frozen', or 'last_N_blocks'")


def _is_retryable_hub_error(exc):
    if isinstance(exc, HTTPError):
        return exc.code in (408, 429) or 500 <= exc.code < 600
    return isinstance(exc, (URLError, ConnectionError, TimeoutError, HTTPException))


def load_dinov2_with_retry(dino_size, max_attempts=DINOV2_HUB_MAX_ATTEMPTS):
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    model_name = f"dinov2_vit{dino_size}14"
    for attempt in range(1, max_attempts + 1):
        try:
            return torch.hub.load(DINOV2_REPO, model_name)
        except Exception as exc:
            if attempt == max_attempts or not _is_retryable_hub_error(exc):
                raise

            delay = DINOV2_HUB_RETRY_DELAY_SECONDS * 2 ** (attempt - 1)
            warnings.warn(
                f"Loading {model_name} from Torch Hub failed with {exc!r}. "
                f"Retrying in {delay} seconds ({attempt}/{max_attempts}).",
                RuntimeWarning,
                stacklevel=2,
            )
            time.sleep(delay)


class L2Norm(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return F.normalize(x, p=2.0, dim=self.dim)


class DinoWrapper(nn.Module):
    """Same as the original DINO model, but with a linear layer on top and a resize to multiple of 14 in the forward pass."""

    def __init__(
        self,
        dino_size,
        feat_dim,
        backbone_tuning=BACKBONE_TUNING_FULL,
        use_cache=False,
        cache_dir=None,
        stml=False,
        stml_g_dim=None,
        stml_normalize_student=False,
    ):
        super().__init__()
        assert dino_size in "sblg"
        backbone_tuning = normalize_backbone_tuning(backbone_tuning)
        if use_cache and backbone_tuning != BACKBONE_TUNING_FROZEN:
            raise ValueError("DINO backbone caching requires backbone_tuning='frozen'")
        self.dinov2 = load_dinov2_with_retry(dino_size)
        self.dino_size = dino_size
        self.backbone_tuning = backbone_tuning
        self.use_cache = bool(use_cache)
        self.stml_enabled = bool(stml)
        self.stml_normalize_student = bool(stml_normalize_student)
        backbone_dim = DINOV2_ARCHS[dino_size]
        if feat_dim is not None or self.stml_enabled:
            self.feat_dim = backbone_dim if feat_dim is None else feat_dim
            self.fc = nn.Linear(backbone_dim, self.feat_dim)
            if self.stml_enabled:
                nn.init.orthogonal_(self.fc.weight)
                nn.init.zeros_(self.fc.bias)
        else:
            self.fc = nn.Identity()
            self.feat_dim = backbone_dim
        if self.stml_enabled:
            self.stml_g_dim = backbone_dim if stml_g_dim is None else int(stml_g_dim)
            if self.stml_g_dim <= 0:
                raise ValueError("stml_g_dim must be positive")
            self.embedding_g = nn.Linear(backbone_dim, self.stml_g_dim)
            nn.init.orthogonal_(self.embedding_g.weight)
            nn.init.zeros_(self.embedding_g.bias)
        else:
            self.stml_g_dim = None
        self.cache_dir = None
        self._memory_feature_cache = {}
        self._cache_stats = {
            "enabled": self.use_cache,
            "fully_cached_batches": 0,
            "batches_with_misses": 0,
            "hit_samples": 0,
            "memory_hit_samples": 0,
            "disk_hit_samples": 0,
            "miss_samples": 0,
            "written_samples": 0,
        }
        self._configure_backbone_tuning()
        if self.backbone_tuning == BACKBONE_TUNING_FROZEN:
            self.dinov2.eval()
        if self.use_cache:
            if cache_dir is None:
                raise ValueError("DINO backbone caching requires a dataset-local cache_dir")
            cache_root = Path(cache_dir)
            self.cache_dir = cache_root / f"v{BACKBONE_CACHE_VERSION}" / f"dinov2_vit{dino_size}14"
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _configure_backbone_tuning(self):
        if self.backbone_tuning == BACKBONE_TUNING_FULL:
            for parameter in self.dinov2.parameters():
                parameter.requires_grad = True
            return

        for parameter in self.dinov2.parameters():
            parameter.requires_grad = False
        if self.backbone_tuning == BACKBONE_TUNING_FROZEN:
            return

        blocks = getattr(self.dinov2, "blocks", None)
        if blocks is None:
            raise ValueError("last_N_blocks tuning requires a DINO backbone with a blocks module")
        num_blocks = int(self.backbone_tuning.split("_")[1])
        if num_blocks > len(blocks):
            raise ValueError(
                f"{self.backbone_tuning} requests {num_blocks} blocks, but this DINO backbone has {len(blocks)}"
            )
        for block in blocks[-num_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
        final_norm = getattr(self.dinov2, "norm", None)
        if final_norm is not None:
            for parameter in final_norm.parameters():
                parameter.requires_grad = True

    def resize_multiple_14(self, images):
        b, c, h, w = images.shape
        # DINO needs height and width as multiple of 14, therefore resize them to the nearest multiple of 14
        h = round(h / 14) * 14
        w = round(w / 14) * 14
        images = tfm.functional.resize(images, [h, w], antialias=True)
        return images

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_tuning == BACKBONE_TUNING_FROZEN:
            # Keep frozen DINO behavior deterministic while allowing the
            # projection head to switch between train/eval modes normally.
            self.dinov2.eval()
        return self

    def forward_backbone(self, images):
        images = self.resize_multiple_14(images)
        if self.backbone_tuning == BACKBONE_TUNING_FROZEN:
            with torch.no_grad():
                return self.dinov2(images)
        return self.dinov2(images)

    def project_features(self, features):
        features = self.fc(features)
        if not self.stml_enabled or self.stml_normalize_student:
            features = F.normalize(features, p=2.0, dim=1)
        return features

    def project_stml_features(self, features):
        """Return the STML background head g and retrieval head f."""

        if not self.stml_enabled:
            raise RuntimeError("STML heads are not enabled for this model")
        return self.embedding_g(features), self.project_features(features)

    def project_stml_teacher_features(self, features):
        """Return only the teacher background head g used by STML."""

        if not self.stml_enabled:
            raise RuntimeError("STML heads are not enabled for this model")
        return self.embedding_g(features)

    def forward(self, images):
        return self.project_features(self.forward_backbone(images))

    def forward_stml(self, images):
        return self.project_stml_features(self.forward_backbone(images))

    def forward_stml_teacher(self, images):
        return self.project_stml_teacher_features(self.forward_backbone(images))

    def forward_cached(self, images, device):
        """Project frozen DINO embeddings, loading or creating per-sample cache entries."""

        if not self.use_cache:
            return self(images.to(device))
        features = self._load_or_compute_cached_backbone_features(images, device)
        return self.project_features(features)

    def forward_stml_cached(self, images, device):
        """Return both STML heads, optionally from cached backbone features."""

        if not self.use_cache:
            return self.forward_stml(images.to(device))
        features = self._load_or_compute_cached_backbone_features(images, device)
        return self.project_stml_features(features)

    def forward_stml_teacher_cached(self, images, device):
        """Return teacher g, optionally from cached backbone features."""

        if not self.use_cache:
            return self.forward_stml_teacher(images.to(device))
        features = self._load_or_compute_cached_backbone_features(images, device)
        return self.project_stml_teacher_features(features)

    def forward_eval(self, images, device):
        """Compatibility alias used by existing evaluation callers."""

        return self.forward_cached(images, device)

    def cache_stats(self):
        total_batches = self._cache_stats["fully_cached_batches"] + self._cache_stats["batches_with_misses"]
        total_samples = self._cache_stats["hit_samples"] + self._cache_stats["miss_samples"]
        return {
            **self._cache_stats,
            "fully_cached_batch_rate": (
                0.0 if total_batches == 0 else self._cache_stats["fully_cached_batches"] / total_batches
            ),
            "sample_hit_rate": 0.0 if total_samples == 0 else self._cache_stats["hit_samples"] / total_samples,
            "cache_dir": None if self.cache_dir is None else str(self.cache_dir),
            "backbone": f"dinov2_vit{self.dino_size}14",
            "cache_version": BACKBONE_CACHE_VERSION,
        }

    def _load_or_compute_cached_backbone_features(self, images, device):
        cpu_images = images.detach().cpu().contiguous()
        cache_keys = [self._cache_key(image) for image in cpu_images]
        cache_paths = [self.cache_dir / f"{cache_key}.pt" for cache_key in cache_keys]
        cached_features = [None] * len(cpu_images)
        missing_indices = []
        for index, (cache_key, cache_path) in enumerate(zip(cache_keys, cache_paths)):
            if cache_key in self._memory_feature_cache:
                cached_features[index] = self._memory_feature_cache[cache_key]
                self._cache_stats["hit_samples"] += 1
                self._cache_stats["memory_hit_samples"] += 1
                continue
            if not cache_path.exists():
                missing_indices.append(index)
                continue
            try:
                features = torch.load(cache_path, map_location="cpu", weights_only=True)
                if features.ndim != 1 or len(features) != DINOV2_ARCHS[self.dino_size]:
                    raise ValueError("cached feature shape does not match DINO backbone output")
                cached_features[index] = features
                self._memory_feature_cache[cache_key] = features
                self._cache_stats["hit_samples"] += 1
                self._cache_stats["disk_hit_samples"] += 1
            except (OSError, RuntimeError, TypeError, ValueError):
                missing_indices.append(index)

        if missing_indices:
            self._cache_stats["batches_with_misses"] += 1
            self._cache_stats["miss_samples"] += len(missing_indices)
            missing_images = cpu_images[missing_indices].to(device)
            missing_features = self.forward_backbone(missing_images).detach().cpu()
            for index, feature in zip(missing_indices, missing_features):
                # A row sliced from a batch retains the batch's entire storage.
                # Clone it before serialization so each cache file owns only
                # one sample's backbone feature vector.
                feature = feature.clone()
                cached_features[index] = feature
                self._memory_feature_cache[cache_keys[index]] = feature
                cache_path = cache_paths[index]
                temp_path = cache_path.with_suffix(f".{os.getpid()}.{uuid.uuid4().hex}.tmp")
                torch.save(feature, temp_path)
                os.replace(temp_path, cache_path)
                self._cache_stats["written_samples"] += 1
        else:
            self._cache_stats["fully_cached_batches"] += 1

        return torch.stack(cached_features).to(device)

    def _cache_key(self, image):
        digest = hashlib.sha256()
        digest.update(f"v{BACKBONE_CACHE_VERSION}:dinov2_vit{self.dino_size}14".encode("ascii"))
        digest.update(str(image.dtype).encode("ascii"))
        digest.update(str(tuple(image.shape)).encode("ascii"))
        digest.update(image.numpy().tobytes())
        return digest.hexdigest()
