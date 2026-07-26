import hashlib
import os
import threading
import time
import uuid
import warnings
import weakref
from contextlib import contextmanager
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tfm

DINOV2_REPO = "facebookresearch/dinov2:main"
DINOV2_HUB_MAX_ATTEMPTS = 4
DINOV2_HUB_RETRY_DELAY_SECONDS = 5
BACKBONE_CACHE_VERSION = 3

DINOV2_ARCHS = {
    "s": 384,
    "b": 768,
    "l": 1024,
    "g": 1536,
}
BACKBONE_TUNING_FULL = "full"
BACKBONE_TUNING_FROZEN = "frozen"
BACKBONE_TUNING_LAST_BLOCKS_PREFIX = "last_"


@contextmanager
def _exclusive_cache_file_lock(path):
    """Serialize mmap creation/commits across local processes."""

    lock_path = Path(f"{path}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_file:
        if os.name == "nt":
            import msvcrt

            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
            try:
                yield
            finally:
                lock_file.seek(0)
                msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class _IndexedFeatureMatrix:
    """One float32 mmap whose rows are addressed by stable dataset indices."""

    def __init__(self, path, num_rows, feature_dim):
        self.path = Path(path)
        self.num_rows = int(num_rows)
        self.feature_dim = int(feature_dim)
        self._lock = threading.Lock()
        self._inflight = {}
        self._matrix = self._open_or_create()
        self._tensor = torch.from_numpy(self._matrix)

    def _open_or_create(self):
        expected_shape = (self.num_rows, self.feature_dim)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with _exclusive_cache_file_lock(self.path):
            if self.path.exists():
                try:
                    matrix = np.load(self.path, mmap_mode="r+")
                    if matrix.dtype == np.float32 and matrix.shape == expected_shape:
                        return matrix
                except (OSError, ValueError):
                    pass
                self.path.unlink(missing_ok=True)

            temp_path = self.path.with_name(
                f"{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
            )
            try:
                matrix = np.lib.format.open_memmap(
                    temp_path,
                    mode="w+",
                    dtype=np.float32,
                    shape=expected_shape,
                )
                matrix[:] = np.nan
                matrix.flush()
                del matrix
                os.replace(temp_path, self.path)
            finally:
                temp_path.unlink(missing_ok=True)
            return np.load(self.path, mmap_mode="r+")

    def _valid_mask(self, rows):
        if len(rows) == 0:
            return np.ones(0, dtype=bool)
        return np.isfinite(self._matrix[rows, 0])

    def _release_claims(self, rows):
        with self._lock:
            for row in rows:
                event = self._inflight.pop(int(row), None)
                if event is not None:
                    event.set()

    def materialize(self, rows, compute_missing):
        """Fill missing unique rows once and return cache-access diagnostics."""

        rows = np.asarray(rows, dtype=np.int64)
        if rows.ndim != 1:
            raise ValueError("cache indices must be one-dimensional")
        if np.any((rows < 0) | (rows >= self.num_rows)):
            raise IndexError("cache index is outside the configured matrix")

        first_local_position = {}
        for local_position, row in enumerate(rows.tolist()):
            first_local_position.setdefault(int(row), int(local_position))
        unique_rows = np.fromiter(first_local_position, dtype=np.int64)
        computed_rows = set()
        written_rows = set()
        waited_rows = set()

        while True:
            with self._lock:
                with _exclusive_cache_file_lock(self.path):
                    valid = self._valid_mask(unique_rows)
                missing_rows = unique_rows[~valid]
                if len(missing_rows) == 0:
                    break

                owned_rows = []
                pending_events = []
                for row in missing_rows.tolist():
                    row = int(row)
                    event = self._inflight.get(row)
                    if event is None:
                        event = threading.Event()
                        self._inflight[row] = event
                        owned_rows.append(row)
                    else:
                        waited_rows.add(row)
                        pending_events.append(event)

            if owned_rows:
                local_positions = np.asarray(
                    [first_local_position[row] for row in owned_rows],
                    dtype=np.int64,
                )
                try:
                    features = torch.as_tensor(
                        compute_missing(local_positions),
                        dtype=torch.float32,
                    ).detach().cpu().contiguous()
                    expected_shape = (len(owned_rows), self.feature_dim)
                    if tuple(features.shape) != expected_shape:
                        raise ValueError(
                            "computed cached feature shape does not match the indexed matrix: "
                            f"expected {expected_shape}, got {tuple(features.shape)}"
                        )
                    if not torch.isfinite(features).all():
                        raise ValueError("refusing to persist non-finite backbone features")
                    feature_array = features.numpy()
                    computed_rows.update(owned_rows)

                    with self._lock:
                        with _exclusive_cache_file_lock(self.path):
                            owned_array = np.asarray(owned_rows, dtype=np.int64)
                            still_missing = ~self._valid_mask(owned_array)
                            rows_to_write = owned_array[still_missing]
                            if len(rows_to_write):
                                values_to_write = feature_array[still_missing]
                                # Column zero is the validity marker. Write it
                                # last so another process never observes a
                                # partially committed row as complete.
                                if self.feature_dim > 1:
                                    self._matrix[rows_to_write, 1:] = values_to_write[:, 1:]
                                    self._matrix.flush()
                                self._matrix[rows_to_write, 0] = values_to_write[:, 0]
                                self._matrix.flush()
                                written_rows.update(int(row) for row in rows_to_write)
                except BaseException:
                    self._release_claims(owned_rows)
                    raise
                self._release_claims(owned_rows)

            for event in set(pending_events):
                event.wait()

        with self._lock:
            with _exclusive_cache_file_lock(self.path):
                if not self._valid_mask(unique_rows).all():
                    raise RuntimeError("indexed backbone cache contains unresolved rows")

        return {
            "computed_rows": computed_rows,
            "written_rows": written_rows,
            "waited_rows": waited_rows,
        }

    @property
    def tensor(self):
        return self._tensor


_INDEXED_FEATURE_MATRICES = weakref.WeakValueDictionary()
_INDEXED_FEATURE_MATRICES_LOCK = threading.Lock()


def _get_indexed_feature_matrix(cache_dir, cache_key, num_rows, feature_dim):
    namespace = hashlib.sha256(str(cache_key).encode("utf-8")).hexdigest()
    path = Path(cache_dir) / f"{namespace}.features.npy"
    registry_key = str(path.resolve())
    with _INDEXED_FEATURE_MATRICES_LOCK:
        matrix = _INDEXED_FEATURE_MATRICES.get(registry_key)
        if matrix is None:
            matrix = _IndexedFeatureMatrix(path, num_rows, feature_dim)
            _INDEXED_FEATURE_MATRICES[registry_key] = matrix
        elif (matrix.num_rows, matrix.feature_dim) != (int(num_rows), int(feature_dim)):
            raise ValueError("indexed backbone cache key was reused with a different matrix shape")
        return matrix


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
        self._cache_stats_lock = threading.Lock()
        self._cache_stats = {
            "enabled": self.use_cache,
            "fully_cached_batches": 0,
            "batches_with_misses": 0,
            "hit_samples": 0,
            "memory_hit_samples": 0,
            "disk_hit_samples": 0,
            "miss_samples": 0,
            "written_samples": 0,
            "waited_samples": 0,
            "uncached_samples": 0,
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
        """Project features for the ordinary supervised/evaluation path."""

        features = self.fc(features)
        return F.normalize(features, p=2.0, dim=1)

    def project_stml_features(self, features):
        """Return the STML background head g and retrieval head f."""

        if not self.stml_enabled:
            raise RuntimeError("STML heads are not enabled for this model")
        retrieval_features = self.fc(features)
        if self.stml_normalize_student:
            retrieval_features = F.normalize(retrieval_features, p=2.0, dim=1)
        return self.embedding_g(features), retrieval_features

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

    def forward_cached(
        self,
        images,
        device,
        *,
        cache_key=None,
        cache_indices=None,
        cache_size=None,
    ):
        """Project frozen DINO embeddings through the indexed mmap cache."""

        if not self.use_cache:
            return self(images.to(device, non_blocking=True))
        features = self._load_or_compute_cached_backbone_features(
            images,
            device,
            cache_key=cache_key,
            cache_indices=cache_indices,
            cache_size=cache_size,
        )
        return self.project_features(features)

    def forward_backbone_cached(
        self,
        images,
        device,
        *,
        cache_key=None,
        cache_indices=None,
        cache_size=None,
    ):
        """Return raw frozen DINO features, optionally using the persistent cache."""

        if not self.use_cache:
            return self.forward_backbone(images.to(device, non_blocking=True))
        return self._load_or_compute_cached_backbone_features(
            images,
            device,
            cache_key=cache_key,
            cache_indices=cache_indices,
            cache_size=cache_size,
        )

    def forward_stml_cached(
        self,
        images,
        device,
        *,
        cache_key=None,
        cache_indices=None,
        cache_size=None,
    ):
        """Return both STML heads, optionally from cached backbone features."""

        if torch.is_tensor(images) and images.ndim == 2:
            return self.project_stml_features(images.to(device, non_blocking=True))
        if not self.use_cache:
            return self.forward_stml(images.to(device, non_blocking=True))
        features = self._load_or_compute_cached_backbone_features(
            images,
            device,
            cache_key=cache_key,
            cache_indices=cache_indices,
            cache_size=cache_size,
        )
        return self.project_stml_features(features)

    def forward_stml_teacher_cached(
        self,
        images,
        device,
        *,
        cache_key=None,
        cache_indices=None,
        cache_size=None,
    ):
        """Return teacher g, optionally from cached backbone features."""

        if torch.is_tensor(images) and images.ndim == 2:
            return self.project_stml_teacher_features(images.to(device, non_blocking=True))
        if not self.use_cache:
            return self.forward_stml_teacher(images.to(device, non_blocking=True))
        features = self._load_or_compute_cached_backbone_features(
            images,
            device,
            cache_key=cache_key,
            cache_indices=cache_indices,
            cache_size=cache_size,
        )
        return self.project_stml_teacher_features(features)

    def forward_eval(
        self,
        images,
        device,
        *,
        cache_key=None,
        cache_indices=None,
        cache_size=None,
    ):
        """Compatibility alias used by existing evaluation callers."""

        return self.forward_cached(
            images,
            device,
            cache_key=cache_key,
            cache_indices=cache_indices,
            cache_size=cache_size,
        )

    def cache_stats(self):
        with self._cache_stats_lock:
            stats = dict(self._cache_stats)
        total_batches = stats["fully_cached_batches"] + stats["batches_with_misses"]
        total_samples = stats["hit_samples"] + stats["miss_samples"]
        return {
            **stats,
            "fully_cached_batch_rate": (
                0.0 if total_batches == 0 else stats["fully_cached_batches"] / total_batches
            ),
            "sample_hit_rate": 0.0 if total_samples == 0 else stats["hit_samples"] / total_samples,
            "cache_dir": None if self.cache_dir is None else str(self.cache_dir),
            "backbone": f"dinov2_vit{self.dino_size}14",
            "cache_version": BACKBONE_CACHE_VERSION,
            "cache_format": "indexed_float32_mmap",
        }

    def _record_cache_access(self, requested_rows, access):
        computed_rows = access["computed_rows"]
        requested_unique = set(int(row) for row in np.asarray(requested_rows).tolist())
        misses = len(computed_rows)
        hits = max(0, len(requested_unique) - misses)
        with self._cache_stats_lock:
            if misses:
                self._cache_stats["batches_with_misses"] += 1
            else:
                self._cache_stats["fully_cached_batches"] += 1
            self._cache_stats["hit_samples"] += hits
            self._cache_stats["disk_hit_samples"] += hits
            self._cache_stats["miss_samples"] += misses
            self._cache_stats["written_samples"] += len(access["written_rows"])
            self._cache_stats["waited_samples"] += len(access["waited_rows"])

    def materialize_cached_backbone_features(
        self,
        *,
        cache_key,
        cache_indices,
        cache_size,
        compute_missing,
    ):
        """Ensure indexed rows exist and expose the shared mmap tensor."""

        if not self.use_cache or self.cache_dir is None:
            raise RuntimeError("indexed backbone materialization requires use_cache=True")
        cache_indices = np.asarray(cache_indices, dtype=np.int64)
        matrix = _get_indexed_feature_matrix(
            self.cache_dir,
            cache_key,
            cache_size,
            DINOV2_ARCHS[self.dino_size],
        )
        access = matrix.materialize(cache_indices, compute_missing)
        self._record_cache_access(cache_indices, access)
        return matrix.tensor, torch.as_tensor(cache_indices, dtype=torch.long)

    def _load_or_compute_cached_backbone_features(
        self,
        images,
        device,
        *,
        cache_key=None,
        cache_indices=None,
        cache_size=None,
    ):
        device = torch.device(device)
        if cache_key is None or cache_indices is None or cache_size is None:
            # A tensor alone has no stable sample identity. Compute normally
            # instead of hashing every pixel or silently reusing the wrong row.
            with self._cache_stats_lock:
                self._cache_stats["uncached_samples"] += len(images)
            return self.forward_backbone(images.to(device, non_blocking=True))

        cache_indices = np.asarray(cache_indices, dtype=np.int64)
        if len(cache_indices) != len(images):
            raise ValueError("cache_indices must contain one row per input image")

        def compute_missing(local_positions):
            positions = torch.as_tensor(local_positions, dtype=torch.long, device=images.device)
            missing_images = images.index_select(0, positions).to(
                device,
                non_blocking=True,
            )
            return self.forward_backbone(missing_images).detach().float().cpu()

        feature_matrix, matrix_rows = self.materialize_cached_backbone_features(
            cache_key=cache_key,
            cache_indices=cache_indices,
            cache_size=cache_size,
            compute_missing=compute_missing,
        )
        return feature_matrix.index_select(0, matrix_rows).to(
            device,
            non_blocking=True,
        )
