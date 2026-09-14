"""Specific PlacEs Dataset (SPED) loader.

SPED was assembled from the Archive of Many Outdoor Scenes (AMOS): every
camera is one specific-place class and its time-lapse frames are observations
of that class under changing illumination, weather, and season.  The original
paper used 2,543 cameras and about 2.5 million images.

The selected SPED image collection is not exposed as a stable, automatically
downloadable archive.  This loader therefore consumes a manually prepared
camera tree and never attempts to reconstruct the authors' camera selection
from the much larger AMOS archive.  Either of these layouts is accepted::

    root/images/<camera_id>/<optional year/month folders>/<image>
    root/<camera_id>/<optional year/month folders>/<image>
    root/{train,val,test}/<camera_id>/<optional folders>/<image>

For large copies, ``sped_manifest.csv`` can replace directory discovery.  It
must contain ``path`` and ``camera_id`` columns, with paths relative to the
dataset root.  A directory scan writes the equivalent hidden
``.sped_index_v1.csv`` atomically, so constructing the test split does not scan
millions of files a second time.

SPED has no released metric-learning class split.  This project ranks camera
ids by a versioned SHA-256 digest and assigns the first half to development and
the other half to held-out testing.  Every observation of a camera remains on
one side, while each development camera still contributes all of its temporal
observations to few-shot label-propagation experiments.
"""

import csv
import hashlib
import os
import warnings
from collections import Counter
from pathlib import Path, PurePosixPath

from torch.utils.data import Dataset
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader


class SPED(Dataset):
    """Load SPED camera observations as specific-place classes."""

    DATASET_PAGE = "https://arxiv.org/abs/1701.05105"
    AMOS_PAGE = "https://mvrl.cse.wustl.edu/datasets/amos/"
    AVAILABLE_SPLITS = ("train", "test", "train+test")
    CLASS_SPLIT_VERSION = "sped_camera_sha256_half_v1"
    CLASS_SPLIT_BASIS = "sha256(<version>:<camera_id>)"
    MANIFEST_FILENAMES = (
        "sped_manifest.csv",
        "SPED_manifest.csv",
        ".sped_index_v1.csv",
    )
    GENERATED_INDEX_FILENAME = ".sped_index_v1.csv"
    PATH_COLUMNS = ("path", "image_path", "filepath", "file")
    CAMERA_COLUMNS = ("camera_id", "place_id", "location_id", "class_id", "label")
    IMAGE_DIRECTORY_NAMES = ("images", "Images")
    SOURCE_SPLIT_DIRECTORY_NAMES = ("train", "training", "val", "validation", "test")
    WRAPPER_DIRECTORY_NAMES = (
        "SPED",
        "sped",
        "SpecificPlacesDataset",
        "Specific_Places_Dataset",
    )
    IGNORED_DIRECTORY_NAMES = {
        "backbone_cache",
        "cache",
        "logs",
        "metadata",
    }
    IMAGE_EXTENSIONS = frozenset(extension.lower() for extension in IMG_EXTENSIONS)

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        download=False,
    ):
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(
                f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}"
            )

        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split

        self.dataset_root = self.find_dataset_root(self.root)
        if self.dataset_root is None:
            if download:
                raise RuntimeError(self._manual_download_message(self.root))
            raise ValueError(self._missing_dataset_message(self.root))

        records, index_source = self._load_or_discover_records(self.dataset_root)
        all_camera_ids = sorted({camera_id for _, camera_id in records})
        if len(all_camera_ids) < 2:
            raise ValueError(
                "SPED needs at least two camera/location directories to create "
                f"development and test splits; found {len(all_camera_ids)} below "
                f"{self.dataset_root}"
            )

        first_path_by_camera = {}
        for relative_path, camera_id in records:
            first_path_by_camera.setdefault(camera_id, relative_path)
        missing_camera_examples = [
            self.dataset_root / relative_path
            for camera_id, relative_path in first_path_by_camera.items()
            if not (self.dataset_root / relative_path).is_file()
        ]
        if missing_camera_examples:
            examples = ", ".join(str(path) for path in missing_camera_examples[:3])
            raise ValueError(
                "SPED manifest paths do not resolve to images for "
                f"{len(missing_camera_examples)} cameras; examples: {examples}"
            )

        development_camera_ids, test_camera_ids = self.partition_camera_ids(
            all_camera_ids
        )
        development_camera_set = set(development_camera_ids)
        test_camera_set = set(test_camera_ids)
        if split == "train":
            selected_camera_set = development_camera_set
        elif split == "test":
            selected_camera_set = test_camera_set
        else:
            selected_camera_set = development_camera_set | test_camera_set

        label_by_camera = {
            camera_id: label for label, camera_id in enumerate(all_camera_ids)
        }
        selected_records = [
            (relative_path, camera_id)
            for relative_path, camera_id in records
            if camera_id in selected_camera_set
        ]
        if not selected_records:
            raise ValueError(f"SPED split {split!r} is empty below {self.dataset_root}")

        self.paths = [
            str(self.dataset_root / relative_path)
            for relative_path, _ in selected_records
        ]
        self.camera_ids = [camera_id for _, camera_id in selected_records]
        self.labels = [label_by_camera[camera_id] for camera_id in self.camera_ids]
        self.targets = self.labels
        self.orig_labels = list(self.labels)
        self.classes = list(all_camera_ids)
        self.class_to_idx = dict(label_by_camera)
        self.selected_camera_ids = sorted(selected_camera_set)
        self.development_camera_ids = list(development_camera_ids)
        self.test_camera_ids = list(test_camera_ids)

        sample_counts = Counter(camera_id for _, camera_id in records)
        development_sample_count = sum(
            sample_counts[camera_id] for camera_id in development_camera_ids
        )
        test_sample_count = sum(sample_counts[camera_id] for camera_id in test_camera_ids)
        self.class_disjoint_split = True
        self.class_split_info = {
            "dataset_root": str(self.dataset_root),
            "dataset_page": self.DATASET_PAGE,
            "source_dataset_page": self.AMOS_PAGE,
            "split": self.split,
            "class_split": "sha256_camera_half",
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "class_split_basis": self.CLASS_SPLIT_BASIS,
            "place_definition": "amos_camera_id",
            "class_disjoint_test": True,
            "official_image_level_split_used": False,
            "source_camera_count": int(len(all_camera_ids)),
            "source_sample_count": int(len(records)),
            "development_class_count": int(len(development_camera_ids)),
            "test_class_count": int(len(test_camera_ids)),
            "development_sample_count": int(development_sample_count),
            "test_sample_count": int(test_sample_count),
            "selected_class_count": int(len(selected_camera_set)),
            "selected_sample_count": int(len(selected_records)),
            "development_camera_digest": self.camera_id_digest(
                development_camera_ids
            ),
            "test_camera_digest": self.camera_id_digest(test_camera_ids),
            "index_source": str(index_source),
        }

    @classmethod
    def _manual_download_message(cls, root):
        return (
            "SPED does not support automatic download. The original 2,543-camera "
            "selection is based on AMOS, whose maintainers require an access "
            f"request. Prepare camera folders or sped_manifest.csv below {root}; "
            f"see {cls.DATASET_PAGE} and {cls.AMOS_PAGE}."
        )

    @classmethod
    def _missing_dataset_message(cls, root):
        return (
            "SPED was not found. Expected sped_manifest.csv, "
            "images/<camera_id>/..., <train|val|test>/<camera_id>/..., or "
            "<camera_id>/... below "
            f"{root} (optionally inside a SPED wrapper directory)."
        )

    # ------------------------------------------------------------------
    # Dataset discovery and scalable indexing
    # ------------------------------------------------------------------

    @classmethod
    def find_dataset_root(cls, root):
        """Return the directory containing a SPED manifest or camera tree."""

        root = Path(root)
        candidates = [root / name for name in cls.WRAPPER_DIRECTORY_NAMES]
        candidates.append(root)
        if root.is_dir():
            visible_children = [
                child
                for child in sorted(root.iterdir())
                if child.is_dir() and not child.name.startswith(".")
            ]
            if len(visible_children) == 1:
                candidates.append(visible_children[0])

        seen = set()
        for candidate in candidates:
            if not candidate.is_dir():
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if cls._find_manifest(candidate) is not None:
                return candidate
            if cls._find_camera_image_roots(candidate):
                return candidate
        return None

    @classmethod
    def is_ready(cls, root):
        dataset_root = cls.find_dataset_root(root)
        if dataset_root is None:
            return False
        manifest_path = cls._find_manifest(dataset_root)
        if manifest_path is None:
            return bool(cls._find_camera_image_roots(dataset_root))

        try:
            camera_probes = {}
            for relative_path, camera_id in cls._iter_manifest_records(manifest_path):
                camera_probes.setdefault(camera_id, relative_path)
                if len(camera_probes) >= 2:
                    break
        except (OSError, ValueError, csv.Error):
            return False
        return len(camera_probes) >= 2 and all(
            (dataset_root / relative_path).is_file()
            for relative_path in camera_probes.values()
        )

    @classmethod
    def _find_manifest(cls, dataset_root):
        for filename in cls.MANIFEST_FILENAMES:
            candidate = Path(dataset_root) / filename
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def _find_camera_image_roots(cls, dataset_root):
        dataset_root = Path(dataset_root)
        for directory_name in cls.IMAGE_DIRECTORY_NAMES:
            candidate = dataset_root / directory_name
            if candidate.is_dir() and cls._has_multiple_camera_directories(candidate):
                return (candidate,)

        split_roots = tuple(
            dataset_root / directory_name
            for directory_name in cls.SOURCE_SPLIT_DIRECTORY_NAMES
            if (dataset_root / directory_name).is_dir()
            and cls._has_any_camera_directory(dataset_root / directory_name)
        )
        if split_roots:
            return split_roots
        if cls._has_multiple_camera_directories(dataset_root):
            return (dataset_root,)
        return ()

    @classmethod
    def _candidate_camera_directories(cls, image_root):
        return [
            child
            for child in sorted(Path(image_root).iterdir())
            if child.is_dir()
            and not child.name.startswith(".")
            and child.name not in cls.IGNORED_DIRECTORY_NAMES
        ]

    @classmethod
    def _has_multiple_camera_directories(cls, image_root):
        found = 0
        try:
            camera_directories = cls._candidate_camera_directories(image_root)
        except OSError:
            return False
        for camera_directory in camera_directories:
            if cls._first_image(camera_directory) is None:
                continue
            found += 1
            if found >= 2:
                return True
        return False

    @classmethod
    def _has_any_camera_directory(cls, image_root):
        try:
            camera_directories = cls._candidate_camera_directories(image_root)
        except OSError:
            return False
        return any(
            cls._first_image(camera_directory) is not None
            for camera_directory in camera_directories
        )

    @classmethod
    def _first_image(cls, directory):
        try:
            for path in Path(directory).rglob("*"):
                if path.is_file() and path.suffix.lower() in cls.IMAGE_EXTENSIONS:
                    return path
        except OSError:
            return None
        return None

    @classmethod
    def _load_or_discover_records(cls, dataset_root):
        manifest_path = cls._find_manifest(dataset_root)
        if manifest_path is not None:
            records = list(cls._iter_manifest_records(manifest_path))
            if not records:
                raise ValueError(f"SPED manifest is empty: {manifest_path}")
            return records, manifest_path

        image_roots = cls._find_camera_image_roots(dataset_root)
        if not image_roots:
            raise ValueError(cls._missing_dataset_message(dataset_root))
        records = cls._scan_camera_tree(dataset_root, image_roots)
        if not records:
            raise ValueError(
                "SPED camera tree contains no supported images: "
                + ", ".join(str(image_root) for image_root in image_roots)
            )

        index_path = Path(dataset_root) / cls.GENERATED_INDEX_FILENAME
        try:
            cls._write_manifest_atomic(index_path, records)
            index_source = index_path
        except OSError as exc:
            warnings.warn(
                f"Could not cache the SPED directory index at {index_path}: {exc}. "
                "This split is usable, but the next construction will scan the tree again.",
                RuntimeWarning,
                stacklevel=2,
            )
            index_source = "+".join(str(image_root) for image_root in image_roots)
        return records, index_source

    @classmethod
    def _scan_camera_tree(cls, dataset_root, image_roots):
        records = []
        for image_root in image_roots:
            for camera_directory in cls._candidate_camera_directories(image_root):
                camera_images = sorted(
                    path
                    for path in camera_directory.rglob("*")
                    if path.is_file() and path.suffix.lower() in cls.IMAGE_EXTENSIONS
                )
                for image_path in camera_images:
                    records.append(
                        (
                            image_path.relative_to(dataset_root).as_posix(),
                            camera_directory.name,
                        )
                    )
        return records

    @classmethod
    def _iter_manifest_records(cls, manifest_path):
        manifest_path = Path(manifest_path)
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"SPED manifest has no header: {manifest_path}")
            normalized_fields = {
                str(field).strip().lower(): field for field in reader.fieldnames
            }
            path_column = next(
                (normalized_fields[name] for name in cls.PATH_COLUMNS if name in normalized_fields),
                None,
            )
            camera_column = next(
                (
                    normalized_fields[name]
                    for name in cls.CAMERA_COLUMNS
                    if name in normalized_fields
                ),
                None,
            )
            if path_column is None or camera_column is None:
                raise ValueError(
                    f"SPED manifest {manifest_path} must contain a path column "
                    f"({cls.PATH_COLUMNS}) and a camera column ({cls.CAMERA_COLUMNS}); "
                    f"found {reader.fieldnames}"
                )

            for line_number, row in enumerate(reader, start=2):
                image_value = str(row.get(path_column, "")).strip().replace("\\", "/")
                camera_id = str(row.get(camera_column, "")).strip()
                if not image_value or not camera_id:
                    raise ValueError(
                        f"SPED manifest has an empty path or camera id at "
                        f"{manifest_path}:{line_number}"
                    )
                relative_path = PurePosixPath(image_value)
                if relative_path.is_absolute() or ".." in relative_path.parts:
                    raise ValueError(
                        f"SPED manifest contains an unsafe image path at "
                        f"{manifest_path}:{line_number}: {image_value!r}"
                    )
                if relative_path.suffix.lower() not in cls.IMAGE_EXTENSIONS:
                    raise ValueError(
                        f"SPED manifest lists an unsupported image extension at "
                        f"{manifest_path}:{line_number}: {image_value!r}"
                    )
                yield relative_path.as_posix(), camera_id

    @classmethod
    def _write_manifest_atomic(cls, destination, records):
        destination = Path(destination)
        temporary = destination.with_name(f"{destination.name}.{os.getpid()}.part")
        temporary.unlink(missing_ok=True)
        try:
            with temporary.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle, lineterminator="\n")
                writer.writerow(("path", "camera_id"))
                writer.writerows(records)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Reproducible camera-level metric-learning split
    # ------------------------------------------------------------------

    @classmethod
    def partition_camera_ids(cls, camera_ids):
        camera_ids = tuple(str(camera_id) for camera_id in camera_ids)
        if len(set(camera_ids)) != len(camera_ids):
            raise ValueError("SPED camera ids must be unique before partitioning")
        if len(camera_ids) < 2:
            raise ValueError("SPED needs at least two cameras to partition")

        ranked = sorted(
            camera_ids,
            key=lambda camera_id: (
                hashlib.sha256(
                    f"{cls.CLASS_SPLIT_VERSION}:{camera_id}".encode("utf-8")
                ).digest(),
                camera_id,
            ),
        )
        midpoint = (len(ranked) + 1) // 2
        return tuple(sorted(ranked[:midpoint])), tuple(sorted(ranked[midpoint:]))

    @staticmethod
    def camera_id_digest(camera_ids):
        payload = "\n".join(str(camera_id) for camera_id in camera_ids).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(self.paths[index])
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
