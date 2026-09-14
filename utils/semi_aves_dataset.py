"""Semi-Aves 2020 datasets and disk-efficient image preparation."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm
from utils.out_of_class_selection import select_out_of_class_records


class _Atomic224Mixin:
    """Small self-contained copy of the project's atomic 224px helpers."""

    TARGET_IMAGE_SIZE = (224, 224)
    JPEG_QUALITY = 90
    USER_AGENT = "metric-learning-semi-aves-downloader/1.0"

    @classmethod
    def _temporary_path(cls, destination):
        destination = Path(destination)
        return destination.with_name(
            f".{destination.name}.{os.getpid()}.part"
        )

    @classmethod
    def _is_target_sized_image(cls, path):
        path = Path(path)
        if not path.is_file():
            return False
        try:
            with Image.open(path) as image:
                return image.size == cls.TARGET_IMAGE_SIZE
        except (OSError, ValueError):
            return False

    @classmethod
    def _resize_image_to_destination(cls, source, destination):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = cls._temporary_path(destination)
        temporary.unlink(missing_ok=True)
        try:
            with Image.open(source) as image:
                image.draft("RGB", cls.TARGET_IMAGE_SIZE)
                rgb_image = image.convert("RGB")
                try:
                    resized = rgb_image.resize(
                        cls.TARGET_IMAGE_SIZE,
                        resample=Image.Resampling.BILINEAR,
                    )
                finally:
                    rgb_image.close()
                try:
                    resized.save(
                        temporary,
                        format="JPEG",
                        quality=cls.JPEG_QUALITY,
                        subsampling=2,
                    )
                finally:
                    resized.close()
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _write_bytes_atomic(cls, destination, content):
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = cls._temporary_path(destination)
        temporary.unlink(missing_ok=True)
        try:
            temporary.write_bytes(content)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _write_json_atomic(cls, destination, payload):
        content = (
            json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            + b"\n"
        )
        cls._write_bytes_atomic(destination, content)

    @classmethod
    def _download_url_to_path(cls, url, destination):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": cls.USER_AGENT},
        )
        with urllib.request.urlopen(request) as response:
            with Path(destination).open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)


class SemiAves(_Atomic224Mixin, Dataset):
    """Semi-Aves with oracle-all and original-known-class protocols.

    ``known`` pools every image whose label falls in the 200 known classes --
    ``l_train_val``, the released challenge test labels, and the in-class
    ``u_train_in`` images under their oracle labels -- and follows the CUB
    convention of holding out half of the classes for final testing. Semi-Aves
    orders its class ids by decreasing frequency, so the halves alternate by
    class id rather than splitting at the midpoint: a contiguous cut would hand
    the entire head of the long tail to the development half. Pooling the
    in-class
    oracle labels lets the semi-supervised label budget decide which development
    images stay labeled instead of hard-coding them as unlabeled. Only the
    out-of-class pool stays unlabeled for the whole run; it is exposed separately
    by :class:`SemiAvesNativeUnlabeledDataset`.

    ``oracle`` pools every released label, including both post-challenge oracle
    files, and assigns complete bird classes to a fixed 500/500 split.
    """

    DATASET_PAGE = "https://github.com/cvl-umass/semi-inat-2020"
    SPLIT_PAGE = (
        "https://github.com/cvl-umass/ssl-evaluation/tree/main/data/semi_aves"
    )
    KAGGLE_PAGE = "https://www.kaggle.com/competitions/semi-inat-2020"
    KAGGLE_COMPETITION = "semi-inat-2020"
    RELEASED_LABELS_COMMIT = "472f904276532c4a28fb03b014aca0314726fe0d"
    RELEASED_LABELS_BASE_URL = (
        "https://raw.githubusercontent.com/cvl-umass/ssl-evaluation/"
        f"{RELEASED_LABELS_COMMIT}/data/semi_aves"
    )

    MODE_KNOWN = "known"
    MODE_ORACLE = "oracle"
    MODES = (MODE_KNOWN, MODE_ORACLE)
    AVAILABLE_SPLITS = ("train", "test", "train+test")

    SPLIT_DIRECTORY = "semi_aves_splits"
    METADATA_FILES = (
        "l_train.txt",
        "val.txt",
        "l_train_val.txt",
        "test.txt",
        "u_train.txt",
        "u_train_in.txt",
        "u_train_in_oracle.txt",
        "u_train_out.txt",
        "u_train_out_oracle.txt",
    )
    EXPECTED_SOURCE_IMAGE_COUNTS = {
        "l_train.txt": 3_959,
        "val.txt": 2_000,
        "l_train_val.txt": 5_959,
        "test.txt": 8_000,
        "u_train.txt": 148_848,
        "u_train_in.txt": 26_640,
        "u_train_in_oracle.txt": 26_640,
        "u_train_out.txt": 122_208,
        "u_train_out_oracle.txt": 122_208,
    }
    IMAGE_DIRECTORIES = (
        "trainval_images",
        "test",
        "u_train_in",
        "u_train_out",
    )
    FILE_RULES = {
        "l_train.txt": ({"trainval_images"}, 0, 199),
        "val.txt": ({"trainval_images"}, 0, 199),
        "l_train_val.txt": ({"trainval_images"}, 0, 199),
        "test.txt": ({"test"}, 0, 199),
        "u_train.txt": ({"u_train_in", "u_train_out"}, -1, -1),
        "u_train_in.txt": ({"u_train_in"}, -1, -1),
        "u_train_in_oracle.txt": ({"u_train_in"}, 0, 199),
        "u_train_out.txt": ({"u_train_out"}, -1, -1),
        "u_train_out_oracle.txt": ({"u_train_out"}, 200, 999),
    }
    ARCHIVE_FILENAME_CANDIDATES = {
        "trainval_images": (
            "l_train_val.tar.gz",
            "trainval_images.tar.gz",
            "train_val.tar.gz",
            "trainval.tar.gz",
        ),
        "test": ("test.tar.gz", "test.tgz"),
        "u_train_in": ("u_train_in.tar.gz", "u_train_in.tgz"),
        "u_train_out": ("u_train_out.tar.gz", "u_train_out.tgz"),
    }

    EXPECTED_KNOWN_CLASS_COUNT = 200
    EXPECTED_OUT_OF_CLASS_COUNT = 800
    EXPECTED_CLASS_COUNT = 1_000
    EXPECTED_UNIQUE_IMAGE_COUNT = 162_807
    LABELED_TRAIN_VAL_IMAGE_COUNT = 5_959
    RELEASED_TEST_IMAGE_COUNT = 8_000
    IN_CLASS_UNLABELED_IMAGE_COUNT = 26_640
    OUT_OF_CLASS_UNLABELED_IMAGE_COUNT = 122_208
    # Every known-class image with a released label, pooled before the known
    # protocol assigns complete classes to its development/test halves.
    KNOWN_POOL_IMAGE_COUNT = (
        LABELED_TRAIN_VAL_IMAGE_COUNT
        + RELEASED_TEST_IMAGE_COUNT
        + IN_CLASS_UNLABELED_IMAGE_COUNT
    )

    ORACLE_DEVELOPMENT_CLASS_COUNT = 500
    ORACLE_TEST_CLASS_COUNT = 500
    KNOWN_DEVELOPMENT_CLASS_COUNT = 100
    KNOWN_TEST_CLASS_COUNT = 100
    CLASS_SPLIT_ALTERNATING = "alternating_class_ids"
    CLASS_SPLIT_SHA256 = "sha256_rank"
    AVAILABLE_CLASS_SPLITS = (CLASS_SPLIT_ALTERNATING, CLASS_SPLIT_SHA256)
    CLASS_SPLIT_BASES = {
        CLASS_SPLIT_ALTERNATING: "alternating_class_ids_like_cub",
        CLASS_SPLIT_SHA256: "sha256_ranked_class_ids",
    }
    # The oracle pool defaults to the same alternating rule the known pool uses.
    # Hash ranking stays available through semi_aves_oracle_hash_500_500, whose
    # version string is unchanged so older records still resolve.
    ORACLE_CLASS_SPLIT_VERSION = (
        "semi_aves_oracle_alternating_class_ids_500_500_v2"
    )
    ORACLE_HASH_CLASS_SPLIT_VERSION = "semi_aves_oracle_sha256_500_500_v1"
    ORACLE_CLASS_SPLIT_VERSIONS = {
        CLASS_SPLIT_ALTERNATING: ORACLE_CLASS_SPLIT_VERSION,
        CLASS_SPLIT_SHA256: ORACLE_HASH_CLASS_SPLIT_VERSION,
    }
    KNOWN_CLASS_SPLIT_VERSION = "semi_aves_known_alternating_class_ids_100_100_v2"
    KNOWN_HASH_CLASS_SPLIT_VERSION = "semi_aves_known_sha256_100_100_v1"
    KNOWN_CLASS_SPLIT_VERSIONS = {
        CLASS_SPLIT_ALTERNATING: KNOWN_CLASS_SPLIT_VERSION,
        CLASS_SPLIT_SHA256: KNOWN_HASH_CLASS_SPLIT_VERSION,
    }
    CUB_PROTOCOL_REFERENCE = (
        "https://github.com/KevinMusgrave/pytorch-metric-learning/"
        "blob/master/src/pytorch_metric_learning/datasets/cub.py"
    )
    OUT_OF_CLASS_SELECTION_VERSION = "semi_aves_out_of_class_sha256_v1"
    STORAGE_VERSION = "semi_aves_released_labels_224_v1"
    COMPLETE_MARKER = ".semi_aves_224_complete.json"

    def __init__(
        self,
        root,
        split="train",
        mode=MODE_KNOWN,
        transform=None,
        target_transform=None,
        download=False,
        class_split=None,
    ):
        self.root = Path(root)
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        class_split = (
            self.CLASS_SPLIT_ALTERNATING if class_split is None else class_split
        )
        if class_split not in self.AVAILABLE_CLASS_SPLITS:
            raise ValueError(
                f"class_split must be one of {self.AVAILABLE_CLASS_SPLITS}, "
                f"got {class_split!r}"
            )
        self.class_split = class_split
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(
                f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}"
            )
        if download and not self.is_ready(self.root):
            self.download_224(self.root)
        if not self.is_ready(self.root):
            raise ValueError(
                "Semi-Aves 2020 224x224 data was not found. Initialize with "
                "download=True, run scripts/download_semi_aves_224.py after "
                "accepting the Kaggle competition terms, or place the four "
                "extracted image directories below the dataset root."
            )

        parsed = self._load_metadata(self.root)
        self.mode = mode
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.class_ids = list(range(self.EXPECTED_CLASS_COUNT))
        self.classes = [f"semi_aves_{class_id:04d}" for class_id in self.class_ids]
        self.class_names = {
            class_id: class_name
            for class_id, class_name in zip(self.class_ids, self.classes)
        }

        if mode == self.MODE_ORACLE:
            records = parsed["oracle_pool"]
            development_classes, test_classes = self.partition_oracle_classes(
                class_split=class_split
            )
            development_set = set(development_classes)
            test_set = set(test_classes)
            if split == "train":
                selected_classes = development_set
            elif split == "test":
                selected_classes = test_set
            else:
                selected_classes = development_set | test_set
            selected_records = [
                record for record in records if record[1] in selected_classes
            ]
            development_sample_count = sum(
                label in development_set for _, label, _ in records
            )
            test_sample_count = len(records) - development_sample_count
            self.development_class_labels = list(development_classes)
            self.test_class_labels = list(test_classes)
            self.class_disjoint_split = True
            self.class_split_info = {
                "source": "pooled_semi_aves_released_oracle_labels",
                "mode": self.MODE_ORACLE,
                "pooled_sources": [
                    "l_train_val",
                    "test",
                    "u_train_in_oracle",
                    "u_train_out_oracle",
                ],
                "pooled_sample_count": len(records),
                "class_disjoint_test": True,
                "class_split_version": self.ORACLE_CLASS_SPLIT_VERSIONS[class_split],
                "class_split": class_split,
                "development_class_count": len(development_classes),
                "test_class_count": len(test_classes),
                "development_sample_count": development_sample_count,
                "test_sample_count": test_sample_count,
                "development_sample_fraction": development_sample_count / len(records),
                "test_sample_fraction": test_sample_count / len(records),
                "development_classes": list(development_classes),
                "held_out_test_classes": list(test_classes),
                "oracle_labels_used": True,
                "native_unlabeled_pool": False,
                "official_image_level_split_used": False,
            }
        else:
            records = parsed["known_pool"]
            development_classes, test_classes = self.partition_known_classes(
                class_split=class_split
            )
            development_set = set(development_classes)
            test_set = set(test_classes)
            if split == "train":
                selected_classes = development_set
            elif split == "test":
                selected_classes = test_set
            else:
                selected_classes = development_set | test_set
            selected_records = [
                record for record in records if record[1] in selected_classes
            ]
            development_sample_count = sum(
                label in development_set for _, label, _ in records
            )
            test_sample_count = len(records) - development_sample_count
            self.development_class_labels = list(development_classes)
            self.test_class_labels = list(test_classes)
            self.class_disjoint_split = True
            self.class_split_info = {
                "source": "pooled_semi_aves_known_class_labels",
                "mode": self.MODE_KNOWN,
                "pooled_sources": [
                    "l_train_val",
                    "test",
                    "u_train_in_oracle",
                ],
                "pooled_sample_count": len(records),
                "class_disjoint_test": True,
                "class_split_version": self.KNOWN_CLASS_SPLIT_VERSIONS[class_split],
                "class_split": class_split,
                "split_basis": self.CLASS_SPLIT_BASES[class_split],
                "protocol_reference": self.CUB_PROTOCOL_REFERENCE,
                "known_class_count": self.EXPECTED_KNOWN_CLASS_COUNT,
                "development_class_count": len(development_classes),
                "test_class_count": len(test_classes),
                "development_sample_count": development_sample_count,
                "test_sample_count": test_sample_count,
                "development_sample_fraction": development_sample_count / len(records),
                "test_sample_fraction": test_sample_count / len(records),
                "development_classes": list(development_classes),
                "held_out_test_classes": list(test_classes),
                "oracle_labels_used": True,
                "in_class_labels_hidden_by": "semi_supervised_label_budget",
                "native_unlabeled_pool": True,
                "native_unlabeled_out_source": "u_train_out_oracle",
                "native_unlabeled_labels_exposed": False,
                "out_of_class_count": self.EXPECTED_OUT_OF_CLASS_COUNT,
                "official_image_level_split_used": False,
            }

        if not selected_records:
            raise ValueError(
                f"Semi-Aves mode={mode!r} split={split!r} contains no images"
            )
        self.paths = [
            str(self.root / Path(*relative_path.parts))
            for relative_path, _, _ in selected_records
        ]
        self.labels = [int(label) for _, label, _ in selected_records]
        self.orig_labels = list(self.labels)
        self.sample_sources = [source for _, _, source in selected_records]

    @classmethod
    def is_ready(cls, root):
        root = Path(root)
        marker_path = root / cls.COMPLETE_MARKER
        required_paths = [
            marker_path,
            *(root / directory for directory in cls.IMAGE_DIRECTORIES),
            *(
                root / cls.SPLIT_DIRECTORY / filename
                for filename in cls.METADATA_FILES
            ),
        ]
        if not all(path.exists() for path in required_paths):
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            image_size = tuple(int(value) for value in marker["image_size"])
            image_count = int(marker["image_count"])
            class_count = int(marker["class_count"])
            storage_version = str(marker["storage_version"])
            source_counts = {
                str(filename): int(count)
                for filename, count in marker["source_image_counts"].items()
            }
        except (
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
            OSError,
        ):
            return False
        return (
            image_size == cls.TARGET_IMAGE_SIZE
            and image_count == cls.EXPECTED_UNIQUE_IMAGE_COUNT
            and class_count == cls.EXPECTED_CLASS_COUNT
            and storage_version == cls.STORAGE_VERSION
            and source_counts == cls.EXPECTED_SOURCE_IMAGE_COUNTS
        )

    @classmethod
    def download_224(cls, root):
        root = Path(root)
        if cls.is_ready(root):
            return root
        root.mkdir(parents=True, exist_ok=True)
        downloader = cls.__new__(cls)
        downloader.root = root
        downloader.download_and_remove()
        return root

    def download_and_remove(self):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_metadata_files()
        parsed = self._load_metadata(self.root)
        records = parsed["oracle_pool"]
        expected_by_directory = {
            directory: set() for directory in self.IMAGE_DIRECTORIES
        }
        for relative_path, _, _ in records:
            expected_by_directory[relative_path.parts[0]].add(
                relative_path.as_posix()
            )

        sources = self._resolve_image_sources(expected_by_directory)
        archive_groups = {}
        stats = {}
        for directory, (source_kind, source_path) in sources.items():
            if source_kind == "directory":
                stats[directory] = self._consume_image_directory(
                    directory,
                    source_path,
                    expected_by_directory[directory],
                )
            else:
                archive_groups.setdefault(Path(source_path), []).append(directory)

        for archive_path, directories in archive_groups.items():
            archive_stats = self._consume_image_archive(
                archive_path,
                {
                    directory: expected_by_directory[directory]
                    for directory in directories
                },
            )
            stats.update(archive_stats)

        image_count = sum(
            int(directory_stats["source_images"])
            for directory_stats in stats.values()
        )
        if image_count != self.EXPECTED_UNIQUE_IMAGE_COUNT:
            raise RuntimeError(
                f"Semi-Aves preparation found {image_count} images; expected "
                f"{self.EXPECTED_UNIQUE_IMAGE_COUNT}"
            )
        marker = {
            "dataset": "SemiAves",
            "dataset_page": self.DATASET_PAGE,
            "split_page": self.SPLIT_PAGE,
            "kaggle_competition": self.KAGGLE_COMPETITION,
            "image_size": list(self.TARGET_IMAGE_SIZE),
            "image_count": image_count,
            "class_count": self.EXPECTED_CLASS_COUNT,
            "known_class_count": self.EXPECTED_KNOWN_CLASS_COUNT,
            "out_of_class_count": self.EXPECTED_OUT_OF_CLASS_COUNT,
            "jpeg_quality": int(self.JPEG_QUALITY),
            "storage_version": self.STORAGE_VERSION,
            "released_labels_commit": self.RELEASED_LABELS_COMMIT,
            "source_image_counts": dict(self.EXPECTED_SOURCE_IMAGE_COUNTS),
            "metadata_sha256": self._metadata_sha256(self.root),
            "written_images": sum(
                int(directory_stats["written_images"])
                for directory_stats in stats.values()
            ),
            "reused_images": sum(
                int(directory_stats["reused_images"])
                for directory_stats in stats.values()
            ),
            "image_sources": {
                directory: str(sources[directory][1])
                for directory in self.IMAGE_DIRECTORIES
            },
            "directory_stats": stats,
            "oracle_class_split_versions": dict(self.ORACLE_CLASS_SPLIT_VERSIONS),
            "known_class_split_version": self.KNOWN_CLASS_SPLIT_VERSION,
            "license_note": (
                "non-commercial research and educational use; original "
                "Semi-Aves and iNaturalist terms apply"
            ),
        }
        self._write_json_atomic(self.root / self.COMPLETE_MARKER, marker)

    def _ensure_metadata_files(self):
        metadata_directory = self.root / self.SPLIT_DIRECTORY
        metadata_directory.mkdir(parents=True, exist_ok=True)
        for filename in self.METADATA_FILES:
            destination = metadata_directory / filename
            if destination.is_file():
                continue
            temporary = self._temporary_path(destination)
            temporary.unlink(missing_ok=True)
            url = f"{self.RELEASED_LABELS_BASE_URL}/{filename}"
            try:
                self._download_url_to_path(url, temporary)
                os.replace(temporary, destination)
            except OSError as error:
                raise RuntimeError(
                    f"Could not download Semi-Aves split metadata from {url}"
                ) from error
            finally:
                temporary.unlink(missing_ok=True)

    @classmethod
    def _load_metadata(cls, root):
        root = Path(root)
        metadata_directory = root / cls.SPLIT_DIRECTORY
        metadata = {}
        for filename in cls.METADATA_FILES:
            path = metadata_directory / filename
            try:
                metadata[filename] = path.read_text(encoding="utf-8")
            except OSError as error:
                raise ValueError(
                    f"Could not read Semi-Aves split file {path}"
                ) from error
        return cls._parse_metadata(metadata)

    @classmethod
    def _parse_metadata(cls, metadata):
        parsed = {}
        for filename in cls.METADATA_FILES:
            try:
                content = metadata[filename]
            except KeyError as error:
                raise ValueError(
                    f"Semi-Aves metadata is missing {filename!r}"
                ) from error
            if not isinstance(content, str):
                raise ValueError(f"Semi-Aves metadata {filename!r} must be text")
            allowed_directories, minimum_label, maximum_label = cls.FILE_RULES[
                filename
            ]
            records = []
            seen_paths = set()
            for line_number, line in enumerate(content.splitlines(), start=1):
                if not line.strip():
                    continue
                tokens = line.split()
                if len(tokens) != 2:
                    raise ValueError(
                        f"Expected '<image path> <class id>' in {filename}:"
                        f"{line_number}, got {line!r}"
                    )
                relative_path = cls._validated_record_path(
                    tokens[0],
                    allowed_directories,
                )
                try:
                    label = int(tokens[1])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid class id in {filename}:{line_number}: "
                        f"{tokens[1]!r}"
                    ) from error
                if not minimum_label <= label <= maximum_label:
                    raise ValueError(
                        f"Class id {label} in {filename}:{line_number} is "
                        f"outside [{minimum_label}, {maximum_label}]"
                    )
                path_key = relative_path.as_posix()
                if path_key in seen_paths:
                    raise ValueError(
                        f"Duplicate Semi-Aves path {path_key!r} in {filename}"
                    )
                seen_paths.add(path_key)
                records.append((relative_path, label, filename.removesuffix(".txt")))
            expected_count = cls.EXPECTED_SOURCE_IMAGE_COUNTS[filename]
            if len(records) != expected_count:
                raise ValueError(
                    f"Semi-Aves {filename} has {len(records)} records; "
                    f"expected {expected_count}"
                )
            parsed[filename] = tuple(records)

        def records_by_path(filename):
            return {
                relative_path.as_posix(): int(label)
                for relative_path, label, _ in parsed[filename]
            }

        l_train = records_by_path("l_train.txt")
        val = records_by_path("val.txt")
        if set(l_train) & set(val):
            raise ValueError("Semi-Aves l_train and val image paths overlap")
        if {**l_train, **val} != records_by_path("l_train_val.txt"):
            raise ValueError(
                "Semi-Aves l_train_val is not the union of l_train and val"
            )

        u_train_in = records_by_path("u_train_in.txt")
        u_train_in_oracle = records_by_path("u_train_in_oracle.txt")
        u_train_out = records_by_path("u_train_out.txt")
        u_train_out_oracle = records_by_path("u_train_out_oracle.txt")
        if set(u_train_in) != set(u_train_in_oracle):
            raise ValueError(
                "Semi-Aves u_train_in and its oracle file contain different images"
            )
        if set(u_train_out) != set(u_train_out_oracle):
            raise ValueError(
                "Semi-Aves u_train_out and its oracle file contain different images"
            )
        if set(u_train_in) & set(u_train_out):
            raise ValueError("Semi-Aves in- and out-of-class unlabeled paths overlap")
        if set(records_by_path("u_train.txt")) != (
            set(u_train_in) | set(u_train_out)
        ):
            raise ValueError(
                "Semi-Aves u_train is not the union of u_train_in and u_train_out"
            )

        known_labeled = []
        for relative_path, label, _ in parsed["l_train_val.txt"]:
            path_key = relative_path.as_posix()
            source = "l_train" if path_key in l_train else "val"
            known_labeled.append((relative_path, label, source))
        known_labeled = tuple(known_labeled)
        known_test = tuple(
            (relative_path, label, "test")
            for relative_path, label, _ in parsed["test.txt"]
        )
        oracle_in = tuple(
            (relative_path, label, "u_train_in_oracle")
            for relative_path, label, _ in parsed["u_train_in_oracle.txt"]
        )
        oracle_out = tuple(
            (relative_path, label, "u_train_out_oracle")
            for relative_path, label, _ in parsed["u_train_out_oracle.txt"]
        )
        # Every image with a known-class label is pooled before splitting: the
        # in-class unlabeled images carry released oracle labels, so the known
        # protocol treats them as labeled and lets the semi-supervised label
        # budget hide labels afterwards. Only the out-of-class pool stays
        # unlabeled for the whole run. The official image-level train/test
        # boundary is deliberately discarded so the split can be class disjoint.
        known_pool = known_labeled + known_test + oracle_in
        oracle_pool = known_pool + oracle_out
        path_sources = {}
        for relative_path, _, source in oracle_pool:
            path_key = relative_path.as_posix()
            if path_key in path_sources:
                raise ValueError(
                    f"Duplicate Semi-Aves image {path_key!r} in {source!r} "
                    f"and {path_sources[path_key]!r}"
                )
            path_sources[path_key] = source
        if len(known_pool) != cls.KNOWN_POOL_IMAGE_COUNT:
            raise ValueError(
                f"Semi-Aves known-class pool has {len(known_pool)} images; "
                f"expected {cls.KNOWN_POOL_IMAGE_COUNT}"
            )
        if len(oracle_pool) != cls.EXPECTED_UNIQUE_IMAGE_COUNT:
            raise ValueError(
                f"Semi-Aves oracle pool has {len(oracle_pool)} images; expected "
                f"{cls.EXPECTED_UNIQUE_IMAGE_COUNT}"
            )

        known_classes = {label for _, label, _ in known_pool}
        out_classes = {label for _, label, _ in oracle_out}
        if known_classes != set(range(cls.EXPECTED_KNOWN_CLASS_COUNT)):
            raise ValueError("Semi-Aves known classes must be contiguous 0..199")
        if out_classes != set(
            range(cls.EXPECTED_KNOWN_CLASS_COUNT, cls.EXPECTED_CLASS_COUNT)
        ):
            raise ValueError("Semi-Aves out-of-class labels must be contiguous 200..999")

        return {
            **parsed,
            "known_labeled": known_labeled,
            "known_test": known_test,
            "known_pool": known_pool,
            "oracle_in": oracle_in,
            "oracle_out": oracle_out,
            "oracle_pool": oracle_pool,
        }

    @classmethod
    def _validated_record_path(cls, raw_path, allowed_directories):
        path = PurePosixPath(str(raw_path).replace("\\", "/"))
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) < 2
            or path.parts[0] not in allowed_directories
            or path.suffix.lower() not in {".jpg", ".jpeg"}
        ):
            raise ValueError(f"Unsafe Semi-Aves image path: {raw_path!r}")
        return path

    @classmethod
    def _partition_classes(cls, class_ids, development_count, version, class_split):
        """Halve ``class_ids`` by alternating them or by ranking their hashes.

        Alternating is the default because Semi-Aves numbers its classes from
        most to least frequent, so taking every other id splits that frequency
        ordering evenly without letting per-class counts drive the partition.
        Hash ranking is arbitrary but reproducible, and stays available for
        checking that a result does not depend on the id ordering.
        """

        class_ids = tuple(sorted(int(class_id) for class_id in class_ids))
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("Semi-Aves class ids must be unique")
        if class_split == cls.CLASS_SPLIT_SHA256:
            ranked = sorted(
                class_ids,
                key=lambda class_id: hashlib.sha256(
                    f"{version}:{class_id}".encode("utf-8")
                ).digest(),
            )
            development = tuple(sorted(ranked[:development_count]))
            test = tuple(sorted(ranked[development_count:]))
        elif class_split == cls.CLASS_SPLIT_ALTERNATING:
            development = class_ids[0::2]
            test = class_ids[1::2]
            if len(development) != development_count:
                raise ValueError(
                    "Semi-Aves alternating split needs an even class count for "
                    f"{development_count} development classes; got {len(class_ids)}"
                )
        else:
            raise ValueError(
                f"class_split must be one of {cls.AVAILABLE_CLASS_SPLITS}, "
                f"got {class_split!r}"
            )
        if set(development) & set(test):
            raise RuntimeError("Semi-Aves class partition is not disjoint")
        if set(development) | set(test) != set(class_ids):
            raise RuntimeError("Semi-Aves class partition is incomplete")
        return development, test

    @classmethod
    def partition_oracle_classes(cls, class_split=None):
        class_split = cls.CLASS_SPLIT_ALTERNATING if class_split is None else class_split
        development, test = cls._partition_classes(
            range(cls.EXPECTED_CLASS_COUNT),
            cls.ORACLE_DEVELOPMENT_CLASS_COUNT,
            cls.ORACLE_CLASS_SPLIT_VERSIONS.get(class_split),
            class_split,
        )
        if len(test) != cls.ORACLE_TEST_CLASS_COUNT:
            raise RuntimeError(
                f"Semi-Aves oracle test split has {len(test)} classes; expected "
                f"{cls.ORACLE_TEST_CLASS_COUNT}"
            )
        return development, test

    @classmethod
    def partition_known_classes(cls, class_split=None):
        """Split the known classes into CUB-sized halves, 100/100.

        Semi-Aves numbers its classes from most to least frequent, so CUB's
        contiguous cut is badly unbalanced here: ids 0-99 carry 26,348 of the
        40,599 pooled images. The default takes every other id, which keeps the
        halves at 100 classes each and splits the frequency ordering evenly
        (20,358 / 20,241 images) without making the partition depend on
        per-class counts.

        ``sha256_rank`` halves the same ids by hash instead. It is arbitrary but
        reproducible, and unlike alternation it leaves the image balance to
        chance -- with only 200 frequency-ordered classes that is a real spread,
        so use it as a robustness check rather than as the headline split.
        """

        class_split = cls.CLASS_SPLIT_ALTERNATING if class_split is None else class_split
        class_ids = tuple(range(cls.EXPECTED_KNOWN_CLASS_COUNT))
        development, test = cls._partition_classes(
            class_ids,
            cls.KNOWN_DEVELOPMENT_CLASS_COUNT,
            cls.KNOWN_CLASS_SPLIT_VERSIONS.get(class_split),
            class_split,
        )
        if len(development) != cls.KNOWN_DEVELOPMENT_CLASS_COUNT:
            raise RuntimeError(
                f"Semi-Aves known development split has {len(development)} "
                f"classes; expected {cls.KNOWN_DEVELOPMENT_CLASS_COUNT}"
            )
        if len(test) != cls.KNOWN_TEST_CLASS_COUNT:
            raise RuntimeError(
                f"Semi-Aves known test split has {len(test)} classes; expected "
                f"{cls.KNOWN_TEST_CLASS_COUNT}"
            )
        if set(development) & set(test):
            raise RuntimeError("Semi-Aves known class partition is not disjoint")
        if set(development) | set(test) != set(class_ids):
            raise RuntimeError("Semi-Aves known class partition is incomplete")
        return development, test

    @classmethod
    def select_out_of_class_records(cls, records, fraction, seed=0):
        """Keep a class-balanced ``fraction`` of the 800-species U-out pool."""

        return select_out_of_class_records(
            records,
            fraction,
            seed,
            selection_version=cls.OUT_OF_CLASS_SELECTION_VERSION,
            dataset_label="Semi-Aves",
        )

    @classmethod
    def _metadata_sha256(cls, root):
        root = Path(root)
        digest = hashlib.sha256()
        for filename in cls.METADATA_FILES:
            digest.update(filename.encode("utf-8"))
            digest.update(b"\0")
            digest.update(
                (root / cls.SPLIT_DIRECTORY / filename).read_bytes()
            )
            digest.update(b"\0")
        return digest.hexdigest()

    def _resolve_image_sources(self, expected_by_directory):
        sources = self._discover_image_sources(
            self.root,
            expected_by_directory,
        )
        if len(sources) == len(self.IMAGE_DIRECTORIES):
            return sources

        try:
            import kagglehub
            kagglehub.login()
        except ImportError as error:
            missing = sorted(set(self.IMAGE_DIRECTORIES) - set(sources))
            raise RuntimeError(
                "kagglehub is required to download Semi-Aves competition "
                f"images; missing sources for {missing}. Install the project "
                "requirements or place extracted images/archives in the "
                "dataset root."
            ) from error
        try:
            competition_path = Path(
                kagglehub.competition_download(self.KAGGLE_COMPETITION)
            )
        except Exception as error:
            raise RuntimeError(
                "Could not download Semi-Aves from Kaggle. Authenticate, "
                f"accept the rules at {self.KAGGLE_PAGE}, and retry."
            ) from error
        discovered = self._discover_image_sources(
            competition_path,
            expected_by_directory,
        )
        sources.update(
            {
                directory: source
                for directory, source in discovered.items()
                if directory not in sources
            }
        )
        missing = sorted(set(self.IMAGE_DIRECTORIES) - set(sources))
        if missing:
            raise RuntimeError(
                f"Could not locate Semi-Aves image sources for {missing} "
                f"below {self.root} or KaggleHub path {competition_path}"
            )
        return sources

    @classmethod
    def _discover_image_sources(cls, location, expected_by_directory):
        location = Path(location)
        sources = {}
        if location.is_file():
            for directory in cls._archive_directories(location):
                sources[directory] = ("archive", location)
            return sources
        if not location.is_dir():
            return sources

        base_candidates = (
            location,
            location / "semi_aves",
            location / "semi-aves",
        )
        for directory in cls.IMAGE_DIRECTORIES:
            for base in base_candidates:
                candidate = base / directory
                if cls._directory_contains_expected(
                    candidate,
                    directory,
                    expected_by_directory[directory],
                ):
                    sources[directory] = ("directory", candidate)
                    break
            if directory in sources:
                continue
            directory_matches = sorted(
                path
                for path in location.rglob(directory)
                if path.is_dir()
                and cls._directory_contains_expected(
                    path,
                    directory,
                    expected_by_directory[directory],
                )
            )
            if len(directory_matches) == 1:
                sources[directory] = ("directory", directory_matches[0])

        archive_files = cls._archive_files_below(location)
        for directory, filenames in cls.ARCHIVE_FILENAME_CANDIDATES.items():
            if directory in sources:
                continue
            matches = [path for path in archive_files if path.name in filenames]
            if len(matches) == 1:
                sources[directory] = ("archive", matches[0])

        missing = set(cls.IMAGE_DIRECTORIES) - set(sources)
        if missing:
            for archive_path in archive_files:
                archive_directories = cls._archive_directories(archive_path)
                for directory in sorted(missing & archive_directories):
                    sources[directory] = ("archive", archive_path)
                missing = set(cls.IMAGE_DIRECTORIES) - set(sources)
                if not missing:
                    break
        return sources

    @staticmethod
    def _archive_files_below(location):
        patterns = ("*.tar.gz", "*.tgz", "*.tar", "*.zip")
        return sorted(
            {
                path
                for pattern in patterns
                for path in Path(location).rglob(pattern)
                if path.is_file()
            }
        )

    @classmethod
    def _archive_directories(cls, archive_path):
        archive_path = Path(archive_path)
        directories = set()
        try:
            if zipfile.is_zipfile(archive_path):
                with zipfile.ZipFile(archive_path, "r") as archive:
                    for info in archive.infolist():
                        directory = cls._member_image_directory(info.filename)
                        if directory is not None:
                            directories.add(directory)
                return directories
            if tarfile.is_tarfile(archive_path):
                with tarfile.open(archive_path, mode="r:*") as archive:
                    for member in archive:
                        directory = cls._member_image_directory(member.name)
                        if directory is not None:
                            directories.add(directory)
                            # Official tarballs contain one top-level source.
                            return directories
        except (OSError, tarfile.TarError, zipfile.BadZipFile):
            return set()
        return directories

    @classmethod
    def _member_image_directory(cls, member_name):
        path = PurePosixPath(str(member_name).replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Unsafe path in Semi-Aves image archive: {member_name!r}"
            )
        for part in path.parts:
            if part in cls.IMAGE_DIRECTORIES:
                return part
        return None

    @staticmethod
    def _directory_contains_expected(source_directory, directory, expected_paths):
        source_directory = Path(source_directory)
        if not source_directory.is_dir():
            return False
        return all(
            (
                source_directory
                / Path(*PurePosixPath(path_key).parts[1:])
            ).is_file()
            for path_key in expected_paths
            if PurePosixPath(path_key).parts[0] == directory
        )

    def _consume_image_directory(
        self,
        directory,
        source_directory,
        expected_paths,
    ):
        source_directory = Path(source_directory)
        written_images = 0
        reused_images = 0
        with tqdm(
            total=len(expected_paths),
            desc=f"Saving 224x224 Semi-Aves {directory}",
            unit="image",
            disable=None,
        ) as progress:
            for path_key in sorted(expected_paths):
                relative_path = PurePosixPath(path_key)
                source_path = source_directory / Path(*relative_path.parts[1:])
                if not source_path.is_file():
                    raise RuntimeError(
                        f"Semi-Aves source {source_directory} is missing "
                        f"{path_key!r}"
                    )
                destination = self.root / Path(*relative_path.parts)
                if self._is_target_sized_image(destination):
                    reused_images += 1
                else:
                    if source_path.resolve() == destination.resolve():
                        with io.BytesIO(source_path.read_bytes()) as image_source:
                            self._resize_image_to_destination(
                                image_source,
                                destination,
                            )
                    else:
                        self._resize_image_to_destination(source_path, destination)
                    written_images += 1
                progress.update(1)
        return {
            "source": str(source_directory),
            "source_images": len(expected_paths),
            "written_images": written_images,
            "reused_images": reused_images,
        }

    def _consume_image_archive(self, archive_path, expected_by_directory):
        archive_path = Path(archive_path)
        seen = {directory: set() for directory in expected_by_directory}
        written = {directory: 0 for directory in expected_by_directory}
        reused = {directory: 0 for directory in expected_by_directory}
        total = sum(len(paths) for paths in expected_by_directory.values())
        with tqdm(
            total=total,
            desc="Saving 224x224 Semi-Aves archive images",
            unit="image",
            disable=None,
        ) as progress:
            if zipfile.is_zipfile(archive_path):
                try:
                    with zipfile.ZipFile(archive_path, "r") as archive:
                        for info in archive.infolist():
                            if info.is_dir():
                                continue
                            relative_path = self._archive_member_path(info.filename)
                            if relative_path is None:
                                continue
                            directory = relative_path.parts[0]
                            if directory not in expected_by_directory:
                                continue
                            path_key = relative_path.as_posix()
                            self._validate_archive_record(
                                directory,
                                path_key,
                                expected_by_directory,
                                seen,
                            )
                            destination = self.root / Path(*relative_path.parts)
                            if self._is_target_sized_image(destination):
                                reused[directory] += 1
                            else:
                                with archive.open(info, "r") as source:
                                    self._resize_image_to_destination(
                                        source,
                                        destination,
                                    )
                                written[directory] += 1
                            progress.update(1)
                except zipfile.BadZipFile as error:
                    raise RuntimeError(
                        f"Could not read Semi-Aves ZIP archive {archive_path}"
                    ) from error
            else:
                try:
                    with tarfile.open(archive_path, mode="r|*") as archive:
                        for member in archive:
                            if not member.isfile():
                                continue
                            relative_path = self._archive_member_path(member.name)
                            if relative_path is None:
                                continue
                            directory = relative_path.parts[0]
                            if directory not in expected_by_directory:
                                continue
                            path_key = relative_path.as_posix()
                            self._validate_archive_record(
                                directory,
                                path_key,
                                expected_by_directory,
                                seen,
                            )
                            destination = self.root / Path(*relative_path.parts)
                            if self._is_target_sized_image(destination):
                                reused[directory] += 1
                            else:
                                source = archive.extractfile(member)
                                if source is None:
                                    raise RuntimeError(
                                        f"Could not read {member.name!r} from "
                                        f"{archive_path}"
                                    )
                                with source:
                                    self._resize_image_to_destination(
                                        source,
                                        destination,
                                    )
                                written[directory] += 1
                            progress.update(1)
                except tarfile.TarError as error:
                    raise RuntimeError(
                        f"Could not read Semi-Aves tar archive {archive_path}"
                    ) from error

        stats = {}
        for directory, expected_paths in expected_by_directory.items():
            missing = expected_paths - seen[directory]
            if missing:
                raise RuntimeError(
                    f"Semi-Aves archive {archive_path} is missing "
                    f"{len(missing)} {directory} images; examples: "
                    f"{sorted(missing)[:5]}"
                )
            stats[directory] = {
                "source": str(archive_path),
                "source_images": len(seen[directory]),
                "written_images": written[directory],
                "reused_images": reused[directory],
            }
        return stats

    @staticmethod
    def _validate_archive_record(
        directory,
        path_key,
        expected_by_directory,
        seen,
    ):
        if path_key not in expected_by_directory[directory]:
            raise RuntimeError(
                f"Unexpected Semi-Aves image in archive: {path_key!r}"
            )
        if path_key in seen[directory]:
            raise RuntimeError(
                f"Duplicate Semi-Aves image in archive: {path_key!r}"
            )
        seen[directory].add(path_key)

    @classmethod
    def _archive_member_path(cls, member_name):
        path = PurePosixPath(str(member_name).replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Unsafe path in Semi-Aves image archive: {member_name!r}"
            )
        image_directory_index = None
        for index, part in enumerate(path.parts):
            if part in cls.IMAGE_DIRECTORIES:
                image_directory_index = index
                break
        if image_directory_index is None:
            return None
        relative_path = PurePosixPath(*path.parts[image_directory_index:])
        if (
            len(relative_path.parts) < 2
            or relative_path.suffix.lower() not in {".jpg", ".jpeg"}
        ):
            return None
        return relative_path

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


class SemiAvesNativeUnlabeledDataset(Dataset):
    """Semi-Aves out-of-class unlabeled pool with every oracle label hidden.

    The in-class images already carry oracle labels inside the known protocol's
    pooled development set, so this dataset supplies only the out-of-class
    (``u_train_out``) images that no protocol ever labels.

    ``out_of_class_fraction`` controls the class-mismatch level: it is the share
    of the released U-out pool that joins the unlabeled data, and the remaining
    out-of-class images are dropped rather than kept with a hidden label. 1.0
    reproduces the original protocol and 0.0 leaves no out-of-class data at all.
    """

    def __init__(
        self,
        root,
        transform=None,
        out_of_class_fraction=1.0,
        out_of_class_seed=0,
    ):
        self.root = Path(root)
        if not SemiAves.is_ready(self.root):
            raise ValueError(
                f"Prepared Semi-Aves data was not found at {self.root}"
            )
        out_of_class_fraction = float(out_of_class_fraction)
        if not 0.0 <= out_of_class_fraction <= 1.0:
            raise ValueError(
                "Semi-Aves out_of_class_fraction must be in [0, 1]: "
                f"{out_of_class_fraction}"
            )
        parsed = SemiAves._load_metadata(self.root)
        available_records = parsed["oracle_out"]
        records = SemiAves.select_out_of_class_records(
            available_records,
            fraction=out_of_class_fraction,
            seed=out_of_class_seed,
        )
        self.paths = [
            str(self.root / Path(*relative_path.parts))
            for relative_path, _, _ in records
        ]
        self.labels = [-1] * len(records)
        self.orig_labels = [int(label) for _, label, _ in records]
        self.oracle_labels = list(self.orig_labels)
        self.sample_sources = ["u_train_out" for _ in records]
        self.transform = transform
        self.out_of_class_fraction = out_of_class_fraction
        self.out_of_class_seed = int(out_of_class_seed)
        self.filter_info = {
            "mode": "semi_aves_out_of_class_only",
            "candidate_source": "released_oracle_files_hidden_at_runtime",
            "labels_exposed_to_training": False,
            "kept_images": len(records),
            "kept_out_of_class_classes": len({label for _, label, _ in records}),
            "out_of_class_behavior": (
                "all_original_u_train_out_images_unlabeled"
                if out_of_class_fraction == 1.0
                else "class_balanced_out_of_class_subset_remainder_excluded"
            ),
            "out_of_class_fraction_requested": out_of_class_fraction,
            "out_of_class_fraction_realized": (
                len(records) / len(available_records)
                if available_records
                else 0.0
            ),
            "available_out_of_class_images": len(available_records),
            "excluded_out_of_class_images": (
                len(available_records) - len(records)
            ),
            "out_of_class_selection_seed": int(out_of_class_seed),
            "out_of_class_selection_version": (
                SemiAves.OUT_OF_CLASS_SELECTION_VERSION
            ),
        }

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(self.paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, -1


SemiAves2020 = SemiAves
