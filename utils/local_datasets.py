import hashlib
import json
import os
import shutil
import statistics
import tarfile
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image
from pytorch_metric_learning.datasets.cub import CUB as _CUB
from pytorch_metric_learning.datasets.inaturalist2018 import (
    INaturalist2018 as _INaturalist2018,
)
from pytorch_metric_learning.datasets.sop import StanfordOnlineProducts as _StanfordOnlineProducts
from tqdm import tqdm
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision.datasets import CIFAR100 as _CIFAR100
from torchvision.datasets import Flowers102 as _Flowers102
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torch.utils.data import Dataset

from .out_of_class_selection import select_out_of_class_records
from .pittsburgh_dataset import (
    Pittsburgh30k,
    Pittsburgh250k,
)
from .semi_aves_dataset import (
    SemiAves,
    SemiAves2020,
    SemiAvesNativeUnlabeledDataset,
)
from .sped_dataset import SPED


class _CIFARSplitMixin:
    def __init__(self, root, split="train", transform=None, target_transform=None, download=False):
        if split not in {"train", "test"}:
            raise ValueError(f"split must be 'train' or 'test', got {split!r}")

        super().__init__(
            root=root,
            train=split == "train",
            transform=transform,
            target_transform=target_transform,
            download=download,
        )
        self.split = split
        self.labels = list(self.targets)


class CIFAR10(_CIFARSplitMixin, _CIFAR10):
    pass


class CIFAR100(_CIFARSplitMixin, _CIFAR100):
    pass


class _DownloadProgressReader:
    """Update a byte progress bar as a streaming archive is read."""

    def __init__(self, response, progress):
        self.response = response
        self.progress = progress

    def read(self, size=-1):
        data = self.response.read(size)
        self.progress.update(len(data))
        return data


class _DinoSizedImageDownloadMixin:
    """Shared atomic 224 x 224 JPEG download helpers."""

    TARGET_IMAGE_SIZE = (224, 224)
    JPEG_QUALITY = 90
    USER_AGENT = "metric-learning-dino-image-downloader/1.0"

    @classmethod
    def _is_target_sized_image(cls, path):
        if not path.is_file():
            return False
        try:
            with Image.open(path) as image:
                return image.size == cls.TARGET_IMAGE_SIZE
        except (OSError, ValueError):
            return False

    @classmethod
    def _resize_image_to_destination(
        cls,
        source,
        destination,
        *,
        crop_bottom_pixels=0,
    ):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = cls._temporary_path(destination)
        temporary.unlink(missing_ok=True)
        try:
            with Image.open(source) as image:
                if crop_bottom_pixels:
                    crop_bottom_pixels = int(crop_bottom_pixels)
                    if crop_bottom_pixels < 0:
                        raise ValueError("crop_bottom_pixels must be non-negative")
                    if image.height <= crop_bottom_pixels:
                        raise ValueError(
                            f"Cannot crop {crop_bottom_pixels} pixels from an "
                            f"image with height {image.height}"
                        )
                    prepared_image = image.crop(
                        (0, 0, image.width, image.height - crop_bottom_pixels)
                    )
                else:
                    image.draft("RGB", cls.TARGET_IMAGE_SIZE)
                    prepared_image = image
                try:
                    rgb_image = prepared_image.convert("RGB")
                finally:
                    if prepared_image is not image:
                        prepared_image.close()
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
    def _open_url(cls, url):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": cls.USER_AGENT},
        )
        return urllib.request.urlopen(request)

    @staticmethod
    def _byte_progress(response, description):
        content_length = response.headers.get("Content-Length")
        total = int(content_length) if content_length is not None else None
        return tqdm(
            total=total,
            desc=description,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            disable=None,
        )

    def _download_url_to_path(self, url, destination, description):
        with self._open_url(url) as response:
            with self._byte_progress(response, description) as progress:
                with destination.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        progress.update(len(chunk))

    @classmethod
    def _write_bytes_atomic(cls, destination, content):
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = cls._temporary_path(destination)
        temporary.unlink(missing_ok=True)
        try:
            with temporary.open("wb") as output:
                output.write(content)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    @classmethod
    def _write_json_atomic(cls, destination, payload):
        serialized = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8") + b"\n"
        cls._write_bytes_atomic(destination, serialized)

    @staticmethod
    def _temporary_path(destination):
        return destination.with_name(f".{destination.name}.{os.getpid()}.part")


class INaturalist2018(_DinoSizedImageDownloadMixin, _INaturalist2018):
    """iNaturalist 2018 with a disk-efficient DINO-sized downloader.

    The upstream dataset is distributed as one 120 GB gzip-compressed tar
    archive. This loader never writes that archive or its full-resolution
    members to disk: it streams each JPEG, resizes it to 224 x 224 in memory,
    and atomically writes only the resized JPEG at the path expected by
    ``pytorch_metric_learning.datasets.INaturalist2018``.
    """

    IMAGE_DIRECTORY = "train_val2018"
    COMPLETE_MARKER = ".inaturalist2018_224_complete.json"
    TRAIN_ANNOTATION = "train2018.json"
    VAL_ANNOTATION = "val2018.json"
    SPLIT_DIRECTORY = "Inat_dataset_splits"
    TRAIN_SPLIT = "Inaturalist_train_set1.txt"
    TEST_SPLIT = "Inaturalist_test_set1.txt"

    def __init__(
        self,
        root,
        split="train+test",
        transform=None,
        target_transform=None,
        download=False,
    ):
        root = Path(root)
        if download and not self.is_ready(root):
            root.mkdir(parents=True, exist_ok=True)
            self.root = str(root)
            self.download_and_remove()
            download = False

        super().__init__(
            root=str(root),
            split=split,
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    @classmethod
    def is_ready(cls, root):
        """Return whether a completed 224 x 224 dataset is present."""

        root = Path(root)
        marker_path = root / cls.COMPLETE_MARKER
        required_paths = (
            root / cls.TRAIN_ANNOTATION,
            root / cls.VAL_ANNOTATION,
            root / cls.SPLIT_DIRECTORY / cls.TRAIN_SPLIT,
            root / cls.SPLIT_DIRECTORY / cls.TEST_SPLIT,
            root / cls.IMAGE_DIRECTORY,
            marker_path,
        )
        if not all(path.exists() for path in required_paths):
            return False

        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            image_size = tuple(int(value) for value in marker["image_size"])
            image_count = int(marker["image_count"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            return False
        return image_size == cls.TARGET_IMAGE_SIZE and image_count > 0

    @classmethod
    def download_224(cls, root):
        """Download the dataset without constructing and parsing a split."""

        root = Path(root)
        if cls.is_ready(root):
            return root
        root.mkdir(parents=True, exist_ok=True)
        downloader = cls.__new__(cls)
        downloader.root = str(root)
        downloader.download_and_remove()
        return root

    def download_and_remove(self):
        """Stream the official archive and retain only 224 x 224 images."""

        root = Path(self.root)
        root.mkdir(parents=True, exist_ok=True)
        self._ensure_annotation(self.TRAIN_ANN_URL, self.TRAIN_ANNOTATION)
        self._ensure_annotation(self.VAL_ANN_URL, self.VAL_ANNOTATION)
        self._ensure_split_files()

        expected_image_count = self._count_split_images()
        if expected_image_count <= 0:
            raise RuntimeError("The iNaturalist metric-learning split files contain no images")

        stats = self._stream_and_resize_images(expected_image_count)
        marker = {
            "dataset": "INaturalist2018",
            "image_size": list(self.TARGET_IMAGE_SIZE),
            "image_count": int(stats["archive_images"]),
            "jpeg_quality": int(self.JPEG_QUALITY),
            "written_images": int(stats["written_images"]),
            "reused_images": int(stats["reused_images"]),
            "source_archive": self.IMG_DOWNLOAD_URL,
            "storage_layout": f"{self.IMAGE_DIRECTORY}/<supercategory>/<category>/<image>.jpg",
        }
        self._write_json_atomic(root / self.COMPLETE_MARKER, marker)

    def _ensure_annotation(self, url, filename):
        destination = Path(self.root) / filename
        if destination.is_file():
            return

        temporary = self._temporary_path(destination)
        temporary.unlink(missing_ok=True)
        found = False
        try:
            with self._open_url(url) as response:
                with self._byte_progress(response, f"Downloading {filename}") as progress:
                    reader = _DownloadProgressReader(response, progress)
                    with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                        for member in archive:
                            if not member.isfile() or PurePosixPath(member.name).name != filename:
                                continue
                            source = archive.extractfile(member)
                            if source is None:
                                raise RuntimeError(f"Could not read {filename} from {url}")
                            destination.parent.mkdir(parents=True, exist_ok=True)
                            with source, temporary.open("wb") as output:
                                shutil.copyfileobj(source, output, length=1024 * 1024)
                            found = True
                            break
            if not found:
                raise RuntimeError(f"{url} did not contain the expected file {filename}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)

    def _ensure_split_files(self):
        split_directory = Path(self.root) / self.SPLIT_DIRECTORY
        train_destination = split_directory / self.TRAIN_SPLIT
        test_destination = split_directory / self.TEST_SPLIT
        if train_destination.is_file() and test_destination.is_file():
            return

        split_directory.mkdir(parents=True, exist_ok=True)
        archive_path = Path(self.root) / ".Inat_dataset_splits.zip.part"
        archive_path.unlink(missing_ok=True)
        try:
            self._download_url_to_path(
                self.SPLITS_URL,
                archive_path,
                description="Downloading iNaturalist metric-learning splits",
            )
            with zipfile.ZipFile(archive_path, "r") as archive:
                for filename, destination in (
                    (self.TRAIN_SPLIT, train_destination),
                    (self.TEST_SPLIT, test_destination),
                ):
                    member_name = self._find_zip_member(archive, filename)
                    self._write_bytes_atomic(destination, archive.read(member_name))
        finally:
            archive_path.unlink(missing_ok=True)

    def _count_split_images(self):
        image_count = 0
        for filename in (self.TRAIN_SPLIT, self.TEST_SPLIT):
            split_path = Path(self.root) / self.SPLIT_DIRECTORY / filename
            with split_path.open("r", encoding="utf-8") as split_file:
                for line_number, line in enumerate(split_file, start=1):
                    image_name = line.strip()
                    if not image_name:
                        continue
                    relative_path = self._validated_image_path(image_name)
                    if relative_path.suffix.lower() not in {".jpg", ".jpeg"}:
                        raise ValueError(
                            f"Unsupported image extension in {split_path}:{line_number}: "
                            f"{relative_path.suffix!r}"
                        )
                    image_count += 1
        return image_count

    def _stream_and_resize_images(self, expected_image_count):
        archive_images = 0
        written_images = 0
        reused_images = 0
        with self._open_url(self.IMG_DOWNLOAD_URL) as response:
            with self._byte_progress(
                response,
                "Streaming official iNaturalist image archive",
            ) as byte_progress:
                reader = _DownloadProgressReader(response, byte_progress)
                with tqdm(
                    total=expected_image_count,
                    desc="Saving 224x224 iNaturalist images",
                    unit="image",
                    disable=None,
                ) as image_progress:
                    with tarfile.open(fileobj=reader, mode="r|gz") as archive:
                        for member in archive:
                            relative_path = self._image_member_path(member)
                            if relative_path is None:
                                continue
                            archive_images += 1
                            destination = Path(self.root) / relative_path
                            if self._is_target_sized_image(destination):
                                reused_images += 1
                            else:
                                source = archive.extractfile(member)
                                if source is None:
                                    raise RuntimeError(
                                        f"Could not read image {member.name!r} from the archive"
                                    )
                                with source:
                                    self._resize_image_to_destination(source, destination)
                                written_images += 1
                            image_progress.update(1)

        if archive_images != expected_image_count:
            raise RuntimeError(
                "The iNaturalist image archive did not match the metric-learning split files: "
                f"expected {expected_image_count} images, found {archive_images}. "
                "The completion marker was not written; rerun the downloader to retry."
            )
        return {
            "archive_images": archive_images,
            "written_images": written_images,
            "reused_images": reused_images,
        }

    @classmethod
    def _image_member_path(cls, member):
        if not member.isfile():
            return None
        path = PurePosixPath(member.name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in iNaturalist image archive: {member.name!r}")
        if not path.parts or path.parts[0] != cls.IMAGE_DIRECTORY:
            return None
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            return None
        return Path(*path.parts)

    @classmethod
    def _validated_image_path(cls, image_name):
        path = PurePosixPath(str(image_name).replace("\\", "/"))
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != cls.IMAGE_DIRECTORY
        ):
            raise ValueError(f"Unsafe path in iNaturalist split file: {image_name!r}")
        return Path(*path.parts)

    @staticmethod
    def _find_zip_member(archive, filename):
        matches = [
            member_name
            for member_name in archive.namelist()
            if PurePosixPath(member_name).name == filename
        ]
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one {filename!r} in the split archive, found {len(matches)}"
            )
        member_path = PurePosixPath(matches[0])
        if member_path.is_absolute() or ".." in member_path.parts:
            raise ValueError(f"Unsafe path in iNaturalist split archive: {matches[0]!r}")
        return matches[0]


class SemiINaturalist2021(_DinoSizedImageDownloadMixin, Dataset):
    """Semi-iNaturalist 2021 with a fixed class-disjoint retrieval split.

    The challenge's five released labeled sources are pooled before splitting:
    ``l_train``, ``val``, ``test``, ``u_train_in``, and ``u_train_out``.  The
    latter two are the post-challenge oracle labels for the images in the
    original ``u_train`` archive.

    Whole species alternate by class id into two halves of roughly equal size,
    the same rule the Semi-Aves known protocol uses. Semi-iNat publishes
    ``kingdom`` as its coarse supercategory, and the stride runs within each
    kingdom separately so that every kingdom is halved exactly rather than
    however its ids happen to fall on the parity. That keeps the development and
    test halves within a few thousand of the 343,219 pooled images
    (172,158 / 171,061) without making the partition depend on per-class counts.

    KaggleHub may return either the four original tarballs or directories
    containing their extracted images. Both layouts are processed directly,
    and only atomic 224 x 224 JPEGs are written to the dataset root.
    """

    DATASET_PAGE = "https://github.com/cvl-umass/semi-inat-2021"
    KAGGLE_PAGE = "https://www.kaggle.com/c/semi-inat-2021"
    KAGGLE_COMPETITION = "semi-inat-2021"
    RELEASED_LABELS_COMMIT = "472f904276532c4a28fb03b014aca0314726fe0d"
    RELEASED_LABELS_BASE_URL = (
        "https://raw.githubusercontent.com/cvl-umass/ssl-evaluation/"
        f"{RELEASED_LABELS_COMMIT}/data/semi_inat"
    )
    SPLIT_DIRECTORY = "semi_inat_splits"
    TAXONOMY_FILE = "all_taxa_info.json"
    SOURCE_SPLITS = (
        "l_train",
        "val",
        "test",
        "u_train_in",
        "u_train_out",
    )
    SOURCE_SPLIT_FILES = {
        source: f"{source}.txt"
        for source in SOURCE_SPLITS
    }
    SOURCE_IMAGE_DIRECTORIES = {
        "l_train": "l_train",
        "val": "val",
        "test": "test",
        "u_train_in": "u_train",
        "u_train_out": "u_train",
    }
    EXPECTED_SOURCE_IMAGE_COUNTS = {
        "l_train": 9721,
        "val": 4050,
        "test": 16200,
        "u_train_in": 91336,
        "u_train_out": 221912,
    }
    ARCHIVE_SOURCES = (
        (
            "l_train",
            "l_train.tar.gz",
            "http://vis-www.cs.umass.edu/semi-inat-2021/l_train.tar.gz",
        ),
        (
            "val",
            "val.tar.gz",
            "http://vis-www.cs.umass.edu/semi-inat-2021/val.tar.gz",
        ),
        (
            "test",
            "test.tar.gz",
            "http://vis-www.cs.umass.edu/semi-inat-2021/test.tar.gz",
        ),
        (
            "u_train",
            "u_train.tar.gz",
            "http://vis-www.cs.umass.edu/semi-inat-2021/u_train.tar.gz",
        ),
    )
    EXPECTED_ARCHIVE_IMAGE_COUNTS = {
        "l_train": 9721,
        "val": 4050,
        "test": 16200,
        "u_train": 313248,
    }
    EXPECTED_IMAGE_COUNT = 343219
    EXPECTED_CLASS_COUNT = 2439
    DEVELOPMENT_CLASS_FRACTION = 0.50
    TEST_CLASS_FRACTION = 0.50
    SPLIT_SUPERCATEGORY_FIELD = "kingdom"
    CLASS_SPLIT_VERSION = "semi_inat_alternating_class_ids_50_50_v2"
    KNOWN_CLASS_SPLIT_VERSION = "semi_inat_known_alternating_class_ids_810_v1"
    MODE_KNOWN = "known"
    MODE_ORACLE = "oracle"
    MODES = (MODE_KNOWN, MODE_ORACLE)
    # The four sources whose species the challenge labels; ``u_train_out`` holds
    # every other species and is the pool the known protocol never labels.
    KNOWN_SOURCE_SPLITS = ("l_train", "val", "test", "u_train_in")
    OUT_OF_CLASS_SOURCE_SPLIT = "u_train_out"
    EXPECTED_KNOWN_CLASS_COUNT = 810
    EXPECTED_OUT_OF_CLASS_COUNT = 1629
    OUT_OF_CLASS_SELECTION_VERSION = "semi_inat_out_of_class_sha256_v1"
    COMPLETE_MARKER = ".semi_inaturalist2021_224_complete.json"
    AVAILABLE_SPLITS = ("train", "test", "train+test")

    def __init__(
        self,
        root,
        split="train+test",
        mode=MODE_ORACLE,
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = Path(root)
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}, got {mode!r}")
        if download and not self.is_ready(self.root):
            self.download_224(self.root)
        if not self.is_ready(self.root):
            raise ValueError(
                "Semi-iNaturalist 2021 224x224 data was not found. Initialize "
                "the dataset with download=True, run "
                "scripts/download_semi_inaturalist2021_224.py, or place the "
                "four official archives in the dataset root before retrying."
            )
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}")

        taxonomy, records = self._load_metadata(self.root)
        if mode == self.MODE_KNOWN:
            # Only the labeled challenge species are pooled; the semi-supervised
            # label budget then decides which of their images stay labeled,
            # exactly as the Semi-Aves known protocol does.
            known_sources = set(self.KNOWN_SOURCE_SPLITS)
            records = [
                record for record in records if record[2] in known_sources
            ]
            known_labels = {int(label) for _, label, _ in records}
            if known_labels != set(range(self.EXPECTED_KNOWN_CLASS_COUNT)):
                raise ValueError(
                    "Semi-iNaturalist 2021 known sources must carry exactly "
                    f"species 0-{self.EXPECTED_KNOWN_CLASS_COUNT - 1}, found "
                    f"{len(known_labels)} species"
                )
            development_classes, test_classes = self.partition_known_classes(
                taxonomy
            )
        else:
            development_classes, test_classes = self.partition_class_ids(taxonomy)
        development_set = set(development_classes)
        test_set = set(test_classes)
        if split == "train":
            selected_classes = development_set
        elif split == "test":
            selected_classes = test_set
        else:
            selected_classes = development_set | test_set

        selected_records = [
            (relative_path, label, source)
            for relative_path, label, source in records
            if label in selected_classes
        ]
        if not selected_records:
            raise ValueError(
                f"Semi-iNaturalist 2021 mode={mode!r} split={split!r} "
                "contains no images"
            )

        self.mode = mode
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.paths = [
            str(self.root / Path(*relative_path.parts))
            for relative_path, _, _ in selected_records
        ]
        self.labels = [int(label) for _, label, _ in selected_records]
        self.orig_labels = list(self.labels)
        self.sample_sources = [source for _, _, source in selected_records]
        self.class_ids = sorted(taxonomy)
        self.classes = [
            str(taxonomy[class_id]["species"])
            for class_id in self.class_ids
        ]
        self.class_names = {
            class_id: str(taxonomy[class_id]["species"])
            for class_id in self.class_ids
        }
        self.taxonomy = taxonomy
        self.development_class_labels = list(development_classes)
        self.test_class_labels = list(test_classes)
        self.class_disjoint_split = True
        self.class_split_info = self._build_class_split_info(
            taxonomy=taxonomy,
            development_classes=development_classes,
            test_classes=test_classes,
            mode=mode,
        )

    @classmethod
    def is_ready(cls, root):
        """Return whether the complete resized dataset and metadata are present."""

        root = Path(root)
        marker_path = root / cls.COMPLETE_MARKER
        required_paths = [
            root / cls.SPLIT_DIRECTORY / cls.TAXONOMY_FILE,
            marker_path,
        ]
        required_paths.extend(
            root / cls.SPLIT_DIRECTORY / filename
            for filename in cls.SOURCE_SPLIT_FILES.values()
        )
        required_paths.extend(
            root / directory
            for directory in sorted(set(cls.SOURCE_IMAGE_DIRECTORIES.values()))
        )
        if not all(path.exists() for path in required_paths):
            return False

        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            image_size = tuple(int(value) for value in marker["image_size"])
            image_count = int(marker["image_count"])
            class_count = int(marker["class_count"])
            source_counts = {
                str(key): int(value)
                for key, value in marker["source_image_counts"].items()
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            return False
        return (
            image_size == cls.TARGET_IMAGE_SIZE
            and image_count == cls.EXPECTED_IMAGE_COUNT
            and class_count == cls.EXPECTED_CLASS_COUNT
            # The marker attests that every archive was fetched and resized, so
            # it deliberately omits CLASS_SPLIT_VERSION: repartitioning the
            # classes must not make a prepared root look undownloaded.
            and source_counts == cls.EXPECTED_SOURCE_IMAGE_COUNTS
        )

    @classmethod
    def download_224(cls, root):
        """Download every released source while retaining only 224px images."""

        root = Path(root)
        if cls.is_ready(root):
            return root
        root.mkdir(parents=True, exist_ok=True)
        downloader = cls.__new__(cls)
        downloader.root = root
        downloader.download_and_remove()
        return root

    def download_and_remove(self):
        """Fetch metadata, stream all four image archives, and mark completion."""

        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._ensure_metadata_files()
        taxonomy, records = self._load_metadata(self.root)
        development_classes, test_classes = self.partition_class_ids(taxonomy)
        expected_paths = self._archive_expected_paths(records)
        image_sources = self._resolve_image_sources()

        archive_stats = {}
        for archive_key, archive_filename, _ in self.ARCHIVE_SOURCES:
            archive_stats[archive_key] = self._resize_image_source(
                archive_key=archive_key,
                source_path=image_sources[archive_key],
                expected_paths=expected_paths[archive_key],
            )

        archive_image_count = sum(
            stats["archive_images"]
            for stats in archive_stats.values()
        )
        if archive_image_count != self.EXPECTED_IMAGE_COUNT:
            raise RuntimeError(
                "The Semi-iNaturalist archives were incomplete: "
                f"expected {self.EXPECTED_IMAGE_COUNT} images, "
                f"found {archive_image_count}"
            )

        marker = {
            "dataset": "SemiINaturalist2021",
            "image_size": list(self.TARGET_IMAGE_SIZE),
            "image_count": archive_image_count,
            "class_count": len(taxonomy),
            "jpeg_quality": int(self.JPEG_QUALITY),
            "written_images": int(
                sum(stats["written_images"] for stats in archive_stats.values())
            ),
            "reused_images": int(
                sum(stats["reused_images"] for stats in archive_stats.values())
            ),
            "source_archives": {
                archive_key: stats["source"]
                for archive_key, stats in archive_stats.items()
            },
            "kaggle_competition": self.KAGGLE_COMPETITION,
            "source_image_counts": dict(self.EXPECTED_SOURCE_IMAGE_COUNTS),
            "pooled_sources": list(self.SOURCE_SPLITS),
            "released_labels_commit": self.RELEASED_LABELS_COMMIT,
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "split_supercategory_field": self.SPLIT_SUPERCATEGORY_FIELD,
            "development_class_fraction": self.DEVELOPMENT_CLASS_FRACTION,
            "test_class_fraction": self.TEST_CLASS_FRACTION,
            "development_class_count": len(development_classes),
            "test_class_count": len(test_classes),
            "development_classes": list(development_classes),
            "held_out_test_classes": list(test_classes),
            "archive_stats": archive_stats,
        }
        self._write_json_atomic(self.root / self.COMPLETE_MARKER, marker)

    def _ensure_metadata_files(self):
        metadata_directory = self.root / self.SPLIT_DIRECTORY
        metadata_directory.mkdir(parents=True, exist_ok=True)
        filenames = [
            self.TAXONOMY_FILE,
            *self.SOURCE_SPLIT_FILES.values(),
        ]
        for filename in filenames:
            destination = metadata_directory / filename
            if destination.is_file():
                continue
            temporary = self._temporary_path(destination)
            temporary.unlink(missing_ok=True)
            url = f"{self.RELEASED_LABELS_BASE_URL}/{filename}"
            try:
                self._download_url_to_path(
                    url,
                    temporary,
                    description=f"Downloading Semi-iNat metadata {filename}",
                )
                os.replace(temporary, destination)
            except OSError as error:
                raise RuntimeError(
                    f"Could not download Semi-iNat metadata from {url}"
                ) from error
            finally:
                temporary.unlink(missing_ok=True)

    @classmethod
    def _load_metadata(cls, root):
        root = Path(root)
        taxonomy_path = root / cls.SPLIT_DIRECTORY / cls.TAXONOMY_FILE
        try:
            raw_taxonomy = json.loads(taxonomy_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as error:
            raise ValueError(f"Could not read Semi-iNat taxonomy file {taxonomy_path}") from error
        if not isinstance(raw_taxonomy, dict):
            raise ValueError(f"Semi-iNat taxonomy must be a JSON object: {taxonomy_path}")

        taxonomy = {}
        for raw_class_id, taxon in raw_taxonomy.items():
            try:
                class_id = int(raw_class_id)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid class id {raw_class_id!r} in {taxonomy_path}"
                ) from error
            if class_id in taxonomy:
                raise ValueError(f"Duplicate class id {class_id} in {taxonomy_path}")
            if not isinstance(taxon, dict):
                raise ValueError(
                    f"Taxonomy entry for class {class_id} must be an object"
                )
            for required_field in ("species", cls.SPLIT_SUPERCATEGORY_FIELD):
                if not str(taxon.get(required_field, "")).strip():
                    raise ValueError(
                        f"Taxonomy entry for class {class_id} has no "
                        f"{required_field!r}"
                    )
            taxonomy[class_id] = dict(taxon)

        if len(taxonomy) != cls.EXPECTED_CLASS_COUNT:
            raise ValueError(
                "Semi-iNat taxonomy has an unexpected number of species: "
                f"expected {cls.EXPECTED_CLASS_COUNT}, found {len(taxonomy)}"
            )
        expected_class_ids = set(range(cls.EXPECTED_CLASS_COUNT))
        if set(taxonomy) != expected_class_ids:
            missing = sorted(expected_class_ids - set(taxonomy))
            unexpected = sorted(set(taxonomy) - expected_class_ids)
            raise ValueError(
                "Semi-iNat class ids must be contiguous from zero: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )

        records = []
        seen_paths = {}
        for source in cls.SOURCE_SPLITS:
            split_path = (
                root
                / cls.SPLIT_DIRECTORY
                / cls.SOURCE_SPLIT_FILES[source]
            )
            source_records = []
            try:
                split_lines = split_path.read_text(encoding="utf-8").splitlines()
            except OSError as error:
                raise ValueError(f"Could not read Semi-iNat label file {split_path}") from error
            for line_number, line in enumerate(split_lines, start=1):
                if not line.strip():
                    continue
                tokens = line.split()
                if len(tokens) != 2:
                    raise ValueError(
                        f"Expected '<image path> <class id>' in "
                        f"{split_path}:{line_number}, got {line!r}"
                    )
                relative_path = cls._validated_record_path(
                    tokens[0],
                    expected_directory=cls.SOURCE_IMAGE_DIRECTORIES[source],
                )
                try:
                    label = int(tokens[1])
                except ValueError as error:
                    raise ValueError(
                        f"Invalid class id in {split_path}:{line_number}: "
                        f"{tokens[1]!r}"
                    ) from error
                if label not in taxonomy:
                    raise ValueError(
                        f"Unknown class id {label} in {split_path}:{line_number}"
                    )
                path_key = relative_path.as_posix()
                if path_key in seen_paths:
                    raise ValueError(
                        f"Duplicate Semi-iNat image path {path_key!r} in "
                        f"{source!r} and {seen_paths[path_key]!r}"
                    )
                seen_paths[path_key] = source
                source_records.append((relative_path, label, source))

            expected_count = cls.EXPECTED_SOURCE_IMAGE_COUNTS[source]
            if len(source_records) != expected_count:
                raise ValueError(
                    f"Semi-iNat source {source!r} has {len(source_records)} "
                    f"records; expected {expected_count}"
                )
            records.extend(source_records)

        if len(records) != cls.EXPECTED_IMAGE_COUNT:
            raise ValueError(
                f"Semi-iNat metadata contains {len(records)} images; "
                f"expected {cls.EXPECTED_IMAGE_COUNT}"
            )
        record_classes = {label for _, label, _ in records}
        if record_classes != set(taxonomy):
            missing = sorted(set(taxonomy) - record_classes)
            unexpected = sorted(record_classes - set(taxonomy))
            raise ValueError(
                "Semi-iNat image labels and taxonomy classes do not match: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        return taxonomy, records

    @classmethod
    def _alternate_within_supercategory(cls, taxonomy, class_ids):
        """Halve ``class_ids`` by taking every other id inside each kingdom."""

        grouped_classes = {}
        for class_id in class_ids:
            supercategory = str(
                taxonomy[int(class_id)][cls.SPLIT_SUPERCATEGORY_FIELD]
            )
            grouped_classes.setdefault(supercategory, []).append(int(class_id))

        development_classes = []
        test_classes = []
        for supercategory in sorted(grouped_classes):
            ranked_class_ids = sorted(grouped_classes[supercategory])
            development_classes.extend(ranked_class_ids[0::2])
            test_classes.extend(ranked_class_ids[1::2])
        return tuple(sorted(development_classes)), tuple(sorted(test_classes))

    @classmethod
    def partition_class_ids(cls, taxonomy):
        """Return the fixed 50/50 split that alternates by id within a kingdom."""

        development_classes, test_classes = cls._alternate_within_supercategory(
            taxonomy, sorted(taxonomy)
        )
        if set(development_classes) & set(test_classes):
            raise RuntimeError("Semi-iNat class partition is not disjoint")
        if set(development_classes) | set(test_classes) != set(taxonomy):
            raise RuntimeError("Semi-iNat class partition does not cover every species")
        return development_classes, test_classes

    @classmethod
    def partition_known_classes(cls, taxonomy):
        """Halve only the 810 species the challenge labels.

        Striding inside each kingdom leaves the halves one species apart per
        kingdom with an odd count, so the split is 406/404 rather than an exact
        405/405; that buys a closer image balance (50.5/49.5 against 49.3/50.7)
        and an evenly halved Fungi. The out-of-class species stay out of the
        partition entirely: they are never labeled by this protocol and reach
        training only through :class:`SemiINatNativeUnlabeledDataset`.
        """

        class_ids = tuple(range(cls.EXPECTED_KNOWN_CLASS_COUNT))
        missing = sorted(set(class_ids) - set(taxonomy))
        if missing:
            raise RuntimeError(
                f"Semi-iNat taxonomy is missing known species {missing[:10]}"
            )
        development, test = cls._alternate_within_supercategory(
            taxonomy, class_ids
        )
        supercategory_count = len(
            {
                str(taxonomy[class_id][cls.SPLIT_SUPERCATEGORY_FIELD])
                for class_id in class_ids
            }
        )
        if abs(len(development) - len(test)) > supercategory_count:
            raise RuntimeError(
                f"Semi-iNat known halves differ by "
                f"{abs(len(development) - len(test))} classes; striding inside "
                f"{supercategory_count} kingdoms cannot exceed one each"
            )
        if set(development) & set(test):
            raise RuntimeError("Semi-iNat known class partition is not disjoint")
        if set(development) | set(test) != set(class_ids):
            raise RuntimeError("Semi-iNat known class partition is incomplete")
        return development, test

    @classmethod
    def _build_class_split_info(
        cls,
        taxonomy,
        development_classes,
        test_classes,
        mode=MODE_ORACLE,
    ):
        development_set = set(development_classes)
        test_set = set(test_classes)
        # Only the partitioned species are reported: the known protocol leaves
        # the out-of-class species out of the split entirely.
        supercategory_counts = {}
        for class_id in sorted(development_set | test_set):
            supercategory = str(
                taxonomy[class_id][cls.SPLIT_SUPERCATEGORY_FIELD]
            )
            counts = supercategory_counts.setdefault(
                supercategory,
                {"total": 0, "development": 0, "test": 0},
            )
            counts["total"] += 1
            if class_id in development_set:
                counts["development"] += 1
            else:
                counts["test"] += 1

        if mode == cls.MODE_KNOWN:
            pooled_sources = list(cls.KNOWN_SOURCE_SPLITS)
        else:
            pooled_sources = list(cls.SOURCE_SPLITS)
        info = {
            "source": (
                "pooled_known_species_class_disjoint_810"
                if mode == cls.MODE_KNOWN
                else "pooled_released_labels_class_disjoint_50_50"
            ),
            "mode": mode,
            "pooled_sources": pooled_sources,
            "source_sample_counts": {
                source: count
                for source, count in cls.EXPECTED_SOURCE_IMAGE_COUNTS.items()
                if source in pooled_sources
            },
            "pooled_sample_count": sum(
                count
                for source, count in cls.EXPECTED_SOURCE_IMAGE_COUNTS.items()
                if source in pooled_sources
            ),
            "class_disjoint_test": True,
            "class_split_version": (
                cls.KNOWN_CLASS_SPLIT_VERSION
                if mode == cls.MODE_KNOWN
                else cls.CLASS_SPLIT_VERSION
            ),
            "split_basis": "alternating_class_ids",
            "split_supercategory_field": cls.SPLIT_SUPERCATEGORY_FIELD,
            "development_class_fraction": cls.DEVELOPMENT_CLASS_FRACTION,
            "test_class_fraction": cls.TEST_CLASS_FRACTION,
            "development_class_count": len(development_classes),
            "test_class_count": len(test_classes),
            "development_classes": list(development_classes),
            "held_out_test_classes": list(test_classes),
            "supercategory_class_counts": supercategory_counts,
            "oracle_labels_used": True,
            "official_image_level_split_used": False,
        }
        if mode == cls.MODE_KNOWN:
            info.update(
                {
                    "known_class_count": cls.EXPECTED_KNOWN_CLASS_COUNT,
                    "in_class_labels_hidden_by": "semi_supervised_label_budget",
                    "native_unlabeled_pool": True,
                    "native_unlabeled_out_source": cls.OUT_OF_CLASS_SOURCE_SPLIT,
                    "native_unlabeled_labels_exposed": False,
                    "out_of_class_count": cls.EXPECTED_OUT_OF_CLASS_COUNT,
                }
            )
        else:
            info["native_unlabeled_pool"] = False
        return info

    @classmethod
    def _archive_expected_paths(cls, records):
        expected_paths = {
            archive_key: set()
            for archive_key, _, _ in cls.ARCHIVE_SOURCES
        }
        for relative_path, _, _ in records:
            archive_key = relative_path.parts[0]
            if archive_key not in expected_paths:
                raise ValueError(
                    f"No Semi-iNat archive is configured for {relative_path}"
                )
            expected_paths[archive_key].add(relative_path.as_posix())

        for archive_key, paths in expected_paths.items():
            expected_count = cls.EXPECTED_ARCHIVE_IMAGE_COUNTS[archive_key]
            if len(paths) != expected_count:
                raise ValueError(
                    f"Semi-iNat archive {archive_key!r} maps to {len(paths)} "
                    f"metadata records; expected {expected_count}"
                )
        return expected_paths

    def _resolve_image_sources(self):
        image_sources = {}
        missing_archives = []
        for archive_key, archive_filename, _ in self.ARCHIVE_SOURCES:
            local_archive = self.root / archive_filename
            if local_archive.is_file():
                image_sources[archive_key] = local_archive
            else:
                missing_archives.append((archive_key, archive_filename))

        if not missing_archives:
            return image_sources

        try:
            import kagglehub
            kagglehub.login()
        except ImportError as error:
            raise RuntimeError(
                "kagglehub is required to download the Semi-iNaturalist 2021 "
                "competition files. Install the project requirements or run "
                "`pip install kagglehub`."
            ) from error

        try:
            competition_path = Path(
                kagglehub.competition_download(self.KAGGLE_COMPETITION)
            )
        except Exception as error:
            raise RuntimeError(
                "Could not download the Semi-iNaturalist 2021 competition "
                f"with kagglehub. Authenticate with Kaggle and accept the "
                f"competition rules at {self.KAGGLE_PAGE}, then rerun."
            ) from error

        for archive_key, archive_filename in missing_archives:
            image_sources[archive_key] = self._find_competition_source(
                competition_path,
                archive_key,
                archive_filename,
            )
        return image_sources

    @staticmethod
    def _find_competition_source(
        competition_path,
        archive_key,
        archive_filename,
    ):
        competition_path = Path(competition_path)
        if competition_path.is_file():
            if competition_path.name == archive_filename:
                return competition_path
            raise RuntimeError(
                f"KaggleHub returned {competition_path}, but "
                f"{archive_filename!r} was requested."
            )
        if not competition_path.is_dir():
            raise RuntimeError(
                f"KaggleHub returned a missing competition path: "
                f"{competition_path}"
            )

        direct_path = competition_path / archive_filename
        if direct_path.is_file():
            return direct_path

        archive_matches = sorted(
            path
            for path in competition_path.rglob(archive_filename)
            if path.is_file()
        )
        if len(archive_matches) == 1:
            return archive_matches[0]
        if len(archive_matches) > 1:
            raise RuntimeError(
                f"Expected exactly one {archive_filename!r} below the "
                f"KaggleHub competition path {competition_path}, found "
                f"{len(archive_matches)}."
            )

        extracted_directory_candidates = (
            competition_path / archive_key / archive_key,
            competition_path / archive_key,
        )
        for extracted_directory in extracted_directory_candidates:
            if extracted_directory.is_dir():
                return extracted_directory

        raise RuntimeError(
            f"KaggleHub returned neither {archive_filename!r} nor an "
            f"extracted {archive_key!r} image directory below "
            f"{competition_path}."
        )

    def _resize_image_source(
        self,
        archive_key,
        source_path,
        expected_paths,
    ):
        source_path = Path(source_path)
        if source_path.is_dir():
            stats = self._consume_image_directory(
                source_directory=source_path,
                archive_key=archive_key,
                expected_paths=expected_paths,
            )
        elif source_path.is_file():
            with source_path.open("rb") as archive_stream:
                stats = self._consume_image_archive(
                    archive_stream=archive_stream,
                    archive_key=archive_key,
                    expected_paths=expected_paths,
                )
        else:
            raise RuntimeError(
                f"Semi-iNat image source does not exist: {source_path}"
            )
        stats["source"] = str(source_path)
        return stats

    def _consume_image_directory(
        self,
        source_directory,
        archive_key,
        expected_paths,
    ):
        written_images = 0
        reused_images = 0
        with tqdm(
            total=len(expected_paths),
            desc=f"Saving 224x224 Semi-iNat {archive_key} images",
            unit="image",
            disable=None,
        ) as image_progress:
            for path_key in expected_paths:
                relative_path = PurePosixPath(path_key)
                if (
                    not relative_path.parts
                    or relative_path.parts[0] != archive_key
                ):
                    raise RuntimeError(
                        f"Unexpected {archive_key!r} metadata path: "
                        f"{path_key!r}"
                    )
                source_path = (
                    source_directory
                    / Path(*relative_path.parts[1:])
                )
                if not source_path.is_file():
                    raise RuntimeError(
                        f"The extracted Semi-iNat {archive_key!r} source is "
                        f"missing {path_key!r} at {source_path}."
                    )

                destination = self.root / Path(*relative_path.parts)
                if self._is_target_sized_image(destination):
                    reused_images += 1
                else:
                    with source_path.open("rb") as source:
                        self._resize_image_to_destination(
                            source,
                            destination,
                        )
                    written_images += 1
                image_progress.update(1)

        return {
            "archive_images": len(expected_paths),
            "written_images": written_images,
            "reused_images": reused_images,
        }

    def _consume_image_archive(
        self,
        archive_stream,
        archive_key,
        expected_paths,
    ):
        seen_paths = set()
        written_images = 0
        reused_images = 0
        with tqdm(
            total=len(expected_paths),
            desc=f"Saving 224x224 Semi-iNat {archive_key} images",
            unit="image",
            disable=None,
        ) as image_progress:
            with tarfile.open(fileobj=archive_stream, mode="r|gz") as archive:
                for member in archive:
                    relative_path = self._image_member_path(
                        member,
                        expected_directory=archive_key,
                    )
                    if relative_path is None:
                        continue
                    path_key = relative_path.as_posix()
                    if path_key not in expected_paths:
                        raise RuntimeError(
                            f"Unexpected image {path_key!r} in the Semi-iNat "
                            f"{archive_key!r} archive"
                        )
                    if path_key in seen_paths:
                        raise RuntimeError(
                            f"Duplicate image {path_key!r} in the Semi-iNat "
                            f"{archive_key!r} archive"
                        )
                    seen_paths.add(path_key)

                    destination = self.root / Path(*relative_path.parts)
                    if self._is_target_sized_image(destination):
                        reused_images += 1
                    else:
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuntimeError(
                                f"Could not read {member.name!r} from the "
                                f"Semi-iNat {archive_key!r} archive"
                            )
                        with source:
                            self._resize_image_to_destination(source, destination)
                        written_images += 1
                    image_progress.update(1)

        missing_paths = expected_paths - seen_paths
        if missing_paths:
            raise RuntimeError(
                f"The Semi-iNat {archive_key!r} archive is missing "
                f"{len(missing_paths)} labeled images; examples: "
                f"{sorted(missing_paths)[:5]}"
            )
        return {
            "archive_images": len(seen_paths),
            "written_images": written_images,
            "reused_images": reused_images,
        }

    @classmethod
    def _image_member_path(cls, member, expected_directory):
        if not member.isfile():
            return None
        path = PurePosixPath(member.name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Unsafe path in Semi-iNat image archive: {member.name!r}"
            )
        if not path.parts or path.parts[0] != expected_directory:
            return None
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            return None
        return path

    @classmethod
    def _validated_record_path(cls, image_name, expected_directory):
        path = PurePosixPath(str(image_name).replace("\\", "/"))
        if (
            path.is_absolute()
            or ".." in path.parts
            or not path.parts
            or path.parts[0] != expected_directory
        ):
            raise ValueError(
                f"Unsafe or unexpected Semi-iNat image path: {image_name!r}"
            )
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            raise ValueError(
                f"Unsupported Semi-iNat image extension: {image_name!r}"
            )
        return path

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


class SemiINatNativeUnlabeledDataset(Dataset):
    """Semi-iNat out-of-class unlabeled pool with every oracle label hidden.

    The known protocol already pools the labeled species' images, so this
    dataset supplies only the ``u_train_out`` images -- the 1,629 species the
    challenge never labels.

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
        if not SemiINaturalist2021.is_ready(self.root):
            raise ValueError(
                f"Prepared Semi-iNaturalist 2021 data was not found at {self.root}"
            )
        out_of_class_fraction = float(out_of_class_fraction)
        if not 0.0 <= out_of_class_fraction <= 1.0:
            raise ValueError(
                "Semi-iNat out_of_class_fraction must be in [0, 1]: "
                f"{out_of_class_fraction}"
            )
        _, all_records = SemiINaturalist2021._load_metadata(self.root)
        out_of_class_source = SemiINaturalist2021.OUT_OF_CLASS_SOURCE_SPLIT
        available_records = [
            record for record in all_records if record[2] == out_of_class_source
        ]
        records = select_out_of_class_records(
            available_records,
            out_of_class_fraction,
            out_of_class_seed,
            selection_version=(
                SemiINaturalist2021.OUT_OF_CLASS_SELECTION_VERSION
            ),
            dataset_label="Semi-iNat",
        )
        self.paths = [
            str(self.root / Path(*relative_path.parts))
            for relative_path, _, _ in records
        ]
        self.labels = [-1] * len(records)
        self.orig_labels = [int(label) for _, label, _ in records]
        self.oracle_labels = list(self.orig_labels)
        self.sample_sources = [out_of_class_source for _ in records]
        self.transform = transform
        self.out_of_class_fraction = out_of_class_fraction
        self.out_of_class_seed = int(out_of_class_seed)
        self.filter_info = {
            "mode": "semi_inat_out_of_class_only",
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
                SemiINaturalist2021.OUT_OF_CLASS_SELECTION_VERSION
            ),
        }

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(self.paths[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, -1


# Short spelling used by some Semi-iNat papers and configurations.
SemiINat2021 = SemiINaturalist2021


class Food101(_DinoSizedImageDownloadMixin, Dataset):
    """Food-101 pooled into a fixed class-disjoint metric-learning split.

    The official image-level train and test sources are pooled because both
    contain all 101 food classes. Complete classes are then assigned to a fixed
    71/30 development/test partition, so retrieval testing uses only classes
    unseen during training.
    """

    KAGGLE_DATASET_PAGE = "https://www.kaggle.com/datasets/dansbecker/food-101"
    KAGGLE_DATASET = "dansbecker/food-101"
    KAGGLE_ARCHIVE_FILENAME = "food-101.zip"
    IMAGE_DIRECTORY = "images"
    COMPLETE_MARKER = ".food101_224_complete.json"
    EXPECTED_IMAGE_COUNT = 101000
    EXPECTED_CLASS_COUNT = 101
    EXPECTED_IMAGES_PER_CLASS = 1000
    EXPECTED_OFFICIAL_SPLIT_COUNTS = {
        "train": 75750,
        "test": 25250,
    }
    EXPECTED_OFFICIAL_IMAGES_PER_CLASS = {
        "train": 750,
        "test": 250,
    }
    DEVELOPMENT_CLASS_FRACTION = 0.70
    TEST_CLASS_FRACTION = 0.30
    DEVELOPMENT_CLASS_COUNT = 71
    TEST_CLASS_COUNT = 30
    CLASS_SPLIT_VERSION = "food101_sha256_71_30_v1"
    AVAILABLE_SPLITS = ("train", "test", "train+test")

    def __init__(
        self,
        root,
        split="train+test",
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = Path(root)
        if download and not self.is_ready(self.root):
            self.download_224(self.root)
        if not self.is_ready(self.root):
            raise ValueError(
                "Food-101 224x224 data was not found. Initialize the dataset "
                "with download=True or run scripts/download_food101_224.py."
            )
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(
                f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}"
            )

        image_root = self.root / self.IMAGE_DIRECTORY
        class_directories = sorted(
            path for path in image_root.iterdir() if path.is_dir()
        )
        class_names = [path.name for path in class_directories]
        if len(class_names) != self.EXPECTED_CLASS_COUNT:
            raise ValueError(
                f"Food-101 must contain {self.EXPECTED_CLASS_COUNT} class "
                f"directories, found {len(class_names)} under {image_root}"
            )

        development_names, test_names = self.partition_class_names(class_names)
        if split == "train":
            selected_names = set(development_names)
        elif split == "test":
            selected_names = set(test_names)
        else:
            selected_names = set(class_names)

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.classes = class_names
        self.class_to_label = {
            class_name: index
            for index, class_name in enumerate(class_names)
        }
        self.class_names = {
            label: class_name.replace("_", " ")
            for class_name, label in self.class_to_label.items()
        }
        self.development_class_names = list(development_names)
        self.test_class_names = list(test_names)
        self.development_class_labels = [
            self.class_to_label[class_name]
            for class_name in development_names
        ]
        self.test_class_labels = [
            self.class_to_label[class_name]
            for class_name in test_names
        ]

        records = []
        for class_directory in class_directories:
            if class_directory.name not in selected_names:
                continue
            image_paths = sorted(
                path
                for path in class_directory.iterdir()
                if path.is_file()
                and path.suffix.lower() in {".jpg", ".jpeg"}
            )
            if len(image_paths) != self.EXPECTED_IMAGES_PER_CLASS:
                raise ValueError(
                    f"Food-101 class {class_directory.name!r} contains "
                    f"{len(image_paths)} images; expected "
                    f"{self.EXPECTED_IMAGES_PER_CLASS}"
                )
            label = self.class_to_label[class_directory.name]
            records.extend((image_path, label) for image_path in image_paths)

        expected_split_images = (
            len(selected_names) * self.EXPECTED_IMAGES_PER_CLASS
        )
        if len(records) != expected_split_images:
            raise ValueError(
                f"Food-101 split {split!r} contains {len(records)} images; "
                f"expected {expected_split_images}"
            )

        self.paths = [str(image_path) for image_path, _ in records]
        self.labels = [int(label) for _, label in records]
        self.orig_labels = list(self.labels)
        self.class_disjoint_split = True
        self.class_split_info = {
            "source": "pooled_official_splits_class_disjoint_71_30",
            "pooled_sources": ["official_train", "official_test"],
            "source_sample_counts": dict(
                self.EXPECTED_OFFICIAL_SPLIT_COUNTS
            ),
            "pooled_sample_count": self.EXPECTED_IMAGE_COUNT,
            "class_disjoint_test": True,
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "development_class_fraction": self.DEVELOPMENT_CLASS_FRACTION,
            "test_class_fraction": self.TEST_CLASS_FRACTION,
            "development_class_count": len(development_names),
            "test_class_count": len(test_names),
            "development_classes": list(self.development_class_labels),
            "held_out_test_classes": list(self.test_class_labels),
            "development_class_names": list(development_names),
            "held_out_test_class_names": list(test_names),
            "official_image_level_split_used": False,
        }

    @classmethod
    def is_ready(cls, root):
        root = Path(root)
        marker_path = root / cls.COMPLETE_MARKER
        if (
            not (root / cls.IMAGE_DIRECTORY).is_dir()
            or not marker_path.is_file()
        ):
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            image_size = tuple(int(value) for value in marker["image_size"])
            image_count = int(marker["image_count"])
            class_count = int(marker["class_count"])
            split_version = str(marker["class_split_version"])
            development_class_count = int(
                marker["development_class_count"]
            )
            test_class_count = int(marker["test_class_count"])
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
            and image_count == cls.EXPECTED_IMAGE_COUNT
            and class_count == cls.EXPECTED_CLASS_COUNT
            and split_version == cls.CLASS_SPLIT_VERSION
            and development_class_count == cls.DEVELOPMENT_CLASS_COUNT
            and test_class_count == cls.TEST_CLASS_COUNT
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
        source = self._resolve_source()

        if source.is_dir():
            classes, records = self._load_directory_metadata(source)
            stats = self._resize_from_directory(source, records)
        elif source.is_file() and zipfile.is_zipfile(source):
            classes, records, dataset_prefix = self._load_zip_metadata(source)
            stats = self._resize_from_zip(
                source,
                dataset_prefix,
                records,
            )
        else:
            raise RuntimeError(
                f"Unsupported Food-101 Kaggle source: {source}"
            )

        development_names, test_names = self.partition_class_names(classes)
        marker = {
            "dataset": "Food101",
            "image_size": list(self.TARGET_IMAGE_SIZE),
            "image_count": len(records),
            "class_count": len(classes),
            "images_per_class": self.EXPECTED_IMAGES_PER_CLASS,
            "jpeg_quality": int(self.JPEG_QUALITY),
            "written_images": int(stats["written_images"]),
            "reused_images": int(stats["reused_images"]),
            "source": str(source),
            "kaggle_dataset": self.KAGGLE_DATASET,
            "kaggle_dataset_page": self.KAGGLE_DATASET_PAGE,
            "pooled_sources": ["official_train", "official_test"],
            "source_image_counts": dict(
                self.EXPECTED_OFFICIAL_SPLIT_COUNTS
            ),
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "development_class_fraction": self.DEVELOPMENT_CLASS_FRACTION,
            "test_class_fraction": self.TEST_CLASS_FRACTION,
            "development_class_count": len(development_names),
            "test_class_count": len(test_names),
            "development_classes": list(development_names),
            "held_out_test_classes": list(test_names),
            "official_image_level_split_used": False,
        }
        self._write_json_atomic(self.root / self.COMPLETE_MARKER, marker)

    def _resolve_source(self):
        local_candidates = (
            self.root / "food-101" / "food-101",
            self.root / "food-101",
        )
        for candidate in local_candidates:
            if self._is_expanded_source(candidate):
                return candidate

        local_archive = self.root / self.KAGGLE_ARCHIVE_FILENAME
        if local_archive.is_file():
            return local_archive

        try:
            import kagglehub
        except ImportError as error:
            raise RuntimeError(
                "kagglehub is required to download Food-101. Install the "
                "project requirements or run `pip install kagglehub`."
            ) from error

        try:
            kaggle_path = Path(
                kagglehub.dataset_download(self.KAGGLE_DATASET)
            )
        except Exception as error:
            raise RuntimeError(
                "Could not download Food-101 from "
                f"{self.KAGGLE_DATASET_PAGE} with KaggleHub."
            ) from error
        return self._find_source_below(kaggle_path)

    @classmethod
    def _find_source_below(cls, kaggle_path):
        kaggle_path = Path(kaggle_path)
        if kaggle_path.is_file():
            if zipfile.is_zipfile(kaggle_path):
                return kaggle_path
            raise RuntimeError(
                f"KaggleHub returned a non-ZIP Food-101 file: {kaggle_path}"
            )
        if not kaggle_path.is_dir():
            raise RuntimeError(
                f"KaggleHub returned a missing Food-101 path: {kaggle_path}"
            )

        direct_candidates = (
            kaggle_path,
            kaggle_path / "food-101",
            kaggle_path / "food-101" / "food-101",
        )
        for candidate in direct_candidates:
            if cls._is_expanded_source(candidate):
                return candidate

        expanded_matches = []
        for train_metadata in kaggle_path.rglob("train.json"):
            candidate = train_metadata.parent.parent
            if cls._is_expanded_source(candidate):
                expanded_matches.append(candidate)
        expanded_matches = sorted(set(expanded_matches))
        if len(expanded_matches) == 1:
            return expanded_matches[0]
        if len(expanded_matches) > 1:
            raise RuntimeError(
                "Expected one expanded Food-101 directory below "
                f"{kaggle_path}, found {len(expanded_matches)}"
            )

        archive_matches = sorted(
            path
            for path in kaggle_path.rglob(cls.KAGGLE_ARCHIVE_FILENAME)
            if path.is_file() and zipfile.is_zipfile(path)
        )
        if len(archive_matches) == 1:
            return archive_matches[0]
        raise RuntimeError(
            "KaggleHub returned neither an expanded Food-101 directory nor "
            f"one {cls.KAGGLE_ARCHIVE_FILENAME!r} below {kaggle_path}."
        )

    @staticmethod
    def _is_expanded_source(candidate):
        candidate = Path(candidate)
        return (
            (candidate / "images").is_dir()
            and (candidate / "meta" / "train.json").is_file()
            and (candidate / "meta" / "test.json").is_file()
        )

    @classmethod
    def _load_directory_metadata(cls, source):
        metadata = {}
        for split in ("train", "test"):
            metadata_path = Path(source) / "meta" / f"{split}.json"
            try:
                metadata[split] = json.loads(
                    metadata_path.read_text(encoding="utf-8")
                )
            except (json.JSONDecodeError, OSError) as error:
                raise ValueError(
                    f"Could not read Food-101 metadata {metadata_path}"
                ) from error
        return cls._validate_official_metadata(metadata)

    @classmethod
    def _load_zip_metadata(cls, archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            train_member = cls._find_zip_suffix(
                archive,
                "meta/train.json",
            )
            dataset_prefix = PurePosixPath(train_member).parent.parent
            test_member = (
                dataset_prefix / "meta" / "test.json"
            ).as_posix()
            if test_member not in archive.namelist():
                raise RuntimeError(
                    f"Food-101 archive has no {test_member!r}"
                )
            try:
                metadata = {
                    "train": json.loads(
                        archive.read(train_member).decode("utf-8")
                    ),
                    "test": json.loads(
                        archive.read(test_member).decode("utf-8")
                    ),
                }
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Could not read Food-101 metadata in {archive_path}"
                ) from error
        classes, records = cls._validate_official_metadata(metadata)
        return classes, records, dataset_prefix

    @staticmethod
    def _find_zip_suffix(archive, suffix):
        suffix_path = PurePosixPath(suffix)
        matches = []
        for member_name in archive.namelist():
            member_path = PurePosixPath(
                member_name.replace("\\", "/")
            )
            if (
                len(member_path.parts) >= len(suffix_path.parts)
                and member_path.parts[-len(suffix_path.parts) :]
                == suffix_path.parts
            ):
                matches.append(member_name)
        if len(matches) != 1:
            raise RuntimeError(
                f"Expected exactly one Food-101 {suffix!r} member, found "
                f"{len(matches)}"
            )
        return matches[0]

    @classmethod
    def _validate_official_metadata(cls, metadata):
        for split in ("train", "test"):
            if not isinstance(metadata.get(split), dict):
                raise ValueError(
                    f"Food-101 {split!r} metadata must be an object"
                )

        train_classes = set(metadata["train"])
        test_classes = set(metadata["test"])
        if train_classes != test_classes:
            raise ValueError(
                "Food-101 official train and test metadata classes differ"
            )
        classes = tuple(sorted(train_classes))
        if len(classes) != cls.EXPECTED_CLASS_COUNT:
            raise ValueError(
                f"Food-101 metadata contains {len(classes)} classes; "
                f"expected {cls.EXPECTED_CLASS_COUNT}"
            )

        records = []
        seen_paths = set()
        for split in ("train", "test"):
            split_count = 0
            expected_per_class = (
                cls.EXPECTED_OFFICIAL_IMAGES_PER_CLASS[split]
            )
            for class_name in classes:
                raw_paths = metadata[split].get(class_name)
                if (
                    not isinstance(raw_paths, list)
                    or len(raw_paths) != expected_per_class
                ):
                    count = (
                        len(raw_paths)
                        if isinstance(raw_paths, list)
                        else "invalid"
                    )
                    raise ValueError(
                        f"Food-101 {split!r} class {class_name!r} has "
                        f"{count} records; expected {expected_per_class}"
                    )
                for raw_path in raw_paths:
                    relative_path = PurePosixPath(
                        str(raw_path).replace("\\", "/")
                    )
                    if (
                        relative_path.is_absolute()
                        or ".." in relative_path.parts
                        or len(relative_path.parts) != 2
                        or relative_path.parts[0] != class_name
                        or relative_path.suffix
                    ):
                        raise ValueError(
                            f"Unsafe Food-101 metadata path: {raw_path!r}"
                        )
                    image_path = PurePosixPath(
                        f"{relative_path.as_posix()}.jpg"
                    )
                    path_key = image_path.as_posix()
                    if path_key in seen_paths:
                        raise ValueError(
                            f"Duplicate Food-101 image path {path_key!r}"
                        )
                    seen_paths.add(path_key)
                    records.append((image_path, class_name, split))
                    split_count += 1

            expected_split_count = cls.EXPECTED_OFFICIAL_SPLIT_COUNTS[split]
            if split_count != expected_split_count:
                raise ValueError(
                    f"Food-101 {split!r} metadata contains {split_count} "
                    f"images; expected {expected_split_count}"
                )

        if len(records) != cls.EXPECTED_IMAGE_COUNT:
            raise ValueError(
                f"Food-101 metadata contains {len(records)} images; expected "
                f"{cls.EXPECTED_IMAGE_COUNT}"
            )
        return classes, records

    def _resize_from_directory(self, source, records):
        written_images = 0
        reused_images = 0
        with tqdm(
            total=len(records),
            desc="Saving 224x224 Food-101 images",
            unit="image",
            disable=None,
        ) as progress:
            for relative_path, _, _ in records:
                source_path = (
                    Path(source)
                    / self.IMAGE_DIRECTORY
                    / Path(*relative_path.parts)
                )
                if not source_path.is_file():
                    raise RuntimeError(
                        f"Food-101 source is missing {source_path}"
                    )
                destination = (
                    self.root
                    / self.IMAGE_DIRECTORY
                    / Path(*relative_path.parts)
                )
                if self._is_target_sized_image(destination):
                    reused_images += 1
                else:
                    with source_path.open("rb") as image_source:
                        self._resize_image_to_destination(
                            image_source,
                            destination,
                        )
                    written_images += 1
                progress.update(1)
        return {
            "written_images": written_images,
            "reused_images": reused_images,
        }

    def _resize_from_zip(self, archive_path, dataset_prefix, records):
        written_images = 0
        reused_images = 0
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive_members = set(archive.namelist())
            with tqdm(
                total=len(records),
                desc="Saving 224x224 Food-101 images",
                unit="image",
                disable=None,
            ) as progress:
                for relative_path, _, _ in records:
                    member_path = (
                        dataset_prefix
                        / self.IMAGE_DIRECTORY
                        / relative_path
                    ).as_posix()
                    if member_path not in archive_members:
                        raise RuntimeError(
                            f"Food-101 archive is missing {member_path!r}"
                        )
                    destination = (
                        self.root
                        / self.IMAGE_DIRECTORY
                        / Path(*relative_path.parts)
                    )
                    if self._is_target_sized_image(destination):
                        reused_images += 1
                    else:
                        with archive.open(member_path, "r") as image_source:
                            self._resize_image_to_destination(
                                image_source,
                                destination,
                            )
                        written_images += 1
                    progress.update(1)
        return {
            "written_images": written_images,
            "reused_images": reused_images,
        }

    @classmethod
    def partition_class_names(cls, class_names):
        class_names = tuple(sorted(str(name) for name in class_names))
        if len(class_names) != cls.EXPECTED_CLASS_COUNT:
            raise ValueError(
                f"Food-101 requires {cls.EXPECTED_CLASS_COUNT} classes, "
                f"found {len(class_names)}"
            )
        if len(class_names) != len(set(class_names)):
            raise ValueError("Food-101 class names must be unique")

        ranked_names = sorted(
            class_names,
            key=lambda class_name: hashlib.sha256(
                f"{cls.CLASS_SPLIT_VERSION}:{class_name}".encode("utf-8")
            ).digest(),
        )
        development_names = tuple(
            sorted(ranked_names[: cls.DEVELOPMENT_CLASS_COUNT])
        )
        test_names = tuple(
            sorted(ranked_names[cls.DEVELOPMENT_CLASS_COUNT :])
        )
        if len(test_names) != cls.TEST_CLASS_COUNT:
            raise RuntimeError(
                f"Food-101 test partition contains {len(test_names)} "
                f"classes; expected {cls.TEST_CLASS_COUNT}"
            )
        if set(development_names) & set(test_names):
            raise RuntimeError("Food-101 class partition is not disjoint")
        if set(development_names) | set(test_names) != set(class_names):
            raise RuntimeError(
                "Food-101 class partition does not cover every class"
            )
        return development_names, test_names

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


class Flowers102(Dataset):
    """Oxford Flowers 102 with the CUB-style class-disjoint DML split.

    Oxford's official train, validation, and test sets all contain every flower
    category. For metric learning, this wrapper pools all three sources and
    follows the CUB convention: the first half of the ordered class IDs is the
    development set and the second half is reserved for final testing.
    """

    DATASET_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102/"
    CUB_PROTOCOL_REFERENCE = (
        "https://github.com/KevinMusgrave/pytorch-metric-learning/"
        "blob/master/src/pytorch_metric_learning/datasets/cub.py"
    )
    BASE_DIRECTORY = "flowers-102"
    IMAGE_DIRECTORY = "jpg"
    METADATA_MD5 = {
        "imagelabels.mat": "e0620be6f572b9609742df49c70aed4d",
        "setid.mat": "a5357ecc9cb78c4bef273ce3793fc85c",
    }
    OFFICIAL_SPLITS = ("train", "val", "test")
    EXPECTED_OFFICIAL_SPLIT_COUNTS = {
        "train": 1_020,
        "val": 1_020,
        "test": 6_149,
    }
    EXPECTED_IMAGE_COUNT = 8_189
    EXPECTED_CLASS_COUNT = 102
    EXPECTED_MIN_IMAGES_PER_CLASS = 40
    EXPECTED_MAX_IMAGES_PER_CLASS = 258
    DEVELOPMENT_CLASS_COUNT = 51
    TEST_CLASS_COUNT = 51
    EXPECTED_DEVELOPMENT_IMAGE_COUNT = 3_493
    EXPECTED_TEST_IMAGE_COUNT = 4_696
    CLASS_SPLIT_VERSION = "flowers102_contiguous_class_ids_51_51_v1"
    AVAILABLE_SPLITS = ("train", "test", "train+test")

    def __init__(
        self,
        root,
        split="train+test",
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = Path(root)
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(
                f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}"
            )
        if download and not self.is_ready(self.root):
            self._download(self.root)
        if not self.is_ready(self.root):
            raise ValueError(
                "Oxford Flowers 102 data was not found or is incomplete. "
                "Initialize with download=True or place the torchvision "
                "Flowers102 files below the dataset root."
            )

        records, source_counts = self._load_pooled_records(self.root)
        development_classes, test_classes = self.partition_classes()
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
        if not selected_records:
            raise ValueError(f"Flowers102 split {split!r} contains no images")

        development_sample_count = sum(
            label in development_set for _, label, _ in records
        )
        test_sample_count = len(records) - development_sample_count
        if (
            development_sample_count
            != self.EXPECTED_DEVELOPMENT_IMAGE_COUNT
            or test_sample_count != self.EXPECTED_TEST_IMAGE_COUNT
        ):
            raise ValueError(
                "Flowers102 51/51 class split contains "
                f"{development_sample_count} development images and "
                f"{test_sample_count} test images; expected "
                f"{self.EXPECTED_DEVELOPMENT_IMAGE_COUNT} and "
                f"{self.EXPECTED_TEST_IMAGE_COUNT}"
            )
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.class_ids = list(range(self.EXPECTED_CLASS_COUNT))
        self.classes = [
            f"flower_{class_id + 1:03d}" for class_id in self.class_ids
        ]
        self.class_names = {
            class_id: class_name
            for class_id, class_name in zip(self.class_ids, self.classes)
        }
        self.development_class_labels = list(development_classes)
        self.test_class_labels = list(test_classes)
        self.paths = [str(path) for path, _, _ in selected_records]
        self.labels = [int(label) for _, label, _ in selected_records]
        self.orig_labels = list(self.labels)
        self.sample_sources = [source for _, _, source in selected_records]
        self.class_disjoint_split = True
        self.class_split_info = {
            "source": (
                "pooled_official_train_val_test_"
                "class_disjoint_51_51"
            ),
            "dataset_page": self.DATASET_PAGE,
            "protocol_reference": self.CUB_PROTOCOL_REFERENCE,
            "pooled_sources": list(self.OFFICIAL_SPLITS),
            "source_sample_counts": dict(source_counts),
            "pooled_sample_count": len(records),
            "class_disjoint_test": True,
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "split_basis": "contiguous_class_ids_like_cub",
            "development_class_count": len(development_classes),
            "test_class_count": len(test_classes),
            "development_classes": list(development_classes),
            "held_out_test_classes": list(test_classes),
            "development_sample_count": development_sample_count,
            "test_sample_count": test_sample_count,
            "development_sample_fraction": (
                development_sample_count / len(records)
            ),
            "test_sample_fraction": test_sample_count / len(records),
            "official_image_level_split_used": False,
        }

    @classmethod
    def _download(cls, root):
        _Flowers102(root=str(root), split="train", download=True)

    @classmethod
    def is_ready(cls, root):
        base_directory = Path(root) / cls.BASE_DIRECTORY
        image_directory = base_directory / cls.IMAGE_DIRECTORY
        if not image_directory.is_dir():
            return False
        if sum(1 for _ in image_directory.glob("image_*.jpg")) != (
            cls.EXPECTED_IMAGE_COUNT
        ):
            return False
        return all(
            cls._file_md5(base_directory / filename) == expected_md5
            for filename, expected_md5 in cls.METADATA_MD5.items()
        )

    @staticmethod
    def _file_md5(path):
        if not path.is_file():
            return None
        digest = hashlib.md5(usedforsecurity=False)
        with path.open("rb") as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest()

    @classmethod
    def _load_pooled_records(cls, root):
        expected_classes = set(range(cls.EXPECTED_CLASS_COUNT))
        records = []
        seen_paths = set()
        source_counts = {}
        for source_split in cls.OFFICIAL_SPLITS:
            source_dataset = _Flowers102(
                root=str(root),
                split=source_split,
                transform=None,
                target_transform=None,
                download=False,
            )
            paths = [Path(path) for path in source_dataset._image_files]
            labels = [int(label) for label in source_dataset._labels]
            expected_count = cls.EXPECTED_OFFICIAL_SPLIT_COUNTS[source_split]
            if len(paths) != expected_count or len(labels) != expected_count:
                raise ValueError(
                    f"Flowers102 official {source_split!r} split contains "
                    f"{len(paths)} paths and {len(labels)} labels; expected "
                    f"{expected_count} of each"
                )
            if set(labels) != expected_classes:
                raise ValueError(
                    f"Flowers102 official {source_split!r} split must contain "
                    f"all {cls.EXPECTED_CLASS_COUNT} classes"
                )
            source_counts[source_split] = len(paths)
            for path, label in zip(paths, labels):
                path_key = str(path)
                if path_key in seen_paths:
                    raise ValueError(
                        f"Flowers102 image {path_key!r} occurs in more than "
                        "one official split"
                    )
                seen_paths.add(path_key)
                records.append((path, label, source_split))

        if len(records) != cls.EXPECTED_IMAGE_COUNT:
            raise ValueError(
                f"Flowers102 pooled source contains {len(records)} images; "
                f"expected {cls.EXPECTED_IMAGE_COUNT}"
            )
        if {label for _, label, _ in records} != expected_classes:
            raise ValueError(
                f"Flowers102 pooled source must contain exactly "
                f"{cls.EXPECTED_CLASS_COUNT} classes"
            )
        class_counts = Counter(label for _, label, _ in records)
        if (
            min(class_counts.values()) != cls.EXPECTED_MIN_IMAGES_PER_CLASS
            or max(class_counts.values()) != cls.EXPECTED_MAX_IMAGES_PER_CLASS
        ):
            raise ValueError(
                "Flowers102 class sizes must range from "
                f"{cls.EXPECTED_MIN_IMAGES_PER_CLASS} to "
                f"{cls.EXPECTED_MAX_IMAGES_PER_CLASS} images"
            )
        records.sort(key=lambda record: record[0].name)
        return tuple(records), source_counts

    @classmethod
    def partition_classes(cls):
        class_ids = tuple(range(cls.EXPECTED_CLASS_COUNT))
        development = class_ids[: cls.DEVELOPMENT_CLASS_COUNT]
        test = class_ids[cls.DEVELOPMENT_CLASS_COUNT :]
        if len(test) != cls.TEST_CLASS_COUNT:
            raise RuntimeError(
                f"Flowers102 test partition contains {len(test)} classes; "
                f"expected {cls.TEST_CLASS_COUNT}"
            )
        if set(development) & set(test):
            raise RuntimeError("Flowers102 class partition is not disjoint")
        if set(development) | set(test) != set(class_ids):
            raise RuntimeError(
                "Flowers102 class partition does not cover every class"
            )
        return development, test

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


class FGVCAircraft(_DinoSizedImageDownloadMixin, Dataset):
    """FGVC-Aircraft with the conventional metric-learning 50/50 split.

    The official train, validation, and test image lists all contain the same
    100 aircraft variants. Following the class-disjoint protocol used by the
    PUMA unified metric-learning benchmark (arXiv:2309.08944), this wrapper
    pools those lists and assigns the first 50 variants in ``variants.txt`` to
    development and the remaining 50 variants to final testing. Each side
    therefore contains 5,000 images from classes unseen on the other side.

    The official archive is streamed and only 224 x 224 JPEGs are retained.
    The 20-pixel copyright banner is removed before resizing, as required by
    the dataset's usage instructions. The source photographs are available
    for non-commercial research only.
    """

    DATASET_PAGE = "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/"
    PROTOCOL_REFERENCE = "https://arxiv.org/abs/2309.08944"
    ARCHIVE_URL = (
        "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
        "fgvc-aircraft-2013b.tar.gz"
    )
    ANNOTATIONS_URL = (
        "https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/archives/"
        "fgvc-aircraft-2013b-annotations.tar.gz"
    )
    ARCHIVE_FILENAME = "fgvc-aircraft-2013b.tar.gz"
    OFFICIAL_DIRECTORY = "fgvc-aircraft-2013b"
    IMAGE_DIRECTORY = "images"
    METADATA_DIRECTORY = "metadata"
    VARIANTS_FILENAME = "variants.txt"
    OFFICIAL_SPLIT_FILENAMES = {
        "train": "images_variant_train.txt",
        "val": "images_variant_val.txt",
        "test": "images_variant_test.txt",
    }
    METADATA_FILENAMES = (
        VARIANTS_FILENAME,
        *OFFICIAL_SPLIT_FILENAMES.values(),
    )
    COMPLETE_MARKER = ".fgvc_aircraft_224_complete.json"
    EXPECTED_IMAGE_COUNT = 10_000
    EXPECTED_CLASS_COUNT = 100
    EXPECTED_IMAGES_PER_CLASS = 100
    EXPECTED_OFFICIAL_SPLIT_COUNTS = {
        "train": 3_334,
        "val": 3_333,
        "test": 3_333,
    }
    DEVELOPMENT_CLASS_COUNT = 50
    TEST_CLASS_COUNT = 50
    COPYRIGHT_BANNER_HEIGHT = 20
    CLASS_SPLIT_VERSION = "first_50_variants_metric_learning_v1"
    AVAILABLE_SPLITS = ("train", "test", "train+test")

    def __init__(
        self,
        root,
        split="train+test",
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = Path(root)
        if download and not self.is_ready(self.root):
            self.download_224(self.root)
        if not self.is_ready(self.root):
            raise ValueError(
                "FGVC-Aircraft 224x224 data was not found. Initialize the "
                "dataset with download=True or run "
                "scripts/download_fgvc_aircraft_224.py. The source images "
                "are licensed for non-commercial research only."
            )
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(
                f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}"
            )

        metadata, classes, records, source_counts = self._load_metadata(
            self.root / self.METADATA_DIRECTORY
        )
        class_to_label = {
            class_name: label
            for label, class_name in enumerate(classes)
        }
        if split == "train":
            selected_labels = set(range(self.DEVELOPMENT_CLASS_COUNT))
        elif split == "test":
            selected_labels = set(
                range(self.DEVELOPMENT_CLASS_COUNT, self.EXPECTED_CLASS_COUNT)
            )
        else:
            selected_labels = set(range(self.EXPECTED_CLASS_COUNT))

        selected_records = [
            (image_id, label)
            for image_id, label, _ in records
            if label in selected_labels
        ]
        expected_split_count = {
            "train": self.DEVELOPMENT_CLASS_COUNT
            * self.EXPECTED_IMAGES_PER_CLASS,
            "test": self.TEST_CLASS_COUNT * self.EXPECTED_IMAGES_PER_CLASS,
            "train+test": self.EXPECTED_IMAGE_COUNT,
        }[split]
        if len(selected_records) != expected_split_count:
            raise ValueError(
                f"FGVC-Aircraft split {split!r} contains "
                f"{len(selected_records)} images; expected "
                f"{expected_split_count}"
            )

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.classes = list(classes)
        self.class_to_idx = dict(class_to_label)
        self.class_to_label = dict(class_to_label)
        self.class_names = {
            label: class_name
            for class_name, label in class_to_label.items()
        }
        self.development_class_names = list(
            classes[: self.DEVELOPMENT_CLASS_COUNT]
        )
        self.test_class_names = list(
            classes[self.DEVELOPMENT_CLASS_COUNT :]
        )
        self.development_class_labels = list(
            range(self.DEVELOPMENT_CLASS_COUNT)
        )
        self.test_class_labels = list(
            range(self.DEVELOPMENT_CLASS_COUNT, self.EXPECTED_CLASS_COUNT)
        )
        self.paths = [
            str(self.root / self.IMAGE_DIRECTORY / f"{image_id}.jpg")
            for image_id, _ in selected_records
        ]
        self.labels = [int(label) for _, label in selected_records]
        self.orig_labels = list(self.labels)
        self.class_disjoint_split = True
        self.class_split_info = {
            "source": "pooled_official_splits_first_50_variants",
            "pooled_sources": list(self.OFFICIAL_SPLIT_FILENAMES),
            "source_sample_counts": dict(source_counts),
            "pooled_sample_count": self.EXPECTED_IMAGE_COUNT,
            "class_disjoint_test": True,
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "class_split_protocol": "first_50_variants_metric_learning",
            "protocol_reference": self.PROTOCOL_REFERENCE,
            "development_class_fraction": 0.5,
            "test_class_fraction": 0.5,
            "development_class_count": self.DEVELOPMENT_CLASS_COUNT,
            "test_class_count": self.TEST_CLASS_COUNT,
            "development_classes": list(self.development_class_labels),
            "held_out_test_classes": list(self.test_class_labels),
            "development_class_names": list(
                self.development_class_names
            ),
            "held_out_test_class_names": list(self.test_class_names),
            "official_image_level_split_used": False,
            "annotation_level": "variant",
            "metadata_sha256": self._metadata_sha256(metadata),
            "copyright_banner_removed_pixels": (
                self.COPYRIGHT_BANNER_HEIGHT
            ),
        }

    @classmethod
    def is_ready(cls, root):
        root = Path(root)
        marker_path = root / cls.COMPLETE_MARKER
        metadata_directory = root / cls.METADATA_DIRECTORY
        required_paths = (
            root / cls.IMAGE_DIRECTORY,
            marker_path,
            *(
                metadata_directory / filename
                for filename in cls.METADATA_FILENAMES
            ),
        )
        if not all(path.exists() for path in required_paths):
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            image_size = tuple(int(value) for value in marker["image_size"])
            image_count = int(marker["image_count"])
            class_count = int(marker["class_count"])
            images_per_class = int(marker["images_per_class"])
            split_version = str(marker["class_split_version"])
            development_class_count = int(
                marker["development_class_count"]
            )
            test_class_count = int(marker["test_class_count"])
            banner_height = int(
                marker["copyright_banner_removed_pixels"]
            )
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
            and image_count == cls.EXPECTED_IMAGE_COUNT
            and class_count == cls.EXPECTED_CLASS_COUNT
            and images_per_class == cls.EXPECTED_IMAGES_PER_CLASS
            and split_version == cls.CLASS_SPLIT_VERSION
            and development_class_count == cls.DEVELOPMENT_CLASS_COUNT
            and test_class_count == cls.TEST_CLASS_COUNT
            and banner_height == cls.COPYRIGHT_BANNER_HEIGHT
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
        metadata, classes, records, source_counts = self._ensure_metadata()
        expected_image_ids = {image_id for image_id, _, _ in records}

        local_source = self._find_local_image_source()
        if local_source is None:
            stats = self._stream_and_resize_archive(
                expected_image_ids
            )
            source_description = self.ARCHIVE_URL
        elif local_source.is_dir():
            stats = self._resize_from_directory(
                local_source,
                expected_image_ids,
            )
            source_description = str(local_source)
        else:
            with local_source.open("rb") as archive_stream:
                stats = self._resize_from_archive_stream(
                    archive_stream,
                    expected_image_ids,
                )
            source_description = str(local_source)

        marker = {
            "dataset": "FGVCAircraft",
            "dataset_page": self.DATASET_PAGE,
            "image_size": list(self.TARGET_IMAGE_SIZE),
            "image_count": len(records),
            "class_count": len(classes),
            "images_per_class": self.EXPECTED_IMAGES_PER_CLASS,
            "jpeg_quality": int(self.JPEG_QUALITY),
            "written_images": int(stats["written_images"]),
            "reused_images": int(stats["reused_images"]),
            "source_archive": source_description,
            "annotations_source": self.ANNOTATIONS_URL,
            "pooled_sources": list(self.OFFICIAL_SPLIT_FILENAMES),
            "source_image_counts": dict(source_counts),
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "class_split_protocol": "first_50_variants_metric_learning",
            "protocol_reference": self.PROTOCOL_REFERENCE,
            "development_class_count": self.DEVELOPMENT_CLASS_COUNT,
            "test_class_count": self.TEST_CLASS_COUNT,
            "development_classes": list(
                classes[: self.DEVELOPMENT_CLASS_COUNT]
            ),
            "held_out_test_classes": list(
                classes[self.DEVELOPMENT_CLASS_COUNT :]
            ),
            "official_image_level_split_used": False,
            "copyright_banner_removed_pixels": (
                self.COPYRIGHT_BANNER_HEIGHT
            ),
            "metadata_sha256": self._metadata_sha256(metadata),
            "license_note": "source photographs: non-commercial research only",
        }
        self._write_json_atomic(self.root / self.COMPLETE_MARKER, marker)

    def _ensure_metadata(self):
        metadata_directory = self.root / self.METADATA_DIRECTORY
        candidate_directories = (
            metadata_directory,
            self.root / self.OFFICIAL_DIRECTORY / "data",
            self.root / "data",
        )
        for candidate in candidate_directories:
            try:
                metadata, classes, records, source_counts = (
                    self._load_metadata(candidate)
                )
            except (OSError, UnicodeError, ValueError):
                continue
            if candidate != metadata_directory:
                self._write_metadata(metadata)
            return metadata, classes, records, source_counts

        metadata = self._download_metadata()
        classes, records, source_counts = self._parse_metadata(metadata)
        self._write_metadata(metadata)
        return metadata, classes, records, source_counts

    @classmethod
    def _load_metadata(cls, metadata_directory):
        metadata_directory = Path(metadata_directory)
        metadata = {
            filename: (metadata_directory / filename).read_text(
                encoding="utf-8"
            )
            for filename in cls.METADATA_FILENAMES
        }
        classes, records, source_counts = cls._parse_metadata(metadata)
        return metadata, classes, records, source_counts

    def _download_metadata(self):
        metadata = {}
        with self._open_url(self.ANNOTATIONS_URL) as response:
            with tarfile.open(fileobj=response, mode="r|gz") as archive:
                for member in archive:
                    filename = self._metadata_member_filename(member)
                    if filename is None:
                        continue
                    if filename in metadata:
                        raise RuntimeError(
                            f"Duplicate FGVC-Aircraft metadata file "
                            f"{filename!r} in the annotation archive"
                        )
                    source = archive.extractfile(member)
                    if source is None:
                        raise RuntimeError(
                            f"Could not read {member.name!r} from the "
                            "FGVC-Aircraft annotation archive"
                        )
                    with source:
                        metadata[filename] = source.read().decode("utf-8")

        missing = set(self.METADATA_FILENAMES) - set(metadata)
        if missing:
            raise RuntimeError(
                "The FGVC-Aircraft annotation archive is missing: "
                f"{sorted(missing)}"
            )
        return metadata

    @classmethod
    def _metadata_member_filename(cls, member):
        if not member.isfile():
            return None
        path = PurePosixPath(member.name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "Unsafe path in FGVC-Aircraft annotation archive: "
                f"{member.name!r}"
            )
        if (
            len(path.parts) < 2
            or path.parts[-2] != "data"
            or path.name not in cls.METADATA_FILENAMES
        ):
            return None
        return path.name

    def _write_metadata(self, metadata):
        metadata_directory = self.root / self.METADATA_DIRECTORY
        for filename in self.METADATA_FILENAMES:
            self._write_bytes_atomic(
                metadata_directory / filename,
                metadata[filename].encode("utf-8"),
            )

    @classmethod
    def _parse_metadata(cls, metadata):
        classes = tuple(
            line.strip()
            for line in metadata[cls.VARIANTS_FILENAME].splitlines()
            if line.strip()
        )
        if len(classes) != cls.EXPECTED_CLASS_COUNT:
            raise ValueError(
                f"FGVC-Aircraft metadata contains {len(classes)} variants; "
                f"expected {cls.EXPECTED_CLASS_COUNT}"
            )
        if len(classes) != len(set(classes)):
            raise ValueError(
                "FGVC-Aircraft variant names must be unique"
            )
        class_to_label = {
            class_name: label
            for label, class_name in enumerate(classes)
        }

        records = []
        seen_image_ids = set()
        source_counts = {}
        class_counts = Counter()
        for source_split, filename in cls.OFFICIAL_SPLIT_FILENAMES.items():
            source_records = []
            for line_number, raw_line in enumerate(
                metadata[filename].splitlines(),
                start=1,
            ):
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    image_id, class_name = line.split(" ", 1)
                except ValueError as error:
                    raise ValueError(
                        f"Invalid FGVC-Aircraft annotation in {filename}:"
                        f"{line_number}: {raw_line!r}"
                    ) from error
                class_name = class_name.strip()
                if len(image_id) != 7 or not image_id.isdigit():
                    raise ValueError(
                        f"Invalid FGVC-Aircraft image id {image_id!r} in "
                        f"{filename}:{line_number}"
                    )
                if class_name not in class_to_label:
                    raise ValueError(
                        f"Unknown FGVC-Aircraft variant {class_name!r} in "
                        f"{filename}:{line_number}"
                    )
                if image_id in seen_image_ids:
                    raise ValueError(
                        f"Duplicate FGVC-Aircraft image id {image_id!r}"
                    )
                seen_image_ids.add(image_id)
                label = class_to_label[class_name]
                class_counts[label] += 1
                source_records.append((image_id, label, source_split))

            expected_source_count = cls.EXPECTED_OFFICIAL_SPLIT_COUNTS[
                source_split
            ]
            if len(source_records) != expected_source_count:
                raise ValueError(
                    f"FGVC-Aircraft {source_split!r} metadata contains "
                    f"{len(source_records)} images; expected "
                    f"{expected_source_count}"
                )
            source_counts[source_split] = len(source_records)
            records.extend(source_records)

        if len(records) != cls.EXPECTED_IMAGE_COUNT:
            raise ValueError(
                f"FGVC-Aircraft metadata contains {len(records)} images; "
                f"expected {cls.EXPECTED_IMAGE_COUNT}"
            )
        unexpected_class_counts = {
            int(label): int(count)
            for label, count in class_counts.items()
            if count != cls.EXPECTED_IMAGES_PER_CLASS
        }
        if (
            len(class_counts) != cls.EXPECTED_CLASS_COUNT
            or unexpected_class_counts
        ):
            raise ValueError(
                "FGVC-Aircraft must contain exactly "
                f"{cls.EXPECTED_IMAGES_PER_CLASS} images for each of "
                f"{cls.EXPECTED_CLASS_COUNT} variants; unexpected counts: "
                f"{unexpected_class_counts}"
            )
        return classes, tuple(records), source_counts

    @classmethod
    def _metadata_sha256(cls, metadata):
        digest = hashlib.sha256()
        for filename in cls.METADATA_FILENAMES:
            digest.update(filename.encode("utf-8"))
            digest.update(b"\0")
            digest.update(metadata[filename].encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _find_local_image_source(self):
        directory_candidates = (
            self.root / self.OFFICIAL_DIRECTORY / "data" / "images",
            self.root / "data" / "images",
        )
        for candidate in directory_candidates:
            if candidate.is_dir():
                return candidate

        archive_candidates = (
            self.root / self.ARCHIVE_FILENAME,
            self.root / self.OFFICIAL_DIRECTORY / self.ARCHIVE_FILENAME,
        )
        for candidate in archive_candidates:
            if candidate.is_file():
                return candidate
        return None

    def _stream_and_resize_archive(self, expected_image_ids):
        with self._open_url(self.ARCHIVE_URL) as response:
            with self._byte_progress(
                response,
                "Streaming official FGVC-Aircraft image archive",
            ) as byte_progress:
                reader = _DownloadProgressReader(response, byte_progress)
                return self._resize_from_archive_stream(
                    reader,
                    expected_image_ids,
                )

    def _resize_from_directory(self, source_directory, expected_image_ids):
        source_directory = Path(source_directory)
        written_images = 0
        reused_images = 0
        with tqdm(
            total=len(expected_image_ids),
            desc="Saving 224x224 FGVC-Aircraft images",
            unit="image",
            disable=None,
        ) as progress:
            for image_id in sorted(expected_image_ids):
                source_path = source_directory / f"{image_id}.jpg"
                if not source_path.is_file():
                    raise RuntimeError(
                        f"FGVC-Aircraft source is missing {source_path}"
                    )
                destination = (
                    self.root / self.IMAGE_DIRECTORY / f"{image_id}.jpg"
                )
                if self._is_target_sized_image(destination):
                    reused_images += 1
                else:
                    with source_path.open("rb") as image_source:
                        self._resize_aircraft_image(
                            image_source,
                            destination,
                        )
                    written_images += 1
                progress.update(1)
        return {
            "written_images": written_images,
            "reused_images": reused_images,
        }

    def _resize_from_archive_stream(
        self,
        archive_stream,
        expected_image_ids,
    ):
        seen_image_ids = set()
        written_images = 0
        reused_images = 0
        with tqdm(
            total=len(expected_image_ids),
            desc="Saving 224x224 FGVC-Aircraft images",
            unit="image",
            disable=None,
        ) as progress:
            with tarfile.open(fileobj=archive_stream, mode="r|gz") as archive:
                for member in archive:
                    image_id = self._image_member_id(member)
                    if image_id is None:
                        continue
                    if image_id not in expected_image_ids:
                        raise RuntimeError(
                            "Unexpected FGVC-Aircraft image "
                            f"{image_id!r} in the archive"
                        )
                    if image_id in seen_image_ids:
                        raise RuntimeError(
                            "Duplicate FGVC-Aircraft image "
                            f"{image_id!r} in the archive"
                        )
                    seen_image_ids.add(image_id)
                    destination = (
                        self.root
                        / self.IMAGE_DIRECTORY
                        / f"{image_id}.jpg"
                    )
                    if self._is_target_sized_image(destination):
                        reused_images += 1
                    else:
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuntimeError(
                                f"Could not read {member.name!r} from the "
                                "FGVC-Aircraft archive"
                            )
                        with source:
                            self._resize_aircraft_image(
                                source,
                                destination,
                            )
                        written_images += 1
                    progress.update(1)

        missing_image_ids = expected_image_ids - seen_image_ids
        if missing_image_ids:
            raise RuntimeError(
                "The FGVC-Aircraft archive is missing "
                f"{len(missing_image_ids)} annotated images; examples: "
                f"{sorted(missing_image_ids)[:5]}"
            )
        return {
            "written_images": written_images,
            "reused_images": reused_images,
        }

    @classmethod
    def _image_member_id(cls, member):
        if not member.isfile():
            return None
        path = PurePosixPath(member.name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                "Unsafe path in FGVC-Aircraft image archive: "
                f"{member.name!r}"
            )
        if (
            len(path.parts) < 3
            or tuple(path.parts[-3:-1]) != ("data", "images")
            or path.suffix.lower() != ".jpg"
        ):
            return None
        return path.stem

    def _resize_aircraft_image(self, source, destination):
        self._resize_image_to_destination(
            source,
            destination,
            crop_bottom_pixels=self.COPYRIGHT_BANNER_HEIGHT,
        )

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


class FGVCFungi(_DinoSizedImageDownloadMixin, Dataset):
    """FGVCx Fungi 2018 with a fixed genus-balanced 50/50 class split.

    The competition released 85,578 training and 4,182 validation images over
    the same 1,394 species and never published labels for its 9,758 test
    images. Both labeled sources ship in one ``images/<class id>.<species>/``
    tree, so this wrapper pools them -- the official image-level train/val
    boundary lives only in the annotation JSONs and is not recoverable from the
    image archive -- and then assigns complete species to a development or a
    final-test half.

    Species alternate into the halves along an ordering sorted by genus and
    then class id, the same idea Semi-iNat applies inside its kingdoms. The
    stride runs once through the concatenated genera instead of restarting per
    genus: Semi-iNat can restart because it has three kingdoms, while 292 of
    Fungi's 418 genera hold an odd number of species, and restarting would hand
    the development half 843 of the 1,394 species. Running the stride through
    keeps the halves at exactly 697 species each while still splitting every
    genus to within one species, so both halves see the same genera.

    Labels are the official category ids parsed from the directory names. Only
    atomic 224 x 224 JPEGs are written to the dataset root. The source
    photographs are licensed for non-commercial research only.
    """

    DATASET_PAGE = "https://github.com/visipedia/fgvcx_fungi_comp"
    KAGGLE_PAGE = "https://www.kaggle.com/c/fungi-challenge-fgvc-2018"
    KAGGLE_COMPETITION = "fungi-challenge-fgvc-2018"
    ARCHIVE_URL = "https://labs.gbif.org/fgvcx/2018/fungi_train_val.tgz"
    ARCHIVE_FILENAME = "fungi_train_val.tgz"
    IMAGE_DIRECTORY = "images"
    COMPLETE_MARKER = ".fgvc_fungi_224_complete.json"
    # 85,578 training plus 4,182 validation images; the unlabeled test images
    # ship in a separate archive that this loader never touches.
    EXPECTED_IMAGE_COUNT = 89_760
    EXPECTED_CLASS_COUNT = 1_394
    POOLED_SOURCES = ("train", "val")
    DEVELOPMENT_CLASS_FRACTION = 0.50
    TEST_CLASS_FRACTION = 0.50
    SPLIT_SUPERCATEGORY_FIELD = "genus"
    CLASS_SPLIT_VERSION = "fungi_alternating_class_ids_within_genus_50_50_v1"
    # Both halves must stay near half the pooled images. The stride cannot
    # control per-class counts, so this only catches a split that collapsed.
    MIN_HALF_IMAGE_FRACTION = 0.40
    IMAGE_EXTENSIONS = (".jpg", ".jpeg")
    AVAILABLE_SPLITS = ("train", "test", "train+test")

    def __init__(
        self,
        root,
        split="train+test",
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = Path(root)
        if download and not self.is_ready(self.root):
            self.download_224(self.root)
        if not self.is_ready(self.root):
            raise ValueError(
                "FGVCx Fungi 2018 224x224 data was not found. Initialize the "
                "dataset with download=True, run "
                "scripts/download_fgvc_fungi_224.py, or place "
                f"{self.ARCHIVE_FILENAME} in the dataset root before "
                "retrying. The source images are licensed for non-commercial "
                "research only."
            )
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(
                f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}"
            )

        taxonomy = self.load_taxonomy(self.root)
        development_classes, test_classes = self.partition_class_ids(taxonomy)
        development_set = set(development_classes)
        test_set = set(test_classes)
        if split == "train":
            selected_classes = development_set
        elif split == "test":
            selected_classes = test_set
        else:
            selected_classes = development_set | test_set

        records = self._scan_records(self.root, taxonomy)
        if len(records) != self.EXPECTED_IMAGE_COUNT:
            raise ValueError(
                f"FGVCx Fungi 2018 contains {len(records)} images below "
                f"{self.root / self.IMAGE_DIRECTORY}; expected "
                f"{self.EXPECTED_IMAGE_COUNT}"
            )
        image_counts = Counter(label for _, label in records)
        development_image_count = sum(
            image_counts[class_id] for class_id in development_classes
        )
        test_image_count = sum(
            image_counts[class_id] for class_id in test_classes
        )
        smallest_half = min(development_image_count, test_image_count)
        if smallest_half < self.MIN_HALF_IMAGE_FRACTION * len(records):
            raise RuntimeError(
                "The FGVCx Fungi class halves are badly unbalanced: "
                f"{development_image_count} development and "
                f"{test_image_count} test images"
            )

        selected_records = [
            (image_path, label)
            for image_path, label in records
            if label in selected_classes
        ]
        if not selected_records:
            raise ValueError(f"FGVCx Fungi split {split!r} contains no images")

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.paths = [str(image_path) for image_path, _ in selected_records]
        self.labels = [int(label) for _, label in selected_records]
        self.orig_labels = list(self.labels)
        self.taxonomy = taxonomy
        self.class_ids = sorted(taxonomy)
        self.classes = [
            taxonomy[class_id]["species"] for class_id in self.class_ids
        ]
        self.class_names = {
            class_id: taxonomy[class_id]["species"]
            for class_id in self.class_ids
        }
        self.class_to_label = {
            taxonomy[class_id]["directory"]: class_id
            for class_id in self.class_ids
        }
        self.development_class_labels = list(development_classes)
        self.test_class_labels = list(test_classes)
        self.class_disjoint_split = True
        self.class_split_info = self._build_class_split_info(
            taxonomy=taxonomy,
            development_classes=development_classes,
            test_classes=test_classes,
            pooled_image_count=len(records),
            development_image_count=development_image_count,
            test_image_count=test_image_count,
        )

    @classmethod
    def is_ready(cls, root):
        """Return whether the complete resized dataset is present."""

        root = Path(root)
        marker_path = root / cls.COMPLETE_MARKER
        if (
            not (root / cls.IMAGE_DIRECTORY).is_dir()
            or not marker_path.is_file()
        ):
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            image_size = tuple(int(value) for value in marker["image_size"])
            image_count = int(marker["image_count"])
            class_count = int(marker["class_count"])
            split_version = str(marker["class_split_version"])
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
            and image_count == cls.EXPECTED_IMAGE_COUNT
            and class_count == cls.EXPECTED_CLASS_COUNT
            and split_version == cls.CLASS_SPLIT_VERSION
        )

    @classmethod
    def download_224(cls, root):
        """Fetch the labeled archive while retaining only 224px images."""

        root = Path(root)
        if cls.is_ready(root):
            return root
        root.mkdir(parents=True, exist_ok=True)
        downloader = cls.__new__(cls)
        downloader.root = root
        downloader.download_and_remove()
        return root

    def download_and_remove(self):
        """Resize every labeled image into the root and mark completion."""

        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._kagglehub_failure = None

        source = self._find_local_image_source()
        if source is None:
            source = self._kagglehub_image_source()
        if source is None:
            stats = self._stream_and_resize_archive()
            source_description = self.ARCHIVE_URL
        elif source.is_dir():
            stats = self._resize_from_directory(source)
            source_description = str(source)
        else:
            with source.open("rb") as archive_stream:
                stats = self._resize_from_archive_stream(archive_stream)
            source_description = str(source)

        taxonomy = self.load_taxonomy(self.root)
        development_classes, test_classes = self.partition_class_ids(taxonomy)
        genus_field = self.SPLIT_SUPERCATEGORY_FIELD
        marker = {
            "dataset": "FGVCFungi",
            "dataset_page": self.DATASET_PAGE,
            "image_size": list(self.TARGET_IMAGE_SIZE),
            "image_count": int(stats["archive_images"]),
            "class_count": len(taxonomy),
            "jpeg_quality": int(self.JPEG_QUALITY),
            "written_images": int(stats["written_images"]),
            "reused_images": int(stats["reused_images"]),
            "source_archive": source_description,
            "kaggle_competition": self.KAGGLE_COMPETITION,
            "pooled_sources": list(self.POOLED_SOURCES),
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "split_basis": "alternating_class_ids_within_genus",
            "split_supercategory_field": genus_field,
            "genus_count": len(
                {taxon[genus_field] for taxon in taxonomy.values()}
            ),
            "development_class_fraction": self.DEVELOPMENT_CLASS_FRACTION,
            "test_class_fraction": self.TEST_CLASS_FRACTION,
            "development_class_count": len(development_classes),
            "test_class_count": len(test_classes),
            "development_classes": list(development_classes),
            "held_out_test_classes": list(test_classes),
            "development_class_names": [
                taxonomy[class_id]["species"]
                for class_id in development_classes
            ],
            "held_out_test_class_names": [
                taxonomy[class_id]["species"] for class_id in test_classes
            ],
            "official_image_level_split_used": False,
            "license_note": (
                "source photographs: non-commercial research only"
            ),
        }
        self._write_json_atomic(self.root / self.COMPLETE_MARKER, marker)

    @classmethod
    def partition_class_ids(cls, taxonomy):
        """Halve the species with one stride through genus-sorted class ids."""

        genus_field = cls.SPLIT_SUPERCATEGORY_FIELD
        ranked_class_ids = sorted(
            taxonomy,
            key=lambda class_id: (
                str(taxonomy[class_id][genus_field]),
                int(class_id),
            ),
        )
        development_classes = tuple(sorted(ranked_class_ids[0::2]))
        test_classes = tuple(sorted(ranked_class_ids[1::2]))
        if set(development_classes) & set(test_classes):
            raise RuntimeError("FGVCx Fungi class partition is not disjoint")
        if set(development_classes) | set(test_classes) != set(taxonomy):
            raise RuntimeError(
                "FGVCx Fungi class partition does not cover every species"
            )
        if abs(len(development_classes) - len(test_classes)) > 1:
            raise RuntimeError(
                "FGVCx Fungi halves differ by "
                f"{abs(len(development_classes) - len(test_classes))} species; "
                "one stride through the genus-sorted ids cannot exceed one"
            )
        development_set = set(development_classes)
        for genus, class_ids in cls._class_ids_by_genus(taxonomy).items():
            development_count = sum(
                1 for class_id in class_ids if class_id in development_set
            )
            if abs(2 * development_count - len(class_ids)) > 1:
                raise RuntimeError(
                    f"FGVCx Fungi genus {genus!r} is split "
                    f"{development_count}/{len(class_ids) - development_count} "
                    "instead of in half"
                )
        return development_classes, test_classes

    @classmethod
    def load_taxonomy(cls, root):
        """Read the species taxonomy out of the prepared image directories."""

        root = Path(root)
        image_root = root / cls.IMAGE_DIRECTORY
        try:
            class_directories = sorted(
                path.name for path in image_root.iterdir() if path.is_dir()
            )
        except OSError as error:
            raise ValueError(
                f"Could not list FGVCx Fungi class directories in {image_root}"
            ) from error
        return cls.build_taxonomy(class_directories)

    @classmethod
    def build_taxonomy(cls, class_directories):
        """Map official class ids to their directory, species, and genus."""

        taxonomy = {}
        for directory_name in class_directories:
            class_id, species = cls._parse_class_directory_name(directory_name)
            if class_id in taxonomy:
                raise ValueError(
                    f"Duplicate FGVCx Fungi class id {class_id} in "
                    f"{directory_name!r} and "
                    f"{taxonomy[class_id]['directory']!r}"
                )
            taxonomy[class_id] = {
                "directory": directory_name,
                "species": species,
                cls.SPLIT_SUPERCATEGORY_FIELD: species.split()[0],
            }

        if len(taxonomy) != cls.EXPECTED_CLASS_COUNT:
            raise ValueError(
                "FGVCx Fungi must contain "
                f"{cls.EXPECTED_CLASS_COUNT} species directories, found "
                f"{len(taxonomy)}"
            )
        expected_class_ids = set(range(cls.EXPECTED_CLASS_COUNT))
        if set(taxonomy) != expected_class_ids:
            missing = sorted(expected_class_ids - set(taxonomy))
            unexpected = sorted(set(taxonomy) - expected_class_ids)
            raise ValueError(
                "FGVCx Fungi class ids must be contiguous from zero: "
                f"missing={missing[:10]}, unexpected={unexpected[:10]}"
            )
        return taxonomy

    @classmethod
    def _parse_class_directory_name(cls, directory_name):
        """Split ``0226.Cortinarius malicorius`` into its id and species."""

        class_id, separator, species = str(directory_name).partition(".")
        # Species names carry their own dots ("Amanita citrina var. citrina"),
        # so only the zero-padded id in front of the first dot is parsed.
        if (
            not separator
            or len(class_id) != 4
            or not class_id.isdigit()
            or not species.strip()
        ):
            raise ValueError(
                "FGVCx Fungi class directories must be named "
                f"'<4-digit class id>.<species>': {directory_name!r}"
            )
        return int(class_id), species.strip()

    @classmethod
    def _class_ids_by_genus(cls, taxonomy):
        grouped_class_ids = {}
        for class_id in sorted(taxonomy):
            genus = str(taxonomy[class_id][cls.SPLIT_SUPERCATEGORY_FIELD])
            grouped_class_ids.setdefault(genus, []).append(int(class_id))
        return grouped_class_ids

    @classmethod
    def _scan_records(cls, root, taxonomy):
        """Return every prepared ``(image path, class id)`` pair."""

        image_root = Path(root) / cls.IMAGE_DIRECTORY
        records = []
        for class_id in sorted(taxonomy):
            class_directory = image_root / taxonomy[class_id]["directory"]
            records.extend(
                (image_path, class_id)
                for image_path in sorted(class_directory.iterdir())
                if image_path.is_file()
                and image_path.suffix.lower() in cls.IMAGE_EXTENSIONS
            )
        return records

    @classmethod
    def _build_class_split_info(
        cls,
        taxonomy,
        development_classes,
        test_classes,
        pooled_image_count,
        development_image_count,
        test_image_count,
    ):
        genus_field = cls.SPLIT_SUPERCATEGORY_FIELD
        development_set = set(development_classes)
        shared_genera = 0
        for class_ids in cls._class_ids_by_genus(taxonomy).values():
            if any(class_id in development_set for class_id in class_ids) and any(
                class_id not in development_set for class_id in class_ids
            ):
                shared_genera += 1
        return {
            "source": "pooled_train_val_images_class_disjoint_50_50",
            "dataset_page": cls.DATASET_PAGE,
            "pooled_sources": list(cls.POOLED_SOURCES),
            "pooled_sample_count": int(pooled_image_count),
            "class_disjoint_test": True,
            "class_split_version": cls.CLASS_SPLIT_VERSION,
            "split_basis": "alternating_class_ids_within_genus",
            "split_supercategory_field": genus_field,
            "development_class_fraction": cls.DEVELOPMENT_CLASS_FRACTION,
            "test_class_fraction": cls.TEST_CLASS_FRACTION,
            "development_class_count": len(development_classes),
            "test_class_count": len(test_classes),
            "development_classes": list(development_classes),
            "held_out_test_classes": list(test_classes),
            "development_sample_count": int(development_image_count),
            "test_sample_count": int(test_image_count),
            "genus_count": len(
                {taxon[genus_field] for taxon in taxonomy.values()}
            ),
            # Genera are halved rather than held out, so a test species can
            # have congeners in development, exactly as Semi-iNat's kingdoms do.
            "genus_disjoint_test": False,
            "genera_shared_by_both_halves": shared_genera,
            "annotation_level": "species",
            "official_image_level_split_used": False,
            "license_note": "source photographs: non-commercial research only",
        }

    def _find_local_image_source(self):
        """Return an already-downloaded archive or complete image tree."""

        extracted_candidates = (
            self.root / self.IMAGE_DIRECTORY,
            self.root / "fungi_train_val" / self.IMAGE_DIRECTORY,
        )
        for candidate in extracted_candidates:
            # A run interrupted mid-archive leaves a partial tree behind, so
            # only a complete one is reused; anything else is fetched again.
            if self._directory_image_count(candidate) == self.EXPECTED_IMAGE_COUNT:
                return candidate

        archive_candidates = (
            self.root / self.ARCHIVE_FILENAME,
            self.root / "fungi_train_val.tar.gz",
        )
        for candidate in archive_candidates:
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def _directory_image_count(cls, directory):
        directory = Path(directory)
        if not directory.is_dir():
            return 0
        return sum(
            1
            for class_directory in directory.iterdir()
            if class_directory.is_dir()
            for image_path in class_directory.iterdir()
            if image_path.is_file()
            and image_path.suffix.lower() in cls.IMAGE_EXTENSIONS
        )

    def _kagglehub_image_source(self):
        """Return the competition copy of the archive, or None if unavailable.

        The official mirror at ``labs.gbif.org`` has been answering 403 since
        at least 2020, so the Kaggle copy is tried before it. Any failure here
        is recorded and reported only if the mirror also fails.
        """

        try:
            import kagglehub
        except ImportError as error:
            self._kagglehub_failure = f"kagglehub is not installed ({error})"
            return None
        try:
            kagglehub.login()
            competition_path = Path(
                kagglehub.competition_download(self.KAGGLE_COMPETITION)
            )
        except Exception as error:
            self._kagglehub_failure = (
                f"kagglehub could not download {self.KAGGLE_COMPETITION!r} "
                f"({error})"
            )
            return None

        if competition_path.is_file():
            return competition_path
        direct_path = competition_path / self.ARCHIVE_FILENAME
        if direct_path.is_file():
            return direct_path
        archive_matches = sorted(
            path
            for path in competition_path.rglob(self.ARCHIVE_FILENAME)
            if path.is_file()
        )
        if len(archive_matches) == 1:
            return archive_matches[0]
        for extracted_directory in (
            competition_path / self.IMAGE_DIRECTORY,
            competition_path / "fungi_train_val" / self.IMAGE_DIRECTORY,
        ):
            if extracted_directory.is_dir():
                return extracted_directory
        self._kagglehub_failure = (
            f"KaggleHub returned neither {self.ARCHIVE_FILENAME!r} nor an "
            f"extracted {self.IMAGE_DIRECTORY!r} tree below {competition_path}"
        )
        return None

    def _stream_and_resize_archive(self):
        try:
            with self._open_url(self.ARCHIVE_URL) as response:
                with self._byte_progress(
                    response,
                    "Streaming official FGVCx Fungi image archive",
                ) as byte_progress:
                    reader = _DownloadProgressReader(response, byte_progress)
                    return self._resize_from_archive_stream(reader)
        except OSError as error:
            kagglehub_failure = getattr(self, "_kagglehub_failure", None)
            kagglehub_note = (
                f" KaggleHub was tried first: {kagglehub_failure}"
                if kagglehub_failure
                else ""
            )
            raise RuntimeError(
                "Could not download the FGVCx Fungi 2018 images from "
                f"{self.ARCHIVE_URL} ({error}). Accept the competition terms "
                f"at {self.KAGGLE_PAGE} and configure a Kaggle API token, or "
                f"place {self.ARCHIVE_FILENAME} in {self.root}, then rerun."
                f"{kagglehub_note}"
            ) from error

    def _resize_from_directory(self, source_directory):
        source_directory = Path(source_directory)
        archive_images = 0
        written_images = 0
        reused_images = 0
        with tqdm(
            total=self.EXPECTED_IMAGE_COUNT,
            desc="Saving 224x224 FGVCx Fungi images",
            unit="image",
            disable=None,
        ) as progress:
            for class_directory in sorted(
                path for path in source_directory.iterdir() if path.is_dir()
            ):
                for source_path in sorted(
                    path
                    for path in class_directory.iterdir()
                    if path.is_file()
                    and path.suffix.lower() in self.IMAGE_EXTENSIONS
                ):
                    archive_images += 1
                    destination = (
                        self.root
                        / self.IMAGE_DIRECTORY
                        / class_directory.name
                        / source_path.name
                    )
                    if self._is_target_sized_image(destination):
                        reused_images += 1
                    else:
                        with source_path.open("rb") as image_source:
                            self._resize_image_to_destination(
                                image_source,
                                destination,
                            )
                        written_images += 1
                    progress.update(1)

        return self._validated_stats(
            archive_images=archive_images,
            written_images=written_images,
            reused_images=reused_images,
        )

    def _resize_from_archive_stream(self, archive_stream):
        seen_paths = set()
        written_images = 0
        reused_images = 0
        with tqdm(
            total=self.EXPECTED_IMAGE_COUNT,
            desc="Saving 224x224 FGVCx Fungi images",
            unit="image",
            disable=None,
        ) as progress:
            with tarfile.open(fileobj=archive_stream, mode="r|gz") as archive:
                for member in archive:
                    relative_path = self._image_member_path(member)
                    if relative_path is None:
                        continue
                    path_key = relative_path.as_posix()
                    if path_key in seen_paths:
                        raise RuntimeError(
                            f"Duplicate image {path_key!r} in the FGVCx Fungi "
                            "archive"
                        )
                    seen_paths.add(path_key)

                    destination = self.root / Path(*relative_path.parts)
                    if self._is_target_sized_image(destination):
                        reused_images += 1
                    else:
                        source = archive.extractfile(member)
                        if source is None:
                            raise RuntimeError(
                                f"Could not read {member.name!r} from the "
                                "FGVCx Fungi archive"
                            )
                        with source:
                            self._resize_image_to_destination(
                                source,
                                destination,
                            )
                        written_images += 1
                    progress.update(1)

        return self._validated_stats(
            archive_images=len(seen_paths),
            written_images=written_images,
            reused_images=reused_images,
        )

    def _validated_stats(self, archive_images, written_images, reused_images):
        if archive_images != self.EXPECTED_IMAGE_COUNT:
            raise RuntimeError(
                "The FGVCx Fungi image source was incomplete: expected "
                f"{self.EXPECTED_IMAGE_COUNT} images, found {archive_images}. "
                "The completion marker was not written; rerun the downloader "
                "to retry."
            )
        return {
            "archive_images": archive_images,
            "written_images": written_images,
            "reused_images": reused_images,
        }

    @classmethod
    def _image_member_path(cls, member):
        if not member.isfile():
            return None
        path = PurePosixPath(member.name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Unsafe path in FGVCx Fungi image archive: {member.name!r}"
            )
        parts = tuple(part for part in path.parts if part != ".")
        if len(parts) < 3 or parts[-3] != cls.IMAGE_DIRECTORY:
            return None
        if path.suffix.lower() not in cls.IMAGE_EXTENSIONS:
            return None
        return PurePosixPath(*parts[-3:])

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


class StanfordDogs(_DinoSizedImageDownloadMixin, Dataset):
    """Stanford Dogs on one of two fixed class-disjoint splits.

    The official image-level train/test lists contain every breed on both
    sides, so a class-disjoint metric-learning split has to be constructed.
    Both rules here pool all 20,580 images and assign complete breeds to one
    half; they differ only in which breeds go where, and both are fixed across
    machines and recorded in run metadata.

    ``sha256_60_40`` (the default) hash-ranks the breed directory names and
    gives the first 60% (72 breeds) to development and the rest (48 breeds) to
    final testing. ``class_id_midpoint`` instead cuts the sorted breed
    directories at the midpoint, 60 breeds against 60, which is the split the
    PUMA unified benchmark uses (arXiv:2309.08944); its loader reads
    ``file_list.mat`` and takes ``range(0, 60)`` for training against
    ``range(60, 120)`` for evaluation. Those ``file_list.mat`` labels are
    ordered by the same alphabetically sorted breed directories this loader
    enumerates, so taking the first 60 of them reproduces PUMA's halves.
    """

    IMAGES_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    IMAGE_DIRECTORY = "Images"
    COMPLETE_MARKER = ".stanford_dogs_224_complete.json"
    EXPECTED_IMAGE_COUNT = 20580
    EXPECTED_CLASS_COUNT = 120
    DEVELOPMENT_CLASS_FRACTION = 0.60
    TEST_CLASS_FRACTION = 0.40
    CLASS_SPLIT_SHA256_60_40 = "sha256_60_40"
    CLASS_SPLIT_CLASS_ID_MIDPOINT = "class_id_midpoint"
    AVAILABLE_CLASS_SPLITS = (
        CLASS_SPLIT_SHA256_60_40,
        CLASS_SPLIT_CLASS_ID_MIDPOINT,
    )
    # The hash-ranked rule shipped first and stays the default so that
    # dataset_protocol=official keeps loading the split its logs record. It
    # also stays the identity written into the download marker, because the
    # marker describes the 224x224 conversion rather than the class partition.
    CLASS_SPLIT_VERSION = "sha256_60_40_v1"
    PUMA_CLASS_SPLIT_VERSION = "stanford_dogs_puma_class_id_midpoint_v1"
    CLASS_SPLIT_VERSIONS = {
        CLASS_SPLIT_SHA256_60_40: CLASS_SPLIT_VERSION,
        CLASS_SPLIT_CLASS_ID_MIDPOINT: PUMA_CLASS_SPLIT_VERSION,
    }
    CLASS_SPLIT_BASES = {
        CLASS_SPLIT_SHA256_60_40: "sha256_ranked_breed_directories",
        CLASS_SPLIT_CLASS_ID_MIDPOINT: "breed_directory_midpoint",
    }
    CLASS_SPLIT_SOURCES = {
        CLASS_SPLIT_SHA256_60_40: "fixed_class_disjoint_60_40",
        CLASS_SPLIT_CLASS_ID_MIDPOINT: "fixed_class_disjoint_60_60",
    }
    CLASS_SPLIT_REFERENCES = {
        CLASS_SPLIT_CLASS_ID_MIDPOINT: "https://arxiv.org/abs/2309.08944",
    }
    CLASS_SPLIT_IMPLEMENTATION_REFERENCES = {
        CLASS_SPLIT_CLASS_ID_MIDPOINT: (
            "https://github.com/sung-yeon-kim/PUMA-WACV25/blob/main/dataset.py"
        ),
    }
    PUMA_DEVELOPMENT_CLASS_COUNT = 60
    AVAILABLE_SPLITS = ("train", "test", "train+test")

    def __init__(
        self,
        root,
        split="train+test",
        transform=None,
        target_transform=None,
        download=False,
        class_split=None,
    ):
        self.root = Path(root)
        if download and not self.is_ready(self.root):
            self.download_224(self.root)
        if not self.is_ready(self.root):
            raise ValueError(
                "Stanford Dogs 224x224 data was not found. Initialize the dataset "
                "with download=True or run scripts/download_stanford_dogs_224.py."
            )
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}")
        class_split = self.CLASS_SPLIT_SHA256_60_40 if class_split is None else class_split
        if class_split not in self.AVAILABLE_CLASS_SPLITS:
            raise ValueError(
                f"class_split must be one of {self.AVAILABLE_CLASS_SPLITS}, "
                f"got {class_split!r}"
            )

        self.split = split
        self.class_split = class_split
        self.transform = transform
        self.target_transform = target_transform
        image_root = self.root / self.IMAGE_DIRECTORY
        class_directories = sorted(path for path in image_root.iterdir() if path.is_dir())
        class_names = [path.name for path in class_directories]
        if len(class_names) != self.EXPECTED_CLASS_COUNT:
            raise ValueError(
                f"Stanford Dogs must contain {self.EXPECTED_CLASS_COUNT} breed directories, "
                f"found {len(class_names)} under {image_root}"
            )

        development_names, test_names = self.partition_class_names(
            class_names, class_split=class_split
        )
        development_set = set(development_names)
        test_set = set(test_names)
        if split == "train":
            selected_names = development_set
        elif split == "test":
            selected_names = test_set
        else:
            selected_names = set(class_names)

        self.classes = class_names
        self.class_to_label = {class_name: index for index, class_name in enumerate(class_names)}
        self.class_names = {
            self.class_to_label[class_name]: self._breed_display_name(class_name)
            for class_name in class_names
        }
        self.development_class_names = list(development_names)
        self.test_class_names = list(test_names)
        self.development_class_labels = [
            self.class_to_label[class_name] for class_name in development_names
        ]
        self.test_class_labels = [self.class_to_label[class_name] for class_name in test_names]

        records = []
        for class_directory in class_directories:
            if class_directory.name not in selected_names:
                continue
            label = self.class_to_label[class_directory.name]
            records.extend(
                (image_path, label)
                for image_path in sorted(class_directory.iterdir())
                if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg"}
            )
        if not records:
            raise ValueError(f"Stanford Dogs split {split!r} contains no images")

        self.paths = [str(image_path) for image_path, _ in records]
        self.labels = [int(label) for _, label in records]
        self.orig_labels = list(self.labels)
        self.class_disjoint_split = True
        self.class_split_info = {
            "source": self.CLASS_SPLIT_SOURCES[class_split],
            "class_disjoint_test": True,
            "class_split": class_split,
            "class_split_version": self.CLASS_SPLIT_VERSIONS[class_split],
            "class_split_basis": self.CLASS_SPLIT_BASES[class_split],
            "development_class_fraction": len(development_names) / len(class_names),
            "test_class_fraction": len(test_names) / len(class_names),
            "development_class_count": len(development_names),
            "test_class_count": len(test_names),
            "development_classes": list(development_names),
            "held_out_test_classes": list(test_names),
            "official_image_level_split_used": False,
        }
        protocol_reference = self.CLASS_SPLIT_REFERENCES.get(class_split)
        if protocol_reference is not None:
            self.class_split_info["protocol_reference"] = protocol_reference
            self.class_split_info["implementation_reference"] = (
                self.CLASS_SPLIT_IMPLEMENTATION_REFERENCES[class_split]
            )

    @classmethod
    def is_ready(cls, root):
        root = Path(root)
        marker_path = root / cls.COMPLETE_MARKER
        if not (root / cls.IMAGE_DIRECTORY).is_dir() or not marker_path.is_file():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            image_size = tuple(int(value) for value in marker["image_size"])
            image_count = int(marker["image_count"])
            class_count = int(marker["class_count"])
            split_version = str(marker["class_split_version"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            return False
        return (
            image_size == cls.TARGET_IMAGE_SIZE
            and image_count == cls.EXPECTED_IMAGE_COUNT
            and class_count == cls.EXPECTED_CLASS_COUNT
            and split_version == cls.CLASS_SPLIT_VERSION
        )

    @classmethod
    def download_224(cls, root):
        root = Path(root)
        if cls.is_ready(root):
            return root
        root.mkdir(parents=True, exist_ok=True)
        downloader = cls.__new__(cls)
        downloader.root = root
        downloader._stream_and_resize_archive()
        return root

    def _stream_and_resize_archive(self):
        archive_images = 0
        written_images = 0
        reused_images = 0
        with self._open_url(self.IMAGES_URL) as response:
            with self._byte_progress(
                response,
                "Streaming official Stanford Dogs image archive",
            ) as byte_progress:
                reader = _DownloadProgressReader(response, byte_progress)
                with tqdm(
                    total=self.EXPECTED_IMAGE_COUNT,
                    desc="Saving 224x224 Stanford Dogs images",
                    unit="image",
                    disable=None,
                ) as image_progress:
                    with tarfile.open(fileobj=reader, mode="r|") as archive:
                        for member in archive:
                            relative_path = self._image_member_path(member)
                            if relative_path is None:
                                continue
                            archive_images += 1
                            destination = self.root / relative_path
                            if self._is_target_sized_image(destination):
                                reused_images += 1
                            else:
                                source = archive.extractfile(member)
                                if source is None:
                                    raise RuntimeError(
                                        f"Could not read image {member.name!r} from the archive"
                                    )
                                with source:
                                    self._resize_image_to_destination(source, destination)
                                written_images += 1
                            image_progress.update(1)

        if archive_images != self.EXPECTED_IMAGE_COUNT:
            raise RuntimeError(
                "The Stanford Dogs archive was incomplete: "
                f"expected {self.EXPECTED_IMAGE_COUNT} images, found {archive_images}. "
                "The completion marker was not written; rerun the downloader to retry."
            )

        class_names = sorted(
            path.name
            for path in (self.root / self.IMAGE_DIRECTORY).iterdir()
            if path.is_dir()
        )
        if len(class_names) != self.EXPECTED_CLASS_COUNT:
            raise RuntimeError(
                "The Stanford Dogs archive had an unexpected breed count: "
                f"expected {self.EXPECTED_CLASS_COUNT}, found {len(class_names)}"
            )
        development_names, test_names = self.partition_class_names(class_names)
        marker = {
            "dataset": "StanfordDogs",
            "image_size": list(self.TARGET_IMAGE_SIZE),
            "image_count": archive_images,
            "class_count": len(class_names),
            "jpeg_quality": self.JPEG_QUALITY,
            "written_images": written_images,
            "reused_images": reused_images,
            "source_archive": self.IMAGES_URL,
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "development_class_fraction": self.DEVELOPMENT_CLASS_FRACTION,
            "test_class_fraction": self.TEST_CLASS_FRACTION,
            "development_classes": list(development_names),
            "held_out_test_classes": list(test_names),
        }
        self._write_json_atomic(self.root / self.COMPLETE_MARKER, marker)

    @classmethod
    def partition_class_names(cls, class_names, class_split=None):
        """Split the breeds in two, by hash rank or at the directory midpoint.

        ``sha256_60_40`` ranks the breed directory names by a versioned
        SHA-256 digest and takes the first 60% for development, which spreads
        the visually similar breeds across both halves rather than letting the
        ImageNet synset order decide.

        ``class_id_midpoint`` takes the first 60 breed directories in sorted
        order instead. That is PUMA's rule: its loader labels images from
        ``file_list.mat`` as ``y = label - 1`` and selects ``range(0, 60)`` for
        training against ``range(60, 120)`` for evaluation. Those labels are
        assigned in sorted breed-directory order, the same order enumerated
        here, so the halves match PUMA's without reading the ``.mat`` file.
        """

        class_split = cls.CLASS_SPLIT_SHA256_60_40 if class_split is None else class_split
        if class_split not in cls.AVAILABLE_CLASS_SPLITS:
            raise ValueError(
                f"class_split must be one of {cls.AVAILABLE_CLASS_SPLITS}, "
                f"got {class_split!r}"
            )
        class_names = sorted(str(class_name) for class_name in class_names)
        if len(class_names) != len(set(class_names)):
            raise ValueError("Stanford Dogs class names must be unique")

        if class_split == cls.CLASS_SPLIT_CLASS_ID_MIDPOINT:
            development_count = cls.PUMA_DEVELOPMENT_CLASS_COUNT
            ranked_names = class_names
        else:
            ranked_names = sorted(
                class_names,
                key=lambda class_name: hashlib.sha256(
                    f"{cls.CLASS_SPLIT_VERSION}:{class_name}".encode("utf-8")
                ).digest(),
            )
            development_count = int(
                round(len(ranked_names) * cls.DEVELOPMENT_CLASS_FRACTION)
            )
        development_names = tuple(sorted(ranked_names[:development_count]))
        test_names = tuple(sorted(ranked_names[development_count:]))
        if not development_names or not test_names:
            raise RuntimeError(
                f"Stanford Dogs {class_split} partition left one half empty; "
                f"{len(class_names)} breeds were offered"
            )
        if set(development_names) & set(test_names):
            raise RuntimeError("Stanford Dogs class partition is not disjoint")
        return development_names, test_names

    @classmethod
    def _image_member_path(cls, member):
        if not member.isfile():
            return None
        path = PurePosixPath(member.name.replace("\\", "/"))
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe path in Stanford Dogs image archive: {member.name!r}")
        if not path.parts or path.parts[0] != cls.IMAGE_DIRECTORY:
            return None
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            return None
        return Path(*path.parts)

    @staticmethod
    def _breed_display_name(class_directory_name):
        _, separator, breed_name = class_directory_name.partition("-")
        if not separator:
            return class_directory_name.replace("_", " ")
        return breed_name.replace("_", " ")

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


class RecursiveUnlabeledImageDataset(Dataset):
    """Load every image below a directory while exposing no semantic labels."""

    def __init__(self, root, transform=None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"External unlabeled image directory does not exist: {self.root}")
        extensions = {extension.lower() for extension in IMG_EXTENSIONS}
        self.paths = sorted(
            path
            for path in self.root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        if not self.paths:
            raise ValueError(f"External unlabeled image directory contains no supported images: {self.root}")
        self.transform = transform
        self.labels = [-1] * len(self.paths)
        self.orig_labels = list(self.labels)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(str(self.paths[index]))
        if self.transform is not None:
            image = self.transform(image)
        return image, -1


class CompCarsModelFilteredUnlabeledImageDataset(Dataset):
    """Load a model-balanced CompCars subset while exposing hidden labels."""

    MODE = "compcars_model_min_count"
    PAPER_LABEL = None
    STRIPPED_PREFIX_DIRS = {"data", "image", "images", "web", "web-nature", "web_nature"}
    WHOLE_CAR_IMAGE_DIRS = {"image", "images"}
    NON_WHOLE_CAR_DIRS = {
        "part",
        "parts",
        "car_part",
        "car_parts",
        "sv_data",
        "surveillance",
        "surveillance-nature",
        "surveillance_nature",
    }

    def __init__(
        self,
        root,
        transform=None,
        min_images_per_model=100,
        candidate_paths=None,
        candidate_source="recursive_images",
        mode=None,
        expected_images=None,
        expected_model_classes=None,
        strict_expected_counts=False,
        calibrate_to_expected_counts=False,
        expected_candidate_images=None,
        expected_candidate_model_classes=None,
        paper_label=None,
    ):
        if min_images_per_model <= 0:
            raise ValueError("min_images_per_model must be positive")
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"External unlabeled image directory does not exist: {self.root}")

        if candidate_paths is None:
            discovered_paths = self.discover_image_paths(self.root)
        else:
            discovered_paths = sorted(Path(path) for path in candidate_paths)
        if not discovered_paths:
            raise ValueError(f"External unlabeled image directory contains no supported images: {self.root}")

        model_keys_by_path = {path: self.infer_model_key(self.root, path) for path in discovered_paths}
        model_counts = Counter(model_keys_by_path.values())
        # The paper filters quote their resulting pool size but not the count
        # threshold that produced it, so the requested threshold may miss the
        # published numbers on a differently packaged copy of the dataset.
        requested_min_images_per_model = int(min_images_per_model)
        min_images_per_model = requested_min_images_per_model
        if calibrate_to_expected_counts and expected_images is not None and expected_model_classes is not None:
            min_images_per_model = self.calibrate_min_images_per_model(
                model_counts,
                requested_min_images_per_model,
                int(expected_images),
                int(expected_model_classes),
            )
        kept_model_keys = sorted(
            model_key
            for model_key, count in model_counts.items()
            if count >= min_images_per_model
        )
        kept_model_key_set = set(kept_model_keys)
        self.paths = [path for path in discovered_paths if model_keys_by_path[path] in kept_model_key_set]
        if not self.paths:
            diagnostics = (
                f"Candidate pool: {len(discovered_paths)} images in {len(model_counts)} model classes "
                f"from candidate_source={candidate_source!r}."
            )
            if expected_images is not None and expected_model_classes is not None:
                diagnostics += (
                    " Nearest count thresholds: "
                    f"{self.nearest_threshold_diagnostics(model_counts, int(expected_images), int(expected_model_classes))}"
                )
            raise ValueError(
                f"CompCars model filtering removed every image at min_images_per_model={min_images_per_model}. "
                f"Lower it or check the directory layout. {diagnostics}"
            )

        self.transform = transform
        self.model_keys = [model_keys_by_path[path] for path in self.paths]
        self.labels = [-1] * len(self.paths)
        self.orig_labels = list(self.labels)
        kept_model_counts = Counter(self.model_keys)
        kept_count_summary = self.summarize_model_counts(kept_model_counts)
        dropped_model_keys = sorted(set(model_counts) - set(kept_model_keys))
        self.filter_info = {
            "mode": self.MODE if mode is None else mode,
            "paper_label": self.PAPER_LABEL if paper_label is None else paper_label,
            "candidate_source": candidate_source,
            "min_images_per_model": int(min_images_per_model),
            "requested_min_images_per_model": requested_min_images_per_model,
            "calibrated_min_images_per_model": min_images_per_model != requested_min_images_per_model,
            "discovered_images": int(len(discovered_paths)),
            "discovered_model_classes": int(len(model_counts)),
            "kept_images": int(len(self.paths)),
            "kept_model_classes": int(len(kept_model_keys)),
            "dropped_images": int(len(discovered_paths) - len(self.paths)),
            "dropped_model_classes": int(len(model_counts) - len(kept_model_keys)),
            "dropped_below_min_count_images": int(sum(model_counts[key] for key in dropped_model_keys)),
            "kept_count_min": kept_count_summary["min"],
            "kept_count_median": kept_count_summary["median"],
            "kept_count_mean": kept_count_summary["mean"],
            "kept_count_max": kept_count_summary["max"],
        }
        self.record_candidate_pool_expectations(
            model_counts=model_counts,
            expected_candidate_images=expected_candidate_images,
            expected_candidate_model_classes=expected_candidate_model_classes,
        )
        self.validate_expected_counts(
            expected_images=expected_images,
            expected_model_classes=expected_model_classes,
            model_counts=model_counts,
            strict_expected_counts=strict_expected_counts,
        )

    @classmethod
    def discover_image_paths(cls, root):
        extensions = {extension.lower() for extension in IMG_EXTENSIONS}
        return sorted(
            path
            for path in Path(root).rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )

    @classmethod
    def infer_model_key(cls, root, image_path):
        relative_parent_parts = list(image_path.relative_to(root).parent.parts)
        while (
            len(relative_parent_parts) > 1
            and relative_parent_parts[0].lower() in cls.STRIPPED_PREFIX_DIRS
        ):
            relative_parent_parts.pop(0)
        if len(relative_parent_parts) >= 2:
            return "/".join(relative_parent_parts[:2])
        if len(relative_parent_parts) == 1:
            return relative_parent_parts[0]
        raise ValueError(
            "CompCars model filtering requires images to be under model directories, "
            f"but found an image directly under {root}: {image_path}"
        )

    @staticmethod
    def calibrate_min_images_per_model(model_counts, requested_min_images, expected_images, expected_model_classes):
        """Return the count threshold that reproduces a paper's published pool size.

        The requested threshold wins whenever it already reproduces the paper,
        so calibration only kicks in for copies of the dataset whose packaging
        shifts the boundary. When no threshold matches, the requested value is
        returned unchanged and the mismatch is reported by
        ``validate_expected_counts``.
        """

        def pool_totals(threshold):
            kept_counts = [count for count in model_counts.values() if count >= threshold]
            return sum(kept_counts), len(kept_counts)

        target = (int(expected_images), int(expected_model_classes))
        if pool_totals(requested_min_images) == target:
            return requested_min_images
        for threshold in sorted(set(model_counts.values())):
            if pool_totals(threshold) == target:
                return int(threshold)
        return requested_min_images

    def record_candidate_pool_expectations(
        self,
        model_counts,
        expected_candidate_images,
        expected_candidate_model_classes,
    ):
        """Record whether the pre-filter candidate pool matches the paper's source subset."""

        if expected_candidate_images is None and expected_candidate_model_classes is None:
            return
        discovered_images = self.filter_info["discovered_images"]
        discovered_model_classes = len(model_counts)
        self.filter_info["expected_candidate_images"] = (
            discovered_images if expected_candidate_images is None else int(expected_candidate_images)
        )
        self.filter_info["expected_candidate_model_classes"] = (
            discovered_model_classes
            if expected_candidate_model_classes is None
            else int(expected_candidate_model_classes)
        )
        self.filter_info["matches_expected_candidate_pool"] = (
            discovered_images == self.filter_info["expected_candidate_images"]
            and discovered_model_classes == self.filter_info["expected_candidate_model_classes"]
        )

    @staticmethod
    def summarize_model_counts(model_counts):
        if not model_counts:
            return {"min": 0, "median": 0.0, "mean": 0.0, "max": 0}
        counts = list(model_counts.values())
        return {
            "min": int(min(counts)),
            "median": float(statistics.median(counts)),
            "mean": float(sum(counts) / len(counts)),
            "max": int(max(counts)),
        }

    def validate_expected_counts(self, expected_images, expected_model_classes, model_counts, strict_expected_counts):
        if expected_images is None and expected_model_classes is None:
            return
        expected_images = len(self.paths) if expected_images is None else int(expected_images)
        expected_model_classes = (
            len(set(self.model_keys)) if expected_model_classes is None else int(expected_model_classes)
        )
        actual_images = len(self.paths)
        actual_model_classes = len(set(self.model_keys))
        self.filter_info["expected_images"] = expected_images
        self.filter_info["expected_model_classes"] = expected_model_classes
        self.filter_info["matches_expected_counts"] = (
            actual_images == expected_images and actual_model_classes == expected_model_classes
        )
        self.filter_info["strict_expected_counts"] = bool(strict_expected_counts)
        if self.filter_info["matches_expected_counts"]:
            return

        threshold_diagnostics = self.nearest_threshold_diagnostics(
            model_counts,
            expected_images,
            expected_model_classes,
        )
        self.filter_info["nearest_count_thresholds"] = threshold_diagnostics
        if not strict_expected_counts:
            return

        paper_label = self.filter_info.get("paper_label") or "paper"
        raise ValueError(
            f"CompCars {paper_label} paper filter did not reproduce the documented subset: "
            f"expected {expected_images} images in {expected_model_classes} model classes, "
            f"got {actual_images} images in {actual_model_classes} model classes. "
            f"candidate_source={self.filter_info['candidate_source']!r}, "
            f"min_images_per_model={self.filter_info['min_images_per_model']}, "
            f"candidate pool={self.filter_info['discovered_images']} images in "
            f"{self.filter_info['discovered_model_classes']} model classes. "
            "Check that external_unlabeled_dir points to the CompCars web-nature whole-car "
            f"classification subset used by {paper_label}. "
            f"Nearest count thresholds: {threshold_diagnostics}"
        )

    @staticmethod
    def nearest_threshold_diagnostics(model_counts, expected_images, expected_model_classes, limit=5):
        rows = []
        for threshold in sorted(set(model_counts.values())):
            kept_counts = [count for count in model_counts.values() if count >= threshold]
            images = int(sum(kept_counts))
            classes = int(len(kept_counts))
            distance = abs(images - expected_images) + 1000 * abs(classes - expected_model_classes)
            rows.append(
                {
                    "threshold": int(threshold),
                    "images": images,
                    "model_classes": classes,
                    "distance": int(distance),
                }
            )
        rows.sort(key=lambda row: (row["distance"], row["threshold"]))
        return rows[:limit]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(str(self.paths[index]))
        if self.transform is not None:
            image = self.transform(image)
        return image, -1


class CompCarsSLADEPaperUnlabeledImageDataset(CompCarsModelFilteredUnlabeledImageDataset):
    """Load the CompCars unlabeled pool described by the SLADE paper.

    SLADE (Duan et al., CVPR 2021) pairs labeled Cars-196 with CompCars as
    unlabeled data and states: "We use CompCars as the unlabeled data. It is
    collected at model level, so we filter out unbalanced categories to avoid
    being biased towards minority classes, resulting in 16,537 images
    categorized into 145 classes."

    The paper pins the resulting pool but neither the source subset nor the
    threshold, and there is no official code release, so this class implements
    the reading that reproduces those counts:

    * candidates come from CompCars' official model-level classification split
      (``train_test_split/classification/{train,test}.txt``), the Part-I subset
      of 30,955 whole-car web-nature images across 431 car models;
    * a category is a make/model directory pair, matching "collected at model
      level" (CompCars annotates make, model and released year);
    * "unbalanced categories" are the models holding fewer than
      ``min_images_per_model`` images, which at the default of 100 keeps 145 of
      the 431 models and 16,537 of the 30,955 images.

    Because the 100-image threshold is inferred rather than quoted, a local copy
    that misses the published counts at that threshold is recalibrated to a
    threshold that hits them exactly (see ``calibrate_min_images_per_model``);
    set ``calibrate_threshold=False`` to keep the requested threshold verbatim.
    ``strict_paper_counts`` turns a remaining mismatch into an error instead of
    a diagnostic recorded in ``filter_info``.
    """

    MODE = "compcars_slade_paper"
    PAPER_LABEL = "SLADE"
    PAPER_MIN_IMAGES_PER_MODEL = 100
    PAPER_TARGET_IMAGES = 16537
    PAPER_TARGET_MODEL_CLASSES = 145
    # CompCars Part-I, the official fine-grained classification split that the
    # paper's model-level filtering starts from.
    CLASSIFICATION_SPLIT_IMAGES = 30955
    CLASSIFICATION_SPLIT_MODEL_CLASSES = 431
    CLASSIFICATION_SPLIT_DIR_NAMES = {"classification"}
    CLASSIFICATION_SPLIT_FILE_NAMES = {"train.txt", "test.txt"}

    def __init__(
        self,
        root,
        transform=None,
        min_images_per_model=PAPER_MIN_IMAGES_PER_MODEL,
        strict_paper_counts=False,
        calibrate_threshold=True,
    ):
        root = Path(root)
        candidate_paths, candidate_source = self.discover_paper_candidate_paths(root)
        super().__init__(
            root=root,
            transform=transform,
            min_images_per_model=min_images_per_model,
            candidate_paths=candidate_paths,
            candidate_source=candidate_source,
            mode=self.MODE,
            expected_images=self.PAPER_TARGET_IMAGES,
            expected_model_classes=self.PAPER_TARGET_MODEL_CLASSES,
            strict_expected_counts=strict_paper_counts,
            calibrate_to_expected_counts=calibrate_threshold,
            expected_candidate_images=(
                self.CLASSIFICATION_SPLIT_IMAGES
                if candidate_source == "classification_split_files"
                else None
            ),
            expected_candidate_model_classes=(
                self.CLASSIFICATION_SPLIT_MODEL_CLASSES
                if candidate_source == "classification_split_files"
                else None
            ),
            paper_label=self.PAPER_LABEL,
        )

    @classmethod
    def discover_paper_candidate_paths(cls, root):
        split_paths = cls.discover_classification_split_paths(root)
        if split_paths:
            return split_paths, "classification_split_files"

        # Without the official split files the closest approximation is every
        # whole-car web-nature image, which covers all 1,716 models rather than
        # the 431 of Part-I; the count check then reports the difference.
        whole_car_paths = [
            path
            for path in cls.discover_image_paths(root)
            if cls.is_whole_car_image_path(root, path)
        ]
        return whole_car_paths, "recursive_whole_car_images"

    @classmethod
    def discover_classification_split_paths(cls, root):
        split_files = cls.find_classification_split_files(root)
        image_paths = []
        seen_paths = set()
        for split_file in split_files:
            for image_path in cls.read_split_file_image_paths(root, split_file):
                if image_path not in seen_paths:
                    image_paths.append(image_path)
                    seen_paths.add(image_path)
        return sorted(image_paths)

    @classmethod
    def find_classification_split_files(cls, root):
        """Return the official classification split lists, else any classification list."""

        text_files = sorted(root.rglob("*.txt"))
        official_split_files = [
            path
            for path in text_files
            if path.parent.name.lower() in cls.CLASSIFICATION_SPLIT_DIR_NAMES
            and path.name.lower() in cls.CLASSIFICATION_SPLIT_FILE_NAMES
        ]
        if official_split_files:
            return official_split_files
        return [
            path
            for path in text_files
            if any("classification" in part.lower() for part in path.parts)
        ]

    @classmethod
    def read_split_file_image_paths(cls, root, split_file):
        paths = []
        image_roots = cls.split_file_image_roots(root, split_file)
        for line in split_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            token = cls.first_image_token(line)
            if token is None:
                continue
            resolved = cls.resolve_split_image_path(image_roots, token)
            if resolved is not None:
                paths.append(resolved)
        return paths

    @classmethod
    def split_file_image_roots(cls, root, split_file):
        """Return the directories a split-file entry can be relative to.

        CompCars lists entries relative to ``image/`` inside the release root,
        so the release root is derived from the split file itself
        (``<release>/train_test_split/classification/train.txt``) to keep the
        lookup working when ``external_unlabeled_dir`` points above or below it.
        """

        bases = [root, root / "data"]
        release_root = split_file.parent.parent.parent
        # Only trust the derived release root while it stays below the
        # configured root, so entries never resolve outside the pool directory.
        if release_root.is_dir() and (release_root == root or root in release_root.parents):
            bases.insert(0, release_root)
        image_roots = []
        for base in bases:
            for image_dir in ("image", "images", None):
                candidate = base if image_dir is None else base / image_dir
                if candidate not in image_roots:
                    image_roots.append(candidate)
        return image_roots

    @classmethod
    def first_image_token(cls, line):
        extensions = {extension.lower() for extension in IMG_EXTENSIONS}
        for token in line.split():
            cleaned = token.strip().strip(",;")
            suffix = Path(cleaned.replace("\\", "/")).suffix.lower()
            if suffix in extensions:
                return cleaned
        return None

    @classmethod
    def resolve_split_image_path(cls, image_roots, token):
        relative_path = Path(token.replace("\\", "/"))
        for image_root in image_roots:
            candidate = image_root / relative_path
            if candidate.is_file():
                return candidate
        return None

    @classmethod
    def is_whole_car_image_path(cls, root, image_path):
        path_parts = [part.lower() for part in Path(root).parts + image_path.relative_to(root).parts[:-1]]
        if any(part in cls.NON_WHOLE_CAR_DIRS for part in path_parts):
            return False
        return (
            Path(root).name.lower() in cls.WHOLE_CAR_IMAGE_DIRS
            or any(part in cls.WHOLE_CAR_IMAGE_DIRS for part in path_parts)
        )


class CompCarsSTMLPaperUnlabeledImageDataset(CompCarsSLADEPaperUnlabeledImageDataset):
    """Load the CompCars unlabeled subset used by STML, which is SLADE's subset.

    STML (Kim et al., CVPR 2022) states that it "directly adopt[s] the
    evaluation protocol of SLADE", including the {Cars, CompCars} pairing, so
    this only relabels :class:`CompCarsSLADEPaperUnlabeledImageDataset` for
    configs that name the STML filter.
    """

    MODE = "compcars_stml_paper"
    PAPER_LABEL = "STML"


class _NABirdsMetadataMixin:
    """Locate an official NABirds directory and read its indexed metadata.

    Shared by the labeled dataset and the unlabeled pool: both accept either the
    official directory itself or a parent holding it, and both must reject
    unsafe paths listed in ``images.txt``.
    """

    REQUIRED_METADATA_FILES = (
        "images.txt",
        "image_class_labels.txt",
        "classes.txt",
    )

    @classmethod
    def find_dataset_root(cls, root):
        root = Path(root)
        preferred_candidates = [root, root / "nabirds", root / "NABirds"]
        for child in root.iterdir():
            if child.is_dir() and child not in preferred_candidates:
                preferred_candidates.append(child)

        matches = []
        for candidate in preferred_candidates:
            if all((candidate / filename).is_file() for filename in cls.REQUIRED_METADATA_FILES):
                if not (candidate / "images").is_dir():
                    continue
                resolved = candidate.resolve()
                if resolved not in matches:
                    matches.append(resolved)

        if not matches:
            expected = ", ".join(cls.REQUIRED_METADATA_FILES)
            raise ValueError(
                "NABirds metadata was not found. Expected an official NABirds directory containing "
                f"images/ and {expected} directly below {root} or one of its immediate subdirectories."
            )
        if len(matches) > 1:
            raise ValueError(
                "Multiple NABirds dataset roots were found below the external directory: "
                + ", ".join(str(path) for path in matches)
            )
        return matches[0]

    @staticmethod
    def read_indexed_text_file(path, value_name, key_type=str):
        """Read a ``<id> <value>`` metadata file into an ``id -> value`` dict.

        NABirds keys its per-image files with opaque UUIDs and its class files
        with integer node ids, so the key type is chosen per file rather than
        assumed.
        """

        records = {}
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            columns = line.split(maxsplit=1)
            if len(columns) != 2:
                raise ValueError(f"Invalid NABirds {value_name} row at {path}:{line_number}")
            try:
                record_id = key_type(columns[0])
            except ValueError as exc:
                raise ValueError(
                    f"Invalid NABirds ID at {path}:{line_number}: {columns[0]!r}"
                ) from exc
            if record_id in records:
                raise ValueError(f"Duplicate NABirds ID {record_id} at {path}:{line_number}")
            records[record_id] = columns[1]
        if not records:
            raise ValueError(f"NABirds metadata file is empty: {path}")
        return records

    @classmethod
    def read_indexed_int_file(cls, path, value_name, key_type=str):
        records = cls.read_indexed_text_file(path, value_name, key_type=key_type)
        try:
            return {record_id: int(value) for record_id, value in records.items()}
        except ValueError as exc:
            raise ValueError(f"NABirds {value_name} values must be integers: {path}") from exc

    def resolve_image_path(self, image_name):
        normalized_name = str(image_name).replace("\\", "/")
        relative_path = Path(normalized_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(f"NABirds images.txt contains an unsafe image path: {image_name!r}")
        return self.dataset_root / "images" / relative_path



class NABirdsUnlabeledImageDataset(_NABirdsMetadataMixin, Dataset):
    """Load every official NABirds image while keeping its annotations hidden.

    NABirds is used as the additional unlabeled bird collection for CUB in the
    SLADE/STML semi-supervised protocol.  Reading ``images.txt`` rather than
    recursively accepting every image makes the pool reproducible and avoids
    accidentally including unrelated files placed below the download root.
    """

    MODE = "nabirds"

    def __init__(self, root, transform=None):
        self.root = Path(root)
        if not self.root.is_dir():
            raise ValueError(f"External unlabeled image directory does not exist: {self.root}")

        self.dataset_root = self.find_dataset_root(self.root)
        image_names = self.read_indexed_text_file(self.dataset_root / "images.txt", "image path")
        source_class_ids = self.read_indexed_int_file(
            self.dataset_root / "image_class_labels.txt",
            "class label",
        )
        class_names = self.read_indexed_text_file(
            self.dataset_root / "classes.txt", "class name", key_type=int
        )

        image_ids = set(image_names)
        label_ids = set(source_class_ids)
        if image_ids != label_ids:
            missing_labels = sorted(image_ids - label_ids)
            missing_images = sorted(label_ids - image_ids)
            raise ValueError(
                "NABirds metadata IDs do not align between images.txt and image_class_labels.txt: "
                f"{len(missing_labels)} images lack labels and {len(missing_images)} labels lack images"
            )

        unknown_class_ids = sorted(set(source_class_ids.values()) - set(class_names))
        if unknown_class_ids:
            raise ValueError(
                "NABirds image_class_labels.txt references class IDs missing from classes.txt: "
                f"{unknown_class_ids[:10]}"
            )

        self.image_ids = sorted(image_names)
        self.image_names = [image_names[image_id] for image_id in self.image_ids]
        self.paths = [self.resolve_image_path(image_name) for image_name in self.image_names]
        missing_paths = [path for path in self.paths if not path.is_file()]
        if missing_paths:
            examples = ", ".join(str(path) for path in missing_paths[:3])
            raise ValueError(
                f"NABirds is missing {len(missing_paths)} image files listed in images.txt; examples: {examples}"
            )

        # Source annotations are used only for integrity diagnostics and then
        # discarded. Every sample exposed to the training pipeline has label -1.
        ordered_source_class_ids = [source_class_ids[image_id] for image_id in self.image_ids]
        self.transform = transform
        self.labels = [-1] * len(self.paths)
        self.orig_labels = list(self.labels)
        used_class_ids = set(ordered_source_class_ids)
        self.filter_info = {
            "mode": self.MODE,
            "candidate_source": "official_metadata",
            "category_unit": "visual categories",
            "dataset_root": str(self.dataset_root),
            "discovered_images": int(len(self.paths)),
            "discovered_model_classes": int(len(used_class_ids)),
            "kept_images": int(len(self.paths)),
            "kept_model_classes": int(len(used_class_ids)),
            "dropped_images": 0,
            "dropped_model_classes": 0,
            "metadata_class_count": int(len(class_names)),
        }

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(str(self.paths[index]))
        if self.transform is not None:
            image = self.transform(image)
        return image, -1


class NABirds(_NABirdsMetadataMixin, _DinoSizedImageDownloadMixin, Dataset):
    """NABirds with a hierarchy-aware class-disjoint DML split.

    The official ``train_test_split.txt`` puts every visual category on both
    sides of its boundary, which is a classification protocol rather than a
    metric learning one. This wrapper pools all 48,562 images and assigns
    complete visual categories to either the development half or the held-out
    test half, following the CUB convention of cutting the ordered class ids in
    two.

    NABirds splits sexes and age classes of one species into separate visual
    categories ("... (Adult male)", "... (Female/juvenile)") and lists them as
    children of a shared hierarchy node. A plain midpoint cut can therefore put
    one plumage of a species in development and another in the held-out test
    set, which leaks a near-identical training class into the final test. Both
    partitions below assign complete hierarchy-parent groups instead, ordered
    by their lowest class id, so no parent node keeps categories on both sides.

    ``class_split`` picks how those groups are dealt out. The default takes a
    prefix of them, which is what ``dataset_protocol=official`` uses.
    ``alternating_parent_groups`` strides them instead and backs
    ``dataset_protocol=nabirds_alternating_277_278``; see
    :meth:`partition_class_ids` for what that buys.

    ``class_id_midpoint`` ignores the hierarchy entirely and cuts the sorted
    class ids at 278/277, reproducing the PUMA benchmark. It is the one
    partition here that lets a species straddle the split, so
    ``class_split_info`` reports ``straddling_parent_count`` for every rule.
    ``docs/nabirds_puma_class_split.json`` lists the categories it selects.

    Cornell releases NABirds only after its terms are accepted, so the
    automatic download reads a Kaggle mirror of the 2015 release and stores it
    as 224x224 JPEGs in the official layout; see :meth:`download_224`. Pointing
    ``data/NABirds`` at an extracted official directory still works and is
    preferred if the terms were accepted at the source.
    """

    DATASET_PAGE = "https://dl.allaboutbirds.org/nabirds"
    CUB_PROTOCOL_REFERENCE = (
        "https://github.com/KevinMusgrave/pytorch-metric-learning/"
        "blob/master/src/pytorch_metric_learning/datasets/cub.py"
    )
    REQUIRED_METADATA_FILES = (
        "images.txt",
        "image_class_labels.txt",
        "classes.txt",
        "hierarchy.txt",
    )
    EXPECTED_IMAGE_COUNT = 48_562
    EXPECTED_CLASS_COUNT = 555
    CLASS_SPLIT_PREFIX_GROUPS = "parent_group_prefix"
    CLASS_SPLIT_ALTERNATING_GROUPS = "alternating_parent_groups"
    CLASS_SPLIT_CLASS_ID_MIDPOINT = "class_id_midpoint"
    AVAILABLE_CLASS_SPLITS = (
        CLASS_SPLIT_PREFIX_GROUPS,
        CLASS_SPLIT_ALTERNATING_GROUPS,
        CLASS_SPLIT_CLASS_ID_MIDPOINT,
    )
    # The prefix rule shipped first and stays the default so that
    # dataset_protocol=official keeps loading the split its logs record.
    CLASS_SPLIT_VERSION = "nabirds_parent_group_class_disjoint_v1"
    ALTERNATING_CLASS_SPLIT_VERSION = "nabirds_alternating_parent_groups_v1"
    PUMA_CLASS_SPLIT_VERSION = "nabirds_puma_class_id_midpoint_v1"
    CLASS_SPLIT_VERSIONS = {
        CLASS_SPLIT_PREFIX_GROUPS: CLASS_SPLIT_VERSION,
        CLASS_SPLIT_ALTERNATING_GROUPS: ALTERNATING_CLASS_SPLIT_VERSION,
        CLASS_SPLIT_CLASS_ID_MIDPOINT: PUMA_CLASS_SPLIT_VERSION,
    }
    CLASS_SPLIT_BASES = {
        CLASS_SPLIT_PREFIX_GROUPS: "hierarchy_parent_groups_in_class_id_order",
        CLASS_SPLIT_ALTERNATING_GROUPS: (
            "alternating_hierarchy_parent_groups_in_class_id_order"
        ),
        CLASS_SPLIT_CLASS_ID_MIDPOINT: "class_id_midpoint_ignoring_hierarchy",
    }
    CLASS_SPLIT_REFERENCES = {
        CLASS_SPLIT_CLASS_ID_MIDPOINT: "https://arxiv.org/abs/2309.08944",
    }
    CLASS_SPLIT_IMPLEMENTATION_REFERENCES = {
        CLASS_SPLIT_CLASS_ID_MIDPOINT: (
            "https://github.com/sung-yeon-kim/PUMA-WACV25/blob/main/dataset.py"
        ),
    }
    # Third-party mirror of the 2015 release. The images are Cornell's and stay
    # under the terms on DATASET_PAGE; the mirror only removes the click-through.
    KAGGLE_DATASET_PAGE = "https://www.kaggle.com/datasets/duyminhle/nabirds"
    KAGGLE_DATASET = "duyminhle/nabirds"
    KAGGLE_ARCHIVE_FILENAME = "archive.zip"
    LOCAL_ARCHIVE_CANDIDATES = ("archive.zip", "nabirds.zip")
    COMPLETE_MARKER = ".nabirds_224_complete.json"
    AVAILABLE_SPLITS = ("train", "test", "train+test")

    def __init__(
        self,
        root,
        split="train+test",
        transform=None,
        target_transform=None,
        download=False,
        class_split=None,
    ):
        self.root = Path(root)
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(
                f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}"
            )
        class_split = (
            self.CLASS_SPLIT_PREFIX_GROUPS if class_split is None else class_split
        )
        if class_split not in self.AVAILABLE_CLASS_SPLITS:
            raise ValueError(
                f"class_split must be one of {self.AVAILABLE_CLASS_SPLITS}, "
                f"got {class_split!r}"
            )
        self.class_split = class_split
        if download and not self.is_ready(self.root):
            self.download_224(self.root)
        self.dataset_root = self.find_dataset_root(self.root)

        image_names = self.read_indexed_text_file(
            self.dataset_root / "images.txt", "image path"
        )
        source_class_ids = self.read_indexed_int_file(
            self.dataset_root / "image_class_labels.txt", "class label"
        )
        class_names = self.read_indexed_text_file(
            self.dataset_root / "classes.txt", "class name", key_type=int
        )
        parents = self._read_hierarchy(self.dataset_root / "hierarchy.txt")

        if set(image_names) != set(source_class_ids):
            missing_labels = sorted(set(image_names) - set(source_class_ids))
            missing_images = sorted(set(source_class_ids) - set(image_names))
            raise ValueError(
                "NABirds metadata IDs do not align between images.txt and "
                f"image_class_labels.txt: {len(missing_labels)} images lack "
                f"labels and {len(missing_images)} labels lack images"
            )
        unknown_class_ids = sorted(set(source_class_ids.values()) - set(class_names))
        if unknown_class_ids:
            raise ValueError(
                "NABirds image_class_labels.txt references class IDs missing "
                f"from classes.txt: {unknown_class_ids[:10]}"
            )
        if len(image_names) != self.EXPECTED_IMAGE_COUNT:
            raise ValueError(
                f"NABirds images.txt lists {len(image_names)} images; expected "
                f"{self.EXPECTED_IMAGE_COUNT}"
            )

        # Only the leaf nodes of the hierarchy carry images; classes.txt also
        # names the interior taxa, which no sample is ever labeled with.
        self.source_class_ids = tuple(sorted(set(source_class_ids.values())))
        if len(self.source_class_ids) != self.EXPECTED_CLASS_COUNT:
            raise ValueError(
                f"NABirds labels cover {len(self.source_class_ids)} visual "
                f"categories; expected {self.EXPECTED_CLASS_COUNT}"
            )
        # Labels are densified to 0..N-1 in class id order; the original NABirds
        # ids stay available through source_class_ids and class_split_info.
        self.class_to_label = {
            class_id: label for label, class_id in enumerate(self.source_class_ids)
        }
        self.class_ids = list(range(len(self.source_class_ids)))
        self.classes = [class_names[class_id] for class_id in self.source_class_ids]
        self.class_names = dict(zip(self.class_ids, self.classes))

        development_source_ids, test_source_ids = self.partition_class_ids(
            self.source_class_ids, parents, class_split=class_split
        )
        development_classes = tuple(
            self.class_to_label[class_id] for class_id in development_source_ids
        )
        test_classes = tuple(
            self.class_to_label[class_id] for class_id in test_source_ids
        )
        development_set = set(development_classes)
        test_set = set(test_classes)
        if split == "train":
            selected_classes = development_set
        elif split == "test":
            selected_classes = test_set
        else:
            selected_classes = development_set | test_set

        ordered_image_ids = sorted(image_names)
        records = []
        development_sample_count = 0
        for image_id in ordered_image_ids:
            label = self.class_to_label[source_class_ids[image_id]]
            if label in development_set:
                development_sample_count += 1
            if label in selected_classes:
                records.append(
                    (self.resolve_image_path(image_names[image_id]), label, image_id)
                )
        if not records:
            raise ValueError(f"NABirds split {split!r} contains no images")
        missing_paths = [path for path, _, _ in records if not path.is_file()]
        if missing_paths:
            examples = ", ".join(str(path) for path in missing_paths[:3])
            raise ValueError(
                f"NABirds is missing {len(missing_paths)} image files listed in "
                f"images.txt; examples: {examples}"
            )

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_ids = [image_id for _, _, image_id in records]
        self.image_names = [image_names[image_id] for image_id in self.image_ids]
        self.paths = [str(path) for path, _, _ in records]
        self.labels = [int(label) for _, label, _ in records]
        self.orig_labels = list(self.labels)
        self.development_class_labels = list(development_classes)
        self.test_class_labels = list(test_classes)
        pooled_sample_count = len(ordered_image_ids)
        test_sample_count = pooled_sample_count - development_sample_count
        straddling_parents = self.count_straddling_parents(
            development_source_ids, test_source_ids, parents
        )
        self.class_disjoint_split = True
        self.class_split_info = {
            "source": "pooled_official_train_test_class_disjoint",
            "dataset_page": self.DATASET_PAGE,
            "protocol_reference": self.CUB_PROTOCOL_REFERENCE,
            "dataset_root": str(self.dataset_root),
            "pooled_sources": ["official_train", "official_test"],
            "pooled_sample_count": pooled_sample_count,
            "class_disjoint_test": True,
            "class_split_version": self.CLASS_SPLIT_VERSIONS[class_split],
            "split_basis": self.CLASS_SPLIT_BASES[class_split],
            "development_class_count": len(development_classes),
            "test_class_count": len(test_classes),
            "development_classes": list(development_classes),
            "held_out_test_classes": list(test_classes),
            "development_source_class_ids": list(development_source_ids),
            "held_out_test_source_class_ids": list(test_source_ids),
            "development_parent_count": len(
                {parents[class_id] for class_id in development_source_ids}
            ),
            "test_parent_count": len(
                {parents[class_id] for class_id in test_source_ids}
            ),
            "development_sample_count": development_sample_count,
            "test_sample_count": test_sample_count,
            "development_sample_fraction": development_sample_count / pooled_sample_count,
            "test_sample_fraction": test_sample_count / pooled_sample_count,
            # Non-zero only for class_id_midpoint: the count of species whose
            # plumages ended up on opposite sides of an otherwise
            # category-disjoint split.
            "straddling_parent_count": len(straddling_parents),
            "straddling_parent_ids": list(straddling_parents),
            "official_image_level_split_used": False,
        }
        if class_split in self.CLASS_SPLIT_REFERENCES:
            self.class_split_info["class_split_reference"] = self.CLASS_SPLIT_REFERENCES[
                class_split
            ]
        if class_split in self.CLASS_SPLIT_IMPLEMENTATION_REFERENCES:
            self.class_split_info["class_split_implementation_reference"] = (
                self.CLASS_SPLIT_IMPLEMENTATION_REFERENCES[class_split]
            )

    @classmethod
    def is_ready(cls, root):
        try:
            cls.find_dataset_root(root)
        except (ValueError, OSError):
            return False
        return True

    @classmethod
    def is_224_complete(cls, root):
        """Whether :meth:`download_224` finished writing this root.

        :meth:`is_ready` only checks that an official directory is present, so
        it is also true for a full resolution copy extracted by hand. This adds
        the marker the resize writes, and is what decides whether the download
        still has work to do.
        """

        root = Path(root)
        marker_path = root / cls.COMPLETE_MARKER
        if not marker_path.is_file():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            image_size = tuple(int(value) for value in marker["image_size"])
            image_count = int(marker["image_count"])
            class_count = int(marker["class_count"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            return False
        return (
            image_size == cls.TARGET_IMAGE_SIZE
            and image_count == cls.EXPECTED_IMAGE_COUNT
            and class_count == cls.EXPECTED_CLASS_COUNT
            and cls.is_ready(root)
        )

    @classmethod
    def download_224(cls, root, source=None):
        """Fetch NABirds from Kaggle and store it as 224x224 JPEGs.

        Cornell releases NABirds only after its terms are accepted, so the
        automatic path pulls the ``duyminhle/nabirds`` Kaggle mirror of the
        2015 release. ``source`` overrides that with an archive or an official
        directory that is already on disk; without it, an unpacked archive
        beside the destination is used before anything is downloaded.

        The metadata files are copied verbatim and the images are written to
        ``root/images/...`` under their official relative paths, so the result
        is an official layout that :meth:`find_dataset_root` accepts. Resizing
        is resumable: images already at the target size are left alone.
        """

        root = Path(root)
        if cls.is_224_complete(root):
            return root
        root.mkdir(parents=True, exist_ok=True)
        downloader = cls.__new__(cls)
        downloader.root = root
        downloader._download_and_resize(source)
        return root

    def _download_and_resize(self, source=None):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        resolved = self._resolve_source(source)

        if resolved.is_dir():
            image_names, class_count = self._read_directory_metadata(resolved)
            self._copy_metadata_from_directory(resolved)
            stats = self._resize_from_directory(resolved, image_names)
        elif resolved.is_file() and zipfile.is_zipfile(resolved):
            image_names, class_count, prefix = self._read_zip_metadata(resolved)
            self._copy_metadata_from_zip(resolved, prefix)
            stats = self._resize_from_zip(resolved, prefix, image_names)
        else:
            raise RuntimeError(f"Unsupported NABirds source: {resolved}")

        marker = {
            "dataset": "NABirds",
            "image_size": list(self.TARGET_IMAGE_SIZE),
            "image_count": len(image_names),
            "class_count": class_count,
            "jpeg_quality": int(self.JPEG_QUALITY),
            "written_images": int(stats["written_images"]),
            "reused_images": int(stats["reused_images"]),
            "source": str(resolved),
            "kaggle_dataset": self.KAGGLE_DATASET,
            "kaggle_dataset_page": self.KAGGLE_DATASET_PAGE,
            "dataset_page": self.DATASET_PAGE,
            "metadata_files": list(self.REQUIRED_METADATA_FILES),
            # The class split is decided at load time from these metadata
            # files, so the download stays split-agnostic.
            "official_image_level_split_used": False,
        }
        self._write_json_atomic(self.root / self.COMPLETE_MARKER, marker)

    def _resolve_source(self, source=None):
        if source is not None:
            source = Path(source)
            if source.is_dir():
                return self.find_dataset_root(source)
            if source.is_file() and zipfile.is_zipfile(source):
                return source
            raise RuntimeError(
                "NABirds source must be an official directory or a ZIP "
                f"archive, got: {source}"
            )

        # Kaggle hands out the mirror as archive.zip; accept one sitting in the
        # destination or beside it so a 10 GB download is never repeated. The
        # name is generic, so an archive that is not NABirds is skipped rather
        # than reported as broken.
        for candidate in self.LOCAL_ARCHIVE_CANDIDATES:
            local_archive = self.root / candidate
            if self._is_nabirds_zip(local_archive):
                return local_archive
        beside_root = self.root.parent / self.KAGGLE_ARCHIVE_FILENAME
        if self._is_nabirds_zip(beside_root):
            return beside_root

        try:
            import kagglehub
        except ImportError as error:
            raise RuntimeError(
                "kagglehub is required to download NABirds. Install the "
                "project requirements or run `pip install kagglehub`."
            ) from error

        try:
            kaggle_path = Path(kagglehub.dataset_download(self.KAGGLE_DATASET))
        except Exception as error:
            raise RuntimeError(
                f"Could not download NABirds from {self.KAGGLE_DATASET_PAGE} "
                "with KaggleHub. Authenticate with Kaggle, or pass an already "
                "downloaded archive as source."
            ) from error
        return self._find_source_below(kaggle_path)

    @classmethod
    def _find_source_below(cls, kaggle_path):
        kaggle_path = Path(kaggle_path)
        if kaggle_path.is_file():
            if zipfile.is_zipfile(kaggle_path):
                return kaggle_path
            raise RuntimeError(
                f"KaggleHub returned a non-ZIP NABirds file: {kaggle_path}"
            )
        if not kaggle_path.is_dir():
            raise RuntimeError(
                f"KaggleHub returned a missing NABirds path: {kaggle_path}"
            )

        # KaggleHub either expands the archive for us or leaves the ZIP behind.
        try:
            return cls.find_dataset_root(kaggle_path)
        except (ValueError, OSError):
            pass
        expanded_matches = sorted(
            {
                metadata.parent
                for metadata in kaggle_path.rglob("images.txt")
                if (metadata.parent / "images").is_dir()
            }
        )
        if len(expanded_matches) == 1:
            return expanded_matches[0]
        if len(expanded_matches) > 1:
            raise RuntimeError(
                f"Expected one expanded NABirds directory below {kaggle_path}, "
                f"found {len(expanded_matches)}"
            )
        archive_matches = sorted(
            path for path in kaggle_path.rglob("*.zip") if cls._is_nabirds_zip(path)
        )
        if len(archive_matches) == 1:
            return archive_matches[0]
        raise RuntimeError(
            "KaggleHub returned neither an expanded NABirds directory nor a "
            f"NABirds ZIP archive below {kaggle_path}"
        )

    @classmethod
    def _is_nabirds_zip(cls, path):
        """Whether ``path`` is a ZIP archive holding the official metadata."""

        path = Path(path)
        if not path.is_file() or not zipfile.is_zipfile(path):
            return False
        try:
            with zipfile.ZipFile(path, "r") as archive:
                cls._find_zip_prefix(archive)
        except (RuntimeError, zipfile.BadZipFile, OSError):
            return False
        return True

    @classmethod
    def _find_zip_prefix(cls, archive):
        """Return the directory inside ``archive`` holding the metadata files.

        Kaggle mirrors sometimes wrap the release in a ``nabirds/`` directory
        and sometimes do not, so the prefix is found rather than assumed.
        """

        members = set(archive.namelist())
        candidates = []
        for member_name in members:
            member_path = PurePosixPath(member_name.replace("\\", "/"))
            if member_path.name != "images.txt":
                continue
            prefix = member_path.parent
            if all(
                (prefix / filename).as_posix() in members
                for filename in cls.REQUIRED_METADATA_FILES
            ):
                candidates.append(prefix)
        if len(candidates) != 1:
            raise RuntimeError(
                "Expected exactly one NABirds metadata directory in the "
                f"archive, found {len(candidates)}"
            )
        return candidates[0]

    @classmethod
    def _read_zip_metadata(cls, archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            prefix = cls._find_zip_prefix(archive)

            def read_member(filename):
                member = (prefix / filename).as_posix()
                try:
                    return archive.read(member).decode("utf-8")
                except (KeyError, UnicodeDecodeError) as error:
                    raise ValueError(
                        f"Could not read NABirds {filename} in {archive_path}"
                    ) from error

            image_names = cls._parse_indexed_text(
                read_member("images.txt"), "image path", archive_path
            )
            labels = cls._parse_indexed_text(
                read_member("image_class_labels.txt"), "class label", archive_path
            )
        return (*cls._validate_source_metadata(image_names, labels), prefix)

    @classmethod
    def _read_directory_metadata(cls, source):
        source = Path(source)
        image_names = cls.read_indexed_text_file(source / "images.txt", "image path")
        labels = cls.read_indexed_int_file(
            source / "image_class_labels.txt", "class label"
        )
        return cls._validate_source_metadata(image_names, labels)

    @staticmethod
    def _parse_indexed_text(content, value_name, origin):
        records = {}
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            columns = line.split(maxsplit=1)
            if len(columns) != 2:
                raise ValueError(
                    f"Invalid NABirds {value_name} row at {origin}:{line_number}"
                )
            if columns[0] in records:
                raise ValueError(
                    f"Duplicate NABirds ID {columns[0]} at {origin}:{line_number}"
                )
            records[columns[0]] = columns[1]
        if not records:
            raise ValueError(f"NABirds metadata is empty in {origin}")
        return records

    @classmethod
    def _validate_source_metadata(cls, image_names, labels):
        """Check the source before spending an hour resizing it."""

        if set(image_names) != set(labels):
            raise ValueError(
                "NABirds metadata IDs do not align between images.txt and "
                f"image_class_labels.txt: {len(set(image_names) - set(labels))} "
                f"images lack labels and {len(set(labels) - set(image_names))} "
                "labels lack images"
            )
        if len(image_names) != cls.EXPECTED_IMAGE_COUNT:
            raise ValueError(
                f"NABirds images.txt lists {len(image_names)} images; expected "
                f"{cls.EXPECTED_IMAGE_COUNT}"
            )
        class_count = len({int(label) for label in labels.values()})
        if class_count != cls.EXPECTED_CLASS_COUNT:
            raise ValueError(
                f"NABirds labels cover {class_count} visual categories; "
                f"expected {cls.EXPECTED_CLASS_COUNT}"
            )
        return image_names, class_count

    def _copy_metadata_from_zip(self, archive_path, prefix):
        with zipfile.ZipFile(archive_path, "r") as archive:
            for filename in self.REQUIRED_METADATA_FILES:
                member = (prefix / filename).as_posix()
                self._write_bytes_atomic(self.root / filename, archive.read(member))

    def _copy_metadata_from_directory(self, source):
        source = Path(source)
        for filename in self.REQUIRED_METADATA_FILES:
            destination = self.root / filename
            origin = source / filename
            if origin.resolve() == destination.resolve():
                continue
            self._write_bytes_atomic(destination, origin.read_bytes())

    def _relative_image_path(self, image_name):
        """Validate an ``images.txt`` path before it is used as a destination."""

        relative_path = PurePosixPath(str(image_name).replace("\\", "/"))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError(
                f"NABirds images.txt contains an unsafe image path: {image_name!r}"
            )
        return relative_path

    def _resize_from_directory(self, source, image_names):
        source = Path(source)
        if source.resolve() == self.root.resolve():
            raise RuntimeError(
                "NABirds cannot resize a directory into itself; this would "
                f"overwrite the full resolution images in {source}"
            )
        written_images = 0
        reused_images = 0
        with tqdm(
            total=len(image_names),
            desc="Saving 224x224 NABirds images",
            unit="image",
            disable=None,
        ) as progress:
            for image_name in image_names.values():
                relative_path = self._relative_image_path(image_name)
                source_path = source / "images" / Path(*relative_path.parts)
                if not source_path.is_file():
                    raise RuntimeError(f"NABirds source is missing {source_path}")
                destination = self.root / "images" / Path(*relative_path.parts)
                if self._is_target_sized_image(destination):
                    reused_images += 1
                else:
                    with source_path.open("rb") as image_source:
                        self._resize_image_to_destination(image_source, destination)
                    written_images += 1
                progress.update(1)
        return {"written_images": written_images, "reused_images": reused_images}

    def _resize_from_zip(self, archive_path, prefix, image_names):
        written_images = 0
        reused_images = 0
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive_members = set(archive.namelist())
            with tqdm(
                total=len(image_names),
                desc="Saving 224x224 NABirds images",
                unit="image",
                disable=None,
            ) as progress:
                for image_name in image_names.values():
                    relative_path = self._relative_image_path(image_name)
                    member = (prefix / "images" / relative_path).as_posix()
                    if member not in archive_members:
                        raise RuntimeError(
                            f"NABirds archive is missing {member!r}"
                        )
                    destination = self.root / "images" / Path(*relative_path.parts)
                    if self._is_target_sized_image(destination):
                        reused_images += 1
                    else:
                        with archive.open(member, "r") as image_source:
                            self._resize_image_to_destination(
                                image_source, destination
                            )
                        written_images += 1
                    progress.update(1)
        return {"written_images": written_images, "reused_images": reused_images}

    @classmethod
    def _read_hierarchy(cls, path):
        """Map every child node id to its parent id from ``hierarchy.txt``."""

        parents = {}
        for line_number, raw_line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line:
                continue
            columns = line.split()
            if len(columns) != 2:
                raise ValueError(f"Invalid NABirds hierarchy row at {path}:{line_number}")
            try:
                child_id, parent_id = (int(column) for column in columns)
            except ValueError as exc:
                raise ValueError(
                    f"NABirds hierarchy ids must be integers: {path}:{line_number}"
                ) from exc
            if child_id in parents:
                raise ValueError(
                    f"Duplicate NABirds hierarchy child {child_id} at {path}:{line_number}"
                )
            parents[child_id] = parent_id
        if not parents:
            raise ValueError(f"NABirds metadata file is empty: {path}")
        return parents

    @classmethod
    def partition_class_ids(cls, class_ids, parents, class_split=None):
        """Halve the visual categories without splitting a hierarchy parent.

        Categories are grouped by their hierarchy parent and the groups are
        ordered by their lowest class id, so sibling plumages of one species
        always land on the same side of the split whichever rule deals them
        out.

        ``parent_group_prefix`` (the default) takes complete groups until the
        development half holds at least half of the categories.
        ``alternating_parent_groups`` takes every other group instead, which
        matters because the low class ids are the sparser taxa: both rules give
        277/278 categories, but the prefix leaves development with 46.4% of the
        images against the stride's 50.5%.

        ``class_id_midpoint`` is the exception: it ignores the hierarchy and
        cuts the sorted class ids at the midpoint, giving development the first
        278 categories (22,909 images) and holding out the last 277 (25,653).
        That is the PUMA benchmark's split, reproduced here so its published
        NABirds numbers are comparable. The rule is PUMA's own, not an
        inference from its reported counts: its loader builds
        ``{k: i for i, k in enumerate(sorted(set(targets)))}`` and takes
        ``range(0, 278)`` for training against ``range(278, 555)`` for
        evaluation, and it never consults ``train_test_split.txt``. NABirds
        does not list the plumages of a
        species under consecutive ids -- none of the 137 multi-category groups
        is contiguous -- so this cut leaves 12 species with categories on both
        sides, among them the Red-tailed Hawk and the Northern Flicker. The
        halves are still disjoint in visual categories; it is the species that
        leak. :meth:`count_straddling_parents` measures it, and every split
        records the count.
        """

        class_split = cls.CLASS_SPLIT_PREFIX_GROUPS if class_split is None else class_split
        if class_split not in cls.AVAILABLE_CLASS_SPLITS:
            raise ValueError(
                f"class_split must be one of {cls.AVAILABLE_CLASS_SPLITS}, "
                f"got {class_split!r}"
            )
        class_ids = tuple(sorted(int(class_id) for class_id in class_ids))
        if len(set(class_ids)) != len(class_ids):
            raise ValueError("NABirds class ids must be unique")
        missing_parents = [class_id for class_id in class_ids if class_id not in parents]
        if missing_parents:
            raise ValueError(
                "NABirds hierarchy.txt has no parent for visual categories "
                f"{missing_parents[:10]}"
            )

        groups = {}
        for class_id in class_ids:
            groups.setdefault(parents[class_id], []).append(class_id)
        ordered_parents = sorted(groups, key=lambda parent: groups[parent][0])
        if class_split == cls.CLASS_SPLIT_CLASS_ID_MIDPOINT:
            # PUMA gives the larger half to development: 278 of 555 categories.
            midpoint = (len(class_ids) + 1) // 2
            development = class_ids[:midpoint]
            test = class_ids[midpoint:]
        elif class_split == cls.CLASS_SPLIT_ALTERNATING_GROUPS:
            development = cls._collect_groups(groups, ordered_parents[0::2])
            test = cls._collect_groups(groups, ordered_parents[1::2])
        else:
            half = len(class_ids) // 2
            development = []
            for parent_id in ordered_parents:
                if len(development) >= half:
                    break
                development.extend(groups[parent_id])
            development = tuple(sorted(development))
            development_set = set(development)
            test = tuple(
                class_id for class_id in class_ids if class_id not in development_set
            )
        if not development or not test:
            raise ValueError(
                "NABirds class partition left one half empty; the hierarchy "
                f"groups {len(class_ids)} categories into {len(groups)} parents"
            )
        return development, test

    @staticmethod
    def _collect_groups(groups, parent_ids):
        """Flatten the chosen hierarchy groups back into sorted class ids."""

        return tuple(
            sorted(class_id for parent_id in parent_ids for class_id in groups[parent_id])
        )

    @staticmethod
    def count_straddling_parents(development_class_ids, test_class_ids, parents):
        """Return the hierarchy parents that kept categories on both sides.

        Zero for every partition that deals out whole groups. A non-empty
        result means the two halves share a species, so a held-out category has
        a near-identical relative in development.
        """

        development_parents = {parents[class_id] for class_id in development_class_ids}
        test_parents = {parents[class_id] for class_id in test_class_ids}
        return sorted(development_parents & test_parents)

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


class VehicleID(_DinoSizedImageDownloadMixin, Dataset):
    """PKU VehicleID on its official identity-disjoint retrieval protocol.

    A class here is one *vehicle*, not one model, and the official
    ``train_test_split`` directory ships the partition: ``train_list.txt`` holds
    the training identities, and three nested test lists hold 800, 1,600 and
    2,400 held-out identities. The three are the published evaluation sizes --
    small, medium and large -- and papers report all of them, so
    ``test_gallery_size`` picks which one ``split="test"`` returns.

    The split logic follows the reference DML implementation
    (``give_VehicleID_datasets`` in Roth and Brattoli's Deep-Metric-Learning
    -Baselines, Apache 2.0), which reads the same list files and never shuffles
    classes. One deliberate difference: that code renumbers train and test from
    zero independently, which would give the two halves overlapping label
    values. Labels here are densified over the union of every listed identity
    instead, so the halves stay disjoint as ``class_disjoint_split`` claims.
    The raw identities remain on ``vehicle_ids``.

    PKU releases VehicleID only against a signed agreement, so the automatic
    path reads a Kaggle mirror; see :meth:`download_224`.
    """

    DATASET_PAGE = "https://www.pkuml.org/resources/pku-vehicleid.html"
    REFERENCE = (
        "https://github.com/Confusezius/Deep-Metric-Learning-Baselines/"
        "blob/master/datasets.py"
    )
    # Third-party mirror. The images are PKU's and stay under the terms on
    # DATASET_PAGE; the mirror only removes the agreement step.
    KAGGLE_DATASET_PAGE = "https://www.kaggle.com/datasets/maphat/vehicleid"
    KAGGLE_DATASET = "maphat/vehicleid"
    LOCAL_ARCHIVE_CANDIDATES = ("archive.zip", "vehicleid.zip", "VehicleID_V1.0.zip")
    SPLIT_DIRECTORY = "train_test_split"
    IMAGE_DIRECTORY = "image"
    TRAIN_LIST = "train_list.txt"
    TEST_GALLERY_SIZES = (800, 1600, 2400)
    # The list files are named for the identity count they carry, and the
    # release's own counts are the check that a mirror shipped them intact.
    EXPECTED_TEST_IDENTITY_COUNTS = {800: 800, 1600: 1600, 2400: 2400}
    DEFAULT_TEST_GALLERY_SIZE = 2400
    COMPLETE_MARKER = ".vehicleid_224_complete.json"
    CLASS_SPLIT_VERSION = "vehicleid_official_train_test_split_v1"
    AVAILABLE_SPLITS = (
        "train",
        "test",
        "test_800",
        "test_1600",
        "test_2400",
        "train+test",
    )

    def __init__(
        self,
        root,
        split="train",
        transform=None,
        target_transform=None,
        download=False,
        test_gallery_size=None,
    ):
        self.root = Path(root)
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(
                f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}"
            )
        # "test_800" and friends name a size directly; "test" defers to the
        # argument so a protocol can pick one without renaming the split.
        if split.startswith("test_"):
            requested_size = int(split.split("_")[1])
            if test_gallery_size is not None and int(test_gallery_size) != requested_size:
                raise ValueError(
                    f"split={split!r} and test_gallery_size={test_gallery_size!r} "
                    "disagree"
                )
            test_gallery_size = requested_size
        test_gallery_size = (
            self.DEFAULT_TEST_GALLERY_SIZE
            if test_gallery_size is None
            else int(test_gallery_size)
        )
        if test_gallery_size not in self.TEST_GALLERY_SIZES:
            raise ValueError(
                f"test_gallery_size must be one of {self.TEST_GALLERY_SIZES}, "
                f"got {test_gallery_size!r}"
            )
        self.test_gallery_size = test_gallery_size

        if download and not self.is_ready(self.root):
            self.download_224(self.root)
        self.dataset_root = self.find_dataset_root(self.root)

        train_records = self.read_list_file(self.split_list_path(self.TRAIN_LIST))
        test_records = {
            size: self.read_list_file(self.test_list_path(size))
            for size in self.TEST_GALLERY_SIZES
        }

        # One label space over every listed identity keeps train and test
        # labels from colliding; the lists themselves are already disjoint.
        self.vehicle_ids = tuple(
            sorted(
                {vehicle_id for _, vehicle_id in train_records}
                | {
                    vehicle_id
                    for records in test_records.values()
                    for _, vehicle_id in records
                }
            )
        )
        self.vehicle_to_label = {
            vehicle_id: label for label, vehicle_id in enumerate(self.vehicle_ids)
        }
        self.class_ids = list(range(len(self.vehicle_ids)))

        development_vehicles = tuple(sorted({v for _, v in train_records}))
        test_vehicles = tuple(sorted({v for _, v in test_records[test_gallery_size]}))
        overlap = set(development_vehicles) & set(test_vehicles)
        if overlap:
            raise ValueError(
                f"VehicleID train and test_{test_gallery_size} lists share "
                f"{len(overlap)} identities; examples: {sorted(overlap)[:5]}"
            )

        selected = test_records[test_gallery_size]
        if split == "train":
            records = train_records
        elif split == "train+test":
            records = train_records + selected
        else:
            records = selected

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.image_ids = [image_id for image_id, _ in records]
        self.paths = [str(self.resolve_image_path(image_id)) for image_id, _ in records]
        self.labels = [self.vehicle_to_label[vehicle_id] for _, vehicle_id in records]
        self.orig_labels = [vehicle_id for _, vehicle_id in records]
        if not self.paths:
            raise ValueError(f"VehicleID split {split!r} contains no images")
        missing_paths = [path for path in self.paths if not Path(path).is_file()]
        if missing_paths:
            examples = ", ".join(missing_paths[:3])
            raise ValueError(
                f"VehicleID is missing {len(missing_paths)} image files listed in "
                f"{self.SPLIT_DIRECTORY}; examples: {examples}"
            )

        self.class_disjoint_split = True
        self.class_split_info = {
            "source": "official_identity_disjoint_train_test_split",
            "dataset_page": self.DATASET_PAGE,
            "protocol_reference": self.REFERENCE,
            "dataset_root": str(self.dataset_root),
            "class_disjoint_test": True,
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "split_basis": "official_train_test_split_lists",
            "test_gallery_size": test_gallery_size,
            "available_test_gallery_sizes": list(self.TEST_GALLERY_SIZES),
            "development_class_count": len(development_vehicles),
            "test_class_count": len(test_vehicles),
            "development_classes": [
                self.vehicle_to_label[vehicle_id] for vehicle_id in development_vehicles
            ],
            "held_out_test_classes": [
                self.vehicle_to_label[vehicle_id] for vehicle_id in test_vehicles
            ],
            "development_sample_count": len(train_records),
            "test_sample_count": len(selected),
            "test_sample_counts": {
                size: len(records) for size, records in test_records.items()
            },
            "official_image_level_split_used": True,
        }

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------

    @classmethod
    def find_dataset_root(cls, root):
        """Locate the directory holding ``train_test_split/`` and ``image/``.

        Mirrors wrap the release in a directory of their own -- and sometimes
        in two -- so the root is searched for rather than assumed.
        """

        root = Path(root)
        candidates = [root]
        try:
            candidates.extend(child for child in sorted(root.iterdir()) if child.is_dir())
        except OSError:
            pass

        matches = []
        for candidate in candidates:
            if cls._is_expanded_source(candidate):
                resolved = candidate.resolve()
                if resolved not in matches:
                    matches.append(resolved)
        if not matches:
            # One more level down, for mirrors that nest twice.
            for candidate in sorted(root.glob("*/*")):
                if candidate.is_dir() and cls._is_expanded_source(candidate):
                    resolved = candidate.resolve()
                    if resolved not in matches:
                        matches.append(resolved)
        if not matches:
            raise ValueError(
                "VehicleID metadata was not found. Expected "
                f"{cls.SPLIT_DIRECTORY}/{cls.TRAIN_LIST} and "
                f"{cls.IMAGE_DIRECTORY}/ below {root}."
            )
        if len(matches) > 1:
            raise ValueError(
                "Multiple VehicleID dataset roots were found below "
                f"{root}: " + ", ".join(str(path) for path in matches)
            )
        return matches[0]

    @classmethod
    def _is_expanded_source(cls, candidate):
        candidate = Path(candidate)
        if not (candidate / cls.IMAGE_DIRECTORY).is_dir():
            return False
        split_directory = candidate / cls.SPLIT_DIRECTORY
        required = [cls.TRAIN_LIST] + [
            cls.test_list_name(size) for size in cls.TEST_GALLERY_SIZES
        ]
        return all((split_directory / filename).is_file() for filename in required)

    @classmethod
    def is_ready(cls, root):
        try:
            cls.find_dataset_root(root)
        except (ValueError, OSError):
            return False
        return True

    @staticmethod
    def test_list_name(size):
        return f"test_list_{int(size)}.txt"

    def split_list_path(self, filename):
        return self.dataset_root / self.SPLIT_DIRECTORY / filename

    def test_list_path(self, size):
        return self.split_list_path(self.test_list_name(size))

    @classmethod
    def read_list_file(cls, path):
        """Read a ``<image id> <vehicle id>`` list into ordered pairs.

        Both columns are integers: the image id names the file as
        ``image/{:07d}.jpg`` and the vehicle id is the identity label.
        """

        return cls.parse_list_file(Path(path).read_text(encoding="utf-8"), path)

    @staticmethod
    def parse_list_file(content, origin):
        records = []
        seen = set()
        for line_number, raw_line in enumerate(content.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            columns = line.split()
            if len(columns) != 2:
                raise ValueError(f"Invalid VehicleID list row at {origin}:{line_number}")
            try:
                image_id, vehicle_id = (int(column) for column in columns)
            except ValueError as exc:
                raise ValueError(
                    f"VehicleID list ids must be integers: {origin}:{line_number}"
                ) from exc
            if image_id < 0 or vehicle_id < 0:
                raise ValueError(
                    f"VehicleID list ids must be non-negative: {origin}:{line_number}"
                )
            if image_id in seen:
                raise ValueError(
                    f"Duplicate VehicleID image id {image_id} at {origin}:{line_number}"
                )
            seen.add(image_id)
            records.append((image_id, vehicle_id))
        if not records:
            raise ValueError(f"VehicleID list file is empty: {origin}")
        return records

    @staticmethod
    def image_filename(image_id):
        return f"{int(image_id):07d}.jpg"

    def resolve_image_path(self, image_id):
        return self.dataset_root / self.IMAGE_DIRECTORY / self.image_filename(image_id)

    # ------------------------------------------------------------------
    # Download
    # ------------------------------------------------------------------

    @classmethod
    def is_224_complete(cls, root):
        """Whether :meth:`download_224` finished writing this root."""

        root = Path(root)
        marker_path = root / cls.COMPLETE_MARKER
        if not marker_path.is_file():
            return False
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            image_size = tuple(int(value) for value in marker["image_size"])
            image_count = int(marker["image_count"])
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, OSError):
            return False
        return (
            image_size == cls.TARGET_IMAGE_SIZE
            and image_count > 0
            and cls.is_ready(root)
        )

    @classmethod
    def download_224(cls, root, source=None):
        """Fetch PKU VehicleID and store it as 224x224 JPEGs.

        PKU distributes VehicleID only against a signed agreement, so the
        automatic path pulls the ``maphat/vehicleid`` Kaggle mirror. ``source``
        overrides that with an archive or an extracted directory already on
        disk; without it, an archive in or beside the destination is used
        before anything is downloaded.

        Only the images the split lists actually reference are written, under
        their official ``image/{:07d}.jpg`` names, and the lists are copied
        verbatim -- so the result is a layout :meth:`find_dataset_root`
        accepts. Resizing is resumable.
        """

        root = Path(root)
        if cls.is_224_complete(root):
            return root
        root.mkdir(parents=True, exist_ok=True)
        downloader = cls.__new__(cls)
        downloader.root = root
        downloader._download_and_resize(source)
        return root

    def _download_and_resize(self, source=None):
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)
        resolved = self._resolve_source(source)

        if resolved.is_dir():
            lists = self._read_directory_lists(resolved)
            self._copy_lists(lists)
            stats = self._resize_from_directory(resolved, self._listed_image_ids(lists))
        elif resolved.is_file() and zipfile.is_zipfile(resolved):
            lists, prefix = self._read_zip_lists(resolved)
            self._copy_lists(lists)
            stats = self._resize_from_zip(
                resolved, prefix, self._listed_image_ids(lists)
            )
        else:
            raise RuntimeError(f"Unsupported VehicleID source: {resolved}")

        marker = {
            "dataset": "VehicleID",
            "image_size": list(self.TARGET_IMAGE_SIZE),
            "image_count": len(self._listed_image_ids(lists)),
            "identity_count": len(
                {vehicle_id for records in lists.values() for _, vehicle_id in records}
            ),
            "jpeg_quality": int(self.JPEG_QUALITY),
            "written_images": int(stats["written_images"]),
            "reused_images": int(stats["reused_images"]),
            "source": str(resolved),
            "kaggle_dataset": self.KAGGLE_DATASET,
            "kaggle_dataset_page": self.KAGGLE_DATASET_PAGE,
            "dataset_page": self.DATASET_PAGE,
            "protocol_reference": self.REFERENCE,
            "list_files": sorted(lists),
            "list_row_counts": {name: len(records) for name, records in lists.items()},
            "official_image_level_split_used": True,
        }
        self._write_json_atomic(self.root / self.COMPLETE_MARKER, marker)

    @classmethod
    def _list_filenames(cls):
        return (cls.TRAIN_LIST,) + tuple(
            cls.test_list_name(size) for size in cls.TEST_GALLERY_SIZES
        )

    @staticmethod
    def _listed_image_ids(lists):
        """Every image id the lists reference, in a stable order, deduplicated."""

        ordered = {}
        for name in sorted(lists):
            for image_id, _ in lists[name]:
                ordered.setdefault(image_id, None)
        return tuple(ordered)

    def _resolve_source(self, source=None):
        if source is not None:
            source = Path(source)
            if source.is_dir():
                return self.find_dataset_root(source)
            if source.is_file() and zipfile.is_zipfile(source):
                return source
            raise RuntimeError(
                "VehicleID source must be an extracted directory or a ZIP "
                f"archive, got: {source}"
            )

        for candidate in self.LOCAL_ARCHIVE_CANDIDATES:
            for directory in (self.root, self.root.parent):
                local_archive = directory / candidate
                if self._is_vehicleid_zip(local_archive):
                    return local_archive

        try:
            import kagglehub
        except ImportError as error:
            raise RuntimeError(
                "kagglehub is required to download VehicleID. Install the "
                "project requirements or run `pip install kagglehub`."
            ) from error

        try:
            kaggle_path = Path(kagglehub.dataset_download(self.KAGGLE_DATASET))
        except Exception as error:
            raise RuntimeError(
                f"Could not download VehicleID from {self.KAGGLE_DATASET_PAGE} "
                "with KaggleHub. Authenticate with Kaggle, or request the "
                f"official release at {self.DATASET_PAGE} and pass it as source."
            ) from error
        return self._find_source_below(kaggle_path)

    @classmethod
    def _find_source_below(cls, kaggle_path):
        kaggle_path = Path(kaggle_path)
        if kaggle_path.is_file():
            if zipfile.is_zipfile(kaggle_path):
                return kaggle_path
            raise RuntimeError(
                f"KaggleHub returned a non-ZIP VehicleID file: {kaggle_path}"
            )
        if not kaggle_path.is_dir():
            raise RuntimeError(
                f"KaggleHub returned a missing VehicleID path: {kaggle_path}"
            )
        try:
            return cls.find_dataset_root(kaggle_path)
        except (ValueError, OSError):
            pass
        archive_matches = sorted(
            path for path in kaggle_path.rglob("*.zip") if cls._is_vehicleid_zip(path)
        )
        if len(archive_matches) == 1:
            return archive_matches[0]
        raise RuntimeError(
            "KaggleHub returned neither an expanded VehicleID directory nor a "
            f"VehicleID ZIP archive below {kaggle_path}"
        )

    @classmethod
    def _is_vehicleid_zip(cls, path):
        path = Path(path)
        if not path.is_file() or not zipfile.is_zipfile(path):
            return False
        try:
            with zipfile.ZipFile(path, "r") as archive:
                cls._find_zip_prefix(archive)
        except (RuntimeError, zipfile.BadZipFile, OSError):
            return False
        return True

    @classmethod
    def _find_zip_prefix(cls, archive):
        """Return the directory inside ``archive`` holding the release."""

        members = set(archive.namelist())
        candidates = []
        for member_name in members:
            member_path = PurePosixPath(member_name.replace("\\", "/"))
            if member_path.name != cls.TRAIN_LIST:
                continue
            if member_path.parent.name != cls.SPLIT_DIRECTORY:
                continue
            prefix = member_path.parent.parent
            if all(
                (prefix / cls.SPLIT_DIRECTORY / filename).as_posix() in members
                for filename in cls._list_filenames()
            ):
                candidates.append(prefix)
        if len(candidates) != 1:
            raise RuntimeError(
                "Expected exactly one VehicleID train_test_split directory in "
                f"the archive, found {len(candidates)}"
            )
        return candidates[0]

    @classmethod
    def _read_zip_lists(cls, archive_path):
        with zipfile.ZipFile(archive_path, "r") as archive:
            prefix = cls._find_zip_prefix(archive)
            lists = {}
            for filename in cls._list_filenames():
                member = (prefix / cls.SPLIT_DIRECTORY / filename).as_posix()
                try:
                    content = archive.read(member).decode("utf-8")
                except (KeyError, UnicodeDecodeError) as error:
                    raise ValueError(
                        f"Could not read VehicleID {filename} in {archive_path}"
                    ) from error
                lists[filename] = cls.parse_list_file(content, member)
        return cls._validate_lists(lists), prefix

    @classmethod
    def _read_directory_lists(cls, source):
        source = Path(source)
        lists = {
            filename: cls.read_list_file(source / cls.SPLIT_DIRECTORY / filename)
            for filename in cls._list_filenames()
        }
        return cls._validate_lists(lists)

    @classmethod
    def _validate_lists(cls, lists):
        """Check the identity partition before spending hours resizing."""

        train_identities = {vehicle_id for _, vehicle_id in lists[cls.TRAIN_LIST]}
        for size in cls.TEST_GALLERY_SIZES:
            name = cls.test_list_name(size)
            identities = {vehicle_id for _, vehicle_id in lists[name]}
            expected = cls.EXPECTED_TEST_IDENTITY_COUNTS[size]
            if len(identities) != expected:
                raise ValueError(
                    f"VehicleID {name} covers {len(identities)} identities; "
                    f"expected {expected}"
                )
            overlap = train_identities & identities
            if overlap:
                raise ValueError(
                    f"VehicleID {name} shares {len(overlap)} identities with "
                    f"{cls.TRAIN_LIST}; examples: {sorted(overlap)[:5]}"
                )
        return lists

    def _copy_lists(self, lists):
        for filename, records in lists.items():
            content = "".join(
                f"{image_id} {vehicle_id}\n" for image_id, vehicle_id in records
            )
            self._write_bytes_atomic(
                self.root / self.SPLIT_DIRECTORY / filename, content.encode("utf-8")
            )

    def _resize_from_directory(self, source, image_ids):
        source = Path(source)
        if source.resolve() == self.root.resolve():
            raise RuntimeError(
                "VehicleID cannot resize a directory into itself; this would "
                f"overwrite the full resolution images in {source}"
            )
        written_images = 0
        reused_images = 0
        with tqdm(
            total=len(image_ids),
            desc="Saving 224x224 VehicleID images",
            unit="image",
            disable=None,
        ) as progress:
            for image_id in image_ids:
                filename = self.image_filename(image_id)
                source_path = source / self.IMAGE_DIRECTORY / filename
                if not source_path.is_file():
                    raise RuntimeError(f"VehicleID source is missing {source_path}")
                destination = self.root / self.IMAGE_DIRECTORY / filename
                if self._is_target_sized_image(destination):
                    reused_images += 1
                else:
                    with source_path.open("rb") as image_source:
                        self._resize_image_to_destination(image_source, destination)
                    written_images += 1
                progress.update(1)
        return {"written_images": written_images, "reused_images": reused_images}

    def _resize_from_zip(self, archive_path, prefix, image_ids):
        written_images = 0
        reused_images = 0
        with zipfile.ZipFile(archive_path, "r") as archive:
            archive_members = set(archive.namelist())
            with tqdm(
                total=len(image_ids),
                desc="Saving 224x224 VehicleID images",
                unit="image",
                disable=None,
            ) as progress:
                for image_id in image_ids:
                    filename = self.image_filename(image_id)
                    member = (prefix / self.IMAGE_DIRECTORY / filename).as_posix()
                    if member not in archive_members:
                        raise RuntimeError(
                            f"VehicleID archive is missing {member!r}"
                        )
                    destination = self.root / self.IMAGE_DIRECTORY / filename
                    if self._is_target_sized_image(destination):
                        reused_images += 1
                    else:
                        with archive.open(member, "r") as image_source:
                            self._resize_image_to_destination(
                                image_source, destination
                            )
                        written_images += 1
                    progress.update(1)
        return {"written_images": written_images, "reused_images": reused_images}

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


class DeepFashionInShop(Dataset):
    """DeepFashion In-shop Clothes Retrieval dataset.

    Expected local layout is the official benchmark directory under ``root``:

    ``Eval/list_eval_partition.txt``
    ``Img/img/...``

    ``split="train"`` uses the official training partition. ``split="test"``
    returns query samples followed by gallery samples and exposes
    ``query_indices``/``gallery_indices`` for canonical query-to-gallery
    evaluation.
    """

    METADATA_FILENAME = "list_eval_partition.txt"
    AVAILABLE_SPLITS = ("train", "query", "gallery", "test", "query+gallery", "train+test")

    def __init__(self, root, split="train", transform=None, target_transform=None, download=False):
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}")
        self.split = split

        metadata_file = self.find_metadata_file(self.root)
        if metadata_file is None:
            if download:
                raise RuntimeError(
                    "DeepFashion In-shop does not support automatic download. "
                    "Prepare the dataset manually with Eval/list_eval_partition.txt "
                    "and extracted Img/img/... under data/DeepFashionInShop."
                )
            raise ValueError(
                "DeepFashion In-shop metadata was not found. Expected "
                "Eval/list_eval_partition.txt under the dataset root."
            )

        self.dataset_root = metadata_file.parent.parent
        records = self._read_partition_file(metadata_file)
        self.records = self._select_records(records, split)
        if not self.records:
            raise ValueError(f"DeepFashion In-shop split {split!r} is empty")
        if not self._records_have_any_image_file(self.records):
            raise ValueError(
                "DeepFashion In-shop images were not found. Expected extracted files like "
                "Img/img/... under the dataset root."
            )

        self.paths = [str(self._resolve_image_path(record["image_name"])) for record in self.records]
        self.image_names = [record["image_name"] for record in self.records]
        self.item_ids = [record["item_id"] for record in self.records]
        self.evaluation_status = [record["evaluation_status"] for record in self.records]
        item_ids = sorted({record["item_id"] for record in records})
        self.class_to_label = self._make_class_to_label(item_ids)
        self.classes = self._make_classes(item_ids, self.class_to_label)
        self.labels = [self.class_to_label[record["item_id"]] for record in self.records]
        self.orig_labels = list(self.labels)

        if split in {"test", "query+gallery"}:
            query_count = sum(1 for record in self.records if record["evaluation_status"] == "query")
            gallery_count = sum(1 for record in self.records if record["evaluation_status"] == "gallery")
            self.query_indices = list(range(query_count))
            self.gallery_indices = list(range(query_count, query_count + gallery_count))

    @classmethod
    def find_metadata_file(cls, root):
        root = Path(root)
        if not root.exists():
            return None

        candidates = [
            root / "Eval" / cls.METADATA_FILENAME,
            root / "In-shop Clothes Retrieval Benchmark" / "Eval" / cls.METADATA_FILENAME,
            root / "DeepFashion" / "In-shop Clothes Retrieval Benchmark" / "Eval" / cls.METADATA_FILENAME,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate

        matches = sorted(root.rglob(cls.METADATA_FILENAME))
        return matches[0] if matches else None

    def _records_have_any_image_file(self, records):
        return any(self._resolve_image_path(record["image_name"]).exists() for record in records)

    @staticmethod
    def _item_id_to_label(item_id):
        if item_id.startswith("id_"):
            suffix = item_id[3:]
            if suffix.isdigit():
                return int(suffix)
        digits = "".join(char for char in item_id if char.isdigit())
        return int(digits) if digits else None

    @classmethod
    def _make_class_to_label(cls, item_ids):
        numeric_labels = [cls._item_id_to_label(item_id) for item_id in item_ids]
        if all(label is not None for label in numeric_labels) and len(set(numeric_labels)) == len(item_ids):
            return dict(zip(item_ids, numeric_labels))
        return {item_id: index for index, item_id in enumerate(item_ids)}

    @staticmethod
    def _make_classes(item_ids, class_to_label):
        max_label = max(class_to_label.values())
        classes = [""] * (max_label + 1)
        for item_id in item_ids:
            classes[class_to_label[item_id]] = item_id
        return classes

    def _read_partition_file(self, metadata_file):
        lines = [line.strip() for line in metadata_file.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(lines) < 3:
            raise ValueError(f"Invalid DeepFashion In-shop partition file: {metadata_file}")

        try:
            expected_count = int(lines[0])
        except ValueError as exc:
            raise ValueError(f"Invalid sample count in {metadata_file}: {lines[0]!r}") from exc

        records = []
        for line_number, line in enumerate(lines[2:], start=3):
            columns = line.split()
            if len(columns) != 3:
                raise ValueError(
                    f"Invalid DeepFashion In-shop row at {metadata_file}:{line_number}: expected 3 columns"
                )
            image_name, item_id, evaluation_status = columns
            if evaluation_status not in {"train", "query", "gallery"}:
                raise ValueError(
                    f"Invalid evaluation_status at {metadata_file}:{line_number}: {evaluation_status!r}"
                )
            records.append(
                {
                    "image_name": image_name,
                    "item_id": item_id,
                    "evaluation_status": evaluation_status,
                }
            )

        if len(records) != expected_count:
            raise ValueError(
                f"DeepFashion In-shop partition count mismatch: header says {expected_count}, "
                f"but {len(records)} rows were read"
            )
        return records

    @staticmethod
    def _select_records(records, split):
        if split == "train":
            statuses = ("train",)
        elif split == "query":
            statuses = ("query",)
        elif split == "gallery":
            statuses = ("gallery",)
        elif split in {"test", "query+gallery"}:
            statuses = ("query", "gallery")
        elif split == "train+test":
            statuses = ("train", "query", "gallery")
        else:
            raise ValueError(f"Unsupported DeepFashion In-shop split: {split}")

        return [
            record
            for status in statuses
            for record in records
            if record["evaluation_status"] == status
        ]

    def _resolve_image_path(self, image_name):
        relative_path = Path(image_name)
        candidates = [
            self.dataset_root / "Img" / relative_path,
            self.dataset_root / relative_path,
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        image = default_loader(self.paths[index])
        label = self.labels[index]
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


### Standford neu erstellen, da auf Windows bspw. kein im Link erlaubt ist
class StanfordOnlineProducts(_DinoSizedImageDownloadMixin, _StanfordOnlineProducts):
    """Stanford Online Products with a readiness-driven download.

    The upstream ``BaseDataset`` only downloads when the root directory is
    missing or empty, so an interrupted download leaves a non-empty root that
    silently suppresses every later attempt. Readiness here is the presence of
    both split files, which makes a partial extraction recoverable.
    """

    FILE_ID = "1TclrpQOF_ullUP99wk_gjGN8pKvtErG8"
    ARCHIVE_NAME = "Stanford_Online_Products.zip"
    SPLIT_DIRECTORY = "Stanford_Online_Products"
    SPLIT_FILES = ("Ebay_train.txt", "Ebay_test.txt")
    ZIP_MAGIC = b"PK\x03\x04"

    def __init__(
        self,
        root,
        split="train+test",
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = str(root)
        if download and not self.is_ready(self.root):
            self.download_and_remove()
        if not self.is_ready(self.root):
            raise ValueError(
                f"Stanford Online Products was not found under {self.root}. "
                "Initialize the dataset with download=True."
            )
        if split not in self.get_available_splits():
            raise ValueError(
                f"Supported splits are: {', '.join(self.get_available_splits())}"
            )

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.generate_split()

    @classmethod
    def is_ready(cls, root):
        split_root = Path(root) / cls.SPLIT_DIRECTORY
        return all((split_root / filename).exists() for filename in cls.SPLIT_FILES)

    def download_and_remove(self):
        root = Path(self.root)
        if self.is_ready(root):
            return

        root.mkdir(parents=True, exist_ok=True)
        archive_path = root / self.ARCHIVE_NAME
        temporary_path = self._temporary_path(archive_path)
        temporary_path.unlink(missing_ok=True)
        try:
            self._download_archive(temporary_path)
            with zipfile.ZipFile(temporary_path, "r") as archive:
                archive.extractall(root)
        finally:
            temporary_path.unlink(missing_ok=True)

        if not self.is_ready(root):
            raise RuntimeError(
                f"The Stanford Online Products archive extracted to {root} "
                f"without {' and '.join(self.SPLIT_FILES)}"
            )

    def _download_archive(self, destination):
        # Drive serves an HTML interstitial instead of the archive once the
        # file exceeds its daily quota, and it answers with 200 either way.
        with self._open_url(self.DOWNLOAD_URL) as response:
            content_length = response.headers.get("Content-Length")
            expected_bytes = int(content_length) if content_length else None
            with self._byte_progress(response, "Stanford Online Products") as progress:
                with destination.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        progress.update(len(chunk))

        downloaded_bytes = destination.stat().st_size
        if expected_bytes is not None and downloaded_bytes != expected_bytes:
            raise RuntimeError(
                f"The Stanford Online Products download stopped after "
                f"{downloaded_bytes} of {expected_bytes} bytes"
            )
        with destination.open("rb") as archive:
            magic = archive.read(len(self.ZIP_MAGIC))
        if magic != self.ZIP_MAGIC:
            raise RuntimeError(
                f"{self.DOWNLOAD_URL} returned {downloaded_bytes} bytes that are "
                "not a zip archive; Google Drive most likely served a quota or "
                "virus-scan page instead of the dataset"
            )


class CUB(_DinoSizedImageDownloadMixin, _CUB):
    """CUB-200-2011 with a readiness-driven, user-agent-bearing download.

    Two upstream defects stop the automatic download. Caltech answers urllib's
    default ``Python-urllib/3.x`` user agent with 403, so the inherited
    downloader writes an error page instead of the archive; and the upstream
    ``BaseDataset`` only downloads when the root is missing or empty, so that
    failed attempt -- or a sibling this project writes itself, such as
    ``backbone_cache`` -- silently suppresses every later attempt and the run
    dies on a missing ``image_class_labels.txt`` instead.

    Readiness is structural rather than a fixed image count: the two metadata
    files must agree on their length, and ``images`` must hold one directory
    per class id they name. The protocol layer already rejects a root whose
    class inventory is not CUB's 200.
    """

    ARCHIVE_NAME = "CUB_200_2011.tgz"
    DATASET_DIRECTORY = "CUB_200_2011"
    IMAGE_DIRECTORY = "images"
    LABEL_FILE = "image_class_labels.txt"
    PATH_FILE = "images.txt"
    GZIP_MAGIC = b"\x1f\x8b"

    def __init__(
        self,
        root,
        split="train+test",
        transform=None,
        target_transform=None,
        download=False,
    ):
        self.root = str(root)
        if download and not self.is_ready(self.root):
            self.download_and_remove()
        if not self.is_ready(self.root):
            raise ValueError(
                f"CUB-200-2011 was not found under {self.root}. "
                "Initialize the dataset with download=True."
            )
        if split not in self.get_available_splits():
            raise ValueError(
                f"Supported splits are: {', '.join(self.get_available_splits())}"
            )

        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.generate_split()

    @classmethod
    def is_ready(cls, root):
        dataset_root = Path(root) / cls.DATASET_DIRECTORY
        class_ids, image_paths = cls._read_metadata(dataset_root)
        if not class_ids or len(class_ids) != len(image_paths):
            return False

        image_root = dataset_root / cls.IMAGE_DIRECTORY
        if not image_root.is_dir():
            return False
        class_directories = sum(1 for entry in image_root.iterdir() if entry.is_dir())
        if class_directories != len(set(class_ids)):
            return False
        # An interrupted extraction keeps the metadata it already wrote and
        # loses the images it never reached, so probing the last listed image
        # catches that case without stat-ing all 11,788 of them.
        return (image_root / image_paths[-1]).is_file()

    @classmethod
    def _read_metadata(cls, dataset_root):
        """Return the listed class ids and image paths, or two empty lists."""

        label_rows = cls._read_rows(dataset_root / cls.LABEL_FILE)
        path_rows = cls._read_rows(dataset_root / cls.PATH_FILE)
        try:
            class_ids = [int(row[1]) for row in label_rows]
        except (IndexError, ValueError):
            return [], []
        image_paths = [row[1] for row in path_rows if len(row) > 1]
        if len(image_paths) != len(path_rows):
            return [], []
        return class_ids, image_paths

    @staticmethod
    def _read_rows(path):
        try:
            with path.open("r", encoding="utf-8", errors="replace") as source:
                return [line.split() for line in source if line.strip()]
        except OSError:
            return []

    def download_and_remove(self):
        root = Path(self.root)
        if self.is_ready(root):
            return

        root.mkdir(parents=True, exist_ok=True)
        archive_path = root / self.ARCHIVE_NAME
        if archive_path.is_file():
            # An archive an earlier run left behind saves the 1.1 GB transfer.
            # A truncated one only shows itself while extracting; it is worth
            # nothing afterwards, so it goes and the download runs again.
            extracted = self._extract_archive(archive_path, root)
            archive_path.unlink()
            if extracted:
                return

        temporary_path = self._temporary_path(archive_path)
        temporary_path.unlink(missing_ok=True)
        try:
            self._download_archive(temporary_path)
            extracted = self._extract_archive(temporary_path, root)
        finally:
            temporary_path.unlink(missing_ok=True)

        if not extracted:
            raise RuntimeError(
                f"The CUB-200-2011 archive from {self.DOWNLOAD_URL} did not "
                f"extract a complete dataset to {root}"
            )

    @classmethod
    def _extract_archive(cls, archive_path, root):
        """Extract in place, reporting whether it produced a usable dataset."""

        try:
            with tarfile.open(archive_path, "r:gz") as archive:
                archive.extractall(root, filter="data")
        except (tarfile.TarError, EOFError, OSError):
            return False
        return cls.is_ready(root)

    def _download_archive(self, destination):
        # Caltech's CDN serves a 403 page to urllib's default user agent, which
        # is what the inherited downloader sends; _open_url names this project.
        with self._open_url(self.DOWNLOAD_URL) as response:
            content_length = response.headers.get("Content-Length")
            expected_bytes = int(content_length) if content_length else None
            with self._byte_progress(response, "CUB-200-2011") as progress:
                with destination.open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        progress.update(len(chunk))

        downloaded_bytes = destination.stat().st_size
        if expected_bytes is not None and downloaded_bytes != expected_bytes:
            raise RuntimeError(
                f"The CUB-200-2011 download stopped after {downloaded_bytes} "
                f"of {expected_bytes} bytes"
            )
        with destination.open("rb") as archive:
            magic = archive.read(len(self.GZIP_MAGIC))
        if magic != self.GZIP_MAGIC:
            raise RuntimeError(
                f"{self.DOWNLOAD_URL} returned {downloaded_bytes} bytes that "
                "are not a gzip archive; Caltech most likely served an error "
                "page instead of the dataset"
            )
