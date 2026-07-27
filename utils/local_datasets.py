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
from pytorch_metric_learning.datasets.inaturalist2018 import (
    INaturalist2018 as _INaturalist2018,
)
from pytorch_metric_learning.datasets.sop import StanfordOnlineProducts as _StanfordOnlineProducts
from pytorch_metric_learning.utils.common_functions import _urlretrieve
from tqdm import tqdm
from torchvision.datasets import CIFAR10 as _CIFAR10
from torchvision.datasets import CIFAR100 as _CIFAR100
from torchvision.datasets.folder import IMG_EXTENSIONS, default_loader
from torch.utils.data import Dataset


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
    def _resize_image_to_destination(cls, source, destination):
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


class StanfordDogs(_DinoSizedImageDownloadMixin, Dataset):
    """Stanford Dogs with a fixed 60/40 class-disjoint split.

    The official image-level train/test lists contain every breed on both
    sides. For metric learning this wrapper pools all 20,580 images and assigns
    complete breeds to either the 60% development split (72 breeds) or the 40%
    final-test split (48 breeds). The hash-ranked partition is fixed across
    machines and is recorded in the download marker and run metadata.
    """

    IMAGES_URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
    IMAGE_DIRECTORY = "Images"
    COMPLETE_MARKER = ".stanford_dogs_224_complete.json"
    EXPECTED_IMAGE_COUNT = 20580
    EXPECTED_CLASS_COUNT = 120
    DEVELOPMENT_CLASS_FRACTION = 0.60
    TEST_CLASS_FRACTION = 0.40
    CLASS_SPLIT_VERSION = "sha256_60_40_v1"
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
                "Stanford Dogs 224x224 data was not found. Initialize the dataset "
                "with download=True or run scripts/download_stanford_dogs_224.py."
            )
        if split not in self.AVAILABLE_SPLITS:
            raise ValueError(f"split must be one of {self.AVAILABLE_SPLITS}, got {split!r}")

        self.split = split
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

        development_names, test_names = self.partition_class_names(class_names)
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
            "source": "fixed_class_disjoint_60_40",
            "class_disjoint_test": True,
            "class_split_version": self.CLASS_SPLIT_VERSION,
            "development_class_fraction": self.DEVELOPMENT_CLASS_FRACTION,
            "test_class_fraction": self.TEST_CLASS_FRACTION,
            "development_class_count": len(development_names),
            "test_class_count": len(test_names),
            "development_classes": list(development_names),
            "held_out_test_classes": list(test_names),
            "official_image_level_split_used": False,
        }

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
    def partition_class_names(cls, class_names):
        class_names = sorted(str(class_name) for class_name in class_names)
        if len(class_names) != len(set(class_names)):
            raise ValueError("Stanford Dogs class names must be unique")
        ranked_names = sorted(
            class_names,
            key=lambda class_name: hashlib.sha256(
                f"{cls.CLASS_SPLIT_VERSION}:{class_name}".encode("utf-8")
            ).digest(),
        )
        development_count = int(round(len(ranked_names) * cls.DEVELOPMENT_CLASS_FRACTION))
        development_names = tuple(sorted(ranked_names[:development_count]))
        test_names = tuple(sorted(ranked_names[development_count:]))
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
        kept_model_keys = sorted(
            model_key
            for model_key, count in model_counts.items()
            if count >= int(min_images_per_model)
        )
        kept_model_key_set = set(kept_model_keys)
        self.paths = [path for path in discovered_paths if model_keys_by_path[path] in kept_model_key_set]
        if not self.paths:
            raise ValueError(
                "CompCars model filtering removed every image. "
                f"Lower min_images_per_model below {min_images_per_model} or check the directory layout."
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
            "candidate_source": candidate_source,
            "min_images_per_model": int(min_images_per_model),
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

        raise ValueError(
            "CompCars STML paper filter did not reproduce the documented subset: "
            f"expected {expected_images} images in {expected_model_classes} model classes, "
            f"got {actual_images} images in {actual_model_classes} model classes. "
            f"candidate_source={self.filter_info['candidate_source']!r}, "
            f"min_images_per_model={self.filter_info['min_images_per_model']}. "
            "Check that external_unlabeled_dir points to the CompCars web-nature whole-car "
            "classification subset used by STML. "
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


class CompCarsSTMLPaperUnlabeledImageDataset(CompCarsModelFilteredUnlabeledImageDataset):
    """Load the STML paper's CompCars unlabeled subset when the local root matches it."""

    MODE = "compcars_stml_paper"
    PAPER_MIN_IMAGES_PER_MODEL = 100
    PAPER_TARGET_IMAGES = 16537
    PAPER_TARGET_MODEL_CLASSES = 145

    def __init__(
        self,
        root,
        transform=None,
        min_images_per_model=PAPER_MIN_IMAGES_PER_MODEL,
        strict_paper_counts=False,
    ):
        root = Path(root)
        candidate_paths, candidate_source = self.discover_stml_candidate_paths(root)
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
        )

    @classmethod
    def discover_stml_candidate_paths(cls, root):
        split_paths = cls.discover_classification_split_paths(root)
        if split_paths:
            return split_paths, "classification_split_files"

        whole_car_paths = [
            path
            for path in cls.discover_image_paths(root)
            if cls.is_whole_car_image_path(root, path)
        ]
        return whole_car_paths, "recursive_whole_car_images"

    @classmethod
    def discover_classification_split_paths(cls, root):
        split_files = sorted(
            path
            for path in root.rglob("*.txt")
            if any("classification" in part.lower() for part in path.parts)
        )
        image_paths = []
        seen_paths = set()
        for split_file in split_files:
            for image_path in cls.read_split_file_image_paths(root, split_file):
                if image_path not in seen_paths:
                    image_paths.append(image_path)
                    seen_paths.add(image_path)
        return sorted(image_paths)

    @classmethod
    def read_split_file_image_paths(cls, root, split_file):
        paths = []
        for line in split_file.read_text(encoding="utf-8", errors="ignore").splitlines():
            token = cls.first_image_token(line)
            if token is None:
                continue
            resolved = cls.resolve_split_image_path(root, token)
            if resolved is not None:
                paths.append(resolved)
        return paths

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
    def resolve_split_image_path(cls, root, token):
        relative_path = Path(token.replace("\\", "/"))
        candidates = [
            root / relative_path,
            root / "image" / relative_path,
            root / "images" / relative_path,
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
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


class NABirdsUnlabeledImageDataset(Dataset):
    """Load every official NABirds image while keeping its annotations hidden.

    NABirds is used as the additional unlabeled bird collection for CUB in the
    SLADE/STML semi-supervised protocol.  Reading ``images.txt`` rather than
    recursively accepting every image makes the pool reproducible and avoids
    accidentally including unrelated files placed below the download root.
    """

    MODE = "nabirds"
    REQUIRED_METADATA_FILES = (
        "images.txt",
        "image_class_labels.txt",
        "classes.txt",
    )

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
        class_names = self.read_indexed_text_file(self.dataset_root / "classes.txt", "class name")

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
    def read_indexed_text_file(path, value_name):
        records = {}
        for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            columns = line.split(maxsplit=1)
            if len(columns) != 2:
                raise ValueError(f"Invalid NABirds {value_name} row at {path}:{line_number}")
            try:
                record_id = int(columns[0])
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
    def read_indexed_int_file(cls, path, value_name):
        records = cls.read_indexed_text_file(path, value_name)
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

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(str(self.paths[index]))
        if self.transform is not None:
            image = self.transform(image)
        return image, -1


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
class StanfordOnlineProducts(_StanfordOnlineProducts):
    FILE_ID = "1TclrpQOF_ullUP99wk_gjGN8pKvtErG8"
    ARCHIVE_NAME = "Stanford_Online_Products.zip"

    def download_and_remove(self):
        os.makedirs(self.root, exist_ok=True)

        extracted_root = os.path.join(self.root, "Stanford_Online_Products")
        train_file = os.path.join(extracted_root, "Ebay_train.txt")
        test_file = os.path.join(extracted_root, "Ebay_test.txt")

        if os.path.exists(train_file) and os.path.exists(test_file):
            return

        download_file_path = os.path.join(self.root, "Stanford_Online_Products.zip")

        _urlretrieve(
            url=StanfordOnlineProducts.DOWNLOAD_URL,
            filename=download_file_path,
        )

        try:
            with zipfile.ZipFile(download_file_path, "r") as zip_ref:
                zip_ref.extractall(self.root)
        finally:
            if os.path.exists(download_file_path):
                os.remove(download_file_path)
