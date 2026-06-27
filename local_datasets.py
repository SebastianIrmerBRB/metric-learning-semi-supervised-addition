import os
import statistics
import zipfile
from collections import Counter
from pathlib import Path

from pytorch_metric_learning.datasets.sop import StanfordOnlineProducts as _StanfordOnlineProducts
from pytorch_metric_learning.utils.common_functions import _urlretrieve
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
