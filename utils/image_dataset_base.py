"""Building blocks shared by the locally prepared image datasets.

Every downloadable dataset here is stored as atomic 224 x 224 JPEGs that are
resized while streaming the source archive, and is read back as a list of
image paths. The helpers below are that common machinery; the dataset modules
only describe their own archives, metadata, and class splits.
"""

import contextlib
import json
import os
import shutil
import tarfile
import urllib.request
from functools import partial
from pathlib import Path, PurePosixPath

from PIL import Image
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader
from tqdm import tqdm


class ImageListDataset(Dataset):
    """Serve ``self.paths`` through torchvision's default loader.

    Labeled subclasses set ``labels`` and ``target_transform``. An unlabeled
    pool sets ``UNLABELED`` and always yields -1, whatever ``labels`` holds.
    """

    UNLABELED = False

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image = default_loader(str(self.paths[index]))
        if self.transform is not None:
            image = self.transform(image)
        if self.UNLABELED:
            return image, -1
        label = self.labels[index]
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label


def safe_posix_path(name, origin):
    """``name`` as a POSIX path, rejecting absolute and ``..`` paths."""

    path = PurePosixPath(str(name).replace("\\", "/"))
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Unsafe path in {origin}: {name!r}")
    return path


def select_class_split(split, development, test):
    """The class ids (or names) a ``train``/``test``/``train+test`` split keeps."""

    if split == "train":
        return set(development)
    if split == "test":
        return set(test)
    return set(development) | set(test)


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
    # FGVC-Aircraft cuts its copyright banner off every image before resizing.
    CROP_BOTTOM_PIXELS = 0

    @classmethod
    def download_224(cls, root):
        """Prepare the dataset without constructing and parsing a split."""

        root = Path(root)
        if cls.is_ready(root):
            return root
        root.mkdir(parents=True, exist_ok=True)
        downloader = cls.__new__(cls)
        downloader.root = root
        downloader.download_and_remove()
        return root

    @classmethod
    def _marker_matches(cls, root, required_paths=(), **expected):
        """Whether the completion marker and every required path exist.

        The marker must also record the target image size and every
        ``expected`` field; see :meth:`_marker_fields_match`.
        """

        return cls._marker_fields_match(cls._read_marker(root, required_paths), **expected)

    @staticmethod
    def _marker_fields_match(marker, **expected):
        """Compare marker fields after coercion to each expected value's type."""

        try:
            return marker is not None and all(
                type(value)(marker[key]) == value for key, value in expected.items()
            )
        except (KeyError, TypeError, ValueError):
            return False

    @classmethod
    def _marker_image_count(cls, root, required_paths=()):
        """The image count a valid completion marker records, else 0."""

        try:
            return int(cls._read_marker(root, required_paths)["image_count"])
        except (KeyError, TypeError, ValueError):
            return 0

    @classmethod
    def _read_marker(cls, root, required_paths=()):
        """The completion marker, or None unless it records the target size."""

        marker_path = Path(root) / cls.COMPLETE_MARKER
        if not all(Path(path).exists() for path in (*required_paths, marker_path)):
            return None
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            image_size = tuple(int(value) for value in marker["image_size"])
        except (KeyError, TypeError, ValueError, OSError):
            return None
        return marker if image_size == cls.TARGET_IMAGE_SIZE else None

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
    def _resize_image_to_destination(
        cls,
        source,
        destination,
        *,
        crop_bottom_pixels=0,
    ):
        destination = Path(destination)
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

    def _resize_unless_done(self, destination, open_source, stats):
        """Write a resized copy of ``open_source()`` unless one is already there."""

        if self._is_target_sized_image(destination):
            stats["reused_images"] += 1
            return
        with open_source() as source:
            self._resize_image_to_destination(
                source, destination, crop_bottom_pixels=self.CROP_BOTTOM_PIXELS
            )
        stats["written_images"] += 1

    def _resize_all(self, sources, total, description):
        """Resize ``(destination, open_source)`` pairs under one progress bar.

        ``sources`` may be a generator that raises on a missing input, which
        then stops the preparation before any completion marker is written.
        """

        stats = {"written_images": 0, "reused_images": 0}
        with tqdm(total=total, desc=description, unit="image", disable=None) as progress:
            for destination, open_source in sources:
                self._resize_unless_done(destination, open_source, stats)
                progress.update(1)
        return stats

    def _resize_files(self, pairs, total, description, label):
        """Resize ``(source path, destination)`` pairs; a missing source raises."""

        def sources():
            for source_path, destination in pairs:
                if not source_path.is_file():
                    raise RuntimeError(f"{label} source is missing {source_path}")
                yield destination, partial(source_path.open, "rb")

        return self._resize_all(sources(), total, description)

    def _resize_zip_members(self, archive, members, total, description, label):
        """Resize ``(member name, destination)`` pairs out of an open ZIP."""

        archive_members = set(archive.namelist())

        def sources():
            for member, destination in members:
                if member not in archive_members:
                    raise RuntimeError(f"{label} archive is missing {member!r}")
                yield destination, partial(archive.open, member, "r")

        return self._resize_all(sources(), total, description)

    def _resize_tar_stream(
        self,
        archive_stream,
        member_key,
        destination_of,
        description,
        label,
        *,
        expected=None,
        total=None,
        mode="r|gz",
    ):
        """Resize every streamed tar member that ``member_key`` maps to a key.

        Keys must be unique. With ``expected`` they must also be exactly that
        set, so an unexpected or missing image stops the preparation before
        any completion marker is written. ``archive_images`` counts the keys.
        """

        seen = set()
        stats = {"written_images": 0, "reused_images": 0}
        with tqdm(
            total=len(expected) if total is None else total,
            desc=description,
            unit="image",
            disable=None,
        ) as progress, tarfile.open(fileobj=archive_stream, mode=mode) as archive:
            for member in archive:
                key = member_key(member)
                if key is None:
                    continue
                if expected is not None and key not in expected:
                    raise RuntimeError(f"Unexpected image {key!r} in the {label} archive")
                if key in seen:
                    raise RuntimeError(f"Duplicate image {key!r} in the {label} archive")
                seen.add(key)
                # member_key only accepts regular files, which extractfile
                # always opens.
                self._resize_unless_done(
                    destination_of(key), partial(archive.extractfile, member), stats
                )
                progress.update(1)

        missing = set() if expected is None else expected - seen
        if missing:
            raise RuntimeError(
                f"The {label} archive is missing {len(missing)} annotated "
                f"images; examples: {sorted(missing)[:5]}. The completion "
                "marker was not written; rerun to resume."
            )
        return {"archive_images": len(seen), **stats}

    def _resize_from_source(self, source, url, items, label, archive_filename=None):
        """Resize ``items`` from a local directory or archive, else stream ``url``.

        Returns the stats and a description of the source that was used. A
        failed download names ``archive_filename`` as the offline fallback.
        """

        if source is not None and source.is_dir():
            return self._resize_from_directory(source, items), str(source)
        if source is not None:
            with source.open("rb") as archive_stream:
                return self._resize_from_archive_stream(archive_stream, items), str(source)
        try:
            with self._download_stream(url, f"Streaming official {label} image archive") as stream:
                return self._resize_from_archive_stream(stream, items), url
        except OSError as error:
            if archive_filename is None:
                raise
            raise RuntimeError(
                f"Could not download {label} images from {url} ({error}). "
                f"Place {archive_filename} in {self.root} and rerun."
            ) from error

    def _ensure_tar_member(self, url, filename, label):
        """Download the ``url`` tar.gz once, keeping only its ``filename``."""

        destination = Path(self.root) / filename
        if not destination.is_file():
            with self._download_stream(url, f"Downloading {filename}") as stream:
                self._extract_tar_member(stream, filename, destination, label)

    @classmethod
    def _extract_tar_member(cls, archive_stream, filename, destination, label):
        """Atomically copy the member named ``filename`` out of a tar.gz stream."""

        temporary = cls._temporary_path(destination)
        temporary.unlink(missing_ok=True)
        try:
            with tarfile.open(fileobj=archive_stream, mode="r|gz") as archive:
                for member in archive:
                    path = safe_posix_path(member.name, f"{label} archive")
                    if member.isfile() and path.name == filename:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        with archive.extractfile(member) as source, temporary.open("wb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                        break
                else:
                    raise RuntimeError(f"The {label} archive did not contain {filename}")
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

    @classmethod
    @contextlib.contextmanager
    def _download_stream(cls, url, description):
        """A readable byte stream of ``url`` that reports its progress."""

        with cls._open_url(url) as response:
            with cls._byte_progress(response, description) as progress:
                yield _DownloadProgressReader(response, progress)

    @classmethod
    def _download_url_to_path(cls, url, destination, description=None):
        """Stream ``url`` into ``destination``; returns the declared byte count."""

        with cls._open_url(url) as response:
            content_length = response.headers.get("Content-Length")
            with cls._byte_progress(response, description) as progress:
                with Path(destination).open("wb") as output:
                    while chunk := response.read(1024 * 1024):
                        output.write(chunk)
                        progress.update(len(chunk))
        return int(content_length) if content_length else None

    def _download_checked_archive(self, destination, label, magic, archive_kind, likely_cause):
        """Download ``DOWNLOAD_URL`` and reject a truncated or non-archive body.

        Hosts answer a refused or over-quota request with an HTML page and a
        200 status, so the length and the leading magic bytes are checked.
        """

        expected_bytes = self._download_url_to_path(self.DOWNLOAD_URL, destination, label)
        downloaded_bytes = destination.stat().st_size
        if expected_bytes is not None and downloaded_bytes != expected_bytes:
            raise RuntimeError(
                f"The {label} download stopped after {downloaded_bytes} "
                f"of {expected_bytes} bytes"
            )
        with destination.open("rb") as archive:
            leading_bytes = archive.read(len(magic))
        if leading_bytes != magic:
            raise RuntimeError(
                f"{self.DOWNLOAD_URL} returned {downloaded_bytes} bytes that "
                f"are not a {archive_kind} archive; {likely_cause} instead of "
                "the dataset"
            )

    def _download_missing_files(self, base_url, directory, filenames, label):
        """Fetch every ``base_url/<filename>`` not yet present in ``directory``."""

        directory.mkdir(parents=True, exist_ok=True)
        for filename in filenames:
            destination = directory / filename
            if destination.is_file():
                continue
            temporary = self._temporary_path(destination)
            temporary.unlink(missing_ok=True)
            url = f"{base_url}/{filename}"
            try:
                self._download_url_to_path(
                    url, temporary, description=f"Downloading {label} {filename}"
                )
                os.replace(temporary, destination)
            except OSError as error:
                raise RuntimeError(f"Could not download {label} from {url}") from error
            finally:
                temporary.unlink(missing_ok=True)

    @classmethod
    def _write_bytes_atomic(cls, destination, content):
        destination = Path(destination)
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
        destination = Path(destination)
        return destination.with_name(f".{destination.name}.{os.getpid()}.part")
