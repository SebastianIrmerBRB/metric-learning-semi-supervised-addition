"""Fetch the external unlabeled pools on first use.

SLADE pairs Cars-196 with CompCars and In-Shop with Fashion200k (Sec. 4.1).
Neither ships with its benchmark, so a run that names one of them and finds its
directory without images downloads it there first. Both sources serve
anonymous, resumable HTTP (checked 2026-10-09):

* CompCars: the CUHK release on Google Drive, as linked together with the
  archive password from
  https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/instruction.txt. Only the
  web-nature parts (data.z01-data.z22 and data.zip, 16.5 GB) are fetched; the
  surveillance set (sv_data.*) is never used. Drive refuses a file for up to a
  day once its download quota is used up, and the Dropbox copy the instructions
  also link holds only README.txt by now, so a refused part comes from the
  Kaggle mirror duythinguyn/compcars instead: the same parts at the same sizes,
  and its data.zip, which holds the CRC of every archived file, is
  byte-identical to Drive's (SHA-256 checked 2026-10-09). The split archive is
  rejoined with ``zip -F`` and decrypted with ``unzip``, as the instructions
  describe, so both tools must be on PATH.
* Fashion200k: the authors' Google Drive no longer holds the image archive
  (women.tar.gz), so the images come from the Kaggle mirror
  mayukh18/fashion200k-dataset (6.1 GB zip). It keeps the original
  ``women/<category>/<subcategory>/<product>/<image>.jpeg`` layout.

Everything is staged in a hidden sibling of the target directory and moved in
only once complete, under a file lock, so concurrent runs download once and a
killed run resumes where it stopped.
"""

import os
import shutil
import subprocess
import time
import urllib.parse
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from filelock import FileLock, Timeout
from loguru import logger
from tqdm import tqdm

from .dataset_composition import (
    EXTERNAL_UNLABELED_FILTERS_COMPCARS_PAPER,
    EXTERNAL_UNLABELED_FILTER_COMPCARS_MODEL_MIN_COUNT,
    EXTERNAL_UNLABELED_FILTER_NONE,
)

# .png: scripts/download_compcars_224.py stores CompCars' paper subset losslessly.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
DOWNLOAD_ATTEMPTS = 5
DOWNLOAD_CHUNK_BYTES = 8 << 20
# Headroom kept free on top of the archive and the tree it extracts to.
FREE_SPACE_MARGIN_BYTES = 2 << 30


class ExternalPoolDownloadError(RuntimeError):
    """A download that retrying will not fix."""


class _IncompleteDownload(OSError):
    """The connection ended before the whole file arrived; resumable."""


@dataclass(frozen=True)
class ArchivePart:
    name: str
    # Tried in order; every later source serves the same bytes.
    urls: tuple
    size: int


def _drive_url(file_id):
    return f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t"


def _kaggle_file_url(dataset, path):
    # Kaggle answers 404 unless the slashes inside the file path are encoded.
    return f"https://www.kaggle.com/api/v1/datasets/download/{dataset}/{urllib.parse.quote(path, safe='')}"


_COMPCARS_PART_BYTES = 734003200
_COMPCARS_KAGGLE_MIRROR = "duythinguyn/compcars"
# (part, Google Drive file id, the Google Takeout folder holding it in the Kaggle mirror)
_COMPCARS_SOURCES = (
    ("data.z01", "1jRgYO7KaX08BjPWpO5m4yjnli9KT40H2", 8),
    ("data.z02", "1wMoVAd2UniiC1k94EVac2MtwJRAlaqEg", 1),
    ("data.z03", "1x8rfnd48g6YeIB2sYZlKR9yUWq0EVTXN", 4),
    ("data.z04", "1J1UKu30fg5DoNq7PIGJDPTsNRccpeAO0", 11),
    ("data.z05", "1pHyr3RmgNSqgS1FJyyYI8Eko6kgzA8yP", 6),
    ("data.z06", "15cQhDjUILTP5JOijWInzN9mtuLSZOTCv", 11),
    ("data.z07", "1Qy9FrgJT-N33HXKrwPAhDe7cg2Uwd95z", 4),
    ("data.z08", "1nd3J1WZ81eaJuNw8LJ_-Azouo7uO8221", 12),
    ("data.z09", "1Td81PVNWAH_LEgjPX3bZiH3-OzwVAfSK", 8),
    ("data.z10", "1i6szbo8UYNmuoA3eLwUd5RyGV6l5C-qt", 5),
    ("data.z11", "1QHGbUiU5CwgSAppJFLpVrTlLm8vI9dK9", 9),
    ("data.z12", "1HfW88qqiuKBQWRnZJE9yWMU7_K3Rnr0_", 5),
    ("data.z13", "1la7Jt8ti2yxRtUG_4YC4zWPId37ja5jF", 3),
    ("data.z14", "1Fo1xpfuRWcLk4Yt-bdFMgmW99e3X1SBM", 7),
    ("data.z15", "1WTajdHaJ0sYo2gBOvCAtWNoM9S9W7aYv", 12),
    ("data.z16", "1BBM7fT0rs_-S15llKPwV2uv7K8lVpG5X", 3),
    ("data.z17", "1HMxnsz5-u4OM9UgLM30PLQYGIl4bwC_L", 10),
    ("data.z18", "1WfENLN1QsmHNTBZjqNkRp9g7hN1tv1wq", 10),
    ("data.z19", "1bweCNot6B8wk6KhcnDmdaguxRD7kxEIp", 2),
    ("data.z20", "1ewPYhpmJY9Ug-GzKZJ_B6eSEsnnoaXeK", 1),
    ("data.z21", "1i9nsf6g_jV46vMaz8ZqshG6vJOJFvDDL", 2),
    ("data.z22", "1Hq53fIctob6Dh3TiyDvNkWI10Z-pDruV", 9),
    ("data.zip", "1T9K-7-K2bHFepntxIz4guYaqPfRWYW8p", 2),
)
COMPCARS_PARTS = tuple(
    ArchivePart(
        name,
        (
            _drive_url(file_id),
            _kaggle_file_url(_COMPCARS_KAGGLE_MIRROR, f"CompCars-20250717T151129Z-1-{folder:03d}/CompCars/{name}"),
        ),
        391417395 if name == "data.zip" else _COMPCARS_PART_BYTES,
    )
    for name, file_id, folder in _COMPCARS_SOURCES
)
# Published by CUHK in the instruction.txt linked from the dataset page.
COMPCARS_ARCHIVE_PASSWORD = "d89551fd190e38"
COMPCARS_TOP_LEVEL = ("data",)

FASHION200K_ARCHIVE = ArchivePart(
    "fashion200k-dataset.zip",
    ("https://www.kaggle.com/api/v1/datasets/download/mayukh18/fashion200k-dataset",),
    6071136101,
)
FASHION200K_TOP_LEVEL = ("women", "labels", "detection", "image_urls.txt")

POOL_COMPCARS = "CompCars"
POOL_FASHION200K = "Fashion200k"
_COMPCARS_FILTERS = (
    EXTERNAL_UNLABELED_FILTER_COMPCARS_MODEL_MIN_COUNT,
    *EXTERNAL_UNLABELED_FILTERS_COMPCARS_PAPER,
)
_FASHION200K_DIR_NAMES = {"fashion200k", "fashion-200k", "fashion_200k"}


def identify_external_pool(external_dir, external_filter=EXTERNAL_UNLABELED_FILTER_NONE):
    """Name the downloadable pool a run points at, or None for any other directory.

    A CompCars filter always means CompCars; otherwise the directory name decides,
    so an unfiltered ``data/CompCars`` or ``data/Fashion200k`` is recognized too.
    """

    name = Path(external_dir).name.casefold()
    if external_filter in _COMPCARS_FILTERS or name == "compcars":
        return POOL_COMPCARS
    if name in _FASHION200K_DIR_NAMES:
        return POOL_FASHION200K
    return None


def _holds_images(root):
    if not Path(root).is_dir():
        return False
    for _, _, file_names in os.walk(root):
        if any(name.lower().endswith(IMAGE_SUFFIXES) for name in file_names):
            return True
    return False


def _count_images(root):
    return sum(
        name.lower().endswith(IMAGE_SUFFIXES)
        for _, _, file_names in os.walk(root)
        for name in file_names
    )


def ensure_external_unlabeled_pool(external_dir, external_filter=EXTERNAL_UNLABELED_FILTER_NONE):
    """Download CompCars or Fashion200k into ``external_dir`` unless it already holds images.

    Returns True when this call downloaded the pool. Unknown pools and directories
    that already hold images are left untouched, so a hand-made copy always wins.
    """

    if external_dir is None:
        return False
    pool = identify_external_pool(external_dir, external_filter)
    if pool is None:
        return False
    root = Path(external_dir)
    if _holds_images(root):
        return False

    root.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(root.parent / f".{root.name}.download.lock"))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        logger.info(f"Another run is downloading {pool} into {root}; waiting for it to finish")
        lock.acquire()
    try:
        # The run that held the lock may have finished the job.
        if _holds_images(root):
            return False
        logger.info(f"{pool} not found at {root}; downloading it before training")
        if pool == POOL_COMPCARS:
            _download_compcars(root)
        else:
            _download_fashion200k(root)
        logger.info(f"{pool} ready at {root}: {_count_images(root)} images")
        return True
    finally:
        lock.release()


def _staging_dir(root):
    staging = root.parent / f".{root.name}.download"
    staging.mkdir(parents=True, exist_ok=True)
    return staging


def _file_size(path):
    return path.stat().st_size if path.exists() else 0


def _require_free_space(directory, needed_bytes, pool):
    free_bytes = shutil.disk_usage(directory).free
    if free_bytes < needed_bytes:
        raise ExternalPoolDownloadError(
            f"Downloading {pool} needs about {needed_bytes / 1e9:.1f} GB free in {directory}, "
            f"but only {free_bytes / 1e9:.1f} GB are free"
        )


def _run_tool(command, cwd, accepted_returncodes=(0,)):
    completed = subprocess.run(
        command,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    output_tail = (completed.stdout + completed.stderr).strip()[-2000:]
    if completed.returncode not in accepted_returncodes:
        # command[0] only: the arguments carry the archive password.
        raise ExternalPoolDownloadError(
            f"{command[0]} failed with exit code {completed.returncode} in {cwd}:\n{output_tail}"
        )
    if completed.returncode != 0:
        logger.warning(f"{command[0]} finished with warnings (exit code {completed.returncode}):\n{output_tail}")


def _require_no_clash(root, top_level_names, pool):
    """Fail before downloading when leftovers would block installing the download."""

    clashes = [name for name in top_level_names if (root / name).exists()]
    if clashes:
        raise ExternalPoolDownloadError(
            f"{root} holds no images but already contains {clashes}, probably from an earlier "
            f"attempt; remove them so {pool} can be installed there."
        )


def _install_extracted_tree(extract_dir, root):
    """Move the extracted top-level entries into ``root`` without overwriting anything."""

    root.mkdir(parents=True, exist_ok=True)
    entries = sorted(extract_dir.iterdir())
    clashes = [entry.name for entry in entries if (root / entry.name).exists()]
    if clashes:
        raise ExternalPoolDownloadError(
            f"{root} already contains {clashes}; the downloaded copy is left in {extract_dir}. "
            "Move those entries away and start the run again."
        )
    for entry in entries:
        shutil.move(str(entry), str(root / entry.name))


def _download_compcars(root):
    missing_tools = [tool for tool in ("zip", "unzip") if shutil.which(tool) is None]
    if missing_tools:
        raise ExternalPoolDownloadError(
            "CompCars ships as a password-protected split zip; rejoining and extracting it needs "
            f"{' and '.join(missing_tools)} on PATH (e.g. `sudo apt install zip unzip`)."
        )
    _require_no_clash(root, COMPCARS_TOP_LEVEL, POOL_COMPCARS)
    staging = _staging_dir(root)
    combined = staging / "combined.zip"
    archive_bytes = sum(part.size for part in COMPCARS_PARTS)
    if not combined.exists():
        downloaded_bytes = sum(_file_size(staging / part.name) for part in COMPCARS_PARTS)
        # The parts, then their rejoined copy, then the tree it extracts to: JPEGs
        # barely compress, so each stage is about the archive's size.
        _require_free_space(staging, 2 * archive_bytes - downloaded_bytes + FREE_SPACE_MARGIN_BYTES, POOL_COMPCARS)
        for index, part in enumerate(COMPCARS_PARTS, start=1):
            logger.info(f"CompCars part {index}/{len(COMPCARS_PARTS)}: {part.name}")
            _download(part.urls, staging / part.name, part.size)
        logger.info("CompCars: rejoining the split archive")
        partial = staging / "combined.partial.zip"
        partial.unlink(missing_ok=True)
        _run_tool(["zip", "-F", "data.zip", "--out", partial.name], cwd=staging)
        partial.rename(combined)
        for part in COMPCARS_PARTS:
            (staging / part.name).unlink()
    else:
        _require_free_space(staging, archive_bytes + FREE_SPACE_MARGIN_BYTES, POOL_COMPCARS)
    logger.info("CompCars: extracting")
    extract_dir = staging / "extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    # unzip exits with 1 for warnings that still extracted everything.
    _run_tool(
        ["unzip", "-q", "-o", "-P", COMPCARS_ARCHIVE_PASSWORD, combined.name, "-d", extract_dir.name],
        cwd=staging,
        accepted_returncodes=(0, 1),
    )
    _install_extracted_tree(extract_dir, root)
    shutil.rmtree(staging)


def _download_fashion200k(root):
    _require_no_clash(root, FASHION200K_TOP_LEVEL, POOL_FASHION200K)
    staging = _staging_dir(root)
    archive = staging / FASHION200K_ARCHIVE.name
    _require_free_space(
        staging,
        2 * FASHION200K_ARCHIVE.size - _file_size(archive) + FREE_SPACE_MARGIN_BYTES,
        POOL_FASHION200K,
    )
    _download(FASHION200K_ARCHIVE.urls, archive, FASHION200K_ARCHIVE.size)
    logger.info("Fashion200k: extracting")
    extract_dir = staging / "extract"
    shutil.rmtree(extract_dir, ignore_errors=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(extract_dir)
    _install_extracted_tree(extract_dir, root)
    shutil.rmtree(staging)


def _download(urls, destination, expected_size, attempts=DOWNLOAD_ATTEMPTS):
    """Fetch ``destination`` from the first of ``urls`` that delivers it, resuming a partial file.

    The sources serve identical bytes, so a file one of them started is finished
    from the next.
    """

    if isinstance(urls, str):
        urls = (urls,)
    for index, url in enumerate(urls):
        try:
            _download_from(url, destination, expected_size, attempts)
            return
        except ExternalPoolDownloadError as exc:
            if index == len(urls) - 1:
                raise
            logger.warning(
                f"{destination.name} is not available from {urllib.parse.urlsplit(url).netloc} ({exc}); "
                f"trying {urllib.parse.urlsplit(urls[index + 1]).netloc}"
            )


def _download_from(url, destination, expected_size, attempts):
    for attempt in range(1, attempts + 1):
        try:
            _download_once(url, destination, expected_size)
            return
        except (requests.RequestException, _IncompleteDownload) as exc:
            if attempt == attempts:
                raise ExternalPoolDownloadError(
                    f"Downloading {destination.name} failed {attempts} times; the partial file is "
                    f"kept and the next run resumes it. Last error: {exc}"
                ) from exc
            delay_seconds = 30 * attempt
            logger.warning(
                f"Downloading {destination.name} failed ({exc}); retry {attempt}/{attempts - 1} "
                f"in {delay_seconds} s"
            )
            time.sleep(delay_seconds)


def _source_size(response, resumed_from):
    if response.status_code == 206:
        content_range = response.headers.get("Content-Range", "")
        total = content_range.rpartition("/")[2]
        return int(total) if total.isdigit() else None
    content_length = response.headers.get("Content-Length")
    return None if content_length is None else int(content_length) + resumed_from


def _download_once(url, destination, expected_size):
    done_bytes = _file_size(destination)
    if done_bytes == expected_size:
        return
    if done_bytes > expected_size:
        destination.unlink()
        done_bytes = 0
    headers = {"Range": f"bytes={done_bytes}-"} if done_bytes else {}
    with requests.get(url, headers=headers, stream=True, timeout=(30, 300)) as response:
        response.raise_for_status()
        if "text/html" in response.headers.get("Content-Type", ""):
            raise ExternalPoolDownloadError(
                f"got a web page instead of {destination.name}; Google Drive serves one while a file's "
                "download quota is used up ('Too many users have viewed or downloaded this file "
                "recently'), for up to a day. Finished files are kept for the next attempt."
            )
        if response.status_code != 206:
            # The server ignored the range request, so the body starts at byte 0.
            done_bytes = 0
        source_size = _source_size(response, done_bytes)
        if source_size is not None and source_size != expected_size:
            raise ExternalPoolDownloadError(
                f"{destination.name} is {source_size} bytes at {url}, expected {expected_size}; "
                "the upstream file changed, so this download needs updating"
            )
        with open(destination, "ab" if done_bytes else "wb") as handle, tqdm(
            total=expected_size,
            initial=done_bytes,
            unit="B",
            unit_scale=True,
            desc=destination.name,
            mininterval=30,
        ) as progress:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_BYTES):
                handle.write(chunk)
                progress.update(len(chunk))
    received_bytes = _file_size(destination)
    if received_bytes != expected_size:
        raise _IncompleteDownload(f"{destination.name}: {received_bytes} of {expected_size} bytes")
