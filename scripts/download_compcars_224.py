"""Download SLADE's CompCars unlabeled pool, stored as 224 x 224 PNGs.

The web-nature release is a 16.5 GB password-protected split zip
(data.z01-data.z22 + data.zip), and unpacking it the way
utils/external_pool_download.py does peaks at about 35 GB. This script streams
it instead: data.zip, which holds the central directory, comes first, then each
part in turn; the images of a part are decrypted, resized and written, and the
part is deleted before the next one is fetched. Peak disk use is data.zip, two
parts and the output, about 3.5 GB.

Only the 16,537 images / 145 make/model/year classes that the
``compcars_slade_paper`` filter keeps are stored (selected with the dataset's
own path rules, see CompCarsSLADEPaperUnlabeledImageDataset). Each one is
converted to RGB and resized to 224 x 224 with PIL bilinear, which is what the
default ``squash`` transform does, and saved losslessly, so the transform's own
resize becomes a no-op and the backbone sees bit-identical input. Runs with
``image_resize_mode=dinov2`` or another ``image_size`` would not.

Rerunning resumes: stored images and finished parts are skipped. Training runs
that start meanwhile wait on the downloader's lock and then find the pool.

    python scripts/download_compcars_224.py --root data/CompCars
"""

import argparse
import io
import json
import os
import shutil
import struct
import sys
import zlib
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from filelock import FileLock
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import external_pool_download as download  # noqa: E402
from utils import local_datasets  # noqa: E402

CompCarsSLADE = local_datasets.CompCarsSLADEPaperUnlabeledImageDataset

STORED_IMAGE_SIZE = (224, 224)
# Generous per-image bound for the free-space check; 224 x 224 PNG car photos average ~71 KiB.
STORED_IMAGE_BYTES_BOUND = 96 << 10
FREE_SPACE_MARGIN_BYTES = 1 << 30
MARKER_NAME = ".compcars_slade_paper_224_complete.json"

LOCAL_HEADER = struct.Struct("<IHHHHHIIIHH")
CENTRAL_HEADER = struct.Struct("<IHHHHHHIIIHHHHHII")
END_OF_DIRECTORY = struct.Struct("<IHHHHIIH")
ZIP64_LOCATOR = struct.Struct("<IIQI")
ZIP64_END_OF_DIRECTORY = struct.Struct("<IQHHIIQQQQ")
LOCAL_HEADER_SIGNATURE = 0x04034B50
CENTRAL_HEADER_SIGNATURE = 0x02014B50
END_OF_DIRECTORY_SIGNATURE = b"PK\x05\x06"
ZIP64_LOCATOR_SIGNATURE = 0x07064B50
ZIP64_END_SIGNATURE = 0x06064B50
FLAG_ENCRYPTED = 0x1
FLAG_DATA_DESCRIPTOR = 0x8
FLAG_STRONG_ENCRYPTION = 0x40
FLAG_UTF8 = 0x800


@dataclass(frozen=True)
class ZipEntry:
    name: str
    flags: int
    method: int
    mod_time: int
    crc: int
    compressed_size: int
    # Offsets into the parts joined end to end: the local header, and where the next record begins.
    start: int
    end: int


def read_split_zip_directory(last_part, part_sizes):
    """Every entry of a split zip, read from the central directory in its last part."""

    part_starts = [sum(part_sizes[:index]) for index in range(len(part_sizes))]
    last_disk = len(part_sizes) - 1
    with open(last_part, "rb") as handle:
        tail_size = min(part_sizes[-1], END_OF_DIRECTORY.size + 0xFFFF + ZIP64_LOCATOR.size)
        handle.seek(part_sizes[-1] - tail_size)
        tail = handle.read(tail_size)
        end_position = tail.rfind(END_OF_DIRECTORY_SIGNATURE)
        if end_position < 0:
            raise ValueError(f"{last_part} has no end-of-central-directory record")
        _, _, directory_disk, _, entry_count, directory_size, directory_offset, _ = END_OF_DIRECTORY.unpack_from(
            tail, end_position
        )
        if 0xFFFF in (directory_disk, entry_count) or 0xFFFFFFFF in (directory_size, directory_offset):
            signature, zip64_disk, zip64_offset, _ = ZIP64_LOCATOR.unpack_from(tail, end_position - ZIP64_LOCATOR.size)
            if signature != ZIP64_LOCATOR_SIGNATURE or zip64_disk != last_disk:
                raise ValueError(f"{last_part}: zip64 directory record is missing or not in the last part")
            handle.seek(zip64_offset)
            fields = ZIP64_END_OF_DIRECTORY.unpack(handle.read(ZIP64_END_OF_DIRECTORY.size))
            if fields[0] != ZIP64_END_SIGNATURE:
                raise ValueError(f"{last_part}: bad zip64 end-of-central-directory record")
            directory_disk, entry_count, directory_size, directory_offset = fields[5], fields[7], fields[8], fields[9]
        if directory_disk != last_disk:
            raise ValueError(f"the central directory starts in part {directory_disk + 1}, not in {last_part.name}")
        handle.seek(directory_offset)
        directory = handle.read(directory_size)

    records = []
    position = 0
    for _ in range(entry_count):
        fields = CENTRAL_HEADER.unpack_from(directory, position)
        if fields[0] != CENTRAL_HEADER_SIGNATURE:
            raise ValueError(f"bad central directory record at byte {position}")
        flags, method, mod_time, crc, compressed_size, file_size = fields[3], fields[4], fields[5], fields[7], fields[8], fields[9]
        name_length, extra_length, comment_length, disk, local_offset = fields[10], fields[11], fields[12], fields[13], fields[16]
        name_start = position + CENTRAL_HEADER.size
        raw_name = directory[name_start:name_start + name_length]
        name = raw_name.decode("utf-8" if flags & FLAG_UTF8 else "cp437")
        extra = directory[name_start + name_length:name_start + name_length + extra_length]
        # Zip64 extra field: only the fields saturated in the fixed record follow, in this order.
        extra_position = 0
        while extra_position + 4 <= len(extra):
            header_id, size = struct.unpack_from("<HH", extra, extra_position)
            if header_id == 0x0001:
                values = extra[extra_position + 4:extra_position + 4 + size]
                value_position = 0
                if file_size == 0xFFFFFFFF:
                    value_position += 8
                if compressed_size == 0xFFFFFFFF:
                    (compressed_size,) = struct.unpack_from("<Q", values, value_position)
                    value_position += 8
                if local_offset == 0xFFFFFFFF:
                    (local_offset,) = struct.unpack_from("<Q", values, value_position)
                    value_position += 8
                if disk == 0xFFFF:
                    (disk,) = struct.unpack_from("<I", values, value_position)
            extra_position += 4 + size
        records.append((name, flags, method, mod_time, crc, compressed_size, part_starts[disk] + local_offset))
        position = name_start + name_length + extra_length + comment_length

    records.sort(key=lambda record: record[-1])
    directory_start = part_starts[last_disk] + directory_offset
    ends = [record[-1] for record in records[1:]] + [directory_start]
    return [ZipEntry(*record, end=end) for record, end in zip(records, ends)], part_starts


def select_paper_subset(entries):
    """The archive entries the compcars_slade_paper filter keeps, checked against the paper's counts."""

    pseudo_root = Path("CompCars")
    candidates = []
    for entry in entries:
        if PurePosixPath(entry.name).suffix.lower() not in download.IMAGE_SUFFIXES:
            continue
        path = pseudo_root / entry.name
        if CompCarsSLADE.is_whole_car_image_path(pseudo_root, path):
            candidates.append((entry, CompCarsSLADE.infer_model_key(pseudo_root, path)))
    category_counts = Counter(key for _, key in candidates)
    if (len(candidates), len(category_counts)) != (CompCarsSLADE.WEB_NATURE_IMAGES, CompCarsSLADE.WEB_NATURE_MODEL_YEAR_CLASSES):
        raise ValueError(
            f"the archive holds {len(candidates)} whole-car images in {len(category_counts)} model-years, expected "
            f"{CompCarsSLADE.WEB_NATURE_IMAGES} in {CompCarsSLADE.WEB_NATURE_MODEL_YEAR_CLASSES}; the release changed"
        )
    kept = [entry for entry, key in candidates if category_counts[key] >= CompCarsSLADE.PAPER_MIN_IMAGES_PER_MODEL]
    kept_categories = {key for key, count in category_counts.items() if count >= CompCarsSLADE.PAPER_MIN_IMAGES_PER_MODEL}
    if (len(kept), len(kept_categories)) != (CompCarsSLADE.PAPER_TARGET_IMAGES, CompCarsSLADE.PAPER_TARGET_MODEL_CLASSES):
        raise ValueError(
            f"the paper filter keeps {len(kept)} images in {len(kept_categories)} model-years, expected "
            f"{CompCarsSLADE.PAPER_TARGET_IMAGES} in {CompCarsSLADE.PAPER_TARGET_MODEL_CLASSES}"
        )
    return kept


def build_crc_table():
    table = []
    for byte in range(256):
        value = byte
        for _ in range(8):
            value = (value >> 1) ^ 0xEDB88320 if value & 1 else value >> 1
        table.append(value)
    return tuple(table)


CRC_TABLE = build_crc_table()


def zipcrypto_decrypt(data, password, check_byte):
    """Decrypt traditional PKWARE (ZipCrypto) data and drop its 12-byte header."""

    table = CRC_TABLE
    key0, key1, key2 = 0x12345678, 0x23456789, 0x34567890
    for byte in password:
        key0 = (key0 >> 8) ^ table[(key0 ^ byte) & 0xFF]
        key1 = ((key1 + (key0 & 0xFF)) * 134775813 + 1) & 0xFFFFFFFF
        key2 = (key2 >> 8) ^ table[(key2 ^ (key1 >> 24)) & 0xFF]
    plain = bytearray(len(data))
    for index, byte in enumerate(data):
        temp = (key2 | 2) & 0xFFFF
        byte ^= ((temp * (temp ^ 1)) >> 8) & 0xFF
        plain[index] = byte
        key0 = (key0 >> 8) ^ table[(key0 ^ byte) & 0xFF]
        key1 = ((key1 + (key0 & 0xFF)) * 134775813 + 1) & 0xFFFFFFFF
        key2 = (key2 >> 8) ^ table[(key2 ^ (key1 >> 24)) & 0xFF]
    if plain[11] != check_byte:
        raise ValueError("wrong archive password")
    return bytes(plain[12:])


def stored_path(output_dir, entry):
    return output_dir / PurePosixPath(entry.name).with_suffix(".png")


def store_image(task):
    """Decrypt, inflate, check and resize one archive entry; runs in a worker process."""

    segments, entry, password, destination = task
    chunks = []
    for part_path, offset, length in segments:
        with open(part_path, "rb") as handle:
            handle.seek(offset)
            chunks.append(handle.read(length))
    record = b"".join(chunks)
    fields = LOCAL_HEADER.unpack_from(record)
    if fields[0] != LOCAL_HEADER_SIGNATURE:
        raise ValueError(f"{entry.name}: no local header at its offset")
    data_start = LOCAL_HEADER.size + fields[9] + fields[10]
    payload = record[data_start:data_start + entry.compressed_size]
    if len(payload) != entry.compressed_size:
        raise ValueError(f"{entry.name}: record is truncated")
    if entry.flags & FLAG_STRONG_ENCRYPTION:
        raise ValueError(f"{entry.name}: strong encryption is not supported")
    if entry.flags & FLAG_ENCRYPTED:
        check = entry.mod_time >> 8 if entry.flags & FLAG_DATA_DESCRIPTOR else entry.crc >> 24
        payload = zipcrypto_decrypt(payload, password, check & 0xFF)
    if entry.method == 8:
        raw = zlib.decompress(payload, -15)
    elif entry.method == 0:
        raw = payload
    else:
        raise ValueError(f"{entry.name}: unsupported compression method {entry.method}")
    if zlib.crc32(raw) != entry.crc:
        raise ValueError(f"{entry.name}: CRC mismatch")
    # torchvision's default_loader, then the squash transform's Resize((224, 224)).
    with Image.open(io.BytesIO(raw)) as image:
        rgb = image.convert("RGB")
    resized = rgb.resize(STORED_IMAGE_SIZE, resample=Image.Resampling.BILINEAR)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".partial")
    resized.save(temporary, format="PNG")
    os.replace(temporary, destination)
    return destination.stat().st_size


def segments_of(entry, part_starts, part_paths):
    segments = []
    for index, (part_start, part_path) in enumerate(zip(part_starts, part_paths)):
        part_end = part_starts[index + 1] if index + 1 < len(part_starts) else float("inf")
        low, high = max(entry.start, part_start), min(entry.end, part_end)
        if low < high:
            segments.append((part_path, low - part_start, high - low))
    return segments


def store_batch(entries, part_starts, part_paths, output_dir, password, pool, description):
    tasks = [
        (segments_of(entry, part_starts, part_paths), entry, password, stored_path(output_dir, entry))
        for entry in entries
    ]
    stored_bytes = 0
    for size in tqdm(pool.map(store_image, tasks, chunksize=16), total=len(tasks), desc=description, mininterval=10):
        stored_bytes += size
    return stored_bytes


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--root", type=Path, default=PROJECT_ROOT / "data" / "CompCars",
                        help="external_unlabeled_dir of the Cars runs (default: data/CompCars).")
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1),
                        help="Decrypting/resizing processes (ZipCrypto in Python runs at ~3 MB/s per process).")
    return parser.parse_args()


def main():
    args = parse_args()
    # absolute(), not resolve(): the lock must sit where the runs look for it, even behind a symlink.
    root = args.root.absolute()
    if download._holds_images(root):
        print(f"{root} already holds images; nothing to do.")
        return
    if (root / "data").exists():
        raise SystemExit(f"{root}/data exists but holds no images (an earlier attempt?); remove it first.")
    root.parent.mkdir(parents=True, exist_ok=True)
    # The same lock as the on-demand downloader, so a run starting meanwhile waits instead of downloading too.
    with FileLock(str(root.parent / f".{root.name}.download.lock")):
        if download._holds_images(root):
            print(f"{root} already holds images; nothing to do.")
            return
        staging = root.parent / f".{root.name}.224.download"
        output_dir = staging / "stored"
        output_dir.mkdir(parents=True, exist_ok=True)
        parts = download.COMPCARS_PARTS
        part_paths = [staging / part.name for part in parts]
        last = len(parts) - 1
        needed = (
            parts[last].size - download._file_size(part_paths[last])
            + 2 * max(part.size for part in parts[:last])
            + CompCarsSLADE.PAPER_TARGET_IMAGES * STORED_IMAGE_BYTES_BOUND
            + FREE_SPACE_MARGIN_BYTES
        )
        download._require_free_space(staging, needed, download.POOL_COMPCARS)

        print(f"[1/3] central directory: {parts[last].name}")
        download._download(parts[last].urls, part_paths[last], parts[last].size)
        entries, part_starts = read_split_zip_directory(part_paths[last], [part.size for part in parts])
        selected = select_paper_subset(entries)
        pending = [entry for entry in selected if not stored_path(output_dir, entry).is_file()]
        print(f"{len(selected)} paper-subset images, {len(selected) - len(pending)} already stored")

        password = download.COMPCARS_ARCHIVE_PASSWORD.encode()
        print(f"[2/3] streaming parts with {args.workers} workers")
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for index in range(last):
                part_end = part_starts[index + 1]
                if not any(entry.start < part_end and entry.end > part_starts[index] for entry in pending):
                    part_paths[index].unlink(missing_ok=True)
                    continue
                download._download(parts[index].urls, part_paths[index], parts[index].size)
                ready = [entry for entry in pending if entry.end <= part_end]
                pending = [entry for entry in pending if entry.end > part_end]
                store_batch(ready, part_starts, part_paths, output_dir, password, pool, parts[index].name)
                next_start = pending[0].start if pending else float("inf")
                for earlier in range(index + 1):
                    if part_starts[earlier + 1] <= next_start:
                        part_paths[earlier].unlink(missing_ok=True)
            store_batch(pending, part_starts, part_paths, output_dir, password, pool, parts[last].name)

        print("[3/3] installing")
        stored = [stored_path(output_dir, entry) for entry in selected]
        missing = [path for path in stored if not path.is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} images were not stored, e.g. {missing[0]}; rerun to resume.")
        for leftover in output_dir.rglob("*.partial"):
            leftover.unlink()
        (output_dir / MARKER_NAME).write_text(json.dumps({
            "pool": "compcars_slade_paper",
            "image_count": len(stored),
            "model_year_classes": CompCarsSLADE.PAPER_TARGET_MODEL_CLASSES,
            "min_images_per_model": CompCarsSLADE.PAPER_MIN_IMAGES_PER_MODEL,
            "image_size": list(STORED_IMAGE_SIZE),
            "resize": "PIL convert('RGB') then bilinear resize, as the squash transform",
            "format": "PNG",
        }, indent=2) + "\n", encoding="utf-8")
        download._install_extracted_tree(output_dir, root)
        shutil.rmtree(staging)
        total_bytes = sum(path.stat().st_size for path in root.rglob("*.png"))
        print(f"done: {len(stored)} images, {total_bytes / 2**30:.2f} GiB in {root}")


if __name__ == "__main__":
    main()
