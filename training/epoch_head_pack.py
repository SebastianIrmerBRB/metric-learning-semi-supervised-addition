"""Pack the epoch heads of one run into a single losslessly compressed file.

``--save_epoch_head_interval`` writes one ``head_state.pt``-format file per saved
epoch. A trained float32 head hardly compresses on its own (xz takes ~7 % off a
768-d linear head): its low mantissa bytes look random. Each tensor is therefore
stored as the difference of its bits, read as integers, from the previous saved
epoch's, zigzag-coded so that small steps either way become small numbers, split
into byte planes (every element's first byte, then every second byte, ...) so the
unchanged high bytes form runs of zeros, and then compressed. Every
``keyframe_interval``-th saved epoch is stored whole, which bounds the deltas
reading one epoch has to replay.

Measured on 2026-10-01 over the 108 per-epoch heads of a Cars LRML+Circle fold:
255 MB -> 170 MB (1.50x) in ~6 s with zlib; XOR deltas gave 1.46x, lzma 1.55x at
six times the time. It is not more because Adam moves every weight by about the
learning rate on every step, so an epoch still changes most mantissa bits: a
late epoch's delta is 1.6 MB against an early one's 1.75 MB. Heads saved as
16-bit (``--epoch_head_dtype``) pack to 4.0x (float16) and 6.0x (bfloat16) of
the float32 files on the same fold.

``pack_epoch_heads`` reads the packed file back and compares every epoch with its
original, bit for bit, before it deletes anything; a run whose heads it cannot
reproduce exactly keeps them as they are.
"""

import bz2
import json
import lzma
import os
import zlib
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from loguru import logger

from .resume import EPOCH_HEADS_DIRNAME, PACKED_EPOCH_HEADS_FILENAME

PACK_FORMAT = "epoch_heads_delta_planes"
PACK_FORMAT_VERSION = 1
DEFAULT_CODEC = "zlib"
DEFAULT_KEYFRAME_INTERVAL = 50
_MANIFEST_KEY = "manifest"

_CODECS = {
    "zlib": (lambda data: zlib.compress(data, 6), zlib.decompress),
    "bz2": (lambda data: bz2.compress(data, 9), bz2.decompress),
    "lzma": (lambda data: lzma.compress(data, preset=6), lzma.decompress),
}
# Bits are differenced as unsigned integers of the element's own width, so every
# dtype (bfloat16 included, which numpy cannot name) round-trips unchanged.
_BITS_DTYPES = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def _chunk_key(epoch_index, tensor_index):
    return f"e{epoch_index:05d}_t{tensor_index:03d}"


def _tensor_bits(tensor):
    """The tensor's bits as unsigned integers of its own width."""

    tensor = tensor.detach().cpu().contiguous()
    signed = tensor.view(_BITS_DTYPES[tensor.element_size()]).numpy().reshape(-1)
    return signed.view(np.dtype(f"u{signed.itemsize}"))


def _delta(bits, previous):
    """Zigzag-coded wrapping difference; unsigned arithmetic keeps it exact."""

    difference = bits - previous
    sign = difference.view(np.dtype(f"i{bits.itemsize}")) >> (bits.itemsize * 8 - 1)
    return (difference << 1) ^ sign.view(difference.dtype)


def _undelta(coded, previous):
    signed = np.dtype(f"i{coded.itemsize}")
    difference = (coded >> 1).view(signed) ^ -((coded & 1).view(signed))
    return previous + difference.view(coded.dtype)


def _to_planes(bits):
    return bits.view(np.uint8).reshape(-1, bits.itemsize).T.tobytes()


def _from_planes(data, bits_dtype, count):
    planes = np.frombuffer(data, dtype=np.uint8).reshape(bits_dtype.itemsize, count)
    return np.ascontiguousarray(planes.T).view(bits_dtype).reshape(-1)


def _epoch_head_files(directory):
    """The directory's heads in epoch order: the baseline (epoch_m001) first."""

    return sorted(
        Path(directory).glob("epoch_*.pt"),
        key=lambda path: (not path.stem.startswith("epoch_m"), path.stem),
    )


def _load(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise ValueError(f"{path} is not a saved projection head")
    return payload


def _write_npz(path, arrays):
    """``np.savez`` into a temporary file renamed into place."""

    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def pack_epoch_heads(
    run_dir,
    codec=DEFAULT_CODEC,
    keyframe_interval=DEFAULT_KEYFRAME_INTERVAL,
    delete=True,
):
    """Pack ``run_dir/epoch_heads/*.pt`` into ``run_dir/epoch_heads.npz``.

    Returns ``(raw_bytes, packed_bytes)``, or None when there is nothing to pack.
    The originals are deleted only once every epoch has been read back
    bit-identical, and only with ``delete``.
    """

    run_dir = Path(run_dir)
    files = _epoch_head_files(run_dir / EPOCH_HEADS_DIRNAME)
    if not files:
        return None
    if codec not in _CODECS:
        raise ValueError(f"unknown codec {codec!r}; known: {sorted(_CODECS)}")
    compress, _ = _CODECS[codec]
    keyframe_interval = max(1, int(keyframe_interval))

    tensors = None
    epochs = []
    metadata = []
    state_metadata = []
    format_versions = []
    arrays = {}
    previous_bits = None
    for epoch_index, path in enumerate(files):
        payload = _load(path)
        state = payload["state_dict"]
        layout = []
        for name, value in state.items():
            if not torch.is_tensor(value):
                raise ValueError(f"{path}: state entry {name!r} is not a tensor")
            layout.append(
                {"name": name, "dtype": str(value.dtype).removeprefix("torch."), "shape": list(value.shape)}
            )
        if tensors is None:
            tensors = layout
        elif layout != tensors:
            raise ValueError(f"{path}: its tensors differ from {files[0].name}'s")
        epochs.append(int(payload["metadata"]["selected_epoch"]))
        metadata.append(payload["metadata"])
        state_metadata.append(getattr(state, "_metadata", None))
        format_versions.append(payload.get("format_version"))
        keyframe = epoch_index % keyframe_interval == 0
        current_bits = [_tensor_bits(value) for value in state.values()]
        for tensor_index, bits in enumerate(current_bits):
            stored = bits if keyframe else _delta(bits, previous_bits[tensor_index])
            arrays[_chunk_key(epoch_index, tensor_index)] = np.frombuffer(
                compress(_to_planes(stored)), dtype=np.uint8
            )
        previous_bits = current_bits

    manifest = {
        "format": PACK_FORMAT,
        "version": PACK_FORMAT_VERSION,
        "codec": codec,
        "keyframe_interval": keyframe_interval,
        "epochs": epochs,
        "files": [path.name for path in files],
        "tensors": tensors,
        "metadata": metadata,
        "state_metadata": state_metadata,
        "format_versions": format_versions,
    }
    arrays[_MANIFEST_KEY] = np.frombuffer(json.dumps(manifest).encode("utf-8"), dtype=np.uint8)
    packed_path = run_dir / PACKED_EPOCH_HEADS_FILENAME
    _write_npz(packed_path, arrays)

    for path, (_, unpacked) in zip(files, iter_packed_epoch_heads(packed_path), strict=True):
        if not _same_payload(_load(path), unpacked):
            packed_path.unlink()
            raise ValueError(f"{path} did not survive packing bit for bit; kept the originals")

    raw_bytes = sum(path.stat().st_size for path in files)
    packed_bytes = packed_path.stat().st_size
    if delete:
        for path in files:
            path.unlink()
        try:
            (run_dir / EPOCH_HEADS_DIRNAME).rmdir()
        except OSError:
            pass  # Something else lives there; it is not ours to remove.
    return raw_bytes, packed_bytes


def _same_payload(original, unpacked):
    if original.get("format_version") != unpacked.get("format_version"):
        return False
    if original["metadata"] != unpacked["metadata"]:
        return False
    original_state, unpacked_state = original["state_dict"], unpacked["state_dict"]
    if getattr(original_state, "_metadata", None) != getattr(unpacked_state, "_metadata", None):
        return False
    if list(original_state) != list(unpacked_state):
        return False
    return all(
        original_state[name].dtype == unpacked_state[name].dtype
        and original_state[name].shape == unpacked_state[name].shape
        and np.array_equal(_tensor_bits(original_state[name]), _tensor_bits(unpacked_state[name]))
        for name in original_state
    )


def _read_manifest(archive):
    manifest = json.loads(archive[_MANIFEST_KEY].tobytes().decode("utf-8"))
    if manifest.get("format") != PACK_FORMAT or manifest.get("version") != PACK_FORMAT_VERSION:
        raise ValueError(
            f"packed epoch heads in format {manifest.get('format')} v{manifest.get('version')}, "
            "which this revision cannot read"
        )
    return manifest


def _payload(manifest, epoch_index, bits):
    state = OrderedDict()
    for spec, tensor_bits in zip(manifest["tensors"], bits):
        dtype = getattr(torch, spec["dtype"])
        # Copied out of the decode buffer so a caller may modify the tensor.
        signed = tensor_bits.view(np.dtype(f"i{tensor_bits.itemsize}")).copy()
        state[spec["name"]] = torch.from_numpy(signed).view(dtype).reshape(spec["shape"])
    if manifest["state_metadata"][epoch_index] is not None:
        state._metadata = manifest["state_metadata"][epoch_index]
    return {
        "format_version": manifest["format_versions"][epoch_index],
        "state_dict": state,
        "metadata": manifest["metadata"][epoch_index],
    }


def _decode(archive, manifest, epoch_index, previous_bits):
    _, decompress = _CODECS[manifest["codec"]]
    keyframe = epoch_index % manifest["keyframe_interval"] == 0
    bits = []
    for tensor_index, spec in enumerate(manifest["tensors"]):
        stored = _from_planes(
            decompress(archive[_chunk_key(epoch_index, tensor_index)].tobytes()),
            np.dtype(f"u{getattr(torch, spec['dtype']).itemsize}"),
            int(np.prod(spec["shape"], dtype=np.int64)),
        )
        bits.append(stored if keyframe else _undelta(stored, previous_bits[tensor_index]))
    return bits


def iter_packed_epoch_heads(path):
    """Yield ``(epoch, payload)`` in epoch order, each payload as ``load_head_state`` returns it."""

    with np.load(path, allow_pickle=False) as archive:
        manifest = _read_manifest(archive)
        bits = None
        for epoch_index, epoch in enumerate(manifest["epochs"]):
            bits = _decode(archive, manifest, epoch_index, bits)
            yield epoch, _payload(manifest, epoch_index, bits)


def load_packed_epoch_head(path, epoch):
    """One epoch's head, replaying at most ``keyframe_interval`` deltas."""

    with np.load(path, allow_pickle=False) as archive:
        manifest = _read_manifest(archive)
        if epoch not in manifest["epochs"]:
            raise KeyError(f"{path} holds no head for epoch {epoch}; it has {manifest['epochs']}")
        target = manifest["epochs"].index(epoch)
        start = target - target % manifest["keyframe_interval"]
        bits = None
        for epoch_index in range(start, target + 1):
            bits = _decode(archive, manifest, epoch_index, bits)
        return _payload(manifest, target, bits)


def packed_epochs(path):
    """The epochs a packed file holds, without decoding any head."""

    with np.load(path, allow_pickle=False) as archive:
        return list(_read_manifest(archive)["epochs"])


def unpack_epoch_heads(path, output_dir=None):
    """Write a packed file back out as one ``.pt`` per epoch; returns the directory."""

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        names = _read_manifest(archive)["files"]
    output_dir = Path(output_dir) if output_dir is not None else path.parent / EPOCH_HEADS_DIRNAME
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, (_, payload) in zip(names, iter_packed_epoch_heads(path), strict=True):
        torch.save(payload, output_dir / name)
    return output_dir


def pack_run_epoch_heads(run_dir):
    """Pack a finished run's epoch heads, logging instead of raising.

    Called from the end of every fold that saved epoch heads: a failure here
    must neither mask the fold's own exception nor lose a head, so whatever
    cannot be packed exactly is left on disk unpacked.
    """

    try:
        sizes = pack_epoch_heads(run_dir)
    except Exception as exc:  # pragma: no cover - packing is not the run
        logger.warning(f"Epoch heads in {run_dir} were left unpacked: {exc}")
        return None
    if sizes is not None:
        raw_bytes, packed_bytes = sizes
        logger.info(
            f"Packed the epoch heads into {Path(run_dir) / PACKED_EPOCH_HEADS_FILENAME}: "
            f"{raw_bytes / 1e6:.1f} MB -> {packed_bytes / 1e6:.1f} MB "
            f"({raw_bytes / max(packed_bytes, 1):.2f}x)"
        )
    return sizes
