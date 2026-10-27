"""Benchmark ``--frozen_feature_residency`` mmap vs ram vs gpu on CUDA.

The three modes differ in where a training step reads its features from, and in
what that costs once rather than per step:

``mmap``
    The view keeps ``feature_indices`` into the shared on-disk matrix and every
    ``__getitem__`` indexes a ``torch.from_numpy(np.load(..., mmap_mode="r+"))``
    tensor. Rows the page cache does not hold cost a page fault. Workers are
    allowed, because each one reopens the file instead of copying it.
``ram``
    ``_make_resident`` copies the unique rows this view uses into a private
    tensor once, then drops the path. No step touches the file, but
    ``dataloader_num_workers_for_dataset`` now returns 0, so collation runs in
    the training process.
``gpu``
    ``ram`` plus ``GpuResidentFeatureLoader``: the whole view is uploaded to the
    device and a step gathers its batch with ``index_select``, with no
    DataLoader and no host-to-device copy.

Because the difference between the first two is entirely about whether the
feature matrix is in the page cache, each mode is measured twice: with the file
warm, and with its pages dropped via ``posix_fadvise(POSIX_FADV_DONTNEED)``
right before the run. The cold number is the regime ``ram`` exists for; the warm
number is what an ordinary rerun on an idle box actually sees.

Everything below the loader is the frozen-backbone step the real runs execute:
one 768->768 projection + L2 norm, ``MPerClassSampler``, bf16 autocast.

Usage::

    python scripts/bench_feature_residency.py --profile cars
    python scripts/bench_feature_residency.py --profile inat --steps 400
"""

import argparse
import ctypes
import gc
import json
import mmap
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Must precede every torch/numpy import: OpenMP and the BLAS backends size their
# pools when their shared libraries load. A real run resolves this budget in
# main.py before the same imports, and it defaults to one thread per process --
# measuring with torch's unclamped 32-thread default would price four concurrent
# trials against contention no real run has.
from training.cpu_threads import (  # noqa: E402
    DEFAULT_THREAD_BUDGET,
    apply_blas_env,
    limit_native_thread_pools,
    resolve_thread_budget,
)

_THREAD_BUDGET = resolve_thread_budget(
    next(
        (
            token.partition("=")[2] or sys.argv[position + 2]
            for position, token in enumerate(sys.argv[1:])
            if token.partition("=")[0] == "--num_threads"
        ),
        DEFAULT_THREAD_BUDGET,
    )
)
apply_blas_env(_THREAD_BUDGET)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch._dynamo  # noqa: E402

torch.set_num_threads(_THREAD_BUDGET)
torch.set_num_interop_threads(1)
limit_native_thread_pools(_THREAD_BUDGET)

import pytorch_metric_learning.losses as losses  # noqa: E402

import utils  # noqa: E402

MODES = (utils.FEATURE_RESIDENCY_MMAP, utils.FEATURE_RESIDENCY_RAM, utils.FEATURE_RESIDENCY_GPU)
# "asis" leaves the page cache alone: concurrent processes sharing one matrix
# would otherwise evict the file out from under each other.
CACHE_STATES = ("warm", "cold", "asis")

# rows in the shared matrix, rows the training view uses, classes
PROFILES = {
    "cars": {"matrix_rows": 8131, "view_rows": 6070, "num_classes": 98},
    "in_shop": {"matrix_rows": 25882, "view_rows": 20897, "num_classes": 3997},
    "cifar": {"matrix_rows": 60000, "view_rows": 45000, "num_classes": 100},
    "inat": {"matrix_rows": 437513, "view_rows": 325000, "num_classes": 8142},
}


class Head(torch.nn.Module):
    """The trainable part of a frozen-backbone run: one projection + L2 norm."""

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fc = torch.nn.Linear(in_dim, out_dim)

    def project_features(self, features):
        return torch.nn.functional.normalize(self.fc(features), p=2.0, dim=1)

    def forward(self, features):
        return self.project_features(features)


def build_matrix_file(path, matrix_rows, feat_dim, num_classes, seed):
    """Write a class-structured feature matrix standing in for a backbone cache."""

    if path.exists():
        existing = np.load(path, mmap_mode="r")
        if existing.shape == (matrix_rows, feat_dim) and existing.dtype == np.float32:
            del existing
            return path
        del existing

    path.parent.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(seed)
    matrix = np.lib.format.open_memmap(
        path,
        mode="w+",
        dtype=np.float32,
        shape=(matrix_rows, feat_dim),
    )
    centers = generator.standard_normal((num_classes, feat_dim), dtype=np.float32) * 4.0
    # Chunked so a large profile never materializes the whole matrix in RAM.
    chunk = 8192
    for start in range(0, matrix_rows, chunk):
        stop = min(start + chunk, matrix_rows)
        rows = np.arange(start, stop)
        matrix[start:stop] = centers[rows % num_classes] + generator.standard_normal(
            (stop - start, feat_dim),
            dtype=np.float32,
        )
    matrix.flush()
    del matrix
    return path


class _PyBuffer(ctypes.Structure):
    _fields_ = [
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    ]


def resident_fraction(path):
    """Fraction of the file currently held in the page cache, via ``mincore``.

    ``POSIX_FADV_DONTNEED`` is advisory and silently does nothing for pages a
    live mapping still holds, so a cold measurement is only trustworthy if it
    reports the residency it actually started from.
    """

    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    pagesize = os.sysconf("SC_PAGE_SIZE")
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mapping = mmap.mmap(fd, 0, flags=mmap.MAP_SHARED, prot=mmap.PROT_READ)
        view = _PyBuffer()
        ctypes.pythonapi.PyObject_GetBuffer(ctypes.py_object(mapping), ctypes.byref(view), 0)
        try:
            pages = (size + pagesize - 1) // pagesize
            vector = (ctypes.c_ubyte * pages)()
            if libc.mincore(ctypes.c_void_p(view.buf), ctypes.c_size_t(size), vector) != 0:
                return None
            return sum(byte & 1 for byte in vector) / pages
        finally:
            ctypes.pythonapi.PyBuffer_Release(ctypes.byref(view))
            mapping.close()
    finally:
        os.close(fd)


def evict_page_cache(path):
    """Drop a file's clean pages, so the next read faults from disk."""

    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def warm_page_cache(path):
    """Read the file end to end so every page is resident before timing."""

    started = time.perf_counter()
    total = 0
    with open(path, "rb") as handle:
        while True:
            block = handle.read(64 << 20)
            if not block:
                break
            total += len(block)
    elapsed = time.perf_counter() - started
    return total / max(elapsed, 1e-9) / 1e9


def rss_bytes():
    """Current resident set size, to price what each mode holds in host memory."""

    with open("/proc/self/statm", "r") as handle:
        return int(handle.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def make_view(matrix_path, profile, feat_dim, num_views, seed):
    """Pick the matrix rows this training view uses, mirroring an index spec."""

    generator = np.random.default_rng(seed + 1)
    matrix_rows = profile["matrix_rows"]
    view_rows = profile["view_rows"]
    # A real view owns a contiguous-in-label but scattered-in-file subset of the
    # shared matrix: the train split of a cache that also holds valid and test.
    chosen = np.sort(generator.choice(matrix_rows, size=view_rows, replace=False))
    labels = (chosen % profile["num_classes"]).astype(np.int64)
    if num_views > 1:
        # num_views > 1 stores one row per (sample, view); reuse nearby rows so
        # the fan-out matches what precompute would have written.
        offsets = generator.integers(0, matrix_rows, size=(view_rows, num_views))
        feature_indices = offsets.astype(np.int64)
        feature_indices[:, 0] = chosen
    else:
        # precompute leaves the index vector 1D for the single-view case, and
        # both the __getitem__ branch and the GPU gather depend on that rank.
        feature_indices = chosen.astype(np.int64)
    return torch.from_numpy(feature_indices), labels.tolist()


def build_dataset(matrix_path, feature_indices, labels, residency, residency_max_bytes):
    """Construct the dataset exactly as ``precompute_backbone_feature_dataset`` does."""

    feature_memmap = np.load(matrix_path, mmap_mode="r+")
    feature_matrix = torch.from_numpy(feature_memmap)
    return utils.PrecomputedBackboneFeatureDataset(
        features=feature_matrix,
        feature_indices=feature_indices,
        labels=labels,
        source_dataset=None,
        feature_matrix_path=str(matrix_path),
        residency=residency,
        residency_max_bytes=residency_max_bytes,
        residency_desc="bench train",
    )


def make_criterion(loss_name, num_classes, feat_dim, learning_rate, device):
    if loss_name == "MultiSimilarityLoss":
        return losses.MultiSimilarityLoss(alpha=2.0, beta=50.0, base=0.5), None
    if loss_name == "ArcFaceLoss":
        criterion = losses.ArcFaceLoss(num_classes=num_classes, embedding_size=feat_dim).to(device)
        return criterion, torch.optim.Adam(criterion.parameters(), lr=learning_rate)
    if loss_name == "ProxyAnchorLoss":
        criterion = losses.ProxyAnchorLoss(num_classes=num_classes, embedding_size=feat_dim).to(device)
        return criterion, torch.optim.Adam(criterion.parameters(), lr=learning_rate)
    raise ValueError(f"unsupported loss for this benchmark: {loss_name}")


def unpack_batch(batch, device):
    """Normalize DataLoader and GPU-resident batches to ``(features, labels)``."""

    features, labels = batch[0], batch[1]
    if features.device != device:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
    return features, labels


def make_forward_loss(model, criterion, compiled):
    """Mirror ``_fused_forward_loss``: one compiled projection+loss callable."""

    def forward_loss(features, labels):
        return criterion(model.project_features(features), labels)

    return torch.compile(forward_loss) if compiled else forward_loss


def eager_step(
    model,
    criterion,
    optimizer,
    criterion_optimizer,
    features,
    labels,
    device_type,
    phase_sync=None,
    forward_loss=None,
):
    """One frozen-backbone step, optionally instrumented like ``--debug_batch_timing``.

    ``phase_sync`` stands in for ``_BatchTimer``: it synchronizes at the phase
    boundaries the real timer measures across. Those syncs are the measurement,
    not an artifact of it -- the timer cannot attribute GPU time without them --
    but they also stop the CPU from running ahead of the queue, so a run reading
    its own timings is not running at the speed it reports.
    """

    def mark():
        if phase_sync is not None:
            # start() and the preceding stop() each synchronize.
            torch.cuda.synchronize(phase_sync)
            torch.cuda.synchronize(phase_sync)

    optimizer.zero_grad(set_to_none=False)
    if criterion_optimizer is not None:
        criterion_optimizer.zero_grad(set_to_none=False)
    with torch.autocast(device_type=device_type, dtype=torch.bfloat16, cache_enabled=False):
        mark()  # unpack_batch -> label_weight_prep
        if forward_loss is not None:
            # The fused path has no embeddings boundary to mark, exactly as the
            # real compiled step has no separate forward/miner_and_loss phases.
            loss = forward_loss(features, labels)
        else:
            embeddings = model.project_features(features)
            mark()  # forward -> miner_and_loss
            loss = criterion(embeddings, labels)
        mark()  # miner_and_loss -> backward
    loss.backward()
    mark()  # backward -> optimizer_step
    optimizer.step()
    if criterion_optimizer is not None:
        criterion_optimizer.step()
    mark()  # optimizer_step -> loss_item
    return loss


def run_case(mode, cache_state, args, matrix_path, profile, device):
    """Time one (residency, page-cache) combination end to end."""

    torch.manual_seed(args.seed)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    gc.collect()

    read_gbps = None
    if cache_state != "asis":
        evict_page_cache(matrix_path)
        if cache_state == "warm":
            read_gbps = warm_page_cache(matrix_path)
    start_resident = resident_fraction(matrix_path)

    feature_indices, labels = make_view(matrix_path, profile, args.feat_dim, args.num_views, args.seed)

    rss_before = rss_bytes()
    setup_started = time.perf_counter()
    dataset = build_dataset(
        matrix_path,
        feature_indices,
        labels,
        mode,
        None if args.residency_max_gb is None else int(args.residency_max_gb * 1e9),
    )
    loader = utils.make_train_loader(
        dataset,
        args.batch_size,
        args.sampler_m,
        seed=args.seed,
        num_workers=args.num_workers,
        start_method=args.dataloader_start_method,
        persistent_workers=True,
        pin_memory=True,
        gpu_resident_device=device if mode == utils.FEATURE_RESIDENCY_GPU else None,
        gpu_resident_max_bytes=(
            None if args.residency_max_gb is None else int(args.residency_max_gb * 1e9)
        ),
    )
    torch.cuda.synchronize(device)
    setup_s = time.perf_counter() - setup_started
    rss_after_setup = rss_bytes()

    from training import gpu_feature_loader

    is_gpu_resident = isinstance(loader, gpu_feature_loader.GpuResidentFeatureLoader)
    effective_workers = 0 if is_gpu_resident else loader.num_workers
    # Recorded before teardown: a view over the residency budget silently stays
    # memory-mapped, and that fallback has to be visible in the result.
    host_resident = bool(dataset.is_resident)

    if args.start_at is not None:
        # Concurrent processes finish setup at different times -- ram and gpu
        # pay a copy that mmap does not -- and a measured window shorter than
        # that skew would compare a contended run against a solo one.
        remaining = args.start_at - time.time()
        if remaining < 0:
            raise SystemExit(f"--start_at already passed by {-remaining:.1f}s; raise the lead time")
        time.sleep(remaining)

    model = Head(args.feat_dim, args.feat_dim).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion, criterion_optimizer = make_criterion(
        args.loss,
        profile["num_classes"],
        args.feat_dim,
        args.lr,
        device,
    )
    device_type = device.type
    if args.compile:
        # Each case compiles its own callable; inheriting another's kernels
        # would hide the compilation this mode has to pay.
        torch._dynamo.reset()
    forward_loss = make_forward_loss(model, criterion, args.compile)

    # Fetch and step are timed separately: the whole mmap-vs-ram question lives
    # in the fetch, which a step-only timer would not see at all.
    fetch_times = []
    step_times = []
    ramp_times = []
    steps_done = 0
    total_steps = args.warmup_steps + args.steps
    torch.cuda.synchronize(device)
    measured_start = None
    run_started = time.perf_counter()
    loop_mark = run_started

    while steps_done < total_steps:
        for batch in loader:
            fetch_elapsed = time.perf_counter() - loop_mark
            batch_features, batch_labels = unpack_batch(batch, device)
            if steps_done == args.warmup_steps:
                torch.cuda.synchronize(device)
                measured_start = time.perf_counter()
            started = time.perf_counter()
            eager_step(
                model,
                criterion,
                optimizer,
                criterion_optimizer,
                batch_features,
                batch_labels,
                device_type,
                phase_sync=device if args.debug_batch_timing else None,
                forward_loss=forward_loss if args.compile else None,
            )
            if args.sync_each_step:
                torch.cuda.synchronize(device)
            step_elapsed = time.perf_counter() - started
            if steps_done < args.ramp_steps:
                ramp_times.append(fetch_elapsed + step_elapsed)
            if steps_done >= args.warmup_steps:
                fetch_times.append(fetch_elapsed)
                step_times.append(step_elapsed)
            steps_done += 1
            loop_mark = time.perf_counter()
            if steps_done >= total_steps:
                break

    torch.cuda.synchronize(device)
    wall_total = time.perf_counter() - measured_start
    peak_bytes = torch.cuda.max_memory_allocated(device)
    rss_peak = rss_bytes()
    total_times = [fetch + step for fetch, step in zip(fetch_times, step_times)]

    utils.shutdown_dataloader_workers(loader)
    del loader, dataset, model, criterion
    gc.collect()
    torch.cuda.empty_cache()
    # After teardown, so no live mapping of this run blocks the next eviction.
    end_resident = resident_fraction(matrix_path)

    return {
        "mode": mode,
        "cache_state": cache_state,
        "steps": len(step_times),
        "steps_per_s": len(step_times) / wall_total,
        "median_total_ms": statistics.median(total_times) * 1e3,
        "p90_total_ms": sorted(total_times)[int(0.9 * len(total_times)) - 1] * 1e3,
        "median_fetch_ms": statistics.median(fetch_times) * 1e3,
        "p90_fetch_ms": sorted(fetch_times)[int(0.9 * len(fetch_times)) - 1] * 1e3,
        "median_step_ms": statistics.median(step_times) * 1e3,
        "p90_step_ms": sorted(step_times)[int(0.9 * len(step_times)) - 1] * 1e3,
        # The cold penalty is a one-time ramp, not a per-step tax: readahead
        # pulls the file in over the first handful of batches. Summing the ramp
        # is the only way to see a cost that a warmed-up median has erased.
        "ramp_steps": len(ramp_times),
        "ramp_total_s": sum(ramp_times),
        "first_batch_ms": ramp_times[0] * 1e3 if ramp_times else None,
        "setup_s": setup_s,
        # What a trial pays before its first useful step, however it is spent.
        "time_to_ready_s": setup_s + sum(ramp_times),
        "host_resident": host_resident,
        "gpu_resident_loader": is_gpu_resident,
        "dataloader_workers": effective_workers,
        "rss_setup_mb": (rss_after_setup - rss_before) / 1e6,
        "rss_peak_mb": (rss_peak - rss_before) / 1e6,
        "peak_device_mb": peak_bytes / 1e6,
        "warm_read_gbps": read_gbps,
        "start_resident_fraction": start_resident,
        "end_resident_fraction": end_resident,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--profile", default="cars", choices=sorted(PROFILES))
    parser.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    parser.add_argument("--cache_states", nargs="+", default=["warm", "cold"], choices=CACHE_STATES)
    parser.add_argument("--loss", default="MultiSimilarityLoss")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--sampler_m", type=int, default=4)
    parser.add_argument("--feat_dim", type=int, default=768)
    parser.add_argument("--num_views", type=int, default=1)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument(
        "--ramp_steps",
        type=int,
        default=50,
        help="leading batches summed into the cold-start ramp; these are warmup steps too",
    )
    parser.add_argument("--repeats", type=int, default=1, help="interleaved passes over the mode grid")
    parser.add_argument(
        "--num_threads",
        default=DEFAULT_THREAD_BUDGET,
        help="per-process CPU thread budget; read before torch is imported",
    )
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--dataloader_start_method", default="spawn")
    parser.add_argument("--residency_max_gb", type=float, default=4.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sync_each_step", action="store_true")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the fused projection+loss, as _fused_forward_loss would",
    )
    parser.add_argument(
        "--debug_batch_timing",
        action="store_true",
        help="reproduce the per-phase CUDA syncs --debug_batch_timing adds to a real run",
    )
    parser.add_argument(
        "--matrix_dir",
        type=Path,
        default=Path("/tmp/claude-1000/bench_feature_residency"),
        help="where the synthetic backbone-cache matrix is written",
    )
    parser.add_argument(
        "--start_at",
        type=float,
        default=None,
        help="unix timestamp to begin the timed loop at; a barrier for concurrent processes",
    )
    parser.add_argument("--json_out", type=Path, default=None)
    parser.add_argument("--label", default="")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("this benchmark measures the CUDA training step; use --device cuda")

    profile = PROFILES[args.profile]
    matrix_path = args.matrix_dir / f"{args.profile}_{profile['matrix_rows']}x{args.feat_dim}.features.npy"
    build_matrix_file(
        matrix_path,
        profile["matrix_rows"],
        args.feat_dim,
        profile["num_classes"],
        args.seed,
    )
    print(
        f"profile={args.profile} matrix={profile['matrix_rows']}x{args.feat_dim} "
        f"({matrix_path.stat().st_size / 1e9:.3f} GB) view_rows={profile['view_rows']} "
        f"classes={profile['num_classes']} workers={args.num_workers} "
        f"threads={_THREAD_BUDGET}",
        flush=True,
    )

    results = []
    # Modes are interleaved across repeats rather than run back to back, so a
    # thermal or background-load drift spreads over all of them instead of
    # landing on whichever mode happened to run last.
    for repeat in range(args.repeats):
        for cache_state in args.cache_states:
            for mode in args.modes:
                result = run_case(mode, cache_state, args, matrix_path, profile, device)
                result["pid"] = os.getpid()
                result["label"] = args.label
                result["repeat"] = repeat
                results.append(result)
                print(
                    f"[{repeat}] {mode:>5}/{cache_state:<4}: {result['steps_per_s']:8.1f} steps/s  "
                    f"total {result['median_total_ms']:6.3f} ms "
                    f"(fetch {result['median_fetch_ms']:6.3f} + step {result['median_step_ms']:6.3f})  "
                    f"p90 {result['p90_total_ms']:6.3f}  "
                    f"batch0 {result['first_batch_ms']:8.1f} ms  "
                    f"ramp {result['ramp_total_s']:5.2f} s  "
                    f"setup {result['setup_s']:5.2f} s  "
                    f"ready {result['time_to_ready_s']:5.2f} s  "
                    f"rss {result['rss_setup_mb']:7.1f} MB  "
                    f"dev {result['peak_device_mb']:7.1f} MB  "
                    f"w={result['dataloader_workers']}",
                    flush=True,
                )

    summary = []
    print("\nmedian over repeats:", flush=True)
    for cache_state in args.cache_states:
        baseline_rate = None
        for mode in args.modes:
            group = [
                item
                for item in results
                if item["cache_state"] == cache_state and item["mode"] == mode
            ]
            if not group:
                continue
            rate = statistics.median(item["steps_per_s"] for item in group)
            if mode == utils.FEATURE_RESIDENCY_MMAP:
                baseline_rate = rate
            entry = {
                "mode": mode,
                "cache_state": cache_state,
                "repeats": len(group),
                "steps_per_s": rate,
                "steps_per_s_spread": (
                    max(item["steps_per_s"] for item in group)
                    - min(item["steps_per_s"] for item in group)
                ),
                "median_total_ms": statistics.median(item["median_total_ms"] for item in group),
                "median_fetch_ms": statistics.median(item["median_fetch_ms"] for item in group),
                "median_step_ms": statistics.median(item["median_step_ms"] for item in group),
                "time_to_ready_s": statistics.median(item["time_to_ready_s"] for item in group),
                "rss_setup_mb": statistics.median(item["rss_setup_mb"] for item in group),
                "peak_device_mb": statistics.median(item["peak_device_mb"] for item in group),
                "speedup_vs_mmap": None if baseline_rate is None else rate / baseline_rate,
            }
            summary.append(entry)
            speedup = "  n/a" if entry["speedup_vs_mmap"] is None else f"{entry['speedup_vs_mmap']:.2f}x"
            print(
                f"{mode:>5}/{cache_state:<4}: {rate:8.1f} steps/s  "
                f"(spread {entry['steps_per_s_spread']:6.1f})  "
                f"total {entry['median_total_ms']:6.3f} ms  "
                f"fetch {entry['median_fetch_ms']:6.3f} ms  "
                f"ready {entry['time_to_ready_s']:5.2f} s  "
                f"rss {entry['rss_setup_mb']:7.1f} MB  "
                f"dev {entry['peak_device_mb']:7.1f} MB  "
                f"{speedup} vs mmap",
                flush=True,
            )

    payload = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "profile": profile,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "summary": summary,
        "results": results,
    }
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
    return payload


if __name__ == "__main__":
    main()
