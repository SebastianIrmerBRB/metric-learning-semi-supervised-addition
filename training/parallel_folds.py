"""Train several cross-validation folds of one run at the same time.

The folds of a cross-validation run are independent pieces of work (the opt-in
SLADE cross-fold teacher handoff is the one exception, and never reaches this
module), so ``--cv_parallel_folds N`` trains up to ``N`` of them at once and
returns once every fold has finished. Two backends exist because they trade
different things:

``process``
    Every fold runs in its own spawned Python process, exactly as a sequential
    fold would run in the parent: its own RNG streams, logger, CUDA context and
    copy of the frozen features. Results are reproducible fold by fold, and the
    folds do not share the GIL. The cost is memory: each process holds a CUDA
    context, a DINO backbone and its evaluation buffers, and roughly one
    training process's worth of host RAM. The GPU memory cannot be shared
    instead: CUDA IPC is unavailable under WSL2 (``invalid resource handle``).

``thread``
    Every fold runs on a thread of the calling process, so the folds share one
    CUDA context and the in-process frozen-feature cache. The global Python,
    NumPy and Torch RNGs are shared too, so which fold draws which random
    number depends on scheduling and the fold results are not reproducible.
    Python-side work is serialized by the GIL.
"""

from __future__ import annotations

import contextvars
import multiprocessing
import pickle
import threading
import time
import traceback
from dataclasses import dataclass
from multiprocessing.connection import wait as wait_for_connections
from typing import Any

from loguru import logger

CV_PARALLEL_BACKEND_PROCESS = "process"
CV_PARALLEL_BACKEND_THREAD = "thread"
CV_PARALLEL_BACKENDS = (CV_PARALLEL_BACKEND_PROCESS, CV_PARALLEL_BACKEND_THREAD)
DEFAULT_CV_PARALLEL_BACKEND = CV_PARALLEL_BACKEND_PROCESS

# A terminated fold process gets this long to exit before it is killed.
_PROCESS_TERMINATE_TIMEOUT_SECONDS = 30.0


@dataclass(frozen=True)
class FoldJob:
    """One fold to train: its index and its private argument namespace."""

    fold_index: int
    args: Any


class FoldTrainingError(RuntimeError):
    """A fold failed in a way that could not be re-raised as its own type."""


def get_cv_parallel_folds(args):
    value = getattr(args, "cv_parallel_folds", 1)
    return 1 if value is None else int(value)


def get_cv_parallel_backend(args):
    backend = str(
        getattr(args, "cv_parallel_backend", None) or DEFAULT_CV_PARALLEL_BACKEND
    ).lower()
    if backend not in CV_PARALLEL_BACKENDS:
        raise ValueError(
            f"cv_parallel_backend must be one of {CV_PARALLEL_BACKENDS}: {backend!r}"
        )
    return backend


def validate_cv_parallel_args(args):
    folds = get_cv_parallel_folds(args)
    if folds < 1:
        raise ValueError(f"cv_parallel_folds must be at least 1, got {folds}")
    get_cv_parallel_backend(args)


def run_fold_jobs(
    jobs,
    *,
    train_fold,
    on_fold_complete,
    max_parallel,
    backend,
    shared_kwargs=None,
    thread_kwargs=None,
):
    """Train ``jobs`` with at most ``max_parallel`` running at once.

    ``train_fold(job.args, cv_fold=job.fold_index, **kwargs)`` trains one fold;
    for the process backend it must be a picklable module-level function.
    ``shared_kwargs`` reach every fold on either backend and ``thread_kwargs``
    only the thread backend: in-process objects such as the frozen-feature cache
    cannot cross a process boundary.
    ``on_fold_complete(result)`` runs in the calling thread, in completion
    order. Returns only after every started fold has stopped; the first error,
    from a fold or from the callback, is re-raised once that is true.
    """

    jobs = list(jobs)
    if not jobs:
        return
    max_parallel = max(1, min(int(max_parallel), len(jobs)))
    shared_kwargs = dict(shared_kwargs or {})
    if backend == CV_PARALLEL_BACKEND_PROCESS:
        _run_fold_jobs_in_processes(
            jobs,
            train_fold=train_fold,
            on_fold_complete=on_fold_complete,
            max_parallel=max_parallel,
            kwargs=shared_kwargs,
        )
    elif backend == CV_PARALLEL_BACKEND_THREAD:
        _run_fold_jobs_in_threads(
            jobs,
            train_fold=train_fold,
            on_fold_complete=on_fold_complete,
            max_parallel=max_parallel,
            kwargs={**shared_kwargs, **dict(thread_kwargs or {})},
        )
    else:
        raise ValueError(
            f"cv_parallel_backend must be one of {CV_PARALLEL_BACKENDS}: {backend!r}"
        )


# ---------------------------------------------------------------------------
# Process backend
# ---------------------------------------------------------------------------


def capture_process_runtime_policies():
    """Record the process-wide settings a fold process must reproduce.

    ``main`` applies these once per process before any run starts. A spawned
    fold process imports the entry module again and inherits the BLAS thread
    environment, but a caller that did not go through ``main`` (a test, a
    notebook) may have set them differently, so the parent's live values are
    copied rather than re-derived from the arguments.
    """

    import torch

    import utils
    from .ssl import algorithms as ssl_algorithms

    return {
        "torch_num_threads": int(torch.get_num_threads()),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "eval_autocast_enabled": bool(utils._EVAL_AUTOCAST_ENABLED),
        "faiss_low_memory_enabled": bool(utils._FAISS_LOW_MEMORY_ENABLED),
        "faiss_low_memory_temp_bytes": utils._FAISS_LOW_MEMORY_TEMP_BYTES,
        "ssl_faiss_low_memory": ssl_algorithms._SSL_FAISS_LOW_MEMORY.get(),
        "tsne": utils.tsne_backend_settings(),
    }


def apply_process_runtime_policies(policies):
    """Apply :func:`capture_process_runtime_policies` output in a fold process."""

    import torch

    import utils
    from .cpu_threads import limit_native_thread_pools
    from .ssl import algorithms as ssl_algorithms

    threads = int(policies["torch_num_threads"])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # Already fixed by the entry module's own import-time setup.
        pass
    limit_native_thread_pools(threads)
    torch.backends.cudnn.allow_tf32 = policies["cudnn_allow_tf32"]
    torch.backends.cuda.matmul.allow_tf32 = policies["cuda_matmul_allow_tf32"]
    torch.set_float32_matmul_precision(policies["float32_matmul_precision"])
    utils.set_eval_autocast_enabled(policies["eval_autocast_enabled"])
    utils.set_faiss_low_memory(
        policies["faiss_low_memory_enabled"],
        temp_memory_bytes=policies["faiss_low_memory_temp_bytes"],
    )
    ssl_algorithms.set_ssl_faiss_low_memory(policies["ssl_faiss_low_memory"])
    utils.set_tsne_backend(
        policies["tsne"]["backend"],
        device=policies["tsne"]["device"],
        max_dense_gpu_bytes=policies["tsne"]["max_dense_gpu_bytes"],
    )


def _fold_process_main(connection, train_fold, job, kwargs, policies):
    """Entry point of one spawned fold process."""

    try:
        apply_process_runtime_policies(policies)
        result = train_fold(job.args, cv_fold=job.fold_index, **kwargs)
        message = ("ok", job.fold_index, result)
    except BaseException as exc:  # noqa: BLE001 - reported to the parent
        message = ("error", job.fold_index, _describe_exception(exc))
    try:
        connection.send(message)
    except Exception as exc:  # noqa: BLE001 - e.g. an unpicklable result
        connection.send(("error", job.fold_index, _describe_exception(exc)))
    finally:
        connection.close()


def _describe_exception(exc):
    text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    try:
        payload = pickle.dumps(exc)
        pickle.loads(payload)
    except Exception:  # noqa: BLE001 - not every exception survives pickling
        payload = None
    return {"type": type(exc).__name__, "message": str(exc), "traceback": text, "pickled": payload}


def _raise_fold_error(fold_index, description):
    exception = None
    if description.get("pickled") is not None:
        try:
            exception = pickle.loads(description["pickled"])
        except Exception:  # noqa: BLE001
            exception = None
    logger.error(
        f"Cross-validation fold {fold_index} failed in its process:\n"
        f"{description['traceback']}"
    )
    if isinstance(exception, BaseException):
        raise exception
    raise FoldTrainingError(
        f"Cross-validation fold {fold_index} failed: "
        f"{description['type']}: {description['message']}"
    )


@dataclass
class _RunningProcess:
    job: FoldJob
    process: Any
    connection: Any
    started: float


def _run_fold_jobs_in_processes(jobs, *, train_fold, on_fold_complete, max_parallel, kwargs):
    context = multiprocessing.get_context("spawn")
    policies = capture_process_runtime_policies()
    queue = list(jobs)
    running = []
    try:
        while queue or running:
            while queue and len(running) < max_parallel:
                job = queue.pop(0)
                receiver, sender = context.Pipe(duplex=False)
                process = context.Process(
                    target=_fold_process_main,
                    args=(sender, train_fold, job, kwargs, policies),
                    name=f"cv-fold-{job.fold_index:02d}",
                    # Not a daemon: a fold may start DataLoader workers.
                    daemon=False,
                )
                process.start()
                # The parent keeps only the reading end; the child's copy is the
                # one whose closing signals EOF.
                sender.close()
                running.append(_RunningProcess(job, process, receiver, time.monotonic()))
                logger.info(
                    f"Started cross-validation fold {job.fold_index} in process "
                    f"{process.pid} ({len(running)} running, {len(queue)} queued)"
                )

            ready = set(
                wait_for_connections(
                    [entry.connection for entry in running]
                    + [entry.process.sentinel for entry in running]
                )
            )
            for entry in list(running):
                if entry.connection not in ready and entry.process.sentinel not in ready:
                    continue
                message = None
                try:
                    # A sentinel alone can fire just before the pipe is read.
                    if entry.connection.poll():
                        message = entry.connection.recv()
                except EOFError:
                    message = None
                if message is None and entry.process.is_alive():
                    continue
                running.remove(entry)
                entry.connection.close()
                entry.process.join()
                fold_index = entry.job.fold_index
                if message is None:
                    raise FoldTrainingError(
                        f"Cross-validation fold {fold_index} process exited with code "
                        f"{entry.process.exitcode} without reporting a result "
                        "(killed, or out of memory?)"
                    )
                status, _, payload = message
                if status != "ok":
                    _raise_fold_error(fold_index, payload)
                logger.info(
                    f"Cross-validation fold {fold_index} finished in "
                    f"{time.monotonic() - entry.started:.0f} s "
                    f"({len(running)} still running, {len(queue)} queued)"
                )
                on_fold_complete(payload)
    except BaseException:
        _terminate_processes(running)
        raise


def _terminate_processes(running):
    for entry in running:
        if entry.process.is_alive():
            logger.warning(
                f"Stopping cross-validation fold {entry.job.fold_index} "
                f"(process {entry.process.pid})"
            )
            entry.process.terminate()
    deadline = time.monotonic() + _PROCESS_TERMINATE_TIMEOUT_SECONDS
    for entry in running:
        entry.process.join(max(0.0, deadline - time.monotonic()))
        if entry.process.is_alive():
            entry.process.kill()
            entry.process.join()
        entry.connection.close()


# ---------------------------------------------------------------------------
# Thread backend
# ---------------------------------------------------------------------------


def _run_fold_jobs_in_threads(jobs, *, train_fold, on_fold_complete, max_parallel, kwargs):
    import utils

    queue = list(jobs)
    completed = []
    condition = threading.Condition()
    running = {}
    first_error = None

    def work(job, log_scope):
        try:
            with utils.scoped_run_logging(log_scope):
                result = train_fold(job.args, cv_fold=job.fold_index, **kwargs)
            outcome = ("ok", result)
        except BaseException as exc:  # noqa: BLE001 - re-raised by the caller
            outcome = ("error", exc)
        with condition:
            completed.append((job, outcome))
            condition.notify_all()

    with utils.shared_console_logging():
        try:
            while queue or running:
                while queue and len(running) < max_parallel and first_error is None:
                    job = queue.pop(0)
                    # Each fold sees the caller's context variables (the SSL FAISS
                    # policy among them) in a copy of its own; a new thread would
                    # otherwise start from their defaults.
                    context = contextvars.copy_context()
                    thread = threading.Thread(
                        target=context.run,
                        args=(work, job, f"fold_{job.fold_index:02d}"),
                        name=f"cv-fold-{job.fold_index:02d}",
                        daemon=True,
                    )
                    running[job.fold_index] = (thread, time.monotonic())
                    thread.start()
                    logger.info(
                        f"Started cross-validation fold {job.fold_index} on a thread "
                        f"({len(running)} running, {len(queue)} queued)"
                    )
                if not running:
                    break
                with condition:
                    while not completed:
                        condition.wait()
                    finished = list(completed)
                    completed.clear()
                for job, (status, payload) in finished:
                    thread, started = running.pop(job.fold_index)
                    thread.join()
                    if status != "ok":
                        logger.error(
                            f"Cross-validation fold {job.fold_index} failed:\n"
                            + "".join(
                                traceback.format_exception(
                                    type(payload), payload, payload.__traceback__
                                )
                            )
                        )
                        if first_error is None:
                            first_error = payload
                        continue
                    logger.info(
                        f"Cross-validation fold {job.fold_index} finished in "
                        f"{time.monotonic() - started:.0f} s "
                        f"({len(running)} still running, {len(queue)} queued)"
                    )
                    # Recorded even after a sibling failed: the thread could not be
                    # stopped, so its finished work is kept for a resumed run.
                    try:
                        on_fold_complete(payload)
                    except BaseException as exc:  # noqa: BLE001
                        if first_error is None:
                            first_error = exc
        except BaseException:
            # A thread cannot be stopped from outside. The fold threads are
            # daemons, so an interrupted process still exits.
            if running:
                logger.warning(
                    f"Abandoning {len(running)} running fold thread(s): "
                    f"{sorted(running)}"
                )
            raise
    if first_error is not None:
        raise first_error
