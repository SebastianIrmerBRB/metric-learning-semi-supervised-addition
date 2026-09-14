"""CPU thread-budget resolution shared by the entry point and the scheduler.

This module must stay importable *before* PyTorch: OpenMP and the BLAS backends
read ``OMP_NUM_THREADS`` and friends when their shared libraries load, so
:mod:`main` resolves and applies the budget here before importing any
torch-dependent module. Nothing below may import torch, numpy, or sklearn.

The budget exists because runs are process-shaped. ``run_scheduler`` starts each
experiment as an independent child and, in pure CPU mode, admits one child at a
time precisely so that child can use the whole host. When several CPU children
are admitted concurrently the host cores have to be divided between them, which
is what ``concurrency`` expresses.

The budget is a *process* total, not a per-library one. That distinction is the
whole reason :func:`limit_native_thread_pools` exists: the environment knobs
below are read by every numeric wheel that bundles its own threading runtime,
and this stack bundles five of them (numpy, scipy and faiss each ship an
OpenBLAS; torch and scikit-learn each ship a libgomp). Publishing the budget
through the environment alone therefore spends it once per runtime, so a budget
of N yields roughly 5N threads rather than N.
"""

from __future__ import annotations

import json
import os
import sys

#: Pre-import channel for the budget. The scheduler sets this per child; a user
#: can export it to make numpy/sklearn BLAS follow the same budget as torch.
THREAD_BUDGET_ENV = "METRIC_LEARNING_NUM_THREADS"

#: Every threading knob the numeric stack reads at load time.
BLAS_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "BLIS_NUM_THREADS",
)

#: ``--num_threads auto`` resolves the budget from the host instead of pinning it.
AUTO = "auto"

#: Budget used when nothing requests one. Deliberately serial: the training step
#: runs on the GPU, so intra-op CPU parallelism buys nothing and every thread it
#: would spawn is multiplied by the runtime count described in the module
#: docstring. ``training.cli`` uses this as the ``--num_threads`` default.
DEFAULT_THREAD_BUDGET = 1

#: Option strings ``--num_threads`` and ``--experiment_config`` are registered
#: under in :mod:`training.cli`. The budget has to be known before torch is
#: imported, which is before argparse can run, so these are scanned by hand.
#: :func:`main.configure_threads` re-resolves from the parsed namespace and
#: clamps again, so a new spelling added to the parser and missed here costs an
#: oversized pool at import, not a wrong final budget.
NUM_THREADS_OPTIONS = ("--num_threads",)
EXPERIMENT_CONFIG_OPTIONS = ("--experiment_config", "--experiment-config")


def _logical_core_count() -> int:
    """Cores this process may actually run on, honouring cpuset/taskset limits."""

    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        try:
            return max(1, len(affinity(0)))
        except OSError:
            pass
    return max(1, os.cpu_count() or 1)


def _physical_cores_from_cpuinfo() -> int | None:
    """Count distinct physical cores on Linux, ignoring SMT siblings."""

    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            text = handle.read()
    except OSError:
        return None

    cores: set[tuple[str, str]] = set()
    physical_id = core_id = None
    for line in text.splitlines():
        key, separator, value = line.partition(":")
        if not separator:
            # A blank line terminates one processor block.
            if physical_id is not None and core_id is not None:
                cores.add((physical_id, core_id))
            physical_id = core_id = None
            continue
        key = key.strip()
        if key == "physical id":
            physical_id = value.strip()
        elif key == "core id":
            core_id = value.strip()
    if physical_id is not None and core_id is not None:
        cores.add((physical_id, core_id))
    return len(cores) or None


def physical_core_count() -> int:
    """Physical cores available to this process.

    Physical cores rather than SMT threads: oversubscribing past them is at best
    flat and at worst a large regression, because the sibling threads contend for
    one core's vector units.
    """

    logical = _logical_core_count()
    physical = _physical_cores_from_cpuinfo()
    if physical is None:
        # Non-Linux, or a kernel that hides topology. Logical is the safe guess;
        # an SMT host can be tuned down explicitly.
        return logical
    # /proc/cpuinfo describes the host, which can exceed a restricted affinity
    # mask inside a container.
    return max(1, min(physical, logical))


def resolve_thread_budget(requested=None, concurrency: int = 1) -> int:
    """Resolve the per-process thread budget.

    ``requested`` is the explicit request (``--num_threads`` or
    :data:`THREAD_BUDGET_ENV`); ``None`` or ``"auto"`` derives it from the host.
    ``concurrency`` is how many such processes share the host at once.
    """

    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    if requested is None or (isinstance(requested, str) and requested.strip().lower() == AUTO):
        available = physical_core_count()
    else:
        if isinstance(requested, bool):
            # bool is an int subclass; True would silently mean "one thread".
            raise ValueError(f"num_threads must be a positive integer or {AUTO!r}")
        try:
            available = int(requested)
        except (TypeError, ValueError):
            raise ValueError(
                f"num_threads must be a positive integer or {AUTO!r}, got {requested!r}"
            ) from None
        if available != float(requested):
            # A fractional request is a mistake, not something to truncate.
            raise ValueError(
                f"num_threads must be a whole number of threads, got {requested!r}"
            )
        if available < 1:
            raise ValueError(f"num_threads must be positive, got {available}")

    return max(1, available // concurrency)


def apply_blas_env(threads: int) -> None:
    """Publish the budget to the numeric stack before it loads.

    ``setdefault`` so an explicitly exported ``OMP_NUM_THREADS`` still wins.
    """

    for name in BLAS_ENV_VARS:
        os.environ.setdefault(name, str(threads))


def budget_from_environment(environ=None):
    """Read the requested budget from the environment, or ``None`` for auto.

    :data:`THREAD_BUDGET_ENV` wins, then ``OMP_NUM_THREADS``. Honouring the
    standard OpenMP knob matters because callers that predate this module — the
    concurrency benchmark in ``scripts/`` among them — express "threads per
    process" by exporting the BLAS variables alone, and torch's intra-op pool
    should follow them rather than silently claim the whole host.
    """

    environ = os.environ if environ is None else environ
    for name in (THREAD_BUDGET_ENV, "OMP_NUM_THREADS"):
        value = environ.get(name)
        if value is not None and value.strip():
            return value
    return None


def _option_value(argv, options):
    """Read ``--option value`` or ``--option=value`` out of a raw argument list."""

    argv = sys.argv[1:] if argv is None else list(argv)
    value = None
    for position, token in enumerate(argv):
        name, separator, inline = token.partition("=")
        if name not in options:
            continue
        if separator:
            value = inline
        elif position + 1 < len(argv):
            value = argv[position + 1]
        # Keep scanning: argparse lets a later occurrence win.
    return value


def _budget_from_experiment_config(argv=None):
    """Read ``num_threads`` out of the experiment config named on the CLI.

    Mirrors ``training.cli.get_experiment_config_path``: the config is loaded
    only when it is passed explicitly. Any failure to read it is swallowed —
    argparse reports a broken config far better than a pre-import crash would,
    and the budget simply falls back to the default until then.
    """

    path = _option_value(argv, EXPERIMENT_CONFIG_OPTIONS)
    if path is None:
        return None
    try:
        with open(path, encoding="utf-8") as config_file:
            config = json.load(config_file)
    except (OSError, ValueError):
        return None
    if not isinstance(config, dict):
        return None
    return config.get("num_threads")


def environment_thread_ceiling(environ=None):
    """The cap the surrounding environment imposes, or ``None`` if it imposes none.

    ``run_scheduler`` exports :data:`THREAD_BUDGET_ENV` to tell a child how much
    of the host it may claim. That is a ceiling rather than a request: a child
    asking for fewer threads keeps its smaller number, and one asking for more
    is held to the share the scheduler carved out for it.

    Must be read before :func:`apply_blas_env` writes ``OMP_NUM_THREADS``, or it
    reads back the budget that was just published.
    """

    requested = budget_from_environment(environ)
    return None if requested is None else resolve_thread_budget(requested)


def resolve_process_budget(argv=None, environ=None) -> tuple[int, int | None]:
    """Resolve ``(budget, ceiling)`` for this process before any library loads.

    Precedence for the request is ``--num_threads``, then the experiment
    config's ``num_threads``, then :data:`DEFAULT_THREAD_BUDGET`; the ceiling
    from :func:`environment_thread_ceiling` then caps the result. The
    environment deliberately does not *raise* the budget, because the pools it
    sizes are per-runtime while the budget is per-process.
    """

    requested = _option_value(argv, NUM_THREADS_OPTIONS)
    if requested is None:
        requested = _budget_from_experiment_config(argv)
    if requested is None:
        requested = DEFAULT_THREAD_BUDGET

    try:
        budget = resolve_thread_budget(requested)
    except ValueError:
        # Let argparse produce the error message for a malformed request.
        budget = DEFAULT_THREAD_BUDGET

    ceiling = environment_thread_ceiling(environ)
    if ceiling is not None:
        budget = min(budget, ceiling)
    return budget, ceiling


def limit_native_thread_pools(threads: int) -> None:
    """Hold every native threading runtime in this process to ``threads``.

    The environment sizes only the runtimes that were loaded after it was
    published, and each of them reads it independently. This clamps them all
    through their own APIs instead, which is what keeps the process total near
    the budget rather than near a multiple of it.

    A pool that has already spawned cannot be shrunk — neither libgomp nor
    OpenBLAS hands threads back — so this is worth calling as early as the
    imports allow. It is deliberately best-effort: a missing ``threadpoolctl``
    degrades to torch and faiss keeping their own budgets.
    """

    if threads < 1:
        raise ValueError(f"threads must be positive, got {threads}")

    try:
        import threadpoolctl
    except ImportError:
        pass
    else:
        threadpoolctl.threadpool_limits(limits=threads)

    # faiss reaches its OpenMP loops through its own bundled runtime, which
    # threadpoolctl sees as an OpenBLAS rather than an OpenMP library. Only
    # touched when faiss is already imported; importing it here would load a
    # heavy CUDA-linked extension for nothing.
    faiss = sys.modules.get("faiss")
    if faiss is not None:
        faiss.omp_set_num_threads(threads)


def configure_process_threads(requested=None, concurrency: int = 1) -> int:
    """Resolve the budget and apply it to the BLAS environment. Returns the budget."""

    threads = resolve_thread_budget(requested, concurrency)
    apply_blas_env(threads)
    return threads
