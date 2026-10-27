"""CPU-friendly debugger harness for the project's SSL objectives.

This stable entry point delegates to the modular implementation in
``scripts/ssl_loss_debugger``. The harness runs the real project
implementations on a bounded synthetic or production-dataset workload and
exercises data loading, graph construction, loss calculation, backward passes,
optimizer updates, and method state updates.

Useful invocations from the repository root::

    python scripts/debug_ssl_losses.py --scenario all
    python scripts/debug_ssl_losses.py --scenario all --visualize
    python scripts/debug_ssl_losses.py --scenario lrml --visualize --visualization-layout pca
    python scripts/debug_ssl_losses.py --scenario lrml --epochs 5 --no-visualize
    python scripts/debug_ssl_losses.py --scenario lrml --backbone-tuning frozen --no-visualize
    python scripts/debug_ssl_losses.py --scenario lrml --break-at before-loader
    python scripts/debug_ssl_losses.py --scenario hoffer_entropy --log-level DEBUG
    python scripts/debug_ssl_losses.py --scenario seraph --dataset cars --no-visualize
    python scripts/debug_ssl_losses.py --scenario seraph --dataset cars --epochs 200 --report-every 20
    python scripts/debug_ssl_losses.py --scenario seraph --dataset cars --seraph-kappa 1.5 \
        --seraph-kappa-mode batch_std --include-labeled-unlabeled
    python scripts/debug_ssl_losses.py --scenario seraph --dataset cub --no-visualize

``--scenario seraph`` exercises the same ``SeraphRegularizer`` a production
``method: "seraph"`` config runs; every ``regularizer_params`` key documented in
``docs/seraph.md`` has a matching ``--seraph-*`` flag. Long runs are meant to be
throttled with ``--report-every``: every epoch contributes a line to the loss and
gradient-ratio series either way, but only reporting epochs print the batch dump
and pair trace and write artifacts.

In PyCharm, use this file as the run target, keep the working directory at the
repository root, and put normal IDE breakpoints inside
``losses/metric_losses.py``, ``training/semi_supervised.py``,
``training/ssl/algorithms.py``, or
``training/ssl/data.py``. ``--break-at`` additionally invokes Python's ``breakpoint()``
at a stable hand-off point so it is easy to step into the project code.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Direct script execution places only scripts/ on sys.path. Add the repository
# root so the debugger package and production modules resolve the same way they
# do in a repository-root Python session.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.ssl_loss_debugger.main import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
