# Frozen cached training: exact-output performance

Measured on 2026-09-10 with an RTX 5070 Ti and PyTorch 2.13.0+cu130.
The changes preserve eager floating-point expressions, reduction order, precision,
batch composition, and RNG consumption. No AMP or precision settings were changed.

## Results

Synthetic workloads use batch size 128, 768 backbone features, a 128-dimensional
normalized linear head, 512 classes, and one CPU thread. Hoffer uses four labeled
references available per class, drawing one per class per batch. Features are
already in host memory with a source-row index mapping; these measurements exclude
image extraction, disk misses, graph refresh, evaluation, and gradient calibration.

| Work | Original | Optimized | Speedup |
| --- | ---: | ---: | ---: |
| Hoffer training step | 6.18 ms | 3.66 ms | **1.69×** |
| Hoffer sampling, eight batches | 4.89 ms | 0.25 ms | **19.56×** |
| Cached batch fetching, across all five paths | 0.19–0.24 ms | 0.10–0.13 ms | **1.81–1.92×** |
| Supervised ArcFace step, including fetching | 1.76 ms | 1.70 ms | 1.03× |
| SLADE + Circle step, including fetching | 5.56 ms | 5.04 ms | 1.10× |
| Existing ArcFace CUDA graph option, inputs already on GPU | 1.86 ms | 1.08 ms | **1.72×** |

An earlier run measured Hoffer at 1.62× and SLADE at 0.99×. Hoffer's improvement
was substantial in both runs; SLADE's small benefit was variable. Sampling benefits
most when consecutive classes have equal support, as in k-shot splits. Unequal
support retains scalar draws when there is nothing to batch.

Training-step measurements include sampling where applicable, fetching, transfers,
forward, backward, and Adam. The benchmark uses zero learning rate to hold the
workload stable while timing Adam; separate tests use nonzero learning rates and
compare optimizer trajectories. They are component benchmarks, not full-fold
speedup estimates. SLADE's reference benchmark retains the new combined finite
check, so it slightly understates the removed synchronization overhead.

## Changes and remaining limits

- **Shared training/utilities:** `PrecomputedBackboneFeatureDataset.__getitems__`
  gathers deterministic rows once per batch. SSL relabeling, unlabeled, Hoffer,
  and LRML wrappers pass bulk requests through. LRML edge collation also fetches
  unique endpoints together, keeping their original order. Stochastic views
  retain the existing per-item random draws.
- **Hoffer:** consecutive classes with equal candidate counts share one CPU
  `randint` call. CPU roles become integer indices before transfer, removing GPU
  boolean compaction and blocking index transfers. Distances and entropy retain
  their original arithmetic; the large distance tensor remains a cost for many
  classes or reference sets.
- **SLADE:** initialization flags and mining thresholds share host transfers;
  each mined mask is compacted once. Running-distribution updates and loss
  reductions are unchanged. Dynamic pair selection and the underlying ranking
  loss still impose synchronization.
- **LRML / label propagation:** bulk fetching helps training and embedding
  extraction. No full graph-solver or full-fold speedup has been established.
  Changing solver, stopping tolerance, graph-refresh frequency, neighbor search,
  or projection batch size would require separate exactness validation.
- **Losses:** existing ArcFace, ProxyAnchor, MultiSimilarity, TripletMargin,
  and Circle fast paths already remove redundant work. Their formulas were left
  intact. NTXent intentionally remains upstream. Replacing compact reductions,
  expanding squared distances into dot products, or using Inductor fusion can
  change floating-point results despite mathematical equivalence.

For eligible supervised phases, `--compile_train_step` already enables the
`cudagraphs` backend. The engine admits ArcFace, ProxyAnchor, and
MultiSimilarityLoss with MultiSimilarityMiner on CUDA with precomputed inputs.
Active regularizers, confidence-aware objectives, and other unsupported paths
remain eager. The measured 1.72× is ArcFace only, excludes setup and data loading,
and should not be multiplied by the fetching speedup.

### `--graph_projection`: capture for the paths a regularizer blocks

`--compile_train_step` bails out on `phase.regularization_active`, so every SSL
epoch after warm-up runs the whole step eagerly. `--graph_projection` (opt-in,
default off) captures the one piece those steps still share with a supervised
one: the projection head. `torch.cuda.make_graphed_callables` graphs
`project_features`'s forward *and* backward, so the joint labeled+unlabeled
forward becomes one replay, while every loss, miner, and regularizer term stays
eager.

Exactness. The graph replays the head's own kernels, so embeddings and their
gradients are the eager ones bit for bit -- there is no fusion and no
reassociation. Two things enforce that rather than assume it: `make_graphed_projection`
compares a replay against an eager `project_features` before the graph is ever
used and falls back with a warning if they differ, and the trainer clones the
graph's output, because a replay writes into static storage that the next replay
overwrites and those embeddings outlive the step wherever a regularizer keeps
them.

What stays eager, and why:

- **A run that takes a second backward through the head.** Gradient surgery,
  GradNorm, target-ratio calibration, and the gradient-contribution diagnostic
  each run extra `autograd.grad` passes over the same node, which hands them
  detached views of the graph's static gradient buffers. That is a coverage
  limit, not a numerical one; lifting it needs a bit-identity test for the
  multi-pass case, not a narrower condition. Each logs the reason once.
- **CPU runs, `--train_amp`** (`make_graphed_callables` rejects autocast unless
  its weight cache is off, and that cache is most of what autocast buys), **image
  batches** (they enter through the backbone, which this does not capture), and
  any model without a `projection_head()` accessor.

One graph is kept per batch shape per fold, so the joint forward, a labeled-only
warm-up epoch, and a partial last batch are separate captures that each amortize
over the epochs replaying them.

Measured on an idle GPU under WSL2, forward plus backward over the projection of
a 256-row joint batch: 720 us eager, 529 us graphed (1.36x) with a toy loss tail.
A real regularizer tail is heavier, so its share of a full step is smaller;
measure before assuming it. Capture also emits one `AccumulateGrad` stream-mismatch
warning from autograd, which is about synchronization, not correctness.

Two things that were checked and rejected. `torch.optim.Adam(capturable=True)` is
**not** bit-identical to the default Adam -- it rearranges the bias correction
into device tensors, measured at 1.7e-3 max parameter deviation after 20 steps at
lr 1e-4 -- so the optimizer step cannot join the graph under this document's rule.
And for a reducer-blocked loss such as Circle, capturing the dense core while
leaving `AvgNonZeroReducer` eager *is* bit-identical (deviation exactly zero), but
through the `cudagraphs` backend the eager tail keeps the graph's outputs alive,
which hits the 128-recording limit and skips the graph -- it landed 2x slower than
eager. A split like that needs manual capture, not the compile backend.

## Reproduce and validate

```bash
venv/bin/python scripts/benchmark_exact_cached_training.py --device cuda --seconds 2
venv/bin/python scripts/benchmark_exact_cached_training.py --device cuda --capture-only --seconds 2
venv/bin/python -m unittest tests.test_exact_cached_speedups -q
venv/bin/python -m unittest tests.test_bit_exact_loss_compile -q
venv/bin/python -m unittest tests.test_experiment_training_flow -q
```

The new tests compare bytes, including signed zeros, for batches, losses,
gradients, Adam parameters/moments, and SLADE running buffers on CPU and CUDA.
They also check Hoffer batches and generator state across epochs, unequal class
support, stochastic views, duplicate indices, graph metadata, and mmap loading
with spawned workers.

The CUDA validation run covering the new checks, Hoffer, existing loss fast
paths, and CUDA graph capture passed all 50 tests. A broader 220-test run had
five failures and six errors; the same failures/errors reproduced with the
original files. These include missing/changed configurations, removed SLRML
APIs, stale capability expectations, and evaluation device expectations.
