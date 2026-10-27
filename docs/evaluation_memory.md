# Where evaluation memory goes

Measured on an RTX 5070 Ti (16 GB) under WSL2, FAISS 1.14.3,
pytorch-metric-learning 2.9.0.

A frozen-backbone CUB run trains a projection head on 5,864 precomputed
768-dimensional features — about 18 MB of data. Two epochs over three folds
cost this much:

| `--evaluation_embedding_residency` | peak VRAM | peak RSS |
| --- | --- | --- |
| `gpu` | 1820 MiB | 2504 MiB |
| `cpu` | 235 MiB | 1887 MiB |

Neither number is about the data. This note records what the memory actually
is, because the intuition that GPU residency "only caches the embeddings" is
correct about the embeddings and wrong about the total.

Reproduce the run-level numbers with
[`scripts/benchmark_gpu_residency_vram.py`](../scripts/benchmark_gpu_residency_vram.py),
which holds the embeddings on the GPU in every variant and changes only the
memory policy of the reusable FAISS wrapper.

## The FAISS scratch arena dominates VRAM

`_ReusableSingleGpuFaissKNN` calls `setTempMemory(0)` only when `low_memory` is
set ([`utils/__init__.py:2174`](../utils/__init__.py)), and `low_memory` is the
negation of GPU residency ([`utils/__init__.py:2605`](../utils/__init__.py)).
So `residency=gpu` leaves `StandardGpuResources` at its default arena:

```
low_memory=False: faiss reserved 1548 MiB
low_memory=True:  faiss reserved   12 MiB
```

This is a process-lifetime cost, not a per-evaluation one. The wrapper is
cached in the module-global `_REUSABLE_FAISS_KNN`
([`utils/__init__.py:2338`](../utils/__init__.py)), so the arena is allocated at
the epoch −1 validation and held through every training epoch that follows. It
comes from `cudaMalloc`, outside the torch caching allocator, so training cannot
reuse it and `torch.cuda.memory_reserved` cannot see it.

Capping the arena costs nothing measurable. Every variant below keeps the
embeddings resident on the GPU, and P@1/MAP@R are identical across all of them
(the runner aborts on any divergence above 1e-9):

```
CUB/test: 5924 samples, k=59, embeddings=17 MiB on GPU
  variant               process VRAM   ctx+faiss  torch peak   median s   vs base
  baseline                   1930 MiB     1836 MiB       94 MiB      0.048     1.00x
  arena-0                     394 MiB      300 MiB       94 MiB      0.049     1.03x
  arena-256                   650 MiB      556 MiB       94 MiB      0.053     1.11x

Food101/test: 30000 samples, k=999, embeddings=88 MiB on GPU
  baseline                   3958 MiB     1904 MiB     2054 MiB      0.164     1.00x
  arena-0                    2435 MiB      381 MiB     2054 MiB      0.165     1.01x
  arena-256                  2681 MiB      627 MiB     2054 MiB      0.172     1.05x

iNat/test: 103271 samples, k=419, embeddings=303 MiB on GPU
  baseline                   5589 MiB     2157 MiB     3432 MiB      1.353     1.00x
  arena-0                    4018 MiB      586 MiB     3432 MiB      1.241     0.92x
  arena-256                  4275 MiB      843 MiB     3432 MiB      1.335     0.99x
```

FAISS falls back to on-demand `cudaMalloc` for search scratch. For a flat L2
index at these values of `k` the demand is small enough that `arena-0` was
slightly *faster* than the default on iNat.

Retaining the gallery index between evaluations is worth keeping. Discarding it
saved a further ~100 MiB but cost 1.27x on Food101, because the index is rebuilt
every evaluation.

## The MAP@R working set is the rest

`residency=gpu` also puts the whole `AccuracyCalculator` on the GPU
([`utils/__init__.py:2599`](../utils/__init__.py)). With `k="max_bin_count"`,
`mean_average_precision` builds a stack of N×k intermediates — `knn_labels`,
`knn_distances`, a bool relevance mask, an int64 `cumsum`, an int64
`arange(k).repeat(N, 1)`, and two float64 products
(`pytorch_metric_learning/utils/accuracy_calculator.py:91-107`). That is roughly
40 bytes per (query, neighbor) cell at peak:

| N | images/class | k | embeddings | MAP@R peak |
| --- | --- | --- | --- | --- |
| 5,924 | 30 | 29 | 12 MiB | 25 MiB |
| 10,000 | 100 | 99 | 20 MiB | 65 MiB |
| 10,000 | 1,000 | 999 | 20 MiB | 469 MiB |
| 50,000 | 500 | 499 | 98 MiB | 1218 MiB |

On the iNat profile above this is the 3432 MiB torch peak, of which only 303 MiB
is embeddings. It scales with class size, not dataset size, so it is invisible on
CUB and Cars and dominant on Food101 and iNat.

**This cannot be moved to the host by setting the calculator device.**
`AccuracyCalculator.get_accuracy` moves query and reference to `self.device`
*before* it calls `knn_func`
(`pytorch_metric_learning/utils/accuracy_calculator.py:429-432`, used at
`:472`). A CPU calculator therefore pulls the embeddings off the GPU, which is
exactly what GPU residency exists to avoid. Shrinking this term means chunking
the metric computation over queries, which pytorch-metric-learning does not
support.

## Host RAM has its own FAISS cost, and the low-memory mode misses it

The fixed floor of a run, measured incrementally in one process:

| after | RSS |
| --- | --- |
| `import torch`, `import numpy` | 666 MiB |
| CUDA context | 777 MiB |
| `import faiss` | 792 MiB |
| `StandardGpuResources` + one search | 1225 MiB |

`StandardGpuResources` costs **433 MiB of host RAM**, most of it a 256 MiB
pinned staging buffer. Capping both buffers brings it down:

```
setTempMemory(0)                        -> 433 MiB host RAM   (production low_memory)
setTempMemory(256 MiB), setPinnedMemory(0) -> 177 MiB host RAM
```

The current low-memory policy sets only `setTempMemory`. So
`--evaluation_embedding_residency cpu` gives back 1.5 GiB of VRAM and **zero**
host RAM — the pinned buffer is never capped in either mode. That is a large
part of why RAM looks high regardless of which residency is selected, and it is
per process, so concurrent HPO runs each pay it.

## Frozen-feature caches grow without bound across trials

`FrozenFeatureDatasetCache`
([`training/frozen_feature_cache.py:385`](../training/frozen_feature_cache.py))
is a plain dict with no eviction and no size cap. `get_or_compute` only ever
inserts, and the `_bytes` total it tracks is reported by `stats()` but never
enforced against anything.

Every distinct cache key holds a full resident feature view for the lifetime of
the process. The key covers `seed`, `data_split_seed`, `cv_k`, `cv_mode`,
`val_mode`, the label budget, and the input transform, among others
(`_SOURCE_ARG_NAMES`). A five-fold study that also varies seed or label budget
therefore accumulates one resident copy per combination.

Each copy can be large: `--frozen_feature_residency` defaults to `ram`
([`training/cli.py:463`](../training/cli.py)), which copies the rows a view uses
into process memory ([`utils/__init__.py:1465`](../utils/__init__.py)), bounded
per view by `--frozen_feature_residency_max_gb` (default 4.0). The budget is
per view, not per study, so N views means up to N × 4 GB.

One related multiplier: when a resident view is used with `num_workers > 0`,
each spawned worker copies the tensor rather than sharing it
([`utils/__init__.py:1498`](../utils/__init__.py)), and `main.py` forces the
`spawn` start method. `--num_workers` defaults to 0, so this is off unless it is
turned on.

## Two measurement artifacts to rule out first

On WSL2, `nvidia-smi --query-compute-apps` reports nothing, and
`nvidia-smi --query-gpu=memory.used` showed 3.2–4.4 GiB with no training process
present in `ps`. That baseline is the Windows host and driver, not the run.
Measure a run as a delta against an ambient reading taken before the process
creates its CUDA context, which is what the benchmark script does.

`free -h` during idle showed 18 Gi in `buff/cache` against 22 Gi available. That
is page cache holding the mmap'd backbone feature matrices. It is reclaimable
and is not run memory.

## `--low_memory_mode`

The first three costs above are bounded by `--low_memory_mode`, **which is on by
default**. It changes no metric and moves nothing off the GPU: embeddings stay
device-resident and the gallery index stays reused between evaluations. It caps
only what is held *between* searches.

```bash
# Restore the historical unbounded buffers:
python main.py --no-low_memory_mode ...
```

The mode sets three things:

| | `--no-low_memory_mode` | default |
| --- | --- | --- |
| FAISS device scratch, GPU-resident embeddings | ~1.5 GiB | 256 MiB (`--faiss_temp_memory_mb`) |
| FAISS device scratch, host-resident embeddings | 0 | 0 |
| FAISS pinned host buffer | 256 MiB | 0 |
| Cross-trial frozen-feature cache | unbounded | 8 GB, LRU (`--frozen_feature_cache_max_gb`) |

Measured on the same CUB run as the table at the top, GPU-resident in both cases:

| | peak VRAM | peak RSS |
| --- | --- | --- |
| `residency=gpu --no-low_memory_mode` | 1820 MiB | 2504 MiB |
| `residency=gpu` (low-memory mode) | 540 MiB | 2391 MiB |

Validation metrics are bit-identical between the two (0.7833333333333333 mean,
folds 0.83/0.70/0.82). `--faiss_temp_memory_mb 0` gives back a further ~256 MiB
and measured no slower on every profile tested; 256 MiB is the default only
because on-demand scratch allocation is the less predictable choice for index
types this repository does not currently use.

The cache budget is enforced by least-recently-used eviction in
`FrozenFeatureDatasetCache`, and `stats()` now reports `evictions` and
`max_bytes` alongside the byte total, so
`frozen_feature_precompute_stats.json` records whether a study was evicting.
A view larger than the whole budget is still returned to its caller rather than
evicted on insertion.

## SSL graph construction pays the FAISS cost per search

Evaluation retrieval is not the only FAISS consumer. `faiss_flat_ip_search`
([`training/ssl/algorithms.py:426`](../training/ssl/algorithms.py)) builds a
**fresh** `StandardGpuResources` on every call and drops it when the search
returns. Iscen label spreading, the kNN majority-vote and self-training pseudo
labelers, and LRML's graph all route through it, so every pseudo-label refresh
re-reserves the default buffers.

Measured at the In-Shop training shape (25,882 × 768, k=11):

| | live VRAM during search | host RSS delta |
| --- | --- | --- |
| default | 1630 MiB | 270 MiB |
| low-memory mode | 342 MiB | 0 MiB |

Neighbor indices and similarities are identical either way. Because the
allocation is transient rather than held, it does not show up in a memory
reading taken between refreshes — but it does collide with concurrent runs.
`configs/run_manifests/in_shop_iscen.json` sets `max_concurrent_per_gpu: 5`
against `default_gpu_memory_mib: 1024`, and five refreshes overlapping at the
default size is ~8 GB of transient demand on a 16 GB card.

`--low_memory_mode` covers this path too: `apply_low_memory_mode` sets the
evaluation policy in `utils` and the per-search policy in
`training/ssl/algorithms.py` from the same flag. The SSL module keeps its own
module-local setting rather than importing `utils`, so it stays dependency-light.

## Label spreading scales with classes, not just data

The dominant Iscen cost is not FAISS and not the size of the unlabeled pool. It
is `one_hot_targets` and the solution matrix in `faiss_label_spreading`
([`training/ssl/algorithms.py:602`](../training/ssl/algorithms.py)), both dense
`N × num_classes` **float64**:

| dataset/split | N | classes | N×C float64, each | peak RSS over baseline |
| --- | --- | --- | --- | --- |
| CUB/test | 5,924 | 100 | 5 MiB | +746 MiB |
| DeepFashionInShop/train | 25,882 | 3,997 | 789 MiB | +2670 MiB |

In-Shop has 4.4× CUB's samples but 40× its classes, and the memory follows the
product. This is why In-Shop looks so much worse than the other datasets rather
than proportionally worse.

The device side of the solve is already bounded —
`_torch_block_cg_right_hand_side_batch_size`
([`training/ssl/algorithms.py:2028`](../training/ssl/algorithms.py)) sizes its
column batch from free VRAM, and the measured GPU delta was only 330 MiB. It is
the host-side dense arrays that are unbounded. LRML is unaffected: it builds a
sparse kNN edge index and never materializes an `N × num_classes` matrix.

## Still open

**Chunk the MAP@R computation** over queries if the remaining torch peak matters
on iNat-sized splits — 3432 MiB there, against 303 MiB of embeddings. This is a
change to the metric path rather than a flag, because
pytorch-metric-learning computes the whole N×k working set at once.
