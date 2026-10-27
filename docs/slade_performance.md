# SLADE training cost and memory

Investigation on 2026-09-17, using the pasted Cars196 configuration (32 labeled,
512 unlabeled, projection dimension 768, 73 basis vectors, gradient calibration
every step). The remote OOM traceback was not available. The local environment
had CPU PyTorch but no usable CUDA device, so these are code findings and CPU
checks, not measured remote GPU peaks.

## Changes

- The distribution update now selects first-observation versus EMA statistics
  on-device. Training reads calibration status and mining thresholds together.
  This removes two device-to-host waits per SLADE batch without changing the
  loss, mining gates, gradient mode or calibration cadence. The condition is
  cloned before updating the checkpointed flags so retained backward passes
  remain valid. CPU regression checks compare exact values, gradients, buffers
  and Adam trajectories against the previous host-branch implementation.
- Target-ratio calibration no longer measures the supervised-plus-ranking
  gradient. Only the cosine diagnostic uses that measurement. SLADE therefore
  needs three component backwards per calibration probe instead of four, plus
  its ordinary training backward. Diagnostic logging retains its pair norm.
- Component norms transfer their scalar summaries to the host together instead
  of synchronizing after each component. Gradient tuples are still released
  individually. FP32 accumulation and Python sqrt/division are unchanged.
- `max_mean` no longer computes unused sums for the max statistic or unused
  maxima for the mean statistic.
- SLADE + MultiSimilarity can pass symmetric mined masks directly into the
  existing loss core and reducer. This removes the two variable-size pair
  compactions and the reconstruction of masks. Loss hyperparameters, selected
  pairs, reducers, calibration cadence, and targets are unchanged.
- The follow-up fix routes SLADE's unlabeled upstream `NTXentLoss` through
  `losses/bounded_ntxent.py`. Negatives are summed once per anchor, eliminating
  the positive-by-negative product. This is automatic; no config changes or
  smaller batches are needed to enable it. The supervised criterion and custom
  NT-Xent subclasses retain their existing implementation.

CPU checks compared old/new norm summaries exactly across 20 randomized cases.
Tests compare dense/indexed MultiSimilarity values and embedding gradients with
zero tolerance, including empty and one-sided mining; CUDA cases run when CUDA
is available. Integration checks compare SLADE's basis gradients and diagnostics.

A synthetic single-thread CPU probe with a 768-dimensional linear head, 32+512
samples, 73 basis vectors, actual SLADE losses, max/mean probes and backward
took median 129.44 ms before versus 102.63 ms after over 12 measured repetitions
following three warm-up repetitions. This is about 21% less time for that probe,
not an end-to-end or GPU speedup. It excluded loading, clustering, validation
and optimizer steps; its norm probe included the basis parameters as well as
the projection, whereas production calibration excludes private basis weights.

## Remaining step cost and probe cadence

Active SLADE does not use the engine's fused supervised CUDA-graph step, even
when `compile_train_step` is enabled. It computes a second ranking objective,
the basis classification/distribution objective, and their gradients. On InShop,
2997 basis vectors of dimension 768 add 2,301,696 trainable parameters and the
associated dense projections despite the frozen backbone.

When ranking and basis target-ratio calibration are both active, each probe
adds three retained-graph backwards over shared parameters. The ordinary
training backward still runs. Gradient diagnostics add four more probes on
their logging batches. These costs are separate from the k-means refresh.

The current local `configs/runs/ssl/slade/paper.json` already uses an EMA with
`regularizer_target_ratio_update_interval_samples=1000`, unlike the earlier
pasted logs with interval zero. This is every 8 steps at labeled batch 128, or
every 4 steps at batch 256. Reducing probe frequency changes the weight-update
schedule and may change the training trajectory; the synchronization change
above does not change that schedule. Do not assume that sparse calibration
keeps the ratio at its target between probes.

`scripts/benchmark_slade_step.py` measures the actual losses, norm probes,
backward and Adam on synthetic frozen features, with stable weights (`lr=0`).
It does not apply measured target weights, run a miner on the supervised loss,
or include loading, pseudo-label extraction, k-means or validation. It isolates
probe overhead rather than predicting a real training trajectory. A CPU run
with MultiSimilarity, 128 labeled + 256 unlabeled samples, dimension 768, 2997
basis vectors and gradient diagnostics every 30 steps measured:

| Gradient probe cadence | Mean step time |
| --- | --- |
| Every step | 122.81 ms |
| Every 4 steps | 94.67 ms |
| Every 8 steps | 89.30 ms |
| No calibration probes | 82.98 ms |

These are CPU timings, not a measured GPU speedup. In particular the current
1000-sample cadence is already near the cheaper end of this comparison. Measure
the server directly, for example:

```bash
python scripts/benchmark_slade_step.py --device cuda --loss ms
python scripts/benchmark_slade_step.py --device cuda --loss ntxent
```

For the real run, `--debug_batch_timing --debug_batch_timing_interval 30`
separates regularizer computation, calibration, backward and diagnostics;
timing itself synchronizes CUDA and should be disabled for throughput runs.
The local SSL config also uses `embedding_batch_size=32`. If refreshes are slow,
a larger **extraction** batch (e.g. 1024 for cached features) can reduce the
number of tiny projection calls and host transfers without enlarging the
training pair matrices. This does not eliminate the per-step basis cost and
is not an automatic config change in this patch.

## Why frozen features can still exhaust VRAM

Frozen features remove backbone activation graphs; they do not remove the
pair-loss intermediates or the gradients needed to train the projection.

The installed upstream `NTXentLoss._compute_loss` compares every directed
positive pair with every directed negative pair, then masks different anchors.
Its intermediates have shape `[P, N]`, not `[batch_size, batch_size]`. With dense
pair sets, both P and N grow quadratically, so this intermediate grows quartically.
`convert_to_triplets` also creates a `[P, N]` comparison when a triplet loss
receives SLADE's four-tensor pair tuple.

For illustration, taking all label-derived pairs from seven approximately
balanced clusters gives:

| Unlabeled batch | Directed positives | Directed negatives | One FP32 P-by-N tensor |
| --- | --- | --- | --- |
| 64 | 522 | 3,510 | 0.0068 GiB |
| 128 | 2,214 | 14,042 | 0.1158 GiB |
| 256 | 9,108 | 56,172 | 1.9059 GiB |
| 512 | 36,938 | 224,694 | 30.9190 GiB |

These are illustrative pair counts, not observed SLADE mining counts. NT-Xent
holds several such intermediates, and autograd saves tensors for backward.
Its Boolean anchor comparison consumes additional memory. An isolated run can
therefore exceed its share of a 90 GB GPU even with a tiny projection model.
The initial speed changes left this allocation intact. The NT-Xent follow-up
now removes it from SLADE's unlabeled ranking path.

The reported follow-up run used at most 256 unlabeled samples, not 512. That
still allows a large product: two balanced groups at 256 produce 32,512
directed positives and 32,768 directed negatives, or **3.96875 GiB for just one
FP32 product**. Multiple intermediates, backward temporaries and allocator
reservations can coexist. The server's 56,682 MiB process reading includes more
than this tensor, so the snapshot alone does not establish the full breakdown.

## Bounded NT-Xent validation

The adapter preserves the same per-positive numerator and same-anchor negatives,
temperature, configured distance, reducer, duplicate-pair multiplicities,
embedding regularization and tiny-probability floor. It does not substitute a
supervised-contrastive denominator containing other positives. Reordered sums
can introduce floating-point differences; it is not bitwise reproducible with
the old implementation. Completed runs remain valid and need not be deleted.
Restart running processes after syncing the new source to activate the fix.

Twelve CPU FP32 parity cases across temperatures 0.0023, 0.07, 0.5 and 10 had
identical scalar losses and a maximum absolute gradient difference of
1.1920928955078125e-7. Other inputs may have small loss differences too.
Regression tests also cover FP64, degenerate mining, custom distances/reducers,
retained-graph probes and SLADE integration. The relevant regression command ran
188 tests successfully, with six CUDA-dependent tests skipped.

The isolated CPU benchmark with dimension 768 and one retained-graph gradient
probe measured unique tensor storage saved for backward as follows:

| Batch, two balanced groups | Upstream | Bounded |
| --- | --- | --- |
| 64 | 35.54 MiB | 0.675 MiB |
| 256 | not executed | 4.06 MiB |

These figures exclude process overhead, temporary allocations, the basis loss,
the backbone and evaluation. They are not total GPU VRAM measurements. At batch
256 the largest saved tensor was the 256-by-768 embedding storage; no tensor
with P-by-N shape was saved. CUDA peak allocated/reserved memory can be measured
on the server with:

```bash
python scripts/benchmark_slade_ntxent.py --device cuda --batch-size 256
```

Following the user's request, the separate "no usable gradient in 50 consecutive
calibration batches" guard and its 1e-6 ratio floor have been removed. Every
finite positive norm can now calibrate a large finite weight, including for
the basis term. Exactly zero or non-finite measurements defer calibration;
they do not trigger a batch-budget exception. Previously pruned trials are not
automatically reset by a code update, but can be retried with this behavior.
The bounded-memory loss itself still preserves the mathematical saturation.

MultiSimilarity uses dense batch-by-batch matrices instead: a single 512-square
FP32 matrix is 1 MiB. Its pair matrices and a 73-by-768 basis do not alone
explain a many-GB training peak. A MultiSimilarity OOM requires checking the
traceback for evaluation, feature extraction, another loss, or concurrent jobs.

Other per-process costs include CUDA contexts, the frozen backbone if resident,
retained warm-up CUDA graph pools, optimizer state, FAISS resources and the
PyTorch caching allocator. FAISS allocations do not appear in PyTorch's memory
statistics. With low-memory mode and `faiss_temp_memory_mb=256`, an active
resource can reserve 256 MiB; clustering and evaluation use separate resources.
Ten processes multiply these costs and can reach their peaks simultaneously.

## Server checks and immediate controls

- Record the failing operation and allocation size from an OOM traceback.
  A MultiSimilarity process can be the victim of another job's NT-Xent spike.
- Confirm the startup message `SLADE NTXentLoss: using bounded-memory per-anchor
  negative sums` appears when the unlabeled ranking term activates.
- Triplet SLADE runs still have the pair-product conversion; an unlabeled batch
  cap can mitigate it. This changes sampling and is not a universal memory
  guarantee. The NT-Xent fix does not require lowering the batch cap.
- Keep the low-memory policy enabled. `--faiss_temp_memory_mb 0` removes the
  persistent scratch reservation, but FAISS can still allocate working memory.
- `--debug_batch_timing --debug_batch_timing_interval 30` distinguishes training
  losses, calibration, ordinary backward, and loading costs in a real run.
  Profile one job before comparing concurrent throughput.
- Measure allocated and reserved PyTorch memory separately from process VRAM
  in `nvidia-smi`. Reserved memory can remain after a large earlier trial.

Probe cadence and smaller unlabeled batches can improve speed further, but they
change balancing or sampling. Increasing the clustering interval can save
clustering time, though the pasted run spent only about 0.2–0.5 seconds there
versus roughly 17 seconds per SSL epoch.
