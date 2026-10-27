# Parallel independent runs on CPU or GPU

Use `launch_runs.py` to run independent experiments or Optuna studies in
parallel. The scheduling unit is one complete `main.py` process, not one Optuna
trial. Keep every TPE configuration at `"n_jobs": 1`; TPE therefore observes
the result of each trial before choosing the next trial in that study, while
other losses or experiment configurations progress independently.

## Four-loss example

The repository includes
[`configs/run_manifests/hpo/four_losses.json`](../configs/run_manifests/hpo/four_losses.json).
It starts separate ArcFace, ProxyAnchor, NT-Xent, and Multi-Similarity studies:

```bash
python launch_runs.py --manifest configs/run_manifests/hpo/four_losses.json
```

The example manifest uses its configured GPU pool continuously for SSL. Training
uses CPU through batch size 128 and the same GPU above that boundary. Cap total
process concurrency when host CPU or RAM is the limiting resource:

```bash
python launch_runs.py \
  --manifest configs/run_manifests/hpo/four_losses.json \
  --device cpu \
  --max-total-runs 2
```

For GPU execution, set `"device": "cuda"` in the manifest or pass
`--device cuda`. CUDA remains the default for manifests without `device`.

With `"device": "cpu"`, a configured `gpus` pool continuously accelerates SSL
embedding extraction, FAISS graph construction, and CUDA-backed solvers. A run
or HPO trial trains on CPU through `batch_size: 128` and uses the same assigned
GPU for training when `batch_size` is greater than 128. The assigned physical
GPU is exposed to the child as logical `cuda`; an optional run-level `gpu` still
pins the assignment. Because an HPO process can change batch size between
trials, it retains that GPU assignment for the lifetime of the study.

### Per-loss batch-size thresholds

Some losses are faster on the GPU well below batch size 128. `launch_runs.py`
holds a hard-coded table for them:

```python
LOSS_GPU_BATCH_SIZE_THRESHOLDS = {
    "NTXentLoss": 0,
}
```

Each value replaces the 128 boundary for runs of that loss. `0` trains the loss
on its assigned GPU at every batch size; a value above every batch size you
launch keeps it on the CPU. Add losses by their `--loss` spelling; an unknown
name fails the child immediately. The table applies only to the CPU + `gpus`
mode above, and the effective values appear in the launcher output, in each run
log header, and in `state.json`.

An outer `--loss_miner_grid` and the warmup-loss substitution of a supervised
`STMLLoss` baseline are both resolved before the lookup, so each scenario and
HPO trial is placed by the loss it actually trains.

To keep training on CPU at every batch size, use `ssl_gpus` instead. That pool
is reserved only for SSL embedding extraction, FAISS graph construction, and
CUDA-backed solvers:

```json
{
  "device": "cpu",
  "ssl_gpus": [0, 1, 2, 3],
  "max_concurrent_per_gpu": 2,
  "max_total_runs": 8
}
```

The equivalent launcher override is `--ssl-gpus 0 1 2 3`. Each child receives
one load-balanced physical GPU as logical `cuda:0`; its training device remains
`cpu` and its resolved `ssl_device` is `cuda`.

For a direct, unscheduled run, select the logical SSL device explicitly:

```bash
python main.py --device cpu --ssl-device cuda:2
```

The same setting can be stored as `"ssl_device": "cuda:2"` in an experiment
config. Direct runs accept one SSL device; plural `ssl_gpus` belongs to the
launcher because it is the component that load-balances independent processes.
The selected device routes FAISS and GPU-capable solver backends; choose
`linear_solver: "cupy_cg"` or `"torch_block_cg"` in the SSL method config when
the solve itself should run on that GPU (`cg` and `cholmod` remain CPU solvers).

Inspect placement without starting anything:

```bash
python launch_runs.py \
  --manifest configs/run_manifests/hpo/four_losses.json \
  --dry-run
```

Each run may set a different `experiment_config`, `hparam_config`, and arbitrary
extra `args`. For example, replace the four `hparam_config` paths with
loss-specific search-space files. Extra arguments have normal CLI precedence
over values in the experiment config.

`save_dir_root` gives each child a deterministic, isolated output directory:

```text
logs/scheduled/cars_four_losses/arcface/...
logs/scheduled/cars_four_losses/proxy_anchor/...
logs/scheduled/cars_four_losses/ntxent/...
logs/scheduled/cars_four_losses/multi_similarity/...
```

This separation is important because two processes must not write the same
Optuna SQLite database or sampler checkpoint. A run-specific `save_dir` can be
set on an individual run when a different layout is needed.

Useful manifest controls are:

| Field | Meaning |
| --- | --- |
| `device` | `cpu` or `cuda`. Defaults to `cuda` for backward compatibility. |
| `gpus` | Eligible physical indexes or GPU UUIDs. In CPU mode, SSL always uses the assigned GPU, while training uses it only above batch size 128. Omit in CUDA mode to use all GPUs visible to the parent process. |
| `ssl_gpus` | With `device: "cpu"`, eligible physical GPUs for SSL-only acceleration. Omit for a pure CPU child. |
| `max_concurrent_per_gpu` | Compute-slot capacity on each GPU. Existing CUDA processes consume slots too. |
| `max_total_runs` | Optional host-wide child-process cap, useful when CPU or RAM is limiting. |
| `default_gpu_memory_mib` | Default admission reservation for each run. |
| `reserve_gpu_memory_mib` | Free-memory headroom retained on every GPU. |
| `save_dir_root` | Deterministic parent directory; the run name is appended automatically. |
| `fail_fast` | After one failure, skip queued runs while allowing active runs to finish. |
| `study_replay` | Shared trial selection, seeds, and final-fit mode for every run that sets `study_dir`. |

Each item in `runs` requires a unique filesystem-safe `name`. It can override
`experiment_config`, `hparam_config`, `save_dir`, `gpu_memory_mib`, and `slots`;
set `gpu` to pin it to one eligible index/UUID, `priority` to schedule it ahead
of lower-priority work, `enabled: false` to retain it without launching it, and
`env` for additional child environment variables. `args` is a JSON list of
extra `main.py` arguments. `ablation_config` expands the run into one child per
ablation variant; see [Ablating replayed studies](#ablating-replayed-studies).

## Replaying finished studies with new seeds

A run that sets `study_dir` replays an existing HPO study instead of starting a
new one: it takes the place of `experiment_config`/`hparam_config`, because the
study directory already contains the dataset, SSL, and search configuration in
`study_config.json`. Manifest-wide `study_replay` then sets the seeds, the trial
selection, and the final train method once for every replay run:

```json
{
  "device": "cpu",
  "gpus": [0],
  "save_dir_root": "ssl/replay/lrml",
  "study_replay": {
    "study_dir_mode": "cross_seed_train_val",
    "final_test_top_n": 10,
    "comparison_seeds": [7, 8, 9],
    "comparison_seed_targets": ["seed", "data_split_seed", "support_seed"]
  },
  "runs": [
    {
      "name": "lrml_nt",
      "study_dir": "ssl/test_reg/lrml_nt/.../hpo/cars_target_ratio"
    },
    {
      "name": "lrml_ms",
      "study_dir": "ssl/test_reg/lrml_ms/.../hpo/cars_target_ratio",
      "study_replay": { "final_test_trial_numbers": [6, 11] }
    }
  ]
}
```

That launches one child per study, each replaying its own top ten trials over
seeds 7, 8, and 9. A relative `study_dir` is resolved against the working
directory and then against `logs/`, exactly like `--final_test_study_dir`; the
launcher fails before starting anything when a directory is missing or holds no
`study_config.json`.

| `study_replay` field | `main.py` option |
| --- | --- |
| `study_dir_mode` | `--study_dir_mode`: `final_train`, `train_val` (default), or `cross_seed_train_val` |
| `final_test_top_n` | `--final_test_top_n`: replay the N best completed trials |
| `final_test_trial_numbers` | `--final_test_trial_numbers`: replay explicit trial numbers |
| `comparison_seeds` | `--comparison_seeds`: the seeds each selected trial is replayed on |
| `comparison_seed_targets` | `--comparison_seed_targets`: which seed channels those values replace |
| `final_test_visualization` | `--final_test_visualization` |
| `k_shot_grid` | `--k_shot_grid`: labeled examples per class to replay the trials under |
| `label_budget_grid` | `--label_budget_grid`: labeled fractions, in (0, 1], to replay the trials under |
| `ssl_label_sampling_modes` | `--ssl_label_sampling_modes`: sampling modes to replay under; omit to keep the study's own mode |

A run-level `study_replay` overrides the manifest-wide block field by field, and
naming either selector there replaces the shared selection, so a run can use
`final_test_trial_numbers` while the manifest sets `final_test_top_n`. Omitted
fields keep the `main.py` default. See
[the study-directory replay section](hyperparameter_search.md#final-test-evaluation-after-hpo)
for what each mode evaluates; `cross_seed_train_val` ranks the selected trials by
mean validation metric across the seeds and then performs a single final
train/test, so it requires at least two distinct `comparison_seeds`.

### Replaying under other label budgets

`k_shot_grid`, `label_budget_grid`, and `ssl_label_sampling_modes` are the same
outer grid dimensions an experiment config uses, so one manifest can replay each
study's trials under several label budgets:

```json
{
  "study_replay": {
    "study_dir_mode": "cross_seed_train_val",
    "final_test_top_n": 10,
    "comparison_seeds": [2, 3, 4, 8, 9],
    "comparison_seed_targets": ["seed", "data_split_seed", "support_seed"],
    "k_shot_grid": [1, 2, 4]
  }
}
```

For `class_subset_k_shot`, `k_shot_grid` varies the labeled examples per class
and `label_budget_grid` varies the fraction of labeled classes; the values the
study itself used stay in force for any dimension left unset. Leave
`ssl_label_sampling_modes` unset unless the replay should also change the
sampling mode — the study's own mode is used then, and a `k_shot_grid` needs
`class_subset_k_shot`.

Every combination becomes its own selection group: with the block above, each
study replays its ten trials over five seeds for k = 1, 2, and 4 separately
(150 validation runs), and each k gets its own winner and its own final
train/test. Group directories and summaries carry the budget in their names
(`class_subset_k_shot_label_1_k_2_...`), so the groups never share results and
an interrupted run resumes per group.

Because these settings change the data condition rather than the model, the
replayed study must not have tuned them itself: a study whose search space
contains label-budget or split parameters is rejected instead of silently
overridden.

Replay settings belong in the manifest rather than in `args`: the launcher
rejects the corresponding options there so one run cannot silently disagree with
the shared block. One study directory represents one method, so give each method
its own run.

### Ablating replayed studies

A run that also sets `ablation_config` replays its study once per variant of an
[ablation config](ablation_config.md), with that variant's changes applied over
the study's tuned parameters:

```json
{
  "save_dir_root": "final/phase_1/ablations/cars",
  "study_replay": {
    "study_dir_mode": "final_train",
    "comparison_seeds": [7, 8, 9, 10, 11],
    "comparison_seed_targets": ["seed", "support_seed"],
    "k_shot_grid": [4, 8, 16, 32]
  },
  "runs": [
    {
      "name": "hoffer_nt",
      "study_dir": "final/phase_1/cars/hoffer/hoffer_nt/.../hpo/in-shop",
      "ablation_config": "configs/ablations/hoffer_training.json"
    }
  ]
}
```

If that config's variants are `sampler_m_2`, `sampler_m_8`, and `no_warmup`, the
launcher schedules `hoffer_nt_sampler_m_2`, `hoffer_nt_sampler_m_8`, and
`hoffer_nt_no_warmup`. Each child is passed `--ablation_config` with its own
`--ablation_variant` and writes below its own name; a run that pins `save_dir`
puts its variants in `<save_dir>/<variant>`. The whole `study_replay` block
applies to every variant, so all of them share the baseline's seeds and label
budgets. The ablation file is validated while the manifest loads, and
`--ablation_config`/`--ablation_variant` are rejected in that run's `args`.

## Placement behavior

In pure CPU mode (neither `gpus` nor `ssl_gpus`), the launcher does not call
`nvidia-smi`. It forces both `--device cpu` and `--ssl-device cpu`, hides CUDA
devices from each child, ignores GPU-only admission settings, and admits up to
`max_total_runs` children. If that limit is omitted, pure CPU mode runs one
child at a time because an individual PyTorch process may already use all host
cores.

With `device: "cpu"` and `gpus`, placement reserves one GPU for each process
using the normal slot, memory, affinity, and least-loaded rules. SSL uses that
GPU at every batch size. Training stays on CPU at batch sizes up to and including
128 and switches to the assigned GPU above that boundary. `gpus` and `ssl_gpus`
cannot both be configured because `gpus` already covers combined SSL and
large-batch training; use `ssl_gpus` only for SSL-only acceleration.

With `device: "cpu"` and `ssl_gpus`, placement uses the same slot, memory,
affinity, and least-loaded selection rules as CUDA training. `max_total_runs`
can still cap host CPU/RAM pressure. A run-level `gpu` pins that run's SSL GPU.

In CUDA mode, the launcher queries GPU identity, free memory, utilization, and
active compute processes through `nvidia-smi`. It then:

1. spreads runs over the least-loaded GPUs before co-locating them;
2. treats an already-running CUDA process as one occupied run slot;
3. honors `gpu_memory_mib` reservations and `reserve_gpu_memory_mib` headroom;
4. starts queued work whenever a child exits and capacity becomes available;
5. exposes exactly one physical GPU UUID to each GPU-accelerated child with
   `CUDA_VISIBLE_DEVICES`, where the child uses logical `cuda:0`;
6. respects an inherited `CUDA_VISIBLE_DEVICES` allocation, such as one set by
   a cluster scheduler.

`gpu_memory_mib` is an admission reservation, not a CUDA memory limit. Use a
larger value for a configuration that consumes more VRAM. `slots` is the
compute-weight control: with `max_concurrent_per_gpu: 3`, a normal run can use
one slot and a heavier run can reserve two.

The launcher rejects an HPO config with `n_jobs != 1` by default. Set
`allow_parallel_trials: true` only when nested Optuna parallelism is deliberate.

## Choosing concurrency for the H100 host

The shown processes use roughly 1.5 GiB each, so VRAM is not the limiting
resource on a 95.8 GiB H100 NVL. The busy GPUs are already commonly at 80--99%
SM utilization, while GPU 3 is idle. For the twelve-run workload shown, start
with:

```json
{
  "gpus": [0, 1, 2, 3],
  "max_concurrent_per_gpu": 3,
  "default_gpu_memory_mib": 2048
}
```

That distributes twelve studies as three per GPU instead of four on GPUs 0--2
and zero on GPU 3. If only the four loss studies are active, the scheduler puts
one on each GPU. Increase the per-GPU value only if aggregate completed
trials/hour improves; instantaneous GPU utilization alone does not measure
throughput, and extra processes can add kernel, CPU, PCIe, and cache-I/O
contention.

Frozen feature tensors are reused in memory within one HPO process. Separate
run processes share the persistent on-disk DINO cache, whose writes are atomic,
but they do not share the process-local materialized tensor. For a cold cache,
pre-warming it with one run avoids redundant simultaneous backbone work. A
small `launch_delay_seconds` can also reduce startup bursts.

## Logs, failures, and resuming

Launcher stdout/stderr and `state.json` are written below
`scheduler_log_dir/<timestamp>_<pid>/`. Training artifacts remain in the normal
`logs/<save_dir>/...` hierarchy. A failed child does not stop unrelated runs
unless `fail_fast` is enabled; fail-fast skips work that is still queued while
already-running children finish. The launcher exits nonzero if any child failed.

Interrupting the launcher terminates only its child process groups. Re-running
the same manifest uses the same deterministic run output directories, so each
Optuna study applies its normal resume logic and schedules only its remaining
trial budget.

Study replays continue in the same way. A restarted `study_dir` run skips the
validation and final runs the interrupted attempt already finished and continues
an interrupted cross-validation run at its first unfinished fold, so a stopped
manifest can simply be started again. Add `--no-resume_interrupted_runs` to a
run's `args` to train every run of that replay again; see
[Resuming an Interrupted Replay](hyperparameter_search.md#resuming-an-interrupted-replay).
