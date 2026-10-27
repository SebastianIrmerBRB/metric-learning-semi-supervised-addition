# Experiment Config

Use `--experiment-config` to load normal CLI argument values from one top-level JSON file:

```powershell
python main.py --experiment-config configs/experiments/class_subset_search.json
```

The JSON keys are the argparse destination names without leading dashes. For example:

```json
{
  "save_dir": "ssl_baseline_check/class_subset_label_0p01",
  "dataset": "StanfordOnlineProducts",
  "dataset_protocol": "official",
  "val_mode": "split_after_apportion",
  "cv_k": 4,
  "cv_mode": "stratified_group_kfold",
  "ssl_label_sampling_modes": [
    "class_subset"
  ],
  "label_budget_grid": [
    0.01
  ],
  "loss_miner_grid": [
    "NTXentLoss:BatchEasyHardMiner",
    "TripletMarginLoss:TripletMarginMiner"
  ],
  "k_shot_grid": null
}
```

Grid arguments use JSON arrays. Use `null` to disable an optional/default grid such as `k_shot_grid`, `label_budget_grid`, or `loss_miner_grid`.

Explicit CLI arguments override the file:

```powershell
python main.py `
  --experiment-config configs/experiments/class_subset_search.json `
  --save_dir ssl_baseline_check/class_subset_label_0p05 `
  --label_budget_grid 0.05
```

The experiment config may include any normal CLI argument, including `ssl_config` and `hparam_config`. Relative paths are resolved from the current working directory, matching direct CLI behavior.

For a direct non-HPO run, set `"evaluate_test": true` in the experiment config or pass `--evaluate-test` to evaluate `D_test` after training. HPO runs continue to use `final_test_after_hpo` for final test evaluation.

Evaluation measurements are list-valued. Precision@1 and MAP@R remain the
default; add normalized mutual information (NMI) and R-precision (RP) with:

```powershell
python main.py --measurements P@1 MAP@R NMI RP
```

The equivalent experiment-config value is:

```json
{
  "measurements": [
    "precision_at_1",
    "mean_average_precision_at_r",
    "NMI",
    "r_precision"
  ]
}
```

The configured columns are written to `metrics.csv` and TensorBoard for every
validation and test evaluation. `--selection-metric` remains separate, so this
list does not change checkpoint selection or early stopping.

For TPE hyperparameter searches, set `"tpe_startup_trials": 5` in this file or pass `--tpe-startup-trials 5` to control Optuna's random startup-trial count. A CLI or experiment-config value overrides the HPO config's `tpe_startup_trials` field and `sampler_params.n_startup_trials`.

For parallel Optuna trials, set `"n_jobs": 2` or another positive integer in the HPO config JSON. Use `"n_jobs": -1` to let Optuna use all available CPU cores. GPU runs usually need a conservative value so simultaneous trials do not compete for the same accelerator memory.

When an SSL configuration has labeled-only warmup epochs, every HPO trial now
uses that trial's resolved main loss and miner for warmup, including tuned
constructor parameters. The same criterion, miner, and classification-loss
optimizer objects continue into the post-warmup phase, preserving trainable
proxy state and optimizer moments. Loss-driven `STMLLoss` is the exception:
its multi-view unlabeled objective cannot consume the labeled-only warmup
batches, so it retains its explicitly configured supervised warmup objective.

## Artifact volume

Every logging artifact below is independently switchable, so an HPO study can
keep only what it reads back. Measure any change with
`python scripts/bench_logging_overhead.py`.

| Flag | Default | Artifact |
| --- | --- | --- |
| `--debug_log` | off | `debug.log` |
| `--log_epoch_diagnostics` | off | epoch/eval rows in `diagnostics.csv` and their TensorBoard scalars |
| `--log_batch_diagnostics` | off | per-batch rows in `diagnostics.csv` |
| `--ssl_gradient_contribution_log_interval` | `0` (off) | gradient-contribution rows in `diagnostics.csv` |
| `--tensorboard` | on | `tensorboard/` event files |
| `--pseudo_label_diagnostics_mode` | SSL config (`save`) | `pseudo_label_diagnostics.jsonl` |

`metrics.csv` always holds the headline losses and retrieval metrics, so
selection and early stopping are unaffected by any of these. `diagnostics.csv`
is created lazily, so a run with every diagnostic disabled writes no file at
all. `--pseudo_label_diagnostics_mode off` also skips computing the
hidden-label audit, whereas `log` still computes it and only skips the file.
Turning TensorBoard off means `react_trial_visualizer` has nothing to read for
that run.

Per-batch diagnostics are disabled by default because gradient norms and
diagnostic writes add measurable training overhead. Enable them only for
investigation runs with `--log-batch-diagnostics` or:

```json
{
  "log_batch_diagnostics": true
}
```

For an interval-sampled decomposition of regularized SSL updates, set
`ssl_gradient_contribution_log_interval` to a positive number of active
regularization batches:

```json
{
  "ssl_gradient_contribution_log_interval": 100
}
```

The first active regularization batch is measured, followed by every 100th
active batch. The logged `train/gradient_contribution/*` diagnostics contain
the weighted supervised, regularizer, and combined model-gradient L2 norms,
the regularizer's fraction of the two component magnitudes, their cosine
similarity, and the regularizer-to-supervised norm ratio. They are written to
TensorBoard and `diagnostics.csv` even when `log_batch_diagnostics` is false.
Each sampled measurement performs two additional autograd traversals, so a
larger interval is preferable for throughput-sensitive runs. Component
gradients are summarized one at a time to limit peak memory. The normal
combined backward pass and optimizer update are unchanged.

Detailed per-batch timing is also opt-in via `--debug-batch-timing`.

## Learning-Rate Scheduler

Learning-rate scheduling is disabled by default (`"lr_scheduler": "none"`).
Enable an epoch-level StepLR schedule with:

```json
{
  "lr_scheduler": "step",
  "lr_scheduler_params": {
    "step_size": 10,
    "gamma": 0.1
  }
}
```

Or use cosine annealing:

```json
{
  "lr_scheduler": "cosine",
  "lr_scheduler_params": {
    "T_max": 100,
    "eta_min": 0.0
  }
}
```

Cosine annealing with warm restarts is available separately:

```json
{
  "lr_scheduler": "cosine_warm_restarts",
  "lr_scheduler_params": {
    "T_0": 10,
    "T_mult": 2,
    "eta_min": 0.0
  }
}
```

The CLI equivalents are `--lr-scheduler step` and
`--lr-scheduler-params '{"step_size": 10, "gamma": 0.1}'`. If the parameter
object is empty, StepLR defaults to `step_size=10, gamma=0.1`; cosine defaults
to `T_max=epochs, eta_min=0`; warm restarts default to
`T_0=10, T_mult=1, eta_min=0`. StepLR and plain cosine advance after each
non-empty training epoch. Warm restarts advance after every optimizer step
using fractional-epoch progress, so `T_0` and each expanded cycle length are
measured in active training epochs. All policies apply to the model optimizer
plus the active classification-loss optimizer, when present.

TensorBoard records the mean rate actually used during an epoch under
`epoch/train/learning_rate/<optimizer>/group_<n>`. The start, mean, minimum,
maximum, and final batch rates are also available below
`epoch/train/learning_rate_summary/<optimizer>/group_<n>/`.

The resolved experiment-config values are included in run/study metadata as `experiment_config_resolved`.

## Frozen Backbone And Cache

Freeze the pretrained DINO backbone while continuing to train the projection
head with:

```json
{
  "backbone_tuning": "frozen",
  "use_cache": true,
  "feat_dim": 256
}
```

The equivalent CLI flags are `--backbone-tuning frozen` and `--use-cache`.

`use_cache` requires `backbone_tuning="frozen"`, and frozen-backbone training
requires a non-null `feat_dim` so the model retains a trainable projection head.

### Projection Head Depth

By default the projection head is a single linear map from the backbone
dimension to `feat_dim`. `projection_layers` deepens it into an MLP, which is
mainly useful with a frozen backbone, where the head is the only trainable part
of the model:

```json
{
  "backbone_tuning": "frozen",
  "feat_dim": 256,
  "projection_layers": 3,
  "projection_hidden_dim": 1024
}
```

`projection_layers` counts linear layers: `N > 1` stacks `N - 1` hidden layers
separated by ReLU before the `feat_dim` output, so `3` gives
`Linear -> ReLU -> Linear -> ReLU -> Linear`. `projection_hidden_dim` sets the
hidden width and defaults to the DINO backbone dimension. `projection_layers`
above 1 requires a non-null `feat_dim`, and the L2 normalization applied to the
embedding is unchanged. The equivalent CLI flags are `--projection-layers` and
`--projection-hidden-dim`.

Backbone caching is unaffected: cached entries hold raw backbone features, so a
deeper head reuses the same cache.

Cached entries contain raw DINO backbone features, not final projected
embeddings. The projection head may therefore change between epochs or HPO
trials while the shared backbone cache remains valid. This makes the cache
particularly useful across repeated HPO trials and CV folds. Entries are stored
under:

```text
data/<dataset>/backbone_cache/v1/dinov2_vit<size>14/
```

Cache mode uses deterministic transforms for training, validation, test, and
SSL feature extraction. Training therefore bypasses RandAugment when
`use_cache` is enabled: DINO outputs are loaded from disk and gradients flow
only through the trainable projection head. Without `use_cache`, normal
training continues to use RandAugment.

For every pseudo-label method, including Iscen and mixed label propagation, the
complete post-split SSL training pool is materialized as one in-memory
raw-feature matrix. Each pseudo-label refresh projects that same matrix with the
current trainable head, so changing the head between epochs does not require
another DINO backbone pass. The same path is used for training samples, including
Iscen's optional two-stream sampler. HPO trials in the same process also share
the materialized matrix when their dataset split and backbone identity match.

To keep most of the frozen-feature speedup while retaining bounded training
augmentation, set `frozen_feature_train_views` above `1`, for example:

```json
{
  "use_cache": true,
  "backbone_tuning": "frozen",
  "feat_dim": 256,
  "frozen_feature_train_views": 5
}
```

This precomputes that many stochastic training-transform backbone features per
training sample in one extraction pass, then samples one cached view whenever
the projection head sees the sample. Validation and test features remain
deterministic. This is a finite augmentation bank, not fresh RandAugment every
epoch, so values around 5-10 are the intended range. Pseudo-label and compatible
regularization methods that use the full training pool always use one
deterministic view and ignore this setting. STML and SimMatchV2 cannot use the
cache because their objectives require stochastic multi-view image augmentation;
those configurations fail validation instead of silently skipping precompute.

Each transformed dataset sample gets its own cached raw DINO embedding,
identified by the transformed input tensor. Deterministic cache mode maximizes
reuse across HPO trials and CV folds. Augmented train-view mode intentionally
creates additional transformed inputs for the training set, trading more cache
entries and memory for bounded augmentation.
The DINO model weights themselves are not cached by this feature.

Every cache-enabled run writes `backbone_cache_stats.json` in its run directory
with sample hits, misses, writes, and hit rates.
The resolved `run_config.json` also records the cache directory and the
trainable/total model parameter counts.
