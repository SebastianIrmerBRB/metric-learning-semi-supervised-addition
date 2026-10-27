# Cross-Validation

The training script supports optional sklearn cross-validation through `--cv_k` and `--cv_mode`.
When `--cv_k` is `1`, cross-validation is disabled and the existing holdout split is used.
When `--cv_k` is greater than `1`, the script runs one complete training job per fold and reports the mean fold metrics.

## CIFAR-10 Unseen-Class Protocol

Use `--dataset_protocol cifar10_unseen_classes` to evaluate CIFAR-10 with class-disjoint final testing.
This mode ignores the official CIFAR-10 train/test boundary and combines all 60,000 images before partitioning by class:

- development pool for training and validation: classes `0-7`
- fixed held-out final test set: classes `8-9`

The existing validation and cross-validation settings operate only inside the eight-class development pool.
The fixed test classes never enter training, validation, SSL pseudo-labeling, or cross-validation.

```powershell
python main.py --dataset CIFAR10 --dataset_protocol cifar10_unseen_classes --cv_k 4 --cv_mode group_kfold
```

## CIFAR Balanced-Fraction Protocol

Use `--dataset_protocol cifar_balanced_fraction` with CIFAR-10 or CIFAR-100 to
recombine the official train/test pools and create sample-disjoint balanced
subsets. The requested fractions are applied per class, so every class
contributes the same number of examples to development/training and final test.
Any remainder is unused.

```powershell
python main.py --dataset CIFAR10 `
  --dataset_protocol cifar_balanced_fraction `
  --cifar_train_fraction 0.6 `
  --cifar_test_fraction 0.2
```

The two fractions must be positive and sum to at most `1`. The selected training
fraction is the development pool; normal holdout validation, cross-validation,
or post-apportion validation is subsequently created from that pool.

## CIFAR-100 Class-Disjoint Protocols

Use `cifar100_fc100` for the exact TADAM FC100 superclass split. Its fixed
final test contains superclasses `0, 7, 12, 14`; the other 16 superclasses form
the development pool. With cross-validation disabled, the paper's canonical
12-superclass train and four-superclass validation split is used by `val_mode=all`
and `val_mode=match_train`. With four-fold
cross-validation, complete-superclass grouping is required:

```powershell
python main.py --dataset CIFAR100 `
  --dataset_protocol cifar100_fc100 `
  --cv_k 4 `
  --cv_mode superclass_group_kfold
```

Each fold then trains on 12 superclasses and validates on four, while the FC100
test superclasses never enter training, validation, SSL pseudo-labeling, or
cross-validation.

Use `cifar100_fine_class_disjoint` for a fixed 60/40 fine-class split that
places related classes from every superclass on both sides. Use
`cifar100_superclass_disjoint` for a harder fixed 10/10 superclass split with
no superclass overlap. Both pool the official splits first, yielding 600
images per fine class. The fine-class protocol uses 36,000 development and
24,000 final-test images; the superclass protocol uses 30,000 images on each
side.

The exact fixed class partitions are documented in
[CIFAR-100 class-disjoint protocols](cifar100_class_disjoint_protocols.md).

## Validation Modes

`--val_mode` controls how much validation data is used:

- `all`: default. Uses the full validation split, which is the current behavior.
- `match_train`: after the labeled/fractioned training split is created, downsamples validation to roughly the same number of samples and classes. For example, if the train loader uses 522 labeled/fractioned samples across 90 labels, validation uses about 522 samples across about 90 held-out validation labels.
- `split_after_apportion`: skips the initial holdout/CV validation split, apportions labels from the full training split first, then holds out 20% of the apportioned classes for validation. All source-training samples from held-out validation classes move to validation.

With the default holdout split, grouped cross-validation modes, and `split_after_apportion`, training and validation classes remain disjoint. `match_train` only samples within the existing validation classes; it does not move classes between train and validation.
Because retrieval metrics need positive pairs, `match_train` caps the selected validation label count when necessary so selected labels have at least two samples.
`split_after_apportion` excludes unlabeled candidates belonging to validation classes so SSL training cannot see held-out validation classes.

Example:

```powershell
python main.py --val_mode match_train
```

Post-apportion validation split:

```powershell
python main.py --val_mode split_after_apportion
```

## Running

Three-fold grouped cross-validation:

```powershell
python main.py --cv_k 3 --cv_mode group_kfold
```

Five-fold stratified sample-level cross-validation:

```powershell
python main.py --cv_k 5 --cv_mode stratified_kfold
```

Cross-validation can also be combined with Optuna:

```powershell
python main.py --hparam_config configs/hparam_search.json --cv_k 3 --cv_mode group_kfold
```

In Optuna mode, the objective value for a trial is the mean of the selected metric across completed folds.

With `--final_test_after_hpo`, cross-validation drives hyperparameter selection
and `--final_fit_mode` decides how the model measured on D_test is produced:

| mode | trains a final model? | epoch count comes from | test score is |
|---|---|---|---|
| `full_train` (default) | yes, on all development data | `--final_epoch_aggregation` over the folds | one model's score |
| `early_stop_holdout` | yes, minus a validation slice | its own early stopping | one model's score |
| `cross_validation_folds` | no | each fold's own early stopping | mean over `k` fold models |

### `full_train` (default): transfer an epoch count from the folds

Every fold stopped at its own selected epoch, so one duration has to stand in
for all of them. `--final_epoch_aggregation` chooses which:

- `mean` (default): the average fold duration, rounded half up. Follows every
  fold, so a single fold that ran far longer than the rest pulls it upward.
- `median`: the middle fold duration, rounded half up when the fold count is
  even. Ignores one unusually short or long fold entirely.
- `max`: the longest fold's duration. Trains for as long as any fold needed,
  which risks overfitting the folds that stopped much earlier.

The model then trains for that fixed duration on the complete
development/training pool with no validation data and no early stopping.

For fold durations `[3, 5, 5, 20]` the three settings give 8, 5, and 20 epochs.
The chosen aggregation, the per-fold durations, and the resulting epoch count
are all recorded in the `epoch_plan` of `final_evaluation.json`.

### `early_stop_holdout`: let the final fit choose its own duration

```powershell
python main.py --hparam_config configs/hparam_search.json --cv_k 3 `
  --final_test_after_hpo --final_fit_mode early_stop_holdout --holdout_val_ratio 0.1
```

This sidesteps the question of which fold duration to transfer. The final fit
keeps a class-disjoint validation slice out of the development pool, early-stops
on it with `--patience`, restores the checkpoint that slice selected, and
evaluates D_test once. `--epochs` becomes only an upper bound.

`--holdout_val_ratio` sizes that slice (for example `0.1` for 10% of the
development classes instead of the built-in 20%); leaving it unset keeps the
80/20 split. The cost is real: those classes are excluded from training, so the
final model sees less data than a `full_train` fit does. In exchange, no epoch
count has to be guessed, and the run reports genuine validation metrics of its
own rather than inheriting the study's.

`--holdout_val_ratio` also applies to any ordinary `--cv_k 1` run, so a direct
`--evaluate_test` run can use the same small-slice protocol without HPO.

### `cross_validation_folds`: test the folds themselves

```powershell
python main.py --hparam_config configs/hparam_search.json --cv_k 4 `
  --final_test_after_hpo --final_fit_mode cross_validation_folds
```

This is the protocol from *A Metric Learning Reality Check* (Musgrave et al.,
ECCV 2020), the one used by
[powerful-benchmarker](https://github.com/KevinMusgrave/powerful-benchmarker/tree/metric-learning).
It is the same idea as `early_stop_holdout`, applied inside cross-validation
instead of to one extra fit: no separate final model is trained at all.

The winning configuration's folds are trained once more with test evaluation
enabled. Each fold trains on `k-1` partitions, early-stops on its own held-out
partition, and that fold's validation-selected checkpoint is evaluated on
D_test. From the paper:

> For the best hyperparameters, the highest-accuracy checkpoint for each
> training set partition is loaded, and its embeddings for the test set are
> computed and L2 normalized.

Two numbers come out of the same `k` models, matching the paper's two columns:

- **Separated** — each fold's embeddings are scored on their own, giving `k`
  test accuracies. Reported as their mean plus the sample standard deviation
  across folds (`test_precision_at_1` / `test_precision_at_1_std`).
- **Concatenated** — each test sample's `k` embeddings are joined into one
  `cv_k * feat_dim` vector, L2 normalized, and evaluated once
  (`concatenated_test_precision_at_1`, with `concatenated_test_embedding_dim`).
  For `cv_k=4` and a 128-dim head this is the paper's "Concatenated (512-dim)".

The concatenated evaluation needs every fold's test embeddings, which are
written to `test_embeddings.npz` in each fold's log directory. Pass
`--no-concatenated_fold_test` to skip both the saving and the joint evaluation
and report only the per-fold mean.

Because each fold is renormalized before joining, no fold can dominate the
concatenated vector, and folds must have embedded the same test samples in the
same order — a mismatch is rejected rather than silently misaligned.

A plain cross-validation run reaches the same two numbers without HPO by asking
for the embeddings explicitly:

```powershell
python main.py --cv_k 4 --cv_mode group_kfold --evaluate_test --save_test_embeddings
```

This mode requires `cv_k > 1`. It is the only mode where the reported test
score is an average over fold models rather than one model's score, so
`test_precision_at_1` in its summary is a mean and should be read with its
`_std` next to it.

### Keeping the fitted models

The `k` models this protocol reports are thrown away once their scores are
written. `--save_final_head_state` keeps the final test models without also
keeping every HPO trial:

```powershell
python main.py --hparam_config configs/hparam_search.json --cv_k 4 `
  --final_test_after_hpo --final_fit_mode cross_validation_folds --save_final_head_state
```

In a run-scheduler manifest, put `"save_head_state": true` in the
`study_replay` block beside `final_test_visualization`. Use the broader
`--save_head_state` only when non-final standalone runs or HPO trials should
also keep their fitted heads.

Every run then writes the checkpoint its own validation selected to
`head_state.pt` in its log directory, and `head_state_path` joins the columns of
`cv_results.csv`. With `backbone_tuning='frozen'` — which the flag requires —
that file is the entire fitted model: the DINO backbone never moved, so the
projection head plus the run's own `run_config.json` fully determine its
embeddings. A single-layer `768 -> 768` head costs about 2.4 MB per fold, a
`768 -> 128` one about 400 KB. The trained criterion is deliberately not saved;
proxies and class weights produce no embeddings.

`--evaluate_saved_heads` then measures those heads again without training:

```powershell
python main.py --evaluate_saved_heads logs/<save_dir>/final/ssl
```

Point it at a `cv_<timestamp>` directory, at a single run directory, or at any
directory containing exactly one of either — a replay's `final/<role>` folder is
the usual choice. Each head is loaded into a model rebuilt from its run's
recorded configuration, embeds the same D_test that run embedded, and is scored
the same way; the folds are then joined into the concatenated evaluation. The
report lands next to the runs as `saved_head_evaluation.csv` and
`saved_head_evaluation.json`, reproducing both of the protocol's columns.

Only where the work runs may be redirected — `--device`, `--num_workers`,
`--use-cache` and their neighbours are taken from the current request when it
names them explicitly. Everything that decides *what* is measured (dataset,
protocol, seeds, folds) comes from the finished run, because those are what
made the test set the head was fitted against.

## Modes

The available modes map directly to sklearn splitters:

- `kfold`: `sklearn.model_selection.KFold`
- `group_kfold`: `sklearn.model_selection.GroupKFold`
- `stratified_kfold`: `sklearn.model_selection.StratifiedKFold`
- `stratified_group_kfold`: `sklearn.model_selection.StratifiedGroupKFold`
- `superclass_group_kfold`: CIFAR-100 `GroupKFold` whose groups are complete superclasses; use this for FC100-compatible folds
- `superclass_balanced_group_kfold`: CIFAR-100 grouped fold builder that holds out complete fine classes while ensuring every training fold keeps at least one fine class from each represented superclass

For grouped modes, this project uses the dataset class label as the group.
That means all samples from the same class stay in the same fold, so train and validation classes are disjoint.
This matches the existing metric-learning holdout split behavior.
When grouped cross-validation is combined with SSL and `split_after_apportion`, unlabeled candidates from validation classes are excluded from training and pseudo-labeling.

For non-grouped modes, splitting happens at the sample level.
Samples from the same class can appear in both train and validation folds.

## Output Layout

A cross-validation run writes an aggregate folder under the selected `--save_dir`:

```text
logs/<save_dir>/cv_<timestamp>/
```

Inside it:

- `cv_results.csv`: one row per completed fold plus a final `mean` row when all folds finish.
- `cv_summary.json`: run args, completed fold count, per-fold result metadata, and aggregate metrics.
- `fold_00/`, `fold_01/`, ...: normal per-fold training run directories.

`cv_summary.json` is rewritten after every fold, so an interrupted run still
reports the folds it finished. A study-directory replay uses that file to
continue in the same `cv_<timestamp>` directory at its first unfinished fold
(see `docs/hyperparameter_search.md`); every other run starts a new directory.

Each fold directory contains the same files as a normal training run, including:

- `metrics.csv`
- `run_config.json`
- `head_state.pt` (only with `--save_head_state`)
- `info.log`
- `debug.log` (only with `--debug_log`)
- `diagnostics.csv` (only when a diagnostic is enabled; see `--log_epoch_diagnostics`)
- `tensorboard/` (unless `--no-tensorboard`)

## Metrics

The aggregate cross-validation result uses means across folds for:

- `best_valid_precision_at_1`
- `best_valid_mean_average_precision_at_r`
- `test_precision_at_1`
- `test_mean_average_precision_at_r`
- `final_train_loss`

`last_epoch` is the maximum fold epoch, and `global_step` is the sum of fold steps.

## Constraints

Grouped modes require:

```text
cv_k <= number of classes
```

`superclass_group_kfold` instead requires `cv_k <= number of represented
superclasses`.

Stratified sample-level mode requires:

```text
cv_k <= samples in the smallest class
```

If these constraints are violated, the script raises a `ValueError` before the fold can run.
