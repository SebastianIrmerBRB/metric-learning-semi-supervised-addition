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

With `--final_test_after_hpo`, cross-validation is used only for hyperparameter
and epoch selection. The selected checkpoint epoch counts from the best trial's
folds are averaged, then one fresh model is trained for that fixed duration on
the complete development/training pool and evaluated once on the final test
set. The final test score is therefore not an average of fold test scores.

## Modes

The available modes map directly to sklearn splitters:

- `kfold`: `sklearn.model_selection.KFold`
- `group_kfold`: `sklearn.model_selection.GroupKFold`
- `stratified_kfold`: `sklearn.model_selection.StratifiedKFold`
- `stratified_group_kfold`: `sklearn.model_selection.StratifiedGroupKFold`
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

Each fold directory contains the same files as a normal training run, including:

- `metrics.csv`
- `run_config.json`
- `info.log`
- `debug.log`
- `tensorboard/`

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

Stratified sample-level mode requires:

```text
cv_k <= samples in the smallest class
```

If these constraints are violated, the script raises a `ValueError` before the fold can run.
