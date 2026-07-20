# Optuna Hyperparameter Search

This project supports Optuna-based hyperparameter search through `--hparam_config`.
The search layer wraps the normal training script: each Optuna trial resolves a set of CLI and SSL config parameters, runs a complete training job, stores the trial outputs, and reports one objective metric back to Optuna.

The default single-run behavior is unchanged when `--hparam_config` is omitted.

## Running

Run a search with the example config:

```powershell
python main.py --hparam_config configs/hparam_search.json --save_dir experiments/cars196_search
```

The command above writes the study under:

```text
logs/experiments/cars196_search/metric_learning_search/
```

The exact subdirectory name comes from:

- `--save_dir`: base directory under `logs/`
- `study_name`: study directory name when `study_dir` is not set

## Config Shape

The search config is a JSON object similar in spirit to the SSL config.
Search-wide Optuna settings live at the top level, while tuned parameters live under `spaces`.

```json
{
  "enabled": true,
  "n_trials": 20,
  "timeout": null,
  "direction": "maximize",
  "metric": "best_valid_mean_average_precision_at_r",
  "study_name": "metric_learning_search",
  "study_dir": null,
  "storage": null,
  "load_if_exists": true,
  "sampler": "tpe",
  "tpe_startup_trials": null,
  "sampler_params": {},
  "pruner": "none",
  "pruner_params": {},
  "spaces": {
    "lr": {
      "type": "float",
      "low": 1e-7,
      "high": 1e-4,
      "log": true
    },
    "batch_size": {
      "type": "categorical",
      "choices": [8, 16, 32]
    },
    "ssl_config.method_params.n_neighbors": {
      "type": "int",
      "low": 1,
      "high": 20
    }
  }
}
```

Top-level fields:

- `enabled`: if `false`, the config is loaded but the script runs a normal single training job.
- `n_trials`: total number of finished trials desired for the study.
- `timeout`: optional wall-clock timeout in seconds for the Optuna optimization call.
- `direction`: `maximize` or `minimize`.
- `metric`: objective metric returned to Optuna.
- `study_name`: stable Optuna study name and default output folder name.
- `study_dir`: optional explicit output directory. Relative paths are resolved under `logs/`.
- `storage`: optional Optuna storage URL. If omitted, the script creates a SQLite DB in the study directory.
- `load_if_exists`: when `true`, rerunning the same study resumes from the existing storage.
- `sampler`: `tpe`, `random`, or `grid`.
- `tpe_startup_trials`: optional number of random startup trials before TPE uses its model. `null` uses `sampler_params.n_startup_trials` or the Optuna default. `--tpe-startup-trials` / the experiment-config `tpe_startup_trials` key overrides this field.
- `sampler_params`: extra keyword arguments passed to the sampler.
- `pruner`: `none`, `median`, `successive_halving`, or `hyperband`.
- `pruner_params`: extra keyword arguments passed to the pruner.
- `spaces`: parameter search spaces.

Supported objective metrics:

- `best_valid_precision_at_1`
- `best_valid_mean_average_precision_at_r`
- `test_precision_at_1`
- `test_mean_average_precision_at_r`
- `final_train_loss`

`--selection_metric` controls which validation metric saves the checkpoint and resets early-stopping patience:

- `precision_at_1`
- `map_at_r`

The HPO config `metric` controls which completed trial Optuna selects. Prefer validation metrics for Optuna selection. Test metrics are recorded for analysis, but using them as an Optuna objective makes the test set part of tuning.

## Search Spaces

Search space keys are parameter names.
Plain names target argparse fields on `main.py`.
Nested SSL names start with `ssl_config.` and override the loaded SSL JSON after it is read.
`loss.<parameter>` and `miner.<parameter>` pass sampled values directly to the selected loss/miner constructor.
Class-qualified keys such as `loss.TripletMarginLoss.margin` apply only when that class is selected; nonmatching class-qualified spaces are removed from that scenario's Optuna study.
The exact keys `loss` and `miner` remain fixed training choices. Set them explicitly with `--loss` and `--miner`, or compare fixed pairs outside Optuna.
Before a study starts, categorical constructor parameters for each selected loss/miner are checked together against the real constructor. If some Cartesian-product combinations are invalid, the runner automatically samples from one joint categorical space containing only valid combinations. Invalid combinations therefore do not consume trials.
This automatic filtering applies to categorical choices. Continuous cross-parameter constraints cannot be exhaustively enumerated and should be represented through a categorical set of valid combinations instead.
Categorical constructor choices that are not scalar, such as range lists, are also converted into a joint serialized Optuna parameter so they can be stored safely.
Use the reserved `batch_sampler` key when `batch_size` and `sampler_m` must be sampled together.
Its choices are strings formatted as `"batch_size:sampler_m"`, for example `"32:16"`.
Do not include `batch_size` or `sampler_m` separately when using `batch_sampler`.

For Iscen's `TwoStreamMPerClassBatchSampler`, the top-level
`labeled_batch_size` HPO key is a convenience alias for
`ssl_config.labeled_batch_size`. When both it and `batch_sampler` use
categorical choices, the runner automatically turns their Cartesian product
into one joint space and removes every pair that does not satisfy
`labeled_batch_size <= batch_size - labeled_batch_size`. A 50/50 split is
therefore valid. It also removes pairs
where either stream size is not divisible by `sampler_m`, so invalid pairs do
not consume trials. Integer strings are accepted and normalized to integers:

```json
{
  "batch_sampler": {
    "type": "categorical",
    "choices": ["16:4", "32:4", "64:4", "128:4", "256:4", "512:8"]
  },
  "labeled_batch_size": {
    "type": "categorical",
    "choices": ["16", "32", "64", "128"]
  }
}
```

With these choices, Optuna sees only the 14 valid pairs. Trial summaries and
replay parameters expose the expanded `batch_sampler` and
`labeled_batch_size` values even though the study stores the valid pair as one
internal categorical parameter.

Examples:

```json
{
  "lr": {
    "type": "float",
    "low": 1e-7,
    "high": 1e-4,
    "log": true
  },
  "batch_sampler": {
    "type": "categorical",
    "choices": ["8:4", "16:8", "32:16"]
  },
  "ssl_config.confidence_threshold": {
    "type": "float",
    "low": 0.5,
    "high": 0.95
  },
  "ssl_config.method_params.n_neighbors": {
    "type": "int",
    "low": 1,
    "high": 20
  },
  "loss.TripletMarginLoss.margin": {
    "type": "float",
    "low": 0.01,
    "high": 0.5
  },
  "miner.TripletMarginMiner.type_of_triplets": {
    "type": "categorical",
    "choices": ["all", "hard", "semihard", "easy"]
  }
}
```

`BatchEasyHardMiner` has local validation for its strategy rules:

- `pos_strategy` and `neg_strategy` must each be one of `all`, `easy`, `hard`, or `semihard`.
- They cannot both be `semihard`.
- `semihard` cannot be paired with `all` in either direction.
- `allowed_pos_range` and `allowed_neg_range` must be `null` or a two-number range.

For an exhaustive miner-only grid over all valid strategy pairs and finite range choices, use:

```bash
python main.py --miner BatchEasyHardMiner --hparam_config configs/hparam_batch_easy_hard_miner.json
```

Supported space types:

- `categorical`: requires `choices`.
- `float`: requires `low` and `high`; supports `log` and `step`.
- `int`: requires integer `low` and `high`; supports `log` and `step`.

As a shorthand, a list is treated as a categorical space:

```json
{
  "batch_size": [8, 16, 32]
}
```

## Resume Behavior

Resuming is supported through persistent Optuna storage.
By default, when `storage` is omitted, the script writes:

```text
old_logs/<save_dir>/<study_name>/optuna_study.db
```

With `load_if_exists: true`, rerunning the same command with the same `--save_dir` and `study_name` loads the existing study.
Trial history is restored from the database and the sampler RNG/state is restored from `sampler.pkl`.

For `n_jobs > 1`, model training still runs in Optuna's worker threads, but parameter suggestion is serialized through one shared sampler. Optuna's per-thread sampler reseeding is suppressed, and `sampler.pkl` is atomically refreshed immediately after each trial's complete parameter set has been written to Optuna storage. This makes an interrupted parallel run continue from the saved sampler state instead of starting a newly reseeded sequence.

`n_trials` is interpreted as a total target, not as "run this many more".
For example, if the database already contains 7 complete or pruned trials and `n_trials` is 20, the next run schedules 13 more trials.

Interrupted trials are not resumed from the middle of training.
If a process stops during an epoch, the next run resets any unfinished Optuna `RUNNING` trials to `WAITING`.
Those trials are rerun with the same trial number and sampled parameters before new trials are suggested.
Per-trial training logs written before the interruption remain on disk, but the model training state is not checkpoint-resumed.
After reload, history-dependent samplers such as TPE combine the restored sampler state with the completed and pruned trials in the database, then incorporate rerun trials once they finish.

## Output Layout

For a search such as:

```powershell
python main.py --hparam_config configs/hparam_search.json --save_dir experiments/cars196_search
```

the study output directory is:

```text
old_logs/experiments/cars196_search/metric_learning_search/
```

Study-level files:

- `optuna_study.db`: SQLite storage for trial history and resume.
- `sampler.pkl`: atomic checkpoint of the Optuna sampler and its RNG state.
- `study_config.json`: base CLI args, hyperparameter config, resolved study name, and resolved storage URL.
- `trials.csv`: flat table of trial number, state, objective value, params, and scalar result attributes.
- `trials.jsonl`: one JSON object per trial, including params, user attrs, resolved args, resolved SSL config, timestamps, and duration.

Each trial also has its own normal training run directory:

```text
old_logs/<save_dir>/<study_name>/trial_0000/<timestamp>/
old_logs/<save_dir>/<study_name>/trial_0001/<timestamp>/
...
```

Trial-level files follow the standard training output format:

- `metrics.csv`: batch losses, epoch losses, validation metrics, and test metrics.
- `run_config.json`: resolved argparse values and resolved SSL config used for that trial.
- `info.log` and `debug.log`: loguru logs.
- `tensorboard/`: TensorBoard event files, including global-step training and
  validation curves plus the epoch-indexed validation analysis dashboard under
  `epoch/valid/`.

## Trial Tracking

The implementation records the following for each completed trial:

- Optuna trial number and state.
- Objective value.
- Suggested hyperparameters.
- Resolved argparse namespace.
- Resolved SSL config after nested overrides.
- Best validation Precision@1.
- Best validation MAP@R.
- Test Precision@1.
- Test MAP@R.
- Final train epoch loss.
- Last epoch and global step.
- Paths to the trial log directory and trial `metrics.csv`.

This gives two levels of traceability:

1. `trials.csv` and `trials.jsonl` compare trials across the whole study.
2. Each trial directory preserves the detailed training metrics and config for that one run.

## Samplers and Pruners

Available samplers:

- `tpe`: Optuna TPE sampler. This is the default Bayesian optimization style sampler.
- `random`: random search.
- `grid`: grid search over categorical spaces only.

Available pruners:

- `none`: disables pruning.
- `median`: Optuna median pruner.
- `successive_halving`: Optuna successive halving pruner.
- `hyperband`: Optuna Hyperband pruner.

During training, the script reports intermediate values for metrics available before test evaluation:

- `best_valid_precision_at_1`
- `best_valid_mean_average_precision_at_r`
- `final_train_loss`

Test metrics are only available after training finishes, so they are not useful for pruning.

## Interaction With SSL Config

The hyperparameter search config does not replace the SSL config.
The normal `--ssl_config` path is still loaded first, then any `ssl_config.*` entries in `spaces` override fields in memory for that trial.

Example:

```powershell
python main.py `
  --ssl_config configs/ssl_faiss_knn.json `
  --hparam_config configs/hparam_search.json
```

If the Optuna trial suggests:

```json
{
  "ssl_config.confidence_threshold": 0.72,
  "ssl_config.method_params.n_neighbors": 5
}
```

then that trial uses the base SSL config from `configs/ssl_faiss_knn.json`, but with those two values replaced.
The resolved SSL config is saved in the trial's `run_config.json` and in the study-level trial attributes.

## Supervised vs SSL Comparison Mode

Use `--compare_supervised_ssl` to run the Oliver-style comparison with two separate HPO studies:

```powershell
python main.py `
  --compare_supervised_ssl `
  --ssl_config configs/ssl_faiss_knn.json `
  --hparam_config configs/hparam_search.json `
  --save_dir experiments/cars196_10pct
```

This mode uses the label sampling settings from `--ssl_config` to define the fixed labeled subset `D_train`.
All remaining samples in the existing training split become unlabeled candidates `D_UL`.
The validation labels remain a separate held-out model-selection set.
With `--val_mode match_train`, that validation set is downsampled to roughly the same sample count and class count as `D_train`; with `--val_mode all`, all validation samples are used.
It then runs:

- supervised baseline: trains only on `D_train`; all `ssl_config.*` HPO spaces are removed from this study.
- SSL: trains on the same `D_train` plus `D_UL`; labels from `D_UL` are not used.

Both studies use the same validation set, test set, objective metric, and `n_trials`.
During HPO, test evaluation is disabled automatically.
After both studies finish, the best validation configuration from each study is retrained once more and evaluated on `D_test`.

When `--comparison_seeds` is supplied, `comparison_seed_targets` controls which
seed channels each value replaces. The available targets are:

- `seed`: training/runtime and SSL-method randomness;
- `data_split_seed`: dataset protocol and validation splits;
- `support_seed`: labeled support/sample selection;
- `hparam_seed`: Optuna sampler randomness.

Omitting `comparison_seed_targets` selects all four targets, preserving the
original behavior. In an experiment JSON config, a sweep that changes training
and labeled-support randomness while holding the dataset split and Optuna sampler
fixed looks like:

```json
{
  "comparison_seeds": [0, 1, 2, 3, 4],
  "comparison_seed_targets": ["seed", "support_seed"]
}
```

Each comparison-seed scenario runs the requested supervised/SSL HPO workflow
with its resolved seed channels.

Outputs are written under:

```text
logs/<save_dir>/<study_name>_supervised/
logs/<save_dir>/<study_name>_ssl/
logs/<save_dir>/final/supervised/
logs/<save_dir>/final/ssl/
logs/<save_dir>/supervised_ssl_comparison/
```

The comparison summary contains the final supervised metrics, final SSL metrics, and:

```text
delta_ssl_minus_supervised = metric(SSL) - metric(supervised)
```

For methodological consistency, comparison mode rejects HPO spaces that would change the dataset or split during tuning:

- `dataset`
- `mode`
- `seed`
- `hparam_seed`
- `data_split_seed`
- `support_seed`
- `cv_k`
- `cv_mode`
- `val_mode`
- `ssl_config.method`
- `ssl_config.label_sampling_mode`
- `ssl_config.labeled_fraction`
- `ssl_config.max_unlabeled_samples`
- `ssl_config.seed`

You can run the combined supervised-vs-SSL outer experiment grid directly:

```powershell
python main.py `
  --compare_supervised_ssl `
  --ssl_config configs/ssl_faiss_knn.json `
  --hparam_config configs/hparam_search.json `
  --label_budget_grid 0.01 0.05 0.10 0.25 0.50 `
  --ssl_label_sampling_modes per_class_min global_budget class_subset `
  --comparison_seeds 0 1 2 3 4 `
  --save_dir experiments/cars196_grid
```

To run the k-shot class-budget mode and use the outer grid for the number of labeled images per selected class:

```powershell
python main.py `
  --mode ssl `
  --ssl_config configs/ssl_faiss_knn_class_k_shot.json `
  --hparam_config configs/hparam_search.json `
  --label_budget_grid 0.5 `
  --k_shot_grid 1 2 5 `
  --ssl_label_sampling_modes class_subset_k_shot `
  --comparison_seeds 0 1 2 3 4 `
  --save_dir experiments/cars196_k_shot_grid
```

In `class_subset_k_shot`, `--label_budget_grid` values are selected-class fractions and `--k_shot_grid` values are k-shot counts. With `--label_budget_grid 0.5` and `--k_shot_grid 1 2 5`, the outer grid creates separate scenarios for 50% of training classes with one, two, and five labeled images per selected class. Each scenario runs its own Optuna study with the configured `n_trials`.

Do not include `ssl_config.labeled_per_class` in the HPO search space when using `--k_shot_grid`. The script rejects that setup because k-shot is controlled by the outer grid.

For CIFAR10, use a smaller sampler-safe HPO config because the dataset has only 10 classes:

```powershell
python main.py `
  --dataset CIFAR10 `
  --mode ssl `
  --ssl_config configs/ssl_faiss_knn_class_k_shot.json `
  --hparam_config configs/hparam_cifar10_k_shot.json `
  --label_budget_grid 0.5 1.0 `
  --k_shot_grid 1 2 4 `
  --ssl_label_sampling_modes class_subset_k_shot `
  --comparison_seeds 7 `
  --save_dir experiments/cifar10_k_shot_grid
```

`MPerClassSampler` needs `batch_size % sampler_m == 0` and `sampler_m * labeled_classes >= batch_size`. For each `class_subset_k_shot` outer-grid scenario, categorical `batch_sampler` and `sampler_m` spaces are filtered to `sampler_m <= k` before the Optuna study starts. For class-subset label budgets, joint `batch_sampler` choices are also filtered against every post-validation training fold so choices that cannot form a complete batch do not consume trials. Classes with fewer than `sampler_m` examples use the sampler's normal replacement behavior.

To compare fixed loss/miner pairs outside Optuna, add `--loss_miner_grid`:

```powershell
python main.py `
  --compare_supervised_ssl `
  --ssl_config configs/ssl_faiss_knn.json `
  --hparam_config configs/hparam_search.json `
  --label_budget_grid 0.01 0.05 0.10 1.0 `
  --ssl_label_sampling_modes global_budget `
  --loss_miner_grid MultiSimilarityLoss:MultiSimilarityMiner TripletMarginLoss:TripletMarginMiner `
  --comparison_seeds 0 1 2 3 4 `
  --save_dir experiments/cars196_grid
```

`--loss_miner_grid` entries are parsed as `LOSS:MINER`.
Use `no_miner` for losses that should run without a miner.
Classification losses such as `ProxyAnchorLoss` should be paired with `no_miner` because miners are ignored by the training loop for classification losses.

This creates one scenario per combination of label budget, label sampling mode, loss/miner pair, and seed.
Do not include the exact keys `loss` or `miner` in HPO spaces. The selected classes are fixed experiment choices, while their constructor parameters can be tuned with `loss.*` and `miner.*` keys.
For grid runs, keep `study_dir` and `storage` as `null` in the HPO config so each scenario gets separate Optuna storage.
Each scenario writes a resolved SSL config JSON under:

```text
logs/<save_dir>/comparison_grid/ssl_configs/
```

Per-scenario outputs are written under:

```text
logs/<save_dir>/<label_sampling_mode>_label_<budget>_seed_<seed>/
logs/<save_dir>/<label_sampling_mode>_label_<budget>_<loss>_<miner>_seed_<seed>/  # with --loss_miner_grid
logs/<save_dir>/class_subset_k_shot_label_<class_fraction>_k_<k>_seed_<seed>/
```

Grid summaries are written to:

```text
logs/<save_dir>/comparison_grid/grid_summary.csv
logs/<save_dir>/comparison_grid/grid_aggregate.csv
logs/<save_dir>/comparison_grid/grid_summary.json
```

The aggregate table reports mean and sample standard deviation across seeds for each `(label_sampling_mode, labeled_fraction, labeled_per_class, loss, miner)` group.

You can also run only one method over the same grid.
The default training mode is `supervised`, so `--mode supervised` can be omitted, but keeping it explicit is clearer in experiment scripts.
Supervised-only uses the SSL config only to define the split and disables SSL training:

```powershell
python main.py `
  --mode supervised `
  --ssl_config configs/ssl_faiss_knn.json `
  --hparam_config configs/hparam_search.json `
  --label_budget_grid 0.01 0.05 0.10 0.25 0.50 `
  --ssl_label_sampling_modes global_budget `
  --comparison_seeds 0 1 2 3 4 `
  --save_dir experiments/cars196_supervised_grid
```

SSL-only uses the same grid with `--mode ssl`:

```powershell
python main.py `
  --mode ssl `
  --ssl_config configs/ssl_faiss_knn.json `
  --hparam_config configs/hparam_search.json `
  --label_budget_grid 0.01 0.05 0.10 0.25 0.50 `
  --ssl_label_sampling_modes global_budget `
  --comparison_seeds 0 1 2 3 4 `
  --save_dir experiments/cars196_ssl_grid
```

Single-method grid summaries are written to:

```text
logs/<save_dir>/experiment_grid/grid_summary.csv
logs/<save_dir>/experiment_grid/grid_aggregate.csv
logs/<save_dir>/experiment_grid/grid_summary.json
```

Every training run also writes split artifacts:

```text
split/labeled_positions.npy
split/unlabeled_positions.npy
split/val_indices.npy
split/train_indices.npy
split/labeled_indices.npy
split/unlabeled_indices.npy
split/split_info.json
split/test_info.json
```

`split/split_info.json` includes the resolved `dataset_split.validation_mode`, including the original validation size and the selected validation size when `--val_mode match_train` is used.

## Recommended Workflow

1. Start with a small `n_trials` and low `--epochs` to validate that the search space is valid.
2. Inspect `trials.csv` for failed or pruned trials.
3. Increase `n_trials` in the same config and rerun the same command to resume.
4. Use validation metrics for Optuna selection.
5. Use the test metrics only once the search space and model selection procedure are fixed.

## Final Test Evaluation After HPO

Use `--final_test_after_hpo` to evaluate the best configuration from every
standalone or single-method-grid HPO study on the held-out test set:

```powershell
python main.py `
  --experiment-config configs/experiments/cifar100_config.json `
  --final_test_after_hpo
```

The equivalent experiment-config field is `"final_test_after_hpo": true`.
Set `"final_test_top_n": 5` to select the five completed HPO trials with the
best objective values. In `final_train` mode all five receive final fits; in
study-directory `train_val` and `cross_seed_train_val` modes, they are
validation candidates and only the winner receives a final fit. Normal
`final_test_after_hpo` also performs final fits directly. The default is `1`. Set
`"final_test_trial_numbers": [6, 11]` to choose explicit candidates instead of
selecting by objective value.

To run final-test evaluation later from an existing study directory without
resuming HPO, pass the literal study directory and reuse the same selectors:

```powershell
python main.py `
  --final_test_study_dir logs/path/to/study `
  --final_test_trial_numbers 6 11

python main.py `
  --final_test_study_dir logs/path/to/study `
  --final_test_top_n 5
```

This loads `study_config.json` and `trials.jsonl` from the study directory, so
you do not need to repeat the original dataset, SSL, or HPO command-line
arguments.

Study-directory replay has three selection modes:

- `final_train`: immediately performs a full-development train/test for every
  selected trial;
- `train_val`: replays the selected trials on one train/validation split,
  chooses the best validation result, then performs one full-development
  train/test for that winner;
- `cross_seed_train_val`: replays every selected trial over all requested
  `comparison_seeds`, chooses the trial with the best mean validation selection
  metric, averages that winner's selected epoch counts, then performs one
  full-development train/test.

For example, this evaluates the original study's top five trials over three
alternative runtime/validation-split seeds (15 validation runs), then runs one
final fit:

```json
{
  "final_test_study_dir": "old_logs/path/to/method/study",
  "study_dir_mode": "cross_seed_train_val",
  "final_test_top_n": 5,
  "comparison_seeds": [2, 7, 8],
  "comparison_seed_targets": ["seed", "data_split_seed"]
}
```

Use only `data_split_seed` when optimization randomness should remain fixed.
The final fit uses the request/study's baseline seeds, not an arbitrarily chosen
validation seed. A study directory represents one method; `loss_miner_grid` is
not applied during study-directory replay, so run this selection once for each
method's own study directory.

The HPO trials continue to skip the test set. After a study finishes, the
runner:

1. selects the best Optuna trial using the configured validation objective;
2. reads the validation-selected checkpoint epoch from each fold;
3. averages the corresponding training epoch counts and rounds to the nearest
   integer;
4. evaluates the freshly initialized epoch-0 model once on the held-out test
   set, before any optimizer step;
5. trains one fresh model with the best hyperparameters on the complete
   development/training pool, without validation or early stopping;
6. evaluates that final model once on the held-out test set.

For `train_val`, the winning replay's selected epoch supplies the final
duration. For `cross_seed_train_val`, the winner's selected epoch counts are
averaged across seed replays and rounded with the same policy as fold CV. Older
resumed studies that do not contain `selected_epoch` metadata fall back to their
recorded `last_epoch` when using direct `final_train`.

Each study directory receives:

```text
final_evaluation.json
final_evaluation.csv
```

When `final_test_top_n` is greater than 1, the best trial still writes the
legacy `final_evaluation.*` files. Additional selected trials write
`final_evaluation_trial_XXXX.*`, and the combined table is written to
`final_evaluation_top_N.*`.

When `final_test_trial_numbers` is set, each selected trial writes
`final_evaluation_trial_XXXX.*`, and the combined table is written to
`final_evaluation_selected_trials.*`.

These files record the best trial and parameters, fold epoch counts, mean and
rounded final epoch count, final run directory, final train loss, optimizer
steps, test Precision@1, and test MAP@R. The final run's `diagnostics.csv`
contains per-class test metrics.

Example smoke test:

```powershell
python main.py `
  --epochs 1 `
  --patience 1 `
  --hparam_config configs/hparam_search.json `
  --save_dir smoke/hparam_search
```

Example continuation:

```powershell
python main.py `
  --hparam_config configs/hparam_search.json `
  --save_dir smoke/hparam_search
```

As long as `study_name`, `study_dir`, and storage location are unchanged, the second command resumes the same study.

## Files Changed

- `main.py`: adds `--hparam_config`, Optuna config parsing, search-space suggestion, trial execution, study resume behavior, trial summaries, and run config export.
- `configs/hparam_search.json`: example Optuna search config.
- `requirements.txt`: adds `optuna`.
