# Training Orchestration

The upstream repository runs one supervised experiment directly from CLI
arguments. This fork turns that single path into a layered orchestration system
for repeatable experiments, comparisons, and hyperparameter search.

Primary files:

- [`main.py`](../../main.py)
- [`training/cli.py`](../../training/cli.py)
- [`training/hpo.py`](../../training/hpo.py)
- [`training/engine.py`](../../training/engine.py)
- [`training/types.py`](../../training/types.py)
- [`training/io.py`](../../training/io.py)

Related deep docs:

- [End-to-end run data flow](../run_data_flow.md)
- [Experiment config](../experiment_config.md)
- [Optuna hyperparameter search](../hyperparameter_search.md)
- [Cross-validation](../cross_validation.md)

## Config Resolution

Upstream has only argparse defaults and explicit CLI arguments. This fork adds
a top-level JSON experiment config:

```powershell
python main.py --experiment-config configs/experiments/class/cars196.json
```

Configuration precedence is:

1. argparse defaults from `training/cli.py`;
2. values loaded from `--experiment-config`;
3. explicit CLI overrides.

Experiment-config keys must match argparse destinations. Unknown keys raise a
`ValueError` instead of being ignored. JSON values are normalized through the
same argparse type and choice checks used by CLI values.

The top-level config can reference:

- `ssl_config`: a JSON file describing label selection and SSL behavior;
- `hparam_config`: a JSON file describing an Optuna study.

The resolved experiment, SSL, and HPO values are persisted in run/study
metadata.

## Top-Level Dispatch

Upstream always does:

```text
parse args -> train once -> test once
```

This fork dispatches by mode:

| Condition | Behavior |
|---|---|
| `compare_supervised_ssl=true` | Build matched scenarios, run separate supervised and SSL HPO studies, then final-evaluate both. |
| Outer grid fields are present | Expand label-budget, k-shot, loss/miner, sampling-mode, and seed scenarios. |
| Enabled `hparam_config` | Run or resume one Optuna study. |
| Otherwise | Run one direct training job. |

The outer grid is handled before HPO. Each grid point gets its own concrete SSL
JSON under the log directory so the exact scenario can be inspected later.

## Supervised Versus SSL Mode

The new `mode` argument separates the label-selection question from the SSL
method question:

- `mode="ssl"` keeps the configured SSL method.
- `mode="supervised"` converts the loaded SSL config to `method="none"` while
  preserving label budget, label sampling mode, k-shot, caps, and support seed.

This lets supervised baselines train on exactly the same labeled support set as
the SSL method, without generating pseudo-labels.

## Matched Supervised-vs-SSL Comparisons

The upstream project has no comparison runner. This fork adds
`--compare_supervised_ssl`, which runs a pair of studies per scenario:

- supervised baseline on the selected labeled subset;
- SSL run on the same selected labeled subset plus eligible unlabeled data.

The comparison code freezes scenario-defining values outside the HPO space so a
study cannot accidentally tune dataset protocol, seed policy, split mode, label
budget, or SSL method. Summary files record per-method results and SSL-minus-
supervised deltas.

## Outer Experiment Grids

The new outer grid can vary:

- `ssl_label_sampling_modes`;
- `label_budget_grid`;
- `k_shot_grid`;
- `loss_miner_grid`;
- `comparison_seeds`, with `comparison_seed_targets` selecting which seed
  channels the sweep changes.

`loss_miner_grid` entries are fixed pairs such as:

```text
TripletMarginLoss:TripletMarginMiner
ProxyAnchorLoss:no_miner
```

The grid validates impossible combinations early. For example,
classification-style losses must be paired with `no_miner`, and k-shot values
are only valid for `class_subset_k_shot`.

## Optuna HPO

Upstream has no HPO. This fork adds Optuna support with:

- persistent SQLite storage by default;
- resumable studies through `load_if_exists`;
- recovery of interrupted `RUNNING` trials;
- TPE, random, and grid samplers;
- optional median, successive-halving, and Hyperband pruners;
- class-qualified loss/miner constructor search spaces;
- nested SSL config search spaces;
- `batch_sampler` as a joint `batch_size:sampler_m` space;
- trial summaries in `trials.csv` and `trials.jsonl`.

Search-space keys can target:

- argparse fields such as `lr`;
- nested SSL values such as `ssl_config.method_params.n_neighbors`;
- selected loss parameters such as `loss.TripletMarginLoss.margin`;
- selected miner parameters such as
  `miner.TripletMarginMiner.type_of_triplets`.

Before a study starts, constructor-parameter categorical combinations are
checked against the selected loss/miner constructors. Invalid categorical
combinations are removed or collapsed into a valid joint categorical space.

## Cross-Validation

Upstream creates one train/validation class split. This fork can run one full
training job per fold when `cv_k > 1`.

Supported modes:

- `kfold`
- `group_kfold`
- `stratified_kfold`
- `stratified_group_kfold`
- `superclass_balanced_group_kfold`

Grouped modes use class labels as groups, preserving class-disjoint validation.
The CIFAR-100 superclass-balanced grouped mode additionally tries to keep
superclass coverage in each training fold.

CV output is written under:

```text
logs/<save_dir>/cv_<timestamp>/
```

with per-fold run directories, `cv_results.csv`, and `cv_summary.json`.

## Final Test After HPO

Upstream evaluates the validation-selected checkpoint from the same run on the
test set. This fork usually skips test evaluation during HPO and can run a
separate final evaluation via `final_test_after_hpo`.

For CV-backed HPO, the selected epoch counts from the best trial's folds are
averaged. A fresh model is then trained on the full development pool for that
fixed epoch count and evaluated once on the final test set.

The runner can final-evaluate:

- the best trial;
- the top N completed trials;
- explicit trial numbers.

This keeps test metrics out of the HPO objective while still producing final
test scores for selected configurations.

Study-directory replay additionally supports `cross_seed_train_val`. It treats
the configured comparison seeds like lightweight validation folds: the top-N
or explicitly selected trials are trained/validated on every seed, ranked by
their mean validation selection metric, and only the winner is retrained on the
full development pool and evaluated on the test set. Ordinary `train_val` uses
the same final-retraining rule but ranks candidates on one validation split.

## Seed Policy

Upstream has no explicit seed argument. This fork separates four seed roles:

- `seed`: runtime randomness such as model training and DataLoader generators;
- `data_split_seed`: dataset protocol, train/validation, and validation
  downsampling splits;
- `support_seed`: labeled support/sample selection;
- `hparam_seed`: Optuna sampler randomness.

By default, `data_split_seed` and `support_seed` are fixed independently from
the runtime seed. This lets run-seed sweeps change optimization randomness
without silently changing the train/test split or labeled support set.

For an outer `comparison_seeds` grid, `comparison_seed_targets` may contain any
non-empty subset of `seed`, `data_split_seed`, `support_seed`, and `hparam_seed`.
It defaults to all four for backward compatibility. For example,
`["seed", "support_seed"]` varies optimization/SSL randomness and the labeled
support draw while preserving the configured dataset split and HPO sampler seed.

## Result Types

The upstream script returns nothing. This fork uses immutable dataclasses for
machine-readable orchestration results:

- `TrainingResult`
- `HParamSearchConfig`
- `HParamStudyResult`
- `ComparisonScenario`

Those objects carry metrics, selected epochs, artifact paths, CV metadata, and
optional visualization paths through the orchestration stack.
