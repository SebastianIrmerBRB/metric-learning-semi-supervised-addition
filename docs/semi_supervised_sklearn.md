# Semi-Supervised Integration

This project supports optional semi-supervised pseudo-labeling before the existing metric-learning training loop starts.
The training loop still optimizes the selected `pytorch_metric_learning` loss; SSL methods only decide which unlabeled training samples should be added with pseudo-labels.

## Running

Supervised run with an explicit seed:

```powershell
python main.py --seed 0
```

Sklearn label-spreading run:

```powershell
python main.py --seed 0 --ssl_config configs/ssl_label_spreading.json
```

Sklearn label-spreading with pseudo-label updates before every epoch:

```powershell
python main.py --seed 0 --ssl_config configs/ssl_label_spreading_every_epoch.json
```

Sklearn label-spreading with five labeled-only warmup epochs:

```powershell
python main.py --seed 0 --ssl_config configs/ssl_label_spreading_every_epoch_warmup.json
```

FAISS CPU k-nearest-neighbor pseudo-labeling:

```powershell
python main.py --seed 0 --ssl_config configs/ssl_faiss_knn.json
```

Omitting `--ssl_config` disables SSL and keeps the supervised baseline.

## Seed Behavior

`--seed` is the run-level seed and is always available, including supervised runs.
It is applied to Python, NumPy, Torch, CUDA, the train/validation class split, and PyTorch `DataLoader` worker seeding.

SSL configs inherit `--seed` by default.
Only add `seed` to the SSL JSON if a method should intentionally use a different split seed than the rest of the run.

Example override:

```json
{
  "method": "sklearn_label_spreading",
  "seed": 12,
  "labeled_fraction": 0.1,
  "method_params": {
    "kernel": "knn",
    "n_neighbors": 10
  }
}
```

The seed improves reproducibility, but exact bitwise determinism is not guaranteed for every CUDA operation.

## CUDA Multiprocessing Safety

PyTorch warns that CUDA is not safe with forked subprocesses after the CUDA runtime has been initialized.
This matters here because the model is moved to CUDA before train, validation, test, and SSL embedding `DataLoader`s are created.

The default is therefore:

```powershell
python main.py --dataloader_start_method spawn --num_workers 8
```

`spawn` avoids CUDA poison-fork issues when worker processes are enabled.
`forkserver` is also accepted on platforms that support it. On Windows, worker
counts are automatically resolved to `0` because spawning workers can fail
while serializing large nested dataset subsets. Configured worker counts remain
active on other platforms.

For the most conservative server run, disable DataLoader multiprocessing:

```powershell
python main.py --num_workers 0 --ssl_config configs/ssl_label_spreading.json
```

When SSL is enabled, the SSL config field `embedding_num_workers` controls the worker count for pseudo-label embedding extraction.
If you set `--dataloader_start_method default` or `fork` with CUDA and any worker count above zero, the script raises an error instead of starting an unsafe run.

## Config Shape

Method-specific parameters live in JSON instead of argparse so new SSL methods do not bloat the CLI.

```json
{
  "method": "sklearn_label_spreading",
  "update_mode": "once",
  "update_interval_epochs": 1,
  "warmup_epochs": 0,
  "label_sampling_mode": "global_budget",
  "labeled_fraction": 0.1,
  "labeled_per_class": null,
  "confidence_threshold": 0.8,
  "pseudo_label_diagnostics_mode": "save",
  "max_unlabeled_samples": null,
  "embedding_batch_size": 32,
  "embedding_num_workers": 8,
  "method_params": {
    "kernel": "knn",
    "n_neighbors": 10,
    "alpha": 0.2,
    "max_iter": 30
  }
}
```

Common fields:

- `method`: SSL method name. Use `none` or omit `--ssl_config` for supervised training.
- `update_mode`: shared rebuild cadence for SSL artifacts. `once` builds pseudo-labels, graphs, or STML sampling once when that phase starts; `every_epoch` rebuilds from the current model before each non-warmup epoch; `every_n_epochs` rebuilds once when the phase starts and then after each `update_interval_epochs` epochs.
- `update_interval_epochs`: positive interval used by `update_mode: "every_n_epochs"`. This key can also be tuned with HPO as `ssl_config.update_interval_epochs`.
- `warmup_epochs`: number of initial epochs trained only on the labeled SSL subset before pseudo-labels are generated.
- `label_sampling_mode`: controls how labeled samples are selected from the training split.
- `labeled_fraction`: budget used by `label_sampling_mode`.
- `labeled_per_class`: fixed labeled sample count per class. Supported with `label_sampling_mode: "per_class_min"` and required by `label_sampling_mode: "class_subset_k_shot"`.
- `confidence_threshold`: drops pseudo-labels below this confidence when the method provides confidences.
- `pseudo_label_rescue_confidence_floor`: for Iscen two-stream batches, the lower absolute confidence floor used only when the global threshold leaves too few predicted classes for the M-per-class pseudo stream.
- `pseudo_label_rescue_top_k`: maximum rejected candidates rescued from each newly admitted predicted class; `null` uses `sampler_m`.
- `pseudo_label_diagnostics_mode`: controls pseudo-label audit diagnostics for methods that generate pseudo-labels. Use `save` to write `pseudo_label_diagnostics.jsonl`, `log` to only log summaries, or `off` to skip diagnostics.
- `max_unlabeled_samples`: optional cap applied after all non-labeled training samples are selected as unlabeled candidates.
- `embedding_batch_size`: batch size for extracting DINO embeddings used by SSL.
- `embedding_num_workers`: worker count for SSL embedding extraction.
- `method_params`: parameters passed to the selected SSL implementation.

Pseudo-label methods such as sklearn label spreading, FAISS label spreading,
and mixed label propagation apply the cadence to pseudo-label regeneration.
Graph regularizers such as LRML/SLRML apply the same cadence to graph rebuilds.
STML applies it to nearest-neighbor sampler rebuilds.

Label sampling modes:

- `global_budget`: keeps at most `floor(labeled_fraction * train_size)` labeled images, assigned across classes in a balanced round-robin order. For SOP with `labeled_fraction: 0.01`, this gives 476 labeled images for the current 47,638-image internal training split.
- `class_subset`: selects `floor(labeled_fraction * num_train_classes)` classes and labels all images from those classes. This creates a smaller normal fully labeled class-disjoint training subset; for SOP with `labeled_fraction: 0.01`, this selects 90 classes in the current internal training split.
- `class_subset_k_shot`: selects `floor(labeled_fraction * num_train_classes)` classes and labels up to `labeled_per_class` images from each selected class. For `labeled_fraction: 0.01` and `labeled_per_class: 1`, this selects 1% of training classes and labels one image per selected class.
- `per_class_min`: For every training class, keeps `max(1, round(labeled_fraction * class_size))` labeled images, or `labeled_per_class` if set. On SOP, small classes mean low fractions still keep at least one image for every class.

Available methods:

- `faiss_knn`
- `sklearn_label_spreading`
- `sklearn_label_propagation`

## Data Flow

1. `main.py` parses run args and calls `utils.seed_everything(args.seed)`.
2. `utils.setup_dataset_bundle(...)` builds train, validation, and test datasets.
3. `semi_supervised.prepare_ssl_split(...)` creates one fixed labeled subset inside the training classes when SSL is enabled; all remaining training samples become unlabeled candidates.
4. If SSL is disabled, the original train dataset is used.
5. For the first `warmup_epochs`, training uses only the labeled part of the SSL split.
6. With `update_mode: "once"`, the SSL artifact is built once, either before training or immediately after warmup.
7. With `update_mode: "every_epoch"` or `update_mode: "every_n_epochs"`, pseudo-labels, graphs, or STML sampling are regenerated from the current model on that cadence before a non-warmup epoch.
8. Pseudo-label methods extract current DINO embeddings and return pseudo-labels for unlabeled positions.
9. `RelabeledSubset` exposes labeled and selected pseudo-labeled samples through the same interface expected by the existing sampler and training loop.

Validation and test data are not used for pseudo-labeling.

## Validation Data Size

The default `--val_mode all` keeps the current behavior and evaluates on the full validation split.
Use `--val_mode match_train` when the validation set should mirror the labeled/fractioned training size and class count.
For example, if the SSL/supervised split produces 522 labeled training samples across 90 labels, `match_train` samples about 522 validation examples across about 90 existing validation classes.
With the default holdout split, train and validation classes remain disjoint.
Use `--val_mode split_after_apportion` when the labeled/fractioned training subset should be selected before validation is carved out. This holds out 20% of the apportioned classes for validation and removes those validation classes from the unlabeled pool.

## Extension Points

New SSL methods should be added in `training/semi_supervised.py`, with focused algorithms and support code placed in `training/ssl/`.

Implement a class with:

```python
def generate_pseudo_labels(self, model, train_dataset, split, device, config, epoch=None, start_method="spawn"):
    ...
    return PseudoLabelResult(positions=positions, mapped_labels=labels, confidences=confidences)
```

Then register it:

```python
METHOD_REGISTRY["my_method"] = MyMethod()
```

The method should return mapped training labels, not original dataset class IDs.
The common filtering code drops pseudo-labels outside known training classes and applies `confidence_threshold` when confidences are available.

## Files Changed

- `main.py`: adds `--seed`, `--ssl_config`, CUDA-safe DataLoader settings, seed initialization, and SSL dataset expansion.
- `utils/`: adds seed helpers, CUDA-safe DataLoader validation, seeded dataset splitting, seeded loaders, and reusable dataset-bundle setup.
- `training/semi_supervised.py`: contains warmup and update-mode handling, the method registry, and method orchestration; focused configuration, algorithms, pseudo-label filtering, and data wrappers live in `training/ssl/`.
