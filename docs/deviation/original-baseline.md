# Original Baseline

This page records the comparison anchor: `gmberton/image-retrieval` at commit
`a9f4f75da22b564049258e17afa12a387987c1f3`.

## Upstream Scope

The original repository is intentionally small. Its README describes it as a
simple way to run metric-learning experiments over retrieval datasets using
`pytorch_metric_learning`. The repository consists of:

- `main.py`: CLI parsing, model construction, dataset setup, loss/miner setup,
  training, validation, early stopping, and test evaluation.
- `models/retrieval_model.py`: DINOv2 wrapper with optional linear projection and L2
  normalization.
- `utils/`: logging, dataset loading/splitting, DataLoaders, and retrieval
  evaluation.
- `requirements.txt`: `loguru`, `numpy`, `pytorch_metric_learning`, `torch`,
  `torchvision`, and `tqdm`.

The upstream project does not have separate experiment config files, Optuna
integration, semi-supervised learning, tests, structured artifact summaries, or
custom dataset protocols.

## Upstream CLI

The original `main.py` exposes a small CLI:

- `batch_size`
- `lr`
- `classifier_lr`
- `sampler_m`
- `dataset`
- `dino_size`
- `loss`
- `miner`
- `feat_dim`
- `device`
- `optim`
- `save_dir`

Supported datasets are:

- `Cars196`
- `CUB`
- `INaturalist2018`
- `StanfordOnlineProducts`

Optimizers are `adam` and `rmsprop`. The device choices are only `cuda` and
`cpu`.

## Upstream Data Flow

The original training flow is:

1. Initialize loguru logging under `logs/<save_dir>/<timestamp>/`.
2. Build `DinoWrapper`.
3. Download/load the selected dataset using `pytorch_metric_learning.datasets`.
4. Split the provider train split into train and validation by class with an
   80/20 class split.
5. Use the provider test split as the final test set.
6. Train with `MPerClassSampler`.
7. Evaluate the off-the-shelf model on validation before training.
8. Train for up to 100 epochs with patience `3`.
9. Save only `model.state_dict()` on validation Precision@1 improvement.
10. Restore the selected model and evaluate once on the test set.
11. Delete the temporary `best_model.pth`.

Validation and test evaluation compute:

- `precision_at_1`
- `mean_average_precision_at_r`

## Upstream Model

The upstream `DinoWrapper`:

- loads `facebookresearch/dinov2` through `torch.hub.load`;
- supports DINO sizes `s`, `b`, `l`, and `g`;
- optionally adds one linear projection when `feat_dim` is not `None`;
- resizes image height/width to the nearest multiple of 14;
- L2-normalizes the final embedding.

There is no retry logic, no frozen-backbone policy, no cache, and no separate
heads for semi-supervised losses.

## Upstream Loss Behavior

The original loop has two loss branches:

- Classification-style metric losses receive `num_classes` and `embedding_size`
  and get a separate criterion optimizer.
- Standard metric losses receive embeddings and labels, and may receive miner
  outputs.

The classification branch does not call miners. The standard branch optionally
calls a miner unless `miner=no_miner`.

## Upstream Limitations That This Fork Addresses

The fork primarily expands areas that the upstream project intentionally kept
minimal:

- experiment reproducibility beyond one CLI invocation;
- fair label-budgeted comparisons;
- semi-supervised pseudo-label generation;
- cross-validation and final-test protocols;
- CIFAR-based controlled protocols;
- hyperparameter search and final retraining;
- structured diagnostics and tests.
