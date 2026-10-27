# Configs, Scripts, Tests, And Dependencies

The upstream repository has no config tree, no tests, and only a minimal
requirements file. This fork adds experiment configuration files, analysis
scripts, generated reports, tests, extra dependencies, local papers, and
third-party attribution.

## Config Tree

Upstream examples are CLI commands in the README. This fork adds JSON configs
under [`configs/`](../../configs).

Important groups:

- `configs/experiments/`: top-level experiment configs for datasets, protocols,
  HPO, and SSL methods.
- `configs/experiments/class/`: classification-style metric-loss experiment
  variants.
- `configs/supervised/`: supervised classification and contrastive baselines.
- `configs/ssl_*.json`: SSL method configs.
- `configs/hparam*.json`: Optuna search spaces.

These configs are first-class inputs to the runner through `--experiment-config`,
`--ssl_config`, and `--hparam_config`.

## Analysis And Utility Scripts

Added root-level and `scripts/` helpers include:

- `class_distribution_report.py`: dataset/split/support distribution report.
- `check_label_budget_classes.py`: support-set class overlap across seeds and
  budgets.
- `check_seed_distributions.py`: seed distribution inspection.
- `check_cifar100_cv_superclass_coverage.py`: CIFAR-100 fold superclass audit.
- `evaluate_epoch0_test_sets.py`: off-the-shelf test evaluation across configs.
- `show_dataset_samples.py`: sample image export for split inspection.
- `scripts/optuna_param_importance.py`: Optuna parameter-importance analysis.
- `scripts/read_split.py`: split-manifest inspection.

Some generated reports and charts are present under `scripts/reports/`.

## Tests

Upstream has no tests. This fork adds tests under [`tests/`](../../tests) for:

- backbone cache behavior;
- CIFAR-10 and CIFAR-100 unseen-class protocols;
- balanced-fraction CIFAR protocol;
- DataLoader worker resolution;
- non-finite embedding evaluation failures;
- experiment-config parsing;
- final HPO evaluation;
- grouped CV leakage;
- k-shot sampler constraints;
- label sampling modes;
- loss/miner HPO parameters;
- mixed label propagation;
- PaCMAP plot grouping and t-SNE projection;
- sampler epoch length;
- STML behavior.

The tests focus on the new framework logic, especially cases where silent data
leakage or invalid label-budget batches would compromise experiments.

## Requirements

Upstream requirements:

```text
loguru
numpy
pytorch_metric_learning>=2.8.1
torch
torchvision
tqdm
```

This fork adds:

- CUDA wheel index entries for local Windows/CUDA setup;
- `matplotlib`;
- `pandas`;
- `pacmap`;
- `scikit-learn`;
- `faiss-cpu`;
- `tensorboard`;
- `optuna`.

These dependencies support SSL graph methods, FAISS kNN pseudo-labeling,
Optuna HPO, PaCMAP and t-SNE visualization, TensorBoard logging, and tabular
reports.

## Documentation

Beyond this deviation directory, the fork adds topic docs under [`docs/`](..):

- run data flow;
- loss-specific data flows;
- experiment config;
- hyperparameter search;
- cross-validation;
- CIFAR protocols;
- long-tail CIFAR;
- DINOv2 retry behavior;
- sklearn SSL;
- mixed label propagation;
- STML;
- sampler epoch length.

The deviation docs are intentionally higher level and link into those detailed
pages rather than duplicating every equation or command.

## Third-Party Attribution

The original MIT license is retained as `LICENSE-image-retrieval`.

The fork adds [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) for
additional adapted material, including long-tailed CIFAR generation from
`richardaecn/class-balanced-loss`.

The repository also contains local reference material such as:

- `papers/Deep_Semi-Supervised_Metric_Learning_with_Mixed_Label_Propagation.pdf`;
- `STML-CVPR22-main/`;
- `realistic-ssl-evaluation-master/`.

Those directories are reference or comparison material, not part of the
original `gmberton/image-retrieval` codebase.
