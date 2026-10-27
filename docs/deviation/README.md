# Deviation From `gmberton/image-retrieval`

This directory documents the important local deviations from the original
repository at:

```text
https://github.com/gmberton/image-retrieval
upstream comparison commit: a9f4f75da22b564049258e17afa12a387987c1f3
```

The upstream project is a compact supervised image-retrieval training script
with seven root files: `.gitignore`, `LICENSE`, `README.md`, `main.py`,
`requirements.txt`, `models/retrieval_model.py`, and `utils/`. It trains a DINOv2
embedding model on one selected retrieval dataset, one selected loss, and one
selected miner, then evaluates retrieval Precision@1 and MAP@R.

This fork has become a research experiment framework for semi-supervised and
label-budgeted metric learning. The highest-impact deviations are:

| Area | Upstream behavior | Local behavior |
|---|---|---|
| Execution model | One direct training run from CLI args | Direct runs, top-level JSON configs, outer experiment grids, Optuna HPO, supervised-vs-SSL comparison mode, CV, and final test retraining |
| Dataset scope | Cars196, CUB, iNaturalist2018, SOP through `pytorch_metric_learning` | Cars196, CUB, DeepFashion In-Shop, CIFAR-100, 224x224 iNaturalist2018/2019/2021-mini, published class-disjoint iNaturalist retrieval splits, Stanford Dogs, custom CIFAR protocols, validation modes, and external unlabeled image pools |
| Supervision | Fully supervised training only | Labeled-budget selection, pseudo-labeling, SSL warmup/update schedules, STML, LRML, and mixed label propagation |
| Model | DINOv2 plus optional projection | DINOv2 retry loading, backbone tuning policies, frozen-backbone feature cache, STML dual heads, device normalization |
| Training loop | One inline loop in `main.py` | Modular training, seeded loaders, sampler capacity checks, classification-style loss handling, configurable loss/miner kwargs, diagnostics |
| Evaluation and artifacts | Console/file logs and temporary best checkpoint | Structured metrics, split manifests, run configs, HPO trial tables, CV summaries, comparison summaries, pseudo-label diagnostics, PaCMAP artifacts |
| Tests and tools | No tests or analysis scripts | Unit tests for split, sampler, HPO, SSL, model-cache, and evaluation behavior plus reporting/inspection scripts |

## Topic Pages

- [Original baseline](original-baseline.md): what the upstream project does and
  which behavior is used as the comparison anchor.
- [Training orchestration](training-orchestration.md): config resolution,
  top-level dispatch, outer grids, HPO, CV, final retraining, and seed policy.
- [Datasets, splits, and label budgets](datasets-splits-label-budgets.md):
  dataset changes, CIFAR protocols, validation modes, CV splitters, and support
  selection.
- [Semi-supervised methods](semi-supervised-methods.md): SSL config shape,
  pseudo-label methods, STML, LRML, and confidence handling.
- [Model and training changes](model-and-training.md): DINO wrapper changes,
  optimizer/loss changes, backbone cache, sampler behavior, and Windows/runtime
  safety.
- [Evaluation, artifacts, and diagnostics](evaluation-artifacts-diagnostics.md):
  metrics, logging, manifests, HPO/CV output, visualizations, and failure
  checks.
- [Configs, scripts, tests, and dependencies](configs-scripts-tests-dependencies.md):
  added config trees, helper scripts, tests, requirements, and attribution.
- [Source map](source-map.md): changed files and directories mapped to their
  deviation topics.

## Stable Core Still Shared With Upstream

The fork still keeps the upstream core idea:

- images are embedded with DINOv2;
- embeddings are optionally projected and used for metric learning;
- training can use `pytorch_metric_learning` losses and miners;
- evaluation is retrieval evaluation over embeddings, not classifier accuracy;
- the main retrieval metrics remain Precision@1 and MAP@R.

Everything else in this directory describes where the fork expands, changes,
or constrains that baseline.
