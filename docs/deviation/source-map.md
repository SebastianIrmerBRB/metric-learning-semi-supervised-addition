# Source Map

This page maps the important local files and directories to the deviation they
implement. It is not a replacement for the topic pages; it is a quick path from
code to documentation.

## Modified Upstream Files

| Local path | Upstream status | Main deviation |
|---|---|---|
| [`main.py`](../../main.py) | Replaces upstream inline runner | Top-level dispatch, outer grids, supervised-vs-SSL comparisons, HPO final evaluation summaries |
| [`utils/`](../../utils) | Expands upstream utilities | Dataset protocols, split logic, CV, DataLoader safety, metrics logging, evaluation, PaCMAP/t-SNE, cache helpers |
| [`models/retrieval_model.py`](../../models/retrieval_model.py) | Expands upstream DINO wrapper | Retry loading, backbone tuning, frozen feature cache, STML heads |
| [`requirements.txt`](../../requirements.txt) | Expands upstream requirements | Optuna, sklearn, FAISS, TensorBoard, pandas, matplotlib, PaCMAP, CUDA wheel index |
| [`README.md`](../../README.md) | Replaces upstream README content | Local project notes, milestone list, documentation links, attribution |

## New Core Modules

| Local path | Role |
|---|---|
| [`training/cli.py`](../../training/cli.py) | Argparse definitions, experiment-config loading, seed normalization, config value validation |
| [`training/engine.py`](../../training/engine.py) | Single-run training, CV, mode resolution, losses/miners/optimizers, STML teacher handling, run artifacts |
| [`training/hpo.py`](../../training/hpo.py) | Optuna config, study creation/resume, trial parameter mapping, search-space validation, trial summaries |
| [`training/types.py`](../../training/types.py) | Shared constants and dataclasses for training, HPO, and comparison results |
| [`training/io.py`](../../training/io.py) | JSON/CSV serialization helpers for experiment artifacts |
| [`training/semi_supervised.py`](../../training/semi_supervised.py) | Label-budget splits, pseudo-label methods, SSL diagnostics, STML/LRML regularizer interfaces |
| [`training/ssl/`](../../training/ssl) | Focused SSL algorithms, data structures, sampling, diagnostics, and interfaces |
| [`losses/metric_losses.py`](../../losses/metric_losses.py) | Project-local `STMLLoss` and `MixedLabelPropagationProxyLoss` |
| [`utils/local_datasets.py`](../../utils/local_datasets.py) | CIFAR wrappers, streaming 224x224 iNaturalist 2018/2019/2021-mini and Stanford Dogs downloads, published class-disjoint iNaturalist retrieval splits, recursive unlabeled image dataset, local SOP behavior |

## Config And Documentation Trees

| Local path | Role |
|---|---|
| [`configs/`](../../configs) | Top-level experiment configs, SSL configs, supervised configs, HPO spaces |
| [`docs/`](..) | Detailed topic docs for run flow, HPO, CV, SSL methods, CIFAR protocols, loss flows, and DINO retry/cache behavior |
| [`docs/deviation/`](.) | High-level comparison against `gmberton/image-retrieval` |

## Analysis Scripts

| Local path | Role |
|---|---|
| [`class_distribution_report.py`](../../class_distribution_report.py) | Generates class/sample distribution reports for dataset protocols and label budgets |
| [`check_label_budget_classes.py`](../../check_label_budget_classes.py) | Audits labeled support class selection across seeds and budgets |
| [`check_cifar100_cv_superclass_coverage.py`](../../check_cifar100_cv_superclass_coverage.py) | Audits CIFAR-100 superclass coverage in split/fold protocols |
| [`check_seed_distributions.py`](../../check_seed_distributions.py) | Inspects seed-dependent class/superclass selections |
| [`evaluate_epoch0_test_sets.py`](../../evaluate_epoch0_test_sets.py) | Evaluates off-the-shelf/epoch-0 models across experiment configs |
| [`show_dataset_samples.py`](../../show_dataset_samples.py) | Exports sample images for dataset/split inspection |
| [`scripts/`](../../scripts) | Additional report, Optuna, and split-inspection helpers |

## Tests

| Local path | Coverage focus |
|---|---|
| [`tests/test_backbone_cache.py`](../../tests/test_backbone_cache.py) | Frozen-backbone cache constraints and behavior |
| [`tests/test_cifar10_unseen_protocol.py`](../../tests/test_cifar10_unseen_protocol.py) | CIFAR-10 class-disjoint protocol |
| [`tests/test_cifar100_unseen_protocol.py`](../../tests/test_cifar100_unseen_protocol.py) | CIFAR-100 class-disjoint protocols |
| [`tests/test_cifar_balanced_fraction_protocol.py`](../../tests/test_cifar_balanced_fraction_protocol.py) | CIFAR balanced sample-disjoint protocol |
| [`tests/test_dataloader_workers.py`](../../tests/test_dataloader_workers.py) | Platform-specific DataLoader worker resolution |
| [`tests/test_evaluate_nonfinite.py`](../../tests/test_evaluate_nonfinite.py) | Non-finite embedding checks |
| [`tests/test_experiment_config.py`](../../tests/test_experiment_config.py) | Top-level experiment config parsing and overrides |
| [`tests/test_final_hpo_evaluation.py`](../../tests/test_final_hpo_evaluation.py) | Final retraining/evaluation after HPO |
| [`tests/test_grouped_cv_leakage.py`](../../tests/test_grouped_cv_leakage.py) | Grouped CV class leakage prevention |
| [`tests/test_k_shot_sampler_constraints.py`](../../tests/test_k_shot_sampler_constraints.py) | k-shot and sampler-capacity validation |
| [`tests/test_label_sampling_modes.py`](../../tests/test_label_sampling_modes.py) | Label-budget support-selection modes |
| [`tests/test_loss_miner_hparams.py`](../../tests/test_loss_miner_hparams.py) | Loss/miner HPO parameter mapping and validation |
| [`tests/test_mixed_label_propagation.py`](../../tests/test_mixed_label_propagation.py) | Mixed label propagation math and integration behavior |
| [`tests/test_pacmap_plot_groups.py`](../../tests/test_pacmap_plot_groups.py) | Dataset-aware PaCMAP grouping |
| [`tests/test_tsne_visualization.py`](../../tests/test_tsne_visualization.py) | Reproducible t-SNE projection, artifact writing, graph layout, and result serialization |
| [`tests/test_sampler_epoch_length.py`](../../tests/test_sampler_epoch_length.py) | Epoch-length derivation for `MPerClassSampler` |
| [`tests/test_stml.py`](../../tests/test_stml.py) | STML loss/regularizer behavior |

## Reference And Attribution Material

| Local path | Role |
|---|---|
| [`LICENSE-image-retrieval`](../../LICENSE-image-retrieval) | Upstream MIT license retained under a renamed file |
| [`THIRD_PARTY_NOTICES.md`](../../THIRD_PARTY_NOTICES.md) | Attribution for additional adapted material |
| [`papers/`](../../papers) | Local papers used for implemented methods |
| [`STML-CVPR22-main/`](../../STML-CVPR22-main) | STML reference implementation/material |
| [`realistic-ssl-evaluation-master/`](../../realistic-ssl-evaluation-master) | Reference material for SSL evaluation |
