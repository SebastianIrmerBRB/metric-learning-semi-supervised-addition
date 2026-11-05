# Documentation

Guides to the methods, data flow, configuration, and evaluation used by this
project. Start with [experiment configuration](experiment_config.md) to run an
experiment, or [run data flow](run_data_flow.md) to understand the pipeline.

## Running experiments

| Guide | Topic |
| --- | --- |
| [Experiment configuration](experiment_config.md) | Settings, learning-rate schedules, and frozen feature caches |
| [Dataset protocols](dataset_protocols.md) | Development/test boundaries and long-tailed CIFAR |
| [CIFAR-100 class-disjoint protocols](cifar100_class_disjoint_protocols.md) | Fine-class, superclass, and FC100 splits |
| [Cross-validation](cross_validation.md) | Folds, validation, final fitting, and test evaluation |
| [Hyperparameter search](hyperparameter_search.md) | Optuna search spaces, resuming, and final retraining |
| [Parallel runs](parallel_runs.md) | Manifests, scheduling, replay, and concurrency |
| [Ablation configuration](ablation_config.md) | Controlled configuration variants |

## Methods and training

| Guide | Topic |
| --- | --- |
| [Run data flow](run_data_flow.md) | Configuration through training and evaluation |
| [Standard metric losses](standard_loss_data_flow.md) | Pair, tuple, and triplet loss inputs and updates |
| [Classification-style losses](classification_loss_data_flow.md) | Proxy/class-weight losses and their optimizers |
| [Semi-supervised integration](semi_supervised_sklearn.md) | Pseudo-labeling, warm-up, and method extension points |
| [LRML](lrml.md) | Graph construction, sampling, and Laplacian regularization |
| [Iscen label propagation](iscen_label_propagation_walkthrough.md) | Worked propagation example and training integration |
| [Mixed label propagation](mixed_label_propagation.md) | Similarity graphs, hard negatives, and confidence |
| [SERAPH](seraph.md) | Pair entropy, metric parameters, and trace regularization |
| [SLADE](slade.md) | Clustering, basis learning, pair mining, and CompCars pools |
| [SSMLPC](ssmlpc.md) | Local reconstruction regularization |
| [STML](stml.md) | Teacher/student training and external unlabeled pools |
| [STML threshold pseudo-labeling](stml_threshold.md) | Converting teacher relations into pseudo-labels |
| [Target-ratio aggregation](target_ratio_aggregation.md) | Gradient calibration and aggregation settings |

## Evaluation and diagnostics

| Guide | Topic |
| --- | --- |
| [Recall@k](recall_at_k.md) | Retrieval metrics and output locations |
| [Epoch tracking](epoch_tracking.md) | Embeddings, graph artifacts, and diagnostic outputs |
