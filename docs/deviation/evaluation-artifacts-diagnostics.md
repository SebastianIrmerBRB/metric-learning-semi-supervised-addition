# Evaluation, Artifacts, And Diagnostics

The upstream project logs to stdout and two log files, saves a temporary best
model during training, evaluates retrieval metrics, and deletes the temporary
checkpoint after test evaluation. This fork adds structured artifacts for
reproducibility, analysis, and failure diagnosis.

Primary files:

- [`utils/`](../../utils)
- [`training/engine.py`](../../training/engine.py)
- [`training/hpo.py`](../../training/hpo.py)
- [`main.py`](../../main.py)
- [`evaluate_epoch0_test_sets.py`](../../evaluate_epoch0_test_sets.py)
- [`class_distribution_report.py`](../../class_distribution_report.py)

Related deep docs:

- [End-to-end run data flow](../run_data_flow.md)
- [Optuna hyperparameter search](../hyperparameter_search.md)
- [Cross-validation](../cross_validation.md)

## Metrics

Upstream evaluates:

- `precision_at_1`;
- `mean_average_precision_at_r`.

This fork keeps those metrics and tracks them in more places:

- initial validation or test where configured;
- per-epoch validation;
- optional direct-run test;
- HPO objective results;
- CV fold and aggregate results;
- final retraining after HPO;
- per-class retrieval metrics where requested.

TensorBoard retains the original optimizer-step curves and also provides an
`Epoch analysis` dashboard. Its epoch-indexed validation view contains:

- Precision@1 and MAP@R;
- validation embedding extraction, retrieval calculation, diagnostic, and
  total durations;
- validation sample, class, batch, and embedding-dimension counts;
- validation embedding-norm minimum, mean, standard deviation, and maximum;
- per-class sample-count, Precision@1, and MAP@R histograms;
- per-class minimum, macro mean, standard deviation, and maximum retrieval
  metrics.

The epoch dashboard uses tags below `epoch/valid/`; existing tags below
`valid/` continue to use global optimizer steps for backward compatibility.
The off-the-shelf validation baseline is recorded at epoch `-1`.

No validation loss is reported. Retrieval validation first embeds the complete
validation set and then evaluates global nearest-neighbor rankings. Applying a
metric-learning training loss to ordinary sequential validation batches would
be batch-composition dependent and is undefined for objectives that require a
particular positive/negative or multi-view sampling structure.

The selected validation metric is configurable:

- `precision_at_1`;
- `map_at_r`.

## Structured Run Artifacts

A normal run can write:

- `info.log`;
- `debug.log` (only with `--debug_log`);
- `metrics.csv`;
- `diagnostics.csv` (only when a diagnostic category is enabled);
- `run_config.json`;
- split manifest JSON;
- `best_model.pth` during training;
- optional `backbone_cache_stats.json`;
- optional `pseudo_label_diagnostics.jsonl`;
- optional TensorBoard event files.

`run_config.json` records resolved argparse values, SSL config values, trainable
parameter counts, and cache metadata where applicable.

The split manifest records the exact development, validation, test, labeled,
and unlabeled positions needed to audit a run.

## HPO Artifacts

Upstream has no study-level output.

For an Optuna study, this fork writes:

- `optuna_study.db`;
- `study_config.json`;
- `trials.csv`;
- `trials.jsonl`;
- one normal run directory per trial.

Trial summaries include:

- trial number and state;
- objective value;
- sampled parameters;
- resolved argparse namespace;
- resolved SSL config;
- scalar validation/test/train metrics;
- trial log directory and metrics path;
- timestamps and duration.

## CV Artifacts

For cross-validation, this fork writes:

- one run directory per fold;
- `cv_results.csv`;
- `cv_summary.json`;
- aggregate mean metrics when all folds complete.

Optuna can receive intermediate reports from CV as fold results finish.

## Comparison Artifacts

Outer-grid and supervised-vs-SSL comparison modes write summary tables for:

- each scenario;
- supervised result;
- SSL result;
- final evaluation result;
- SSL-minus-supervised metric deltas;
- aggregate rows across seeds and scenario dimensions.

This is a major deviation from upstream's one-run-only logging model.

## PaCMAP and t-SNE Visualizations

The fork can write either PaCMAP or t-SNE embedding visualizations for final
test evaluation. Select the projection with `final_test_visualization` set to
`"pacmap"`, `"tsne"`, or `"none"`. The result dataclass can carry:

- `test_pacmap_coordinates`;
- `test_pacmap_plot`;
- `test_tsne_coordinates`;
- `test_tsne_plot`.

Plot grouping can use dataset-specific metadata such as CIFAR-100 superclasses
or SOP coarse classes when available. t-SNE uses the experiment seed and
automatically lowers its perplexity for small sample sets.

## Epoch-0 Test Evaluation

`evaluate_epoch0_test_sets.py` is a standalone helper for evaluating the
untrained or off-the-shelf model on final test sets derived from experiment
configs. It can also write PaCMAP artifacts with `--pacmap`, t-SNE artifacts
with `--tsne`, or both.

This is separate from the upstream script's initial validation check and is
intended for baseline inspection across multiple dataset protocols.

## Diagnostics

The fork adds diagnostics that do not exist upstream:

- mined pair/triplet counts;
- gradient norms;
- optimizer learning rates;
- pseudo-label counts, confidence summaries, audit accuracy, and stability;
- cache hit/miss/write statistics;
- non-finite embedding detection;
- class-distribution and split-overlap reports.

`NonFiniteEmbeddingError` is raised if evaluation embeddings contain NaN or
infinite values, preventing invalid retrieval metrics from being logged as if
they were meaningful.

## Test Evaluation Policy

Upstream always evaluates test once after training.

This fork separates policies:

- direct runs evaluate test only with `evaluate_test=true` or `--evaluate-test`;
- HPO trials usually skip test through `skip_test_during_hpo`;
- selected HPO trials can be evaluated through `final_test_after_hpo`.

The intent is to keep the final test set out of model and hyperparameter
selection while still supporting final test reporting.
