# Semi-Supervised Methods

The upstream project has no semi-supervised learning path. This fork adds a
semi-supervised layer that first selects labeled support samples, then either
generates pseudo-labels for unlabeled samples or applies an unlabeled
regularizer during training.

Primary files:

- [`training/semi_supervised.py`](../../training/semi_supervised.py)
- [`training/ssl/`](../../training/ssl)
- [`losses/metric_losses.py`](../../losses/metric_losses.py)
- [`training/engine.py`](../../training/engine.py)
- [`configs/`](../../configs)

Related deep docs:

- [Semi-supervised sklearn graph baselines](../semi_supervised_sklearn.md)
- [Deep mixed label propagation](../mixed_label_propagation.md)
- [STML training](../stml.md)

## SSL Config

Semi-supervised behavior is configured through a JSON file referenced by
`ssl_config`.

Common fields:

- `method`
- `update_mode`
- `warmup_epochs`
- `label_sampling_mode`
- `labeled_fraction`
- `labeled_per_class`
- `seed`
- `support_seed`
- `confidence_threshold`
- `max_unlabeled_samples`
- `embedding_batch_size`
- `embedding_num_workers`
- `method_params`

Top-level SSL keys are validated against the `SemiSupervisedConfig` dataclass.
Method-specific options must live under `method_params`.

## Labeled/Unlabeled Split

The SSL pipeline uses positions inside the current training subset:

```text
current training subset
-> labeled support positions
-> unlabeled candidate positions
```

True labels for unlabeled candidates are hidden from SSL generation and
regularizers. Hidden labels may be used only after prediction for diagnostics.

The split is reused for supervised and SSL runs in matched comparisons so both
methods receive the same known-label support set.

## Pseudo-Label Dataset Contract

Accepted pseudo-labels are combined with true-labeled samples through
`RelabeledSubset`.

`RelabeledSubset` exposes:

- `labels`: dense mapped labels for `MPerClassSampler`;
- `orig_labels`: original class IDs or pseudo-label class IDs returned to the
  training loop;
- `confidences`: per-sample weights in `[0, 1]`.

Ordinary losses ignore confidences. Confidence-aware losses, currently
`MixedLabelPropagationProxyLoss`, consume them as sample weights.

## Update Modes And Warmup

The upstream loop always trains on one fixed dataset.

This fork supports:

- `update_mode="once"`: generate pseudo-labels once before training, or once
  after warmup.
- `update_mode="every_epoch"`: regenerate pseudo-labels from the current model
  before every non-warmup epoch.
- `warmup_epochs`: train only on true-labeled support before pseudo-labeling or
  loss-driven SSL begins.

Regularization methods and loss-driven STML require `update_mode="once"` even
though their regularizer batches may be rebuilt by the training loop.

## Registered Pseudo-Label Methods

`METHOD_REGISTRY` currently includes:

| Method key | Behavior |
|---|---|
| `faiss_majority_vote_knn` | Embeds labeled and unlabeled samples, finds labeled nearest neighbors with FAISS, and assigns majority-vote labels. |
| `self_training_knn` | Implements Sahito et al. Algorithm 1 as cumulative exact squared-Euclidean 1-NN promotion rounds. |
| `sklearn_label_spreading` | Uses sklearn `LabelSpreading` over extracted embeddings. |
| `sklearn_label_propagation` | Uses sklearn `LabelPropagation` over extracted embeddings. |
| `mixed_label_propagation` | Implements sparse mixed label propagation from Zhuang and Moulin, CVPR 2023. |
| `regularized` | Composes a supervised loss with a registered unlabeled regularizer instead of generating pseudo-labels. |

`none` disables SSL. `stml` is handled as a loss-driven method for faithful
legacy STML experiments.

## Confidence Filtering And Diagnostics

Pseudo-label methods return `PseudoLabelResult`:

- positions;
- mapped predicted labels;
- optional confidences.

Common filtering:

- drops predictions outside the known training label set;
- applies `confidence_threshold` when confidences are available;
- enforces optional `max_unlabeled_samples`.

`PseudoLabelDiagnosticsTracker` writes `pseudo_label_diagnostics.jsonl` with
raw and accepted counts, confidence summaries, audit accuracy against hidden
labels, and pseudo-label stability across generations.

## Mixed Label Propagation

`mixed_label_propagation` is a substantial algorithmic addition. It:

1. extracts current DINO embeddings;
2. builds a sparse kNN affinity graph;
3. runs initial label propagation;
4. estimates dissimilarity edges;
5. solves a mixed similarity/dissimilarity propagation system;
6. emits pseudo-labels and entropy-based confidences.

The companion `MixedLabelPropagationProxyLoss` owns trainable class proxies and
uses pseudo-label confidence as a sample weight. It is treated as a
classification-style metric loss and gets its own criterion optimizer.

The detailed equation-to-code mapping is in
[Deep mixed label propagation](../mixed_label_propagation.md).

## STML

The fork adds semi-supervised STML support in two forms:

- loss-driven legacy STML: `method="stml"` with `loss="STMLLoss"`;
- composed regularization: `method="regularized"` with
  `method_params.regularizer="stml"`.

STML adds:

- supervised warmup;
- student/teacher models;
- EMA teacher updates;
- unlabeled nearest-neighbor batches;
- multiple augmented views per unlabeled sample;
- separate retrieval/background heads in `DinoWrapper`.

The STML implementation is intentionally adapted to this project's DINOv2
backbone and evaluation policy rather than the original STML paper's exact
ResNet benchmark setup.

## LRML Regularizer

The `regularized` method can also use `regularizer="lrml"`. The implementation
adapts Laplacian Regularized Metric Learning to deep embeddings by applying a
graph-Laplacian smoothness energy over network embeddings.

This is a regularization path, not a pseudo-label generator. It pairs
regularizer batches with supervised batches through `CombinedTrainingLoader`.

## External Unlabeled Data

Upstream only trains on the selected dataset. This fork can append external
unlabeled images for STML-style training:

- `unlabeled_source="split"`: use unlabeled samples from the current
  dataset split.
- `unlabeled_source="external"`: use only images from
  `external_unlabeled_dir`.
- `unlabeled_source="split_and_external"`: use both sources.

External images are loaded recursively by `RecursiveUnlabeledImageDataset`.
Directory names and annotations are ignored.

## Supervised Baselines Reuse SSL Splits

The local "supervised" path in label-budget experiments is not the upstream
fully supervised path. It intentionally uses the SSL config to choose a labeled
support set, then disables pseudo-labeling. This is the main mechanism for fair
supervised-vs-SSL comparisons under the same label budget.
