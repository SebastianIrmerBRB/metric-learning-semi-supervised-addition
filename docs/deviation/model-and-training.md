# Model And Training Changes

The upstream model and training loop are compact: DINOv2, optional projection,
L2 normalization, one optimizer for the model, an optional classifier-loss
optimizer, and a direct epoch loop. This fork keeps that core but adds runtime
safety, model-tuning policies, caching, configurable loss construction,
regularized training paths, and richer sampler handling.

Primary files:

- [`models/retrieval_model.py`](../../models/retrieval_model.py)
- [`training/engine.py`](../../training/engine.py)
- [`losses/metric_losses.py`](../../losses/metric_losses.py)
- [`utils/`](../../utils)

Related deep docs:

- [Standard metric-loss flow](../standard_loss_data_flow.md)
- [Classification-style loss flow](../classification_loss_data_flow.md)
- [DINOv2 hub retry](../dinov2_hub_retry.md)
- [Experiment config: frozen backbone and cache](../experiment_config.md#frozen-backbone-and-cache)

## DINOv2 Loading

Upstream calls:

```python
torch.hub.load("facebookresearch/dinov2", f"dinov2_vit{dino_size}14")
```

This fork wraps hub loading in retry logic:

- repository string: `facebookresearch/dinov2:main`;
- retryable HTTP/network failures are retried;
- backoff delay increases per attempt;
- non-retryable failures are re-raised.

This avoids failing long experiment batches on transient Torch Hub errors.

## Backbone Tuning Policies

Upstream implicitly fine-tunes the full DINO backbone.

This fork adds `backbone_tuning`:

| Policy | Trainable parameters |
|---|---|
| `full` | All DINO parameters plus projection head. |
| `frozen` | Projection head only. |
| `last_N_blocks` | Final N DINO transformer blocks, final norm, and projection head. |

Frozen mode keeps the DINO backbone in eval mode even when the full wrapper is
set to train mode.

## Frozen-Backbone Feature Cache

Upstream recomputes DINO features for every batch.

This fork adds `use_cache` for frozen-backbone runs. Cache behavior:

- requires `backbone_tuning="frozen"`;
- requires a non-null `feat_dim` so a trainable projection head remains;
- stores raw DINO backbone features before projection;
- hashes deterministic transformed CPU image tensors into per-sample cache
  keys;
- reuses cache entries across folds, HPO trials, and batch orders;
- writes cache statistics to run artifacts.

When cache mode is enabled, training uses deterministic transforms and disables
RandAugment, because cached features must correspond to stable inputs.

## STML Heads

Upstream has one output embedding.

This fork can enable two STML heads:

- retrieval head `f`, implemented by the normal projection path;
- background head `g`, implemented by `embedding_g`.

`DinoWrapper` adds:

- `forward_stml`;
- `forward_stml_teacher`;
- cached variants for both;
- optional STML-specific normalization behavior.

These paths are used by `STMLLoss` and the STML regularizer.

## Optimizers

Upstream supports `adam` and `rmsprop`.

This fork adds:

- `adamw` as the default optimizer;
- configurable `classifier_lr`;
- configurable global `weight_decay`, defaulting to `0.0` (disabled);
- optional epoch-level `step` and `cosine` learning-rate schedulers plus
  batch-level `cosine_warm_restarts`, with `none` as the default;
- separate optimizers for classification-style metric losses.

The selected optimizer kind is shared by the model optimizer and, when present,
the criterion optimizer. The same `weight_decay` value and selected scheduler
policy are applied to both. Step and plain cosine schedules advance once after
a non-empty epoch. Cosine warm restarts advance after every optimizer step at
fractional-epoch resolution.

## Loss And Miner Construction

Upstream constructs losses and miners with no constructor kwargs except the
automatic `num_classes` and `embedding_size` for classification-style losses.

This fork adds:

- `loss_params`;
- `miner_params`;
- `warmup_loss`;
- `warmup_miner`;
- `warmup_loss_params`;
- `warmup_miner_params`;
- class-qualified HPO spaces for loss/miner constructor kwargs;
- project-local losses in `losses/metric_losses.py`.

Added local losses include:

- `STMLLoss`;
- `MixedLabelPropagationProxyLoss`.

Classification-style losses still do retrieval evaluation only. Their proxies
or class weights are used for training but ignored during validation and test.

## Sample Weights

Upstream batches contain only images and labels.

This fork normalizes batches through `unpack_training_batch`. A batch may
contain:

- images;
- labels;
- optional confidence/sample weights.

Ordinary losses ignore weights. A loss can opt in through a
`supports_sample_weights` flag. `MixedLabelPropagationProxyLoss` uses this to
scale pseudo-labeled samples by their SSL confidence.

## Sampler Behavior

Upstream uses `MPerClassSampler` directly with the chosen `batch_size` and
`sampler_m`.

This fork adds:

- explicit sampler-capacity validation;
- derived epoch length from active dataset size;
- optional `length_before_new_iter`;
- HPO filtering for impossible `batch_size:sampler_m` choices;
- STML nearest-neighbor batch sampling for unlabeled STML batches.

The validation is important for label-budgeted runs because small support sets
can make a batch shape impossible even when the full dataset would support it.

## Runtime And DataLoader Safety

Upstream sets Torch sharing strategy to avoid too-many-open-files errors.

This fork adds:

- `seed_everything` for Python, NumPy, Torch, and CUDA;
- seeded DataLoader generators and worker initialization;
- configurable DataLoader start method;
- CUDA/fork safety validation;
- Windows worker resolution to `0` where nested datasets cannot be serialized
  safely;
- indexed CUDA device normalization such as `cuda:2`.

These changes are required because SSL embedding extraction and HPO/CV runs
create many DataLoaders under different process and device conditions.

## Training Loop Structure

Upstream keeps the full loop inside `main.py`.

This fork moves training into `training/engine.py`:

- `run_experiment` chooses single-fold versus CV;
- `run_training` owns one concrete train/validation/test run;
- helper functions create losses, miners, optimizers, teachers, loaders, and
  run configs;
- Optuna reporting is integrated without making Optuna required for direct
  runs.

The loop now handles:

- supervised, SSL, loss-driven, and regularized modes;
- warmup phases;
- every-epoch pseudo-label regeneration;
- optional test evaluation;
- selected validation metric for checkpointing;
- non-finite embedding failures during evaluation.
