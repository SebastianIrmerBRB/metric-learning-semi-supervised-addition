# STML Semi-Supervised Training

There are two supported STML modes:

- Legacy faithful STML: `method: "stml"` with `loss: "STMLLoss"`. After
  supervised warm-up, STML replaces the supervised objective and trains on
  hidden-label labeled plus unlabeled samples.
- Composed regularization: `method: "regularized"` with
  `method_params.regularizer: "stml"`. After warm-up, every optimizer step
  combines any configured supervised loss with STML on a separate unlabeled
  batch.

The generic method lives in `semi_supervised.METHOD_REGISTRY`; concrete
regularizers live in `semi_supervised.REGULARIZER_REGISTRY`. Legacy `stml`
remains a separate loss-driven method for backward compatibility and faithful
STML experiments.

The training phases are:

1. For `warmup_epochs`, train only the labeled split. Legacy STML uses
   `warmup_loss` and `warmup_miner`; composed mode uses `loss` and `miner`.
2. At the first STML epoch, initialize the teacher from the fully fine-tuned
   student. The teacher does not exist and receives no EMA updates during
   supervised warm-up.
3. After warm-up, legacy mode uses `STMLLoss` alone. Composed mode uses:
   `supervised_weight * supervised_loss + regularizer_weight * STMLLoss`.
4. Before every STML epoch, the current student `g` space builds the
   nearest-neighbor batches over only the unlabeled pool.
5. Every sampled image is augmented into multiple views. Their shared instance
   ID is passed to the contextual-similarity calculation.
6. The student has the paper's two heads: background head `g` and retrieval
   head `f`. At the STML boundary, the teacher starts with the student's fully
   fine-tuned backbone and an independently initialized `g` head, then follows
   the student by EMA after each student optimizer step.
7. The loss is the full upstream objective: relaxed contrastive loss for both
   student heads plus KL self-distillation from detached student `g` to `f`.

This first mode is the paper's semi-supervised recipe: fine-tune on the labeled
dataset with MS loss, then run STML over the union of labeled and unlabeled
images while hiding every label from STML. The composed regularizer is an
additional project variant, not the semi-supervised recipe evaluated in the
STML paper.

Example:

```powershell
python main.py --experiment_config configs/experiments/stml_cifar100.json
```

The SSL config selects the label budget and switch epoch:

```json
{
  "method": "stml",
  "update_mode": "every_epoch",
  "warmup_epochs": 5,
  "label_sampling_mode": "global_budget",
  "labeled_fraction": 0.1,
  "method_params": {}
}
```

The legacy experiment config selects STML and its warm-up objective:

```json
{
  "loss": "STMLLoss",
  "loss_params": {
    "sigma": 3.0,
    "delta": 1.0,
    "num_views": 2,
    "num_neighbors": 5,
    "teacher_momentum": 0.999,
    "normalize_student": false
  },
  "miner": "no_miner",
  "warmup_loss": "MultiSimilarityLoss",
  "warmup_miner": "MultiSimilarityMiner"
}
```

`num_neighbors` controls both the nearest-neighbor batch sampler and the
contextual STML neighborhood. The effective contextual top-k is
`num_views * num_neighbors`, matching the upstream implementation.

## Frozen DINO Features

STML can train with a frozen DINO backbone as long as the two STML projection
heads remain trainable. Deterministic backbone features can also be cached and
precomputed:

```json
{
  "backbone_tuning": "frozen",
  "use_cache": true
}
```

In cached mode, the deterministic feature for an image is reused for each of its
configured STML views. The full relaxed-contrastive and self-distillation losses
still train `f` and `g`, but augmentation-invariance is no longer supplied as an
extra training signal. This is a deliberate frozen-feature adaptation; the
published recipe uses independently augmented image views. Fully freezing the
`f` and `g` heads remains unsupported because the loss would then have no
trainable embedding parameters.

To compose STML with any supervised loss, use
`configs/ssl_regularized_stml.json` and configure the supervised objective
normally:

```json
{
  "method": "regularized",
  "method_params": {
    "regularizer": "stml",
    "regularizer_weight": 1.0,
    "supervised_weight": 1.0,
    "regularizer_params": {
      "sigma": 3.0,
      "delta": 1.0,
      "num_views": 2,
      "num_neighbors": 5,
      "teacher_momentum": 0.999
    }
  }
}
```

Adding another regularizer requires implementing `BaseTrainingRegularizer` and
registering it in `REGULARIZER_REGISTRY`; the supervised loss path does not
change.

## Using STML's Relations With Another Loss

STML's objective is the only consumer of its soft pairwise weight `w`, which
ties the method to `STMLLoss`. The `stml_threshold` pseudo-label method keeps `w`
and cuts it at a confidence threshold instead, so any configured metric loss can
train on the resulting class ids. See
[stml_threshold.md](stml_threshold.md).

## External Cars Unlabeled Data

The default `unlabeled_source` is `split`, which preserves the existing
behavior: the configured label budget divides the current training dataset into
labeled and unlabeled samples.

For the paper-style Cars196 + CompCars setup, place CompCars images below one
directory and use:

```json
{
  "dataset": "Cars196",
  "ssl_config": "configs/ssl_stml_external.json",
  "unlabeled_source": "external",
  "external_unlabeled_dir": "data/CompCars",
  "external_unlabeled_filter": "compcars_stml_paper",
  "compcars_min_model_images": 100
}
```

STML adopts SLADE's semi-supervised protocol, so `compcars_stml_paper` and
`compcars_slade_paper` build the same pool: the official CompCars classification
split, filtered at model level to 16,537 images across 145 classes, verified
against those counts. [slade_compcars.md](slade_compcars.md) documents that
filter, its layout expectations, the threshold calibration, and
`scripts/check_compcars_slade_subset.py` for checking a local copy.

`compcars_model_min_count` is the unverified variant for CompCars copies that
lack the split files: images are discovered recursively, common container
folders such as `data`, `image`, or `web-nature` are skipped, then the next two
path components are treated as the CompCars make/model key. One-level
model-folder layouts are also accepted. Classes with fewer than
`compcars_min_model_images` images are dropped, and no count check is applied.

After filtering, retained CompCars samples are appended only after Cars
train/validation splitting and are still exposed as hidden-label STML samples
with label `-1`. The realized external image count and model-class count are
written to `split/split_info.json` under `external_unlabeled_pool.filter_info`.

Available modes:

- `split`: use only unlabeled samples from the current dataset split.
- `external`: use labeled current-dataset samples plus only external unlabeled
  images, matching the paper's labeled Cars + additional CompCars setup.
- `split_and_external`: use both split-derived and external unlabeled images.

The example [stml_cars196_compcars.json](../configs/experiments/stml_cars196_compcars.json)
also uses the paper's semi-supervised STML values `sigma=2` and
`teacher_momentum=0.9999`.

## External CUB Unlabeled Data

The paper-style bird pairing uses labeled CUB-200-2011 data and the complete
NABirds collection as additional unlabeled data. Download NABirds V1 from the
[official Cornell Lab page](https://dl.allaboutbirds.org/nabirds), extract it
under `data/NABirds`, and retain the official layout:

```text
data/NABirds/nabirds/
  classes.txt
  image_class_labels.txt
  images.txt
  images/
```

The outer `nabirds/` directory is optional, so placing those metadata files and
the image directory directly under `data/NABirds` also works. Run the included
configuration with:

```powershell
python main.py --experiment_config configs/experiments/stml_cub_nabirds.json
```

`external_unlabeled_filter: "nabirds"` reads the official `images.txt` order,
checks that image IDs, class annotations, and files agree, and then hides every
NABirds label as `-1`. Extra image files that are not listed in the metadata are
not added. The source class count is retained only in
`split/split_info.json` for integrity auditing; it is never supplied to the
training loss. This matches the CUB + NABirds dataset pairing used by SLADE and
the semi-supervised STML evaluation, while retaining this project's DINOv2 and
validation settings.

`warmup_epochs` is the complete supervised-only phase, while `epochs` counts
both warm-up and STML epochs. The examples still use the project's DINOv2
backbone and validation policy rather than the paper's ResNet50 benchmark
setup.
