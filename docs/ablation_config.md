# Ablation Config

An ablation replays a tuned run with one thing changed. `--ablation_config`
names a JSON file of named **variants**. Each variant holds only what it
changes, and it is applied after every other source -- the experiment config,
the SSL config file, the HPO study's winning parameters, and the CLI -- so it
wins against the tuned value it contradicts and keeps everything it does not
mention.

## File format

```json
{
  "name": "hoffer_training",
  "description": "Batch composition and loss temperature around the tuned Hoffer NT-Xent run.",
  "sweep": {
    "sampler_m": [2, 8],
    "batch_size": [128, 512],
    "loss.NTXentLoss.temperature": [0.01, 0.1]
  },
  "variants": {
    "no_warmup": {"ssl_config": {"warmup_epochs": 0}},
    "pool_labeled": {"unlabeled_source": "labeled"},
    "alt_method": {
      "ssl_config": "configs/ssl_methods/hoffer_v2.json",
      "batch_sampler": "512:8"
    }
  }
}
```

- `sweep` moves one key at a time. Every value becomes its own variant, named
  `<last key segment>_<value>`: `sampler_m_2`, `batch_size_512`,
  `temperature_0p01`. In the value, `.` becomes `p`, `-` becomes `m`, `256:8`
  becomes `256-8`, and an SSL config path becomes its file stem.
- `variants` names change sets explicitly, for anything that has to move
  several keys together.
- Both may appear in one file, and names must be unique across them. The
  example above expands to nine variants.

## What a variant can change

| Variant key | Changes |
| --- | --- |
| `"sampler_m": 8`, `"batch_size": 512`, `"lr": 1e-4`, `"unlabeled_source": "labeled"` | That CLI argument; any argparse destination is accepted. |
| `"batch_sampler": "512:8"` | `batch_size` and `sampler_m` together. |
| `"loss_params": {"temperature": 0.1}` | The named parameters of the run's loss; its other parameters keep their tuned values. |
| `"loss.NTXentLoss.temperature": 0.1` | The same, class-qualified. Rejected for a run with a different loss. |
| `"miner_params": {...}`, `"miner.<Miner>.<param>"` | Miner parameters, likewise. |
| `"ssl_config": {"method_params": {"regularizer_weight": 0.1}}` | Only the named SSL config fields, at any depth. |
| `"ssl_config.method_params.regularizer_weight": 0.1` | The same, dotted (`ssl.` works too). |
| `"ssl_config": "configs/ssl_methods/hoffer_v2.json"` | The SSL config file itself; see below. |

Nested objects are sparse: `loss_params` and an object-valued `ssl_config`
change the keys they name and nothing else. Spelling the same value twice in
one variant (`loss_params.temperature` beside `loss.NTXentLoss.temperature`,
or `batch_sampler` beside `sampler_m`) is rejected rather than resolved.

Changing one half of a tuned pair keeps the other half: ablating
`"sampler_m": 8` against a study that tuned `batch_sampler: "128:4"` trains
with batch size 128 and `sampler_m` 8. A combination the sampler cannot build
fails when the run starts, as it would in a hand-written experiment config.

### Swapping the SSL config file

A path-valued `ssl_config` replaces the method file while the run keeps its
data condition from the config it replaces: `label_sampling_mode`,
`labeled_fraction`, `labeled_per_class`, `seed`, `support_seed`,
`unlabeled_class_scope`, `unlabeled_fraction`, and `max_unlabeled_samples`. For
a study replay that is the study's own scenario config, so its k-shot split
survives the swap. The study's tuned `ssl_config.*` parameters are then applied
to the new file just as they were to the old one; a tuned key the new method
does not accept fails validation instead of being dropped. SSL field changes in
the same variant apply on top.

The merged config is written to `logs/<save_dir>/ablation/<variant>_ssl_config.json`,
and that is the `ssl_config` the run records.

## What a variant cannot change

These are rejected when the file is loaded:

- **Seeds, outer-grid dimensions, and the labeled split**: `seed`,
  `support_seed`, `data_split_seed`, `hparam_seed`, `comparison_seeds`,
  `comparison_seed_targets`, `k_shot_grid`, `label_budget_grid`,
  `ssl_label_sampling_modes`, `loss_miner_grid`, and
  `ssl_config.{label_sampling_mode, labeled_fraction, labeled_per_class, seed, support_seed}`.
  A replay varies these per scenario after the variant is chosen, so a variant
  that pinned one would collapse every scenario onto the same value. Set them in
  the manifest's `study_replay` block instead.
- **The baseline itself**: `dataset`, `dataset_protocol`, `cv_k`, `cv_mode`,
  `val_mode`, `mode`, `loss`, and `miner`. Every variant is compared against the
  same baseline, and a different loss would not accept the tuned loss
  parameters. Change `ssl_config.method` by swapping the file.
- **Launch plumbing**: `save_dir`, `device`, `ssl_device`,
  `experiment_config`, `hparam_config`, `final_test_study_dir`,
  `study_dir_mode`, `final_test_top_n`, and `final_test_trial_numbers`.
- **`labeled_batch_fraction`**, which exists only inside a search space. Set
  `ssl_config.labeled_batch_size` instead.

The unlabeled pool (`unlabeled_source`, `ssl_config.unlabeled_class_scope`,
`ssl_config.unlabeled_fraction`, `ssl_config.max_unlabeled_samples`) may be
ablated.

Unknown argument names are rejected when the run starts, before any data is
loaded. SSL fields the config does not have, and loss or miner keys the run's
components do not accept, are rejected when the variant is applied.

## Running an ablation

### From a run manifest

```json
{
  "save_dir_root": "final/phase_1/ablations/cars",
  "study_replay": {
    "study_dir_mode": "final_train",
    "comparison_seeds": [7, 8, 9, 10, 11],
    "comparison_seed_targets": ["seed", "support_seed"],
    "k_shot_grid": [4, 8, 16, 32]
  },
  "runs": [
    {
      "name": "hoffer_nt",
      "study_dir": "final/phase_1/cars/hoffer/hoffer_nt/class_subset_k_shot_label_1_k_16_NTXentLoss_no_miner_tune_seed_7/hpo/in-shop",
      "ablation_config": "configs/ablations/hoffer_training.json"
    }
  ]
}
```

The launcher expands the run into one child per variant (`hoffer_nt_sampler_m_2`,
`hoffer_nt_sampler_m_8`, ..., `hoffer_nt_alt_method`). Each child is started
with `--ablation_config <file> --ablation_variant <variant>` and writes to
`<save_dir_root>/<child name>`, or to `<save_dir>/<variant>` when the run pins
its own `save_dir`. Every child replays the study under the full
`study_replay` block, so each variant gets the same seeds and k-shot grid as the
baseline. A malformed ablation file fails the manifest before anything
launches, and `--ablation_config`/`--ablation_variant` cannot also be passed in
that run's `args`.

A class-qualified loss key has to match the loss of every run that uses the
file, so give each loss its own ablation file. `loss_params` follows whichever
loss the run uses, but its parameters then have to exist on each of them.

### A single run

```bash
python main.py \
  --final_test_study_dir final/phase_1/cars/hoffer/hoffer_nt/.../hpo/in-shop \
  --study_dir_mode final_train \
  --ablation_config configs/ablations/hoffer_training.json \
  --ablation_variant sampler_m_8
```

`--ablation_variant` may be omitted when the file defines exactly one variant.

## What a run records

`run_config.json` carries `ablation_config`, `ablation_variant`, and
`ablation_config_resolved`: the ablation's name, description, selected variant,
and that variant's flattened changes. An interrupted cross-validation run only
resumes folds recorded under the same ablation config and variant.

## Single-ablation files

A file with a top-level `overrides` object instead of `sweep`/`variants` -- the
earlier format -- still loads, as one variant named after the file's `name`.
