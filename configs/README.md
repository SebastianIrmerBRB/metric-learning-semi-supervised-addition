# Thesis configurations

`used/` contains the settings traced to the **current** thesis data in
`/mnt/f/Thesis/daten`. The collection uses recorded training inputs and the
thesis's selected reruns, rather than assuming that a similarly named config
still has its historical contents. `other/` preserves the remaining local
and server config versions, with their relative config links relocated.

The collection includes 110 Phase-0 HPO studies, 125 Phase-1 HPO studies and
4,957 distinct evaluated CV runs. The latter include 2,250 Phase-1 runs,
917 main Phase-2 runs, 518 Phase-2 runs at k=8, 350 geometry runs, 470
external-image runs, 100 additional k=4 ensemble runs, and both sides of 339
patience comparisons. Categories overlap. The 25 supervised references in the
k=4 comparison reuse Phase-1 runs; their links are recorded separately.
The 55 external-image observations explicitly excluded by the thesis are not
included as used runs.

## Layout

```text
used/
  _templates/                 Shared phase/dataset/method settings
  phase_0/
    hpo/                      Recorded search spaces and Optuna settings
    hpo_experiments/          Search recipes accepted by main.py
    hpo_ssl/                  Fixed SSL inputs for those recipes
    hpo_trials/               Every recorded trial, including resumed studies
    recorded_studies/         Complete inherited study_config snapshots
    run_manifests/            HPO schedules by dataset and device
  phase_1/                    HPO files plus recorded evaluation runs
    experiments/              One concrete evaluation run per config
    ssl/                      Its complete recorded SSL settings
    recorded/                 Complete training-time run_config snapshots
    run_manifests/
  phase_2/                    Main, k=8, geometry, external, ensemble, comparisons
    experiments/
    ssl/
    recorded/
    run_manifests/
  originals/                  Referenced original config files and dependencies
  catalog/                    Source hashes, run identities, links and audits
other/
  quickstart/                 Small public-data examples for checking installation
  local/                      Other configs recovered from the local workspace
  server/                     Other configs recovered from dipa
archive_catalog.jsonl         Original hashes, classification and path rewrites
```

Each executable leaf inherits phase, dataset and method settings through
`extends`. Parent paths are relative to the child. Multiple parents merge left
to right; child settings win. Nested dictionaries merge; lists and scalars
replace their parent values, and `{}` explicitly clears a dictionary. HPO
keeps its established merge rules: only `spaces` and `per_loss` merge deeply;
other dictionaries, including `sampler_params`, replace the inherited object.

## Use

Run commands from the repository root. For example, a recovered HPO recipe is:

```bash
python main.py --experiment_config configs/used/phase_1/hpo_experiments/cars/sup_arc.json
```

The evaluation file paths and group memberships are listed in
`used/catalog/runs.jsonl`. Pass the row's `experiment` path to
`python main.py --experiment_config <path>`. These files contain the selected
training parameters directly and do not require the original Optuna database.
They clear the old HPO request and outer comparison grids, retain the recorded
split/support/run seeds, and write into a new `thesis/` output namespace.

Choose a manifest from the corresponding `run_manifests/` directory:

```bash
python launch_runs.py --manifest configs/used/phase_1/run_manifests/hpo/cars_cpu_ssl_cuda.json --dry-run
```

Manifests preserve the recorded training/SSL device combination, use GPU 0
where needed, and schedule one process at a time by default. Adjust device
selection and concurrency for the machine. The HPO recipes run the archived
search protocol; their automatic final-test loop is disabled because the
thesis's selected evaluation runs are provided separately.

`recorded/` and `recorded_studies/` are exact metadata snapshots after resolving
inheritance; they are not CLI experiment files. `originals/` preserves the
source versions available now, including original manifests whose output paths
match selected runs. These source versions may have been edited since training;
the reconstructed executable files and recorded snapshots take precedence.
Historical configs in `other/` remain archives and may need their original
datasets, study directories or obsolete code paths. Moving them does not make
those external dependencies available.

`other/quickstart/` is the exception: these are new, runnable demonstration
configs documented in the root README. They are not thesis experiments.

## Historical details retained

- **297 epoch-cap discrepancies:** later summaries say 1,000 epochs while the
  selected fold configs and initial training logs say 600. Replay files retain
  the training-time cap, and the catalog records the summary differences.
- **Automatic epoch lengths:** a fold's resolved `length_before_new_iter` is
  not reused as the input for every fold. Replay configs restore the configured
  value, normally `null`, so the engine computes each fold's length.
- **Mixed Phase-0 searches:** 23 resumed studies contain 47 fixed-SSL regimes.
  A single HPO recipe cannot reproduce all their historical changes. The recipe
  uses the archived study protocol and recovered grid/first available trial
  inputs. `hpo_trials/*.jsonl` keeps every recorded trial number, state, parameter
  set and objective, while `*.inputs.jsonl.gz` also keeps the exact recorded
  `resolved_args` and `resolved_ssl_config`. Their trial mappings are documented
  in `catalog/hpo_regimes.json`; no later regime is silently substituted for
  an earlier one. There are 15,529 archived trial records, including failed and
  pruned trials and resumed searches exceeding the nominal trial budget.
- The 110 Phase-0 studies are those selected for the current thesis's study
  comparison. Additional historical pilot config files remain traceable in
  the input archives; this is not a claim that every exploratory pilot produced
  a published quantitative result.
- Source paths and SHA-256 hashes remain provenance. Training logs, datasets,
  embeddings and model checkpoints are not bundled here.

## Validate

```bash
python scripts/validate_thesis_configs.py --native
# Also check the original source hashes on the recovery machine:
python scripts/validate_thesis_configs.py --native --verify-sources
```

Validation resolves every generated leaf, checks its recorded content hash,
verifies training settings and trial snapshots, and checks every manifest and
config reference with the native loaders. It starts no training jobs. No test
suite is added to this repository.
