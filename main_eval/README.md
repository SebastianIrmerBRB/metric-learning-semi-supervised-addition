# main_eval/ — Phase-2 analysis

Scripts for the thesis' Phase 2: *which* hyperparameters matter, *why* a value
works, and *why* a method gains. They read what training already left on disk
and write tables and figures below `main_eval_out/` (`--out` to change it).
Nothing here trains, and nothing is written into a run directory.

The section letters (A1, B0, …) refer to `Phase2_Analyseplan.md` in the thesis
folder.

## Data the scripts read

| Source | Default location | Used by |
| --- | --- | --- |
| Phase-0 random-search studies (incl. `pseudo_label_diagnostics.jsonl`) | `sorted_logs/phase_0` | `hpo_sensitivity`, `pseudo_label_quality` |
| Phase-1 TPE studies (40 random start-up + 40 TPE trials) | `/mnt/f/server_logs/final` | `hpo_sensitivity`, `make_phase2_ablations` |
| Phase-1 seed replays (`cv_summary.json`, per-fold `test_embeddings.npz`, `split/`) | `/mnt/f/server_logs/seeded_runs_{cars,cub,in_shop}`, `/mnt/f/server_logs/seeded_runs/cars` | everything else |

For the seed replays the later root wins, which reproduces the thesis series
(LRML/Hoffer/SLADE on Cars196 from `seeded_runs/cars`).

## The scripts

| Script | Plan | Question | Needs GPU/data |
| --- | --- | --- | --- |
| `hpo_sensitivity.py` | A1–A4 | Which HPs matter, and how flat is the optimum? Spearman on the random trials only, top-γ vs. rest, per-bin best, TPE concentration, plateau size, best values across losses/datasets, optional PED-ANOVA (`--ped-anova`). | no |
| `pseudo_label_quality.py` | B2 | How correct and complete were LP's pseudo-labels, does MAP@R follow accuracy or coverage, and which HPs move them? (Phase-0 LP trials) | no |
| `seeded_runs.py` | B1 | Paired differences per seed: method vs. supervised, ablation variant vs. unmodified replay. | no |
| `frozen_feature_diagnostics.py` | B0 | What can the methods see before training? kNN purity, LP accuracy/coverage curve, Hoffer assignment, k-means purity on the frozen DINOv2 features; writes `backbone_test_<dataset>.npz`. | yes (rebuilds the fold) |
| `embedding_geometry.py` | B6 | What did training change in the test space? Spectrum, effective rank, spectral decay ρ, alignment/uniformity, intra/inter-class distances. | CPU is fine, slow-ish |
| `fold_diversity.py` | B7 | Why does Concat help? CKA and neighbour overlap between fold models vs. Concat−Mean gain. | GPU helps |
| `per_query_gains.py` | B8 | On which queries/classes does a method gain, and does the gain depend on how hard the class already is for the backbone? (Cars196/CUB) | GPU helps |
| `make_phase2_ablations.py` | A5, A6, B1, B2, B5 | Writes `--ablation_config` files and run manifests: one-at-a-time sweeps, the supervised→SSL HP path, mechanism controls, the SLADE paper's ablations (`paper`). | no |
| `common.py` | – | Shared discovery, parsing, retrieval metrics, CKA, spectra. | – |

## Suggested order

Cheap, from what exists already:

```bash
source venv/bin/activate
python main_eval/hpo_sensitivity.py                               # ~all studies; add --no-figures for speed
python main_eval/pseudo_label_quality.py
python main_eval/seeded_runs.py
python main_eval/frozen_feature_diagnostics.py --dataset cars --k 4 8 16 32 --folds 0
python main_eval/frozen_feature_diagnostics.py --dataset cub  --k 4 8 16 32 --folds 0
python main_eval/frozen_feature_diagnostics.py --dataset in_shop --k 2 4 --folds 0
python main_eval/embedding_geometry.py --dataset cars --k 4 16 --losses ms \
    --backbone-npz main_eval_out/frozen/backbone_test_Cars196.npz
python main_eval/fold_diversity.py --dataset cars --k 4 16
python main_eval/per_query_gains.py --dataset cars --k 4 --combo iscen_ms \
    --backbone-npz main_eval_out/frozen/backbone_test_Cars196.npz
```

Then new runs, via the existing replay machinery:

```bash
# 2x2 table only: the two new cells in one manifest (--variants filters, --append merges kinds)
python main_eval/make_phase2_ablations.py hp-path  --dataset cars --combo iscen_ms --variants sup_all_shared --name cars_ms_2x2
python main_eval/make_phase2_ablations.py controls --dataset cars --combo iscen_ms lrml_ms --variants pool_labeled \
    --name cars_ms_2x2 --append
# ...and the corner cells, re-run on the same code: unmodified replays (+ LRML's sup_all_shared)
python main_eval/make_phase2_ablations.py baseline --dataset cars --combo sup_ms iscen_ms lrml_ms --name cars_ms_2x2_rest
python main_eval/make_phase2_ablations.py hp-path  --dataset cars --combo lrml_ms --variants sup_all_shared \
    --name cars_ms_2x2_rest --append
# unlabeled-data controls for every SSL method (zero_weight: pool kept, weight 0; pool_labeled: no unlabeled data)
python main_eval/make_phase2_ablations.py controls --dataset cars --combo lrml_ms hoffer_ms slade_ms iscen_ms \
    --variants zero_weight pool_labeled --name cars_ms_unlabeled_controls
# full HP path / all mechanism controls
python main_eval/make_phase2_ablations.py hp-path  --dataset cars --combo iscen_ms
python main_eval/make_phase2_ablations.py controls --dataset cars --combo iscen_ms --diagnostics
# one-at-a-time sensitivity with pseudo-label diagnostics
python main_eval/make_phase2_ablations.py sweep --dataset cars --combo iscen_ms --diagnostics \
    --params ssl_config.method_params.n_neighbors ssl_config.confidence_threshold ssl_config.warmup_epochs
# look at the embedding space over training
python main_eval/make_phase2_ablations.py controls --dataset cars --combo lrml_ms --track
# SLADE: the paper's Tables 4-6 and settings, then frozen / oracle clusters (docs/slade.md, section 7)
python main_eval/make_phase2_ablations.py paper    --dataset cars --combo slade_cir --seeds 2 --name cars_cir_slade_paper
python main_eval/make_phase2_ablations.py controls --dataset cars --combo slade_cir --seeds 2 --name cars_cir_slade_controls \
    --variants clusters_once oracle_noise_0 oracle_noise_10 oracle_noise_20 oracle_noise_40 oracle_noise_60

python -m training.run_scheduler --manifest configs/run_manifests/phase2/<name>.json --dry-run --gpus 0
python -m training.run_scheduler --manifest configs/run_manifests/phase2/<name>.json --gpus 0

# afterwards: pair the variants with the original replays
python main_eval/seeded_runs.py --include-variants --seeded-roots \
    /mnt/f/server_logs/seeded_runs_cars /mnt/f/server_logs/seeded_runs/cars logs/final/phase_2/cars
python main_eval/pseudo_label_quality.py --roots logs/final/phase_2/cars     # if --diagnostics was on
python evaluate/report.py --root logs/final/phase_2/cars --out evaluate_out  # if --track was on
```

## Things to keep in mind

- **Why not fANOVA.** TPE samples densely around the optimum, so a surrogate
  fitted on its trials extrapolates elsewhere (Moosbauer et al., NeurIPS 2021).
  `hpo_sensitivity.py` keeps unbiased statistics on the random trials apart
  from the "where were good trials found" statistics on all trials.
- **The fold SE in `study_stats.csv` is conservative.** Folds hold different
  classes, so part of the fold spread is class difficulty that every trial
  shares. Differences *between* trials on the same folds are less noisy.
- **Pseudo-label diagnostics exist only in Phase 0.** The Phase-1 seed replays
  ran with them off; `make_phase2_ablations.py --diagnostics` adds a `diag`
  variant that replays the tuned config with them on.
- **Seeds.** The Phase-1 replays use seeds 2, 8, 9, 10, 11; ablations default to
  the same seeds so every variant pairs with an existing run.
- **Oracle variants** (`graph_oracle_pruned`, `graph_random_pruned`) read hidden
  labels. They are mechanism diagnostics, never SSL results.
- **In-Shop** uses a query/gallery test protocol; `per_query_gains.py` does not
  support it, and the geometry/diversity numbers there treat query and gallery
  as one set.
- **Runtime.** `embedding_geometry.py` takes about 5 minutes for 20 Cars196 runs
  (Concat embeddings have 3072 dimensions); `--no-concat` halves that.
