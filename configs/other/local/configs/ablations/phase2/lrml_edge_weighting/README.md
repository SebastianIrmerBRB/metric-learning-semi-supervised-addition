# LRML edge weighting

The shared `lrml.json` compares the existing LRML graph weighting modes. Phase 2
manifests replay each study's tuned parameters; the Cars Phase 1 HPO manifest
uses a separate config restricted to Iscen cosine weighting.

| Variant | Undirected edge weight |
| --- | --- |
| `binary` | 1 if either endpoint selected the other among its k nearest neighbors |
| `cosine_gamma_1` | Positive-part cosine similarity, summed over the available directions |
| `cosine_gamma_3` | Positive-part cosine similarity cubed, summed over the available directions (Iscen affinity) |

The binary variant provides an explicit reference. Cosine modes use the same
directed neighbor candidates, remove nonpositive similarities, and give mutual
neighbors twice a one-way edge's weight. This compares the implemented weighting
mechanisms, including that mutual-neighbor factor. With a normalized Laplacian,
degrees are recomputed from each variant's weights. All variants inherit the
study's target-ratio weighting and set `regularizer_max_ratio` to `null`, as in
the other Phase 2 LRML ablations.

Each Phase 2 dataset manifest expands into 15 jobs: three variants for ArcFace,
CircleLoss, MultiSimilarity, NT-Xent, and ProxyAnchor. The manifests follow the
existing LRML contrastive-margin replay settings and source studies:

| Dataset | k-shot | Comparison seed | Epoch limit |
| --- | --- | --- | --- |
| Cars196 | 16 | 2 (training and support) | Inherited from the study |
| In-Shop | 2 | 2 (training and support) | 1000 |
| Semi-iNat | 16 | 2 (training and support) | 1000 |

From the repository root:

```bash
python launch_runs.py --manifest configs/run_manifests/phase2/ablations/cars_lrml_edge_weighting.json
python launch_runs.py --manifest configs/run_manifests/phase2/ablations/in_shop_lrml_edge_weighting.json
python launch_runs.py --manifest configs/run_manifests/phase2/ablations/semi_inat_lrml_edge_weighting.json
```

The source study directories must be available at the paths in the manifests,
as for the existing Phase 2 replays. GPU selection and concurrency follow the
sibling manifests; use the launcher's `--gpus`, `--max-concurrent-per-gpu`, or
`--max-total-runs` options to override them. Add `--dry-run` to check placement
without starting training.

Outputs are separated by dataset under
`final/phase_2/lrml_edge_weighting/<dataset>/<loss>_<variant>`; scheduler logs are
under `final/phase_2/lrml_edge_weighting/_scheduler/<dataset>`.

## Cars Phase 1 HPO

The HPO manifest expands into five independent studies: one per loss, using
only Iscen label propagation's cosine weighting. Each study gets 80 TPE trials
(400 trials total), with 40 startup trials and sequential trials inside each
process. It uses the existing Cars Phase 1 protocol: 16-shot, four-fold group CV, frozen backbone,
600 epochs maximum, patience 20, and the default training and HPO seed of 7.
Final evaluation after HPO follows the existing experiment configuration.

`configs/runs/lrml/cars_edge_weighting.json` inherits the Phase 1 Cars loss and
learning-rate ranges from `configs/runs/cars.json` and the shared LRML ranges
from `configs/runs/lrml/standardized_ssl_params.json`, using relative paths.
Learning rates, supervised loss/miner parameters, batch sampler, graph batch
size, neighbor count (2 to 100), target ratio (0.5 to 1.5), and warmup (0 to 30
epochs, step 5) are searched independently for each loss. Edge weighting stays
fixed to `cosine`; its exponent gamma is tuned from 1 to 10, matching
`configs/runs/iscen/standardized_ssl_params.json`. The single HPO variant is
defined in `configs/ablations/phase1/lrml_edge_weighting/lrml_iscen.json`.

```bash
python launch_runs.py --manifest configs/run_manifests/hpo/cars_lrml_edge_weighting.json
```

Training outputs go under
`logs/ssl/phase_1/cars/lrml_edge_weighting/<loss>_iscen_weighting`, with study name
`cars_lrml_edge_weighting`; scheduler logs go under
`logs/run_scheduler/phase_1/cars/lrml_edge_weighting`. This manifest starts new
HPO studies and does not require the Phase 2 source studies.

## Transfer all manifests

From the repository root:

```bash
./transfer_lrml_edge_weighting.sh irmers@dipa --dry-run
./transfer_lrml_edge_weighting.sh irmers@dipa
```

This copies all four manifests, the Phase 2 variants, the single Iscen HPO
variant, the Cars experiment and SSL config, and the HPO config with its
inherited files, plus the `main.py` fix that accepts an explicit k-shot grid
when the base SSL config has no `labeled_per_class` default, preserving their
relative paths. Set `METRIC_LEARNING_SERVER` to supply the host without a positional
argument, or `METRIC_LEARNING_REMOTE_ROOT` to change the default destination
`/home/irmers/metric-learning`. Training code and the existing Phase 2 source
studies must already be on the server. The transfer does not launch training.
