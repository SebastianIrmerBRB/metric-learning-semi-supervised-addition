# SSL five-loss run matrix

These manifests cover the six composable SSL methods that already have
method-specific HPO search spaces in this repository:

- `iscen`
- `lrml`
- `slrml`
- `hoffer`
- `seraph`
- `simmatch_v2`

Each method has Phase 0 and Phase 1 manifests for CIFAR-100, CUB, Cars196, and
DeepFashion In-Shop. Every manifest schedules NT-Xent, Multi-Similarity,
Circle, ArcFace, and ProxyAnchor as separate runs.

## Template policy

The generated configs treat the existing files as authoritative:

- Cars, CIFAR, and CUB experiment configs are copied from their matching
  `configs/phase_0/<dataset>/<dataset>_<loss>.json` or
  `configs/phase_1/<dataset>/<dataset>_<loss>.json` file.
- Phase 0 HPO settings come from the existing dataset phase-0 search.
- Phase 1 Cars and CIFAR settings come from their loss-specific
  `configs/hpo/<dataset>/supervised_*.json` files.
- CUB inherits the Cars loss-specific HPO settings because no CUB-specific
  loss HPO files exist.
- In-Shop retains both `configs/experiments/class/in-shop.json` and
  `configs/supervised/classification/in-shop.json`.
- Method search spaces are copied exactly from the existing method HPO file;
  only its `ssl_config.*` entries are added to the supervised template.

Consequently, existing CV folds, patience, epochs, final-test flags,
learning-rate ranges, loss parameters, sampler choices, trial counts, samplers,
and scheduler settings remain unchanged. CIFAR Phase 0 therefore retains its
explicit four HPO workers; those manifests set `allow_parallel_trials` so the
run scheduler accepts that intentional nested parallelism. SimMatchV2 is the
one experiment-level exception: `use_cache` is set to `false` because that
implementation requires stochastic weak and strong image views.

The generated SLRML base config also replaces the obsolete unsupported
`slrmml` regularizer name with `slrml`. Stale unsupported
`normalize_embeddings` fields are removed from SLRML and Hoffer.

## Layout

- Base SSL configs: `configs/ssl_methods/<method>.json`
- Experiments:
  `configs/<phase>/ssl_methods/<method>/<dataset>/<dataset>_<loss>.json`
- HPO configs:
  `configs/hpo/ssl_methods/<phase>/<method>/<dataset>/<dataset>_<loss>.json`
- Run manifests:
  `configs/run_manifests/ssl_methods/<phase>/<method>_<dataset>.json`

For example:

```powershell
python launch_runs.py `
  --manifest configs/run_manifests/ssl_methods/phase_0/iscen_cars.json
```

Regenerate or verify the matrix with:

```powershell
python scripts/generate_ssl_method_configs.py
python scripts/generate_ssl_method_configs.py --check
```
