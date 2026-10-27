{
  "cells": [
    {
      "cell_type": "markdown",
      "id": "24eb00c3",
      "metadata": {},
      "source": "# SSL five-loss run matrix\n\nThese manifests cover the six composable SSL methods that already have\nmethod-specific HPO search spaces in this repository:\n\n- `iscen`\n- `lrml`\n- `slrml`\n- `hoffer`\n- `seraph`\n- `simmatch_v2`\n\nEach method has Phase 0 and Phase 1 manifests for CIFAR-100, CUB, Cars196, and\nDeepFashion In-Shop. Every manifest schedules NT-Xent, Multi-Similarity,\nCircle, ArcFace, and ProxyAnchor as separate runs.\n\n## Template policy\n\nThe generated configs treat the existing files as authoritative:\n\n- Cars, CIFAR, and CUB experiment configs are copied from their matching\n  `configs/phase_0/<dataset>/<dataset>_<loss>.json` or\n  `configs/phase_1/<dataset>/<dataset>_<loss>.json` file.\n- Phase 0 HPO settings come from the existing dataset phase-0 search.\n- Phase 1 Cars and CIFAR settings come from their loss-specific\n  `configs/hpo/<dataset>/supervised_*.json` files.\n- CUB inherits the Cars loss-specific HPO settings because no CUB-specific\n  loss HPO files exist.\n- In-Shop retains both `configs/experiments/class/in-shop.json` and\n  `configs/supervised/classification/in-shop.json`.\n- Method search spaces are copied exactly from the existing method HPO file;\n  only its `ssl_config.*` entries are added to the supervised template.\n\nConsequently, existing CV folds, patience, epochs, final-test flags,\nlearning-rate ranges, loss parameters, sampler choices, trial counts, samplers,\nand scheduler settings remain unchanged. CIFAR Phase 0 therefore retains its\nexplicit four HPO workers; those manifests set `allow_parallel_trials` so the\nrun scheduler accepts that intentional nested parallelism. SimMatchV2 is the\none experiment-level exception: `use_cache` is set to `false` because that\nimplementation requires stochastic weak and strong image views.\n\nThe generated SLRML base config also replaces the obsolete unsupported\n`slrmml` regularizer name with `slrml`. Stale unsupported\n`normalize_embeddings` fields are removed from SLRML and Hoffer.\n\n## Layout\n\n- Base SSL configs: `configs/ssl_methods/<method>.json`\n- Experiments:\n  `configs/<phase>/ssl_methods/<method>/<dataset>/<dataset>_<loss>.json`\n- HPO configs:\n  `configs/hpo/ssl_methods/<phase>/<method>/<dataset>/<dataset>_<loss>.json`\n- Run manifests:\n  `configs/run_manifests/ssl_methods/<phase>/<method>_<dataset>.json`\n\nFor example:"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "5b49389b",
      "metadata": {},
      "outputs": [],
      "source": "python launch_runs.py `\n  --manifest configs/run_manifests/ssl_methods/phase_0/iscen_cars.json"
    },
    {
      "cell_type": "markdown",
      "id": "cabb5e3e",
      "metadata": {},
      "source": "Regenerate or verify the matrix with:"
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "id": "7ac9be75",
      "metadata": {},
      "outputs": [],
      "source": "python scripts/generate_ssl_method_configs.py\npython scripts/generate_ssl_method_configs.py --check"
    }
  ],
  "metadata": {
    "jupytext": {
      "cell_metadata_filter": "-all",
      "main_language": "powershell",
      "notebook_metadata_filter": "-all",
      "text_representation": {
        "extension": ".md",
        "format_name": "markdown"
      }
    }
  },
  "nbformat": 4,
  "nbformat_minor": 5
}
