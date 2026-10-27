# Semi-supervised metric learning

Current training, evaluation, and analysis code for semi-supervised deep metric
learning with DINOv2 features. The project builds on
[gmberton/image-retrieval](https://github.com/gmberton/image-retrieval).

This snapshot contains the current source code and general-purpose scripts.
Recovered thesis configs and run manifests are organized under `configs/used/`;
other local and server config versions are under `configs/other/`. See the
[configuration guide](configs/README.md) for the phase structure, inheritance,
recorded run provenance, and validation commands. Tests, datasets, checkpoints,
training-result archives, machine-specific launch scripts, and generated
frontend snapshots are excluded.

## Setup

Use Python 3.12 and create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1`.
Install a PyTorch build compatible with your device when using CUDA. The base
requirements use CPU FAISS; optional GPU backends such as CUDA FAISS, CuPy,
cuML, TorchDR, and Triton depend on your hardware and chosen training options.

Inspect the available experiment and scheduler options with:

```bash
python main.py --help
python launch_runs.py --help
```

The supported config format is described in
[docs/experiment_config.md](docs/experiment_config.md), with scheduler usage in
[docs/parallel_runs.md](docs/parallel_runs.md). References in the detailed docs
to historical experiment paths describe local workflows; recovered files now
live under `configs/used/` or `configs/other/`.

## Code

- `main.py` dispatches experiments; `launch_runs.py` starts the independent-run scheduler.
- `training/` implements training, HPO, scheduling, tracking, and SSL algorithms.
- `models/`, `losses/`, and `utils/` contain models, losses, datasets, and shared helpers.
- `scripts/` contains download, diagnostic, benchmarking, and analysis tools.
- `evaluate/`, `main_eval/`, and `exporter/` process experiment results.
- `react_trial_visualizer/` contains the experiment browser source and its package configs.

The visualizer is optional. Start it with `npm ci` and `npm run dev` inside
`react_trial_visualizer/`; consult its [README](react_trial_visualizer/README.md)
for local data sources and exports.

Dataset downloaders and full training runs require their corresponding data,
dependencies, and device resources.

## Attribution

Retained upstream licensing and third-party attribution are in
[LICENSE-image-retrieval](LICENSE-image-retrieval) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
