# Semi-supervised metric learning

Training, evaluation, and thesis figure generation for semi-supervised deep
metric learning with DINOv2 features. Built on
[gmberton/image-retrieval](https://github.com/gmberton/image-retrieval).

## Install and run a small training job

Use Python **3.12**. These commands are for Linux or WSL and use a CPU build of
PyTorch; an NVIDIA GPU is not required.

```bash
git clone https://github.com/SebastianIrmerBRB/metric-learning-semi-supervised-addition.git
cd metric-learning-semi-supervised-addition
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_environment.py
python main.py --experiment_config configs/other/quickstart/ssl.json
```

The example downloads the public **Oxford Flowers 102** dataset and DINOv2-S
weights on first use, so the first run needs internet access. No dataset account
or local research files are needed. It trains a 128-dimensional projection head
for two epochs: supervised warm-up followed by LRML semi-supervised training.
The small labeled support, capped unlabeled pool, and four batches per epoch
make this an installation check rather than a thesis benchmark.

For a supervised example:

```bash
python main.py --experiment_config configs/other/quickstart/supervised.json
```

Logs and resolved settings are written below `logs/quickstart/`, including
`info.log`, `metrics.csv`, and `run_config.json`. Downloads and backbone features
are cached in `data/` and the standard PyTorch cache for subsequent runs. Both
examples perform validation; the held-out final test set is not evaluated by
default.

On Windows, create the environment with `py -3.12 -m venv .venv` and activate it
with `.venv\Scripts\Activate.ps1`. The validated environment is Linux/Python
3.12; GPU builds on other platforms should follow the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).

## NVIDIA GPU installation

Use a **separate** environment for CUDA rather than installing CPU and CUDA
builds on top of each other. The supplied CUDA requirements select CUDA 13
PyTorch wheels and CuPy; the NVIDIA driver must support that CUDA generation.

```bash
python3.12 -m venv .venv-cuda
source .venv-cuda/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-cuda.txt
CUDA_VISIBLE_DEVICES=0 python scripts/check_environment.py --cuda
CUDA_VISIBLE_DEVICES=0 python main.py --experiment_config configs/other/quickstart/ssl.json --device cuda --ssl-device cuda
```

Change `CUDA_VISIBLE_DEVICES` to an available GPU on a shared server. CPU FAISS
is sufficient for these examples and is also used in the CUDA environment.
For another CUDA generation, choose matching PyTorch and CuPy builds using the
upstream installation instructions.

## Reproduce thesis figures and tables

The complete generator and compact data are included in
[`thesis_generator/`](thesis_generator/README.md). The root requirements cover
both training and figure generation; no extra installation is needed.

```bash
python thesis_generator/generate.py
```

This validates the bundled data and writes PDFs, LaTeX tables, PNG previews,
and script logs under `thesis_generator/ausgabe/`. It requires neither image
datasets nor training logs, model weights, LaTeX, or network access.

```bash
python thesis_generator/generate.py --only fig_phase2 tab_extern
python thesis_generator/generate.py --output /path/to/figures
```

The data are the compact experiment measurements and detailed inputs from the
provided generator package. They are separate from the image datasets used for
training. Raw training logs, full image datasets, checkpoints, and test suites
are not included in this repository.

## Run the recovered thesis experiments

[`configs/used/`](configs/README.md) contains inherited Phase-0, Phase-1 and
Phase-2 settings and run manifests, with source hashes and historical trial
inputs. `configs/other/local/` and `configs/other/server/` preserve the remaining
config versions. These full experiments need their corresponding datasets and
can take substantially longer than the quickstart.

From the repository root, select the `experiment` path in
`configs/used/catalog/runs.jsonl`, then run:

```bash
python main.py --experiment_config <experiment-path>
# Or inspect one of the recovered schedules before launching it:
python launch_runs.py --manifest <manifest-path> --dry-run
```

The dataset loaders download supported public datasets automatically. In-Shop,
Semi-iNaturalist, and other restricted datasets require the corresponding
source access and terms; see [dataset protocols](docs/dataset_protocols.md) and
the download scripts in `scripts/`. Existing datasets can be placed or linked
under `data/<dataset-name>/`. Never delete a dataset while a training job uses it.

Historical Phase-0 searches sometimes resumed with changed fixed settings;
those regimes are retained in the recorded trial inputs. The
[configuration guide](configs/README.md) explains which files are runnable
recipes and which preserve historical metadata.

## Checks and troubleshooting

```bash
python -m pip check
python scripts/check_environment.py
python main.py --help
python launch_runs.py --help
python scripts/validate_thesis_configs.py --native
```

- **CUDA unavailable:** use the CPU quickstart, or install the CUDA requirements
  in a fresh environment and check the driver with `nvidia-smi`.
- **Not enough home-directory quota:** put the virtual environment and pip cache
  on a filesystem with space. For example, set `PIP_CACHE_DIR` to a scratch
  directory and create the venv there; do not change a shared training environment.
- **First run is slower:** dataset/model downloads and initial backbone feature
  extraction happen once and are reused on later runs. CPU feature extraction
  can take several minutes, even though each quickstart epoch has four batches.
- **xFormers warnings:** DINOv2 can report that this optional accelerator is
  unavailable. Both quickstarts work without it.
- **Dataset access or download failure:** check the dataset's access requirements;
  a partial/incomplete dataset is not a successful training run.

## Verification

Verified on Linux with fresh Python 3.12.3 environments: the supervised and
LRML examples completed real two-epoch training runs on CPU; the CUDA example
completed on an NVIDIA H100 with CUDA 13 and driver 580.173.02. The CUDA check
also covered CuPy and a fresh DINOv2 model-download cache. TensorBoard logging,
all recovered config/manifests, and all 25 figure/table generator groups passed.

## Code and optional visualizer

`main.py` dispatches experiments; `launch_runs.py` schedules independent runs.
`training/`, `models/`, `losses/`, and `utils/` implement training, SSL, model,
loss, and dataset functionality. `evaluate/`, `main_eval/`, `exporter/`, and
`scripts/` contain evaluation and analysis tools.

`react_trial_visualizer/` is an optional experiment browser. Its Node/npm
installation is separate from Python training; consult its
[README](react_trial_visualizer/README.md).

## Attribution

See [LICENSE-image-retrieval](LICENSE-image-retrieval) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). The figure bundle includes its
plotting font license inside `thesis_generator/daten/details.zip`.
