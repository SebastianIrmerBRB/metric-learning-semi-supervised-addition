# metric-learning-semi-supervised-addition

Original repository: https://github.com/gmberton/image-retrieval

For CUDA support on Windows, install PyTorch wheels with (done automatically now, in requirements.txt):
`--extra-index-url https://download.pytorch.org/whl/cu128`

Sind nicht unbedingt in richtiger Reihenfolge erledigt (Skript erledigt einige der Sachen schon). 

| Status | Phase | Meilenstein                             |
|--------|-------|-----------------------------------------|
| ☑      | 1     | Forschungsfragen                        |
| ☑      | 2     | Literaturreview aufbauen                |
| ☑      | 3     | Hypothesen und Variablen definieren     |
| ☑      | 4     | Evaluationsdesign festlegen             |
| ☑/2    | 5     | Datensätze auswählen                    |
| ☑      | 6     | Baseline-Methoden implementieren        |
| ☑      | 7     | ViT-Models finden (CLIP/DinoV2)         |
| ☑      | 8     | Pilotexperimente durchführen            |
| ☑      | 9     | Log erweitern                           |
| ☑      | 10    | SSL Approaches finden                   |
| ☑      | 11    | SSL Approaches implementieren           |
| ☑      | 12    | Baseline Runs - Hyperparameter tuning?  |
| ☐      | 13    | Baseline Runs - Darstellungen           |
| ☑/2    | 14    | SSDML Runs - Parameter to tune?         |
| ☑      | 15    | SSDML Runs - Manual / automatic tuning? |
| ☐      | 16    | SSDML Runs - Gegenüberstellung Arten    |


...
## Problem zur Zeit:
- SSDML-Methoden sind nicht offen implementiert (LMNR ist bekannt, aber nicht direkt auf GitHub veröffentlicht)
- SSDML-Methoden sind oft nur transductiv (SSDML-Papers beschreiben Affinity Propagation, Label Propagation, ...), weil sonst Class-Predictions notwendig sind
- Evtl. Classifier-Head dazu machen? 

## Documentation (Work in progress, Daten dazu werden noch nicht mit hochgeladen)

- Sampler epoch length: [docs/length_before_new_iter.md](docs/length_before_new_iter.md)
- Cross-validation and validation modes: [docs/cross_validation.md](docs/cross_validation.md)
- Top-level experiment config: [docs/experiment_config.md](docs/experiment_config.md)
- Parallel independent runs across GPUs: [docs/parallel_runs.md](docs/parallel_runs.md)
- Long-tailed CIFAR generation: [docs/cifar_long_tail.md](docs/cifar_long_tail.md)
- Semi-supervised FixMatch training: [docs/semi_supervised_fixmatch.md](docs/semi_supervised_fixmatch.md)
- Semi-supervised sklearn graph baselines: [docs/semi_supervised_sklearn.md](docs/semi_supervised_sklearn.md)
- LP-DeepSSL / Iscen label spreading: [docs/iscen_label_spreading.md](docs/iscen_label_spreading.md)
- Deep mixed label propagation: [docs/mixed_label_propagation.md](docs/mixed_label_propagation.md)
- STML with supervised warm-up: [docs/stml.md](docs/stml.md)
- Example config values: [docs/example_config.yaml](docs/example_config.yaml)


- graph / fixmatch propagation repository: https://github.com/thomasbohm/semi-supervised-dml https://github.com/google-research/fixmatch

## Code organization

- `main.py` is the executable entry point and top-level experiment dispatcher.
- `training/` contains CLI/HPO orchestration, the training engine, shared result types, and the semi-supervised implementation. Focused SSL building blocks live in `training/ssl/`.
- `utils/` contains the shared utility API plus dataset composition, protocol, split, and local-dataset helpers.
- `models/` contains retrieval model implementations.
- `losses/` contains project-local metric-learning losses.

Use the package paths directly, for example `from training import semi_supervised` and `from models.retrieval_model import DinoWrapper`.

Add new behavior to the focused module for its responsibility, and keep cross-module re-exports intentional and limited.

## Download iNaturalist 2018 at DINO resolution

`--dataset iNat2018` uses a local wrapper around
`pytorch_metric_learning.datasets.INaturalist2018`. It streams the official
train/validation image archive, resizes one image at a time, and stores only
224x224 JPEGs under `data/iNat2018/train_val2018`. The 120 GB source archive and
full-resolution extracted images are never written to disk.

Download it separately before an experiment with:

```powershell
python scripts/download_inaturalist2018_224.py
```

Starting an experiment with `--dataset iNat2018` also starts the same download
automatically when the completion marker is absent. An interrupted run can be
started again; already completed 224x224 images are reused. Because the official
image release is one gzip archive, a retry must stream the archive again, and
the initial network transfer is still approximately 120 GB. Only persistent and
peak disk use are reduced.

The downloader retains the official metric-learning train/test split and accepts
`INaturalist2018` as an alias for `iNat2018`. `--dataset iNat` is reserved for
the distinct Semi-iNaturalist 2021 dataset. Downloading iNaturalist 2018 means
accepting the terms published with the official release.

## Stanford Dogs class-disjoint protocol

`--dataset StanfordDogs` uses the
[Stanford Dogs](http://vision.stanford.edu/aditya86/ImageNetDogs/) images at
224x224 resolution. The standalone download is:

```powershell
python scripts/download_stanford_dogs_224.py
```

An experiment also downloads the data automatically when needed. The official
image archive is streamed and each image is atomically resized, so the original
archive and full-resolution extracted images are not retained.

The [dataset paper](https://people.csail.mit.edu/khosla/papers/fgvc2011.pdf)
uses an image-level split containing all 120 breeds on both sides. For
class-disjoint metric-learning evaluation, this project instead pools the
20,580 images and uses a fixed hash-ranked partition:

- development: 72 complete breeds (60%);
- final test: 48 different complete breeds (40%).

The normal project holdout or cross-validation then operates only inside the 72
development breeds. The exact breed lists and partition version are saved in
the dataset completion marker and each run's dataset-protocol metadata.

## Oxford Flowers 102 class-disjoint protocol

`--dataset Flowers102` downloads the official
[Oxford Flowers 102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)
images and metadata through torchvision. The official train, validation, and
test sources contain all 102 flower categories, so they are pooled before the
metric-learning split is applied. Following the same contiguous-class
convention as CUB, class IDs 1-51 form the development pool and class IDs
52-102 are reserved for final testing.

This is an exact 51/51 class split, not a sample-balanced split. The dataset is
class-imbalanced, so the two sides contain 3,493 and 4,696 images respectively.
The project records both class lists, the realized sample counts, and the split
version in each run's dataset-protocol metadata. `Flowers`, `Flowers-102`,
`OxfordFlowers`, and `OxfordFlowers102` are accepted aliases.

## Semi-Aves 2020 protocols

`--dataset SemiAves` uses the released
[Semi-Aves split and oracle files](https://github.com/cvl-umass/ssl-evaluation/tree/main/data/semi_aves).
Two explicit protocols are available:

- `semi_aves_oracle_500_500` pools `l_train_val`, the released test labels,
  `u_train_in_oracle`, and `u_train_out_oracle` into 162,807 images from 1,000
  species, then assigns 500 complete species to development and 500 to final
  testing.
- `semi_aves_known_140_60` exposes only the 200 genuinely labeled target
  classes to the supervised pool. It creates a fixed class-disjoint 140-class
  train / 60-class validation split and retains the released 200-class test
  set for final evaluation.

In the known-class protocol, SSL automatically receives the original
unlabeled data. `u_train_in` is filtered to the 140 training classes so the 60
validation classes cannot leak into training. All 122,208 `u_train_out` images
from the 800 novel species are included with labels hidden, matching the
original Semi-Aves out-of-class setting.

After accepting the
[Semi-Aves Kaggle terms](https://www.kaggle.com/competitions/semi-inat-2020)
and configuring Kaggle credentials, prepare the 224x224 copy with:

```powershell
python scripts/download_semi_aves_224.py
```

Example protocol selection:

```powershell
python main.py --dataset SemiAves --dataset_protocol semi_aves_oracle_500_500
python main.py --dataset SemiAves --dataset_protocol semi_aves_known_140_60
```

`Semi-Aves`, `Semi_Aves`, and `SemiAves2020` are accepted aliases. The original
non-commercial research and educational-use terms continue to apply.

## Third-Party Attribution

Long-tailed CIFAR generation is adapted from
[richardaecn/class-balanced-loss](https://github.com/richardaecn/class-balanced-loss)
by Yin Cui et al. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the
upstream MIT license and citation details.

## Download DeepInFashion

1. kaggle datasets download -d hserdaraltan/deepfashion-inshop-clothes-retrieval -p data/DeepFashionInShop --unzip
2. then download the list partion  ```gdown --fuzzy "https://drive.google.com/file/d/0B7EVK8r0v71pYVBqLXpRVjhHeWM/view?usp=drive_link&resourcekey=0-rxJ2QcImN-IRo_Bv9QSXmg" -O list_eval_partition.txt```
3. put the list partition in data/DeepFashionInShop/In-shop Clothes Retrieval Benchmark/Eval
4. put the img_highres files into data/DeepFashionInShop/In-shop Clothes Retrieval Benchmark/Eval/Img 

## In-Shop With Fashion200K Unlabeled Images

External unlabeled images can be appended to SSL runs without changing the
official In-Shop query/gallery test set. Put Fashion200K images under a
recursive image root such as `data/Fashion200K`, then run:

```powershell
python main.py --experiment_config configs/experiments/class/in-shop-fashion200k.json
```

The config uses `unlabeled_source: split_and_external`, so pseudo-label SSL sees
both the In-Shop unlabeled candidates from the training split and all images
found below `external_unlabeled_dir`. Use `unlabeled_source: external` to train
with only the external unlabeled pool.

## CUB With NABirds Unlabeled Images

For the CUB + NABirds semi-supervised metric-learning protocol, download
NABirds from the [official dataset page](https://dl.allaboutbirds.org/nabirds),
extract the official metadata and `images/` directory under `data/NABirds`, and
run:

```powershell
python main.py --experiment_config configs/experiments/stml_cub_nabirds.json
```

The dedicated NABirds loader validates `images.txt`,
`image_class_labels.txt`, and `classes.txt`, then hides all source labels before
the images enter the SSL pool. See [docs/stml.md](docs/stml.md) for the expected
layout and protocol details.
