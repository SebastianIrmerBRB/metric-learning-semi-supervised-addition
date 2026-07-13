# metric-learning-semi-supervised-addition

Original repository: https://github.com/gmberton/image-retrieval

For CUDA support on Windows, install PyTorch wheels with (done automatically now, in requirements.txt):
`--extra-index-url https://download.pytorch.org/whl/cu128`

Sind nicht unbedingt in richtiger Reihenfolge erledigt (Skript erledigt einige der Sachen schon). 

| Status | Phase | Meilenstein                             |
|--------|-------|-----------------------------------------|
| ☑/2    | 1     | Forschungsfragen                        |
| ☑/2    | 2     | Literaturreview aufbauen                |
| ☑/2    | 3     | Hypothesen und Variablen definieren     |
| ☑      | 4     | Evaluationsdesign festlegen             |
| ☑/2    | 5     | Datensätze auswählen                    |
| ☑      | 6     | Baseline-Methoden implementieren        |
| ☑      | 7     | ViT-Models finden (CLIP/DinoV2)         |
| ☑      | 8     | Pilotexperimente durchführen            |
| ☑      | 9     | Log erweitern                           |
| ☑/2    | 10    | SSL Approaches finden                   |
| ☐      | 11    | SSL Approaches implementieren           |
| ☐      | 12    | Baseline Runs - Hyperparameter tuning?  |
| ☐      | 13    | Baseline Runs - Darstellungen           |
| ☐      | 14    | SSDML Runs - Parameter to tune?         |
| ☐      | 15    | SSDML Runs - Manual / automatic tuning? |
| ☐      | 16    | SSDML Runs - Gegenüberstellung Arten    |


...
## Problem zur Zeit:
- SSDML-Methoden sind nicht offen implementiert (LMNR, SERAPH sind bekannt, aber nicht direkt auf GitHub veröffentlicht)
- SSDML-Methoden sind oft nur transductiv (SSDML-Papers beschreiben Affinity Propagation, Label Propagation, ...), weil sonst Class-Predictions notwendig sind
- Evtl. Classifier-Head dazu machen? 

## Documentation (Work in progress, Daten dazu werden noch nicht mit hochgeladen)

- Sampler epoch length: [docs/length_before_new_iter.md](docs/length_before_new_iter.md)
- Cross-validation and validation modes: [docs/cross_validation.md](docs/cross_validation.md)
- Top-level experiment config: [docs/experiment_config.md](docs/experiment_config.md)
- Long-tailed CIFAR generation: [docs/cifar_long_tail.md](docs/cifar_long_tail.md)
- Semi-supervised FixMatch training: [docs/semi_supervised_fixmatch.md](docs/semi_supervised_fixmatch.md)
- Semi-supervised sklearn graph baselines: [docs/semi_supervised_sklearn.md](docs/semi_supervised_sklearn.md)
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
