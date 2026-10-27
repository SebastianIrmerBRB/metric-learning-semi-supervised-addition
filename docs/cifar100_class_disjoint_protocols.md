# CIFAR-100 Class-Disjoint Protocols

CIFAR-100 has no single canonical deep metric learning split. CIFAR-FS uses a
64/16/20 fine-class split, while FC100 uses a superclass-based 60/20/20 split.
This repository provides the exact FC100 split plus two alternative fixed
class-disjoint protocols and records the resolved IDs in every run's
`split/split_info.json`.

Both protocols discard the official train/test boundary and pool all images
first. Every fine class contributes its 500 official-train plus 100
official-test images, so the fine-class protocol uses:

```text
600 images/fine class
60 development fine classes = 36,000 development images
40 test fine classes        = 24,000 final-test images
```

The per-fine-class ceiling is therefore 600. An optional
`cifar_imbalance_factor` subsamples only the development side after this fixed
class partition; final test remains balanced.

## FC100 (TADAM)

Use:

```text
dataset_protocol=cifar100_fc100
```

This reproduces the split from the
[TADAM paper and supplementary material](https://proceedings.neurips.cc/paper/2018/hash/66808e327dc79d135ba18e051673d906-Abstract.html).
It is compatible with the 12/4/4 split exposed by
[learn2learn's FC100 dataset](https://github.com/learnables/learn2learn/blob/master/learn2learn/vision/datasets/fc100.py):

```text
canonical train superclasses:      1, 2, 3, 4, 5, 6, 9, 10, 15, 17, 18, 19
canonical validation superclasses: 8, 11, 13, 16
fixed final-test superclasses:      0, 7, 12, 14
```

With `cv_k=1` and `val_mode=all` or `match_train`, the canonical 12/4
train/validation split is used. For cross-validation, the canonical train and
validation sides form one
16-superclass development pool while the four test superclasses remain fixed.
`cv_mode=superclass_group_kfold` is required; with `cv_k=4`, every fold trains on 12
superclasses (60 fine classes) and validates on four superclasses (20 fine
classes).

```powershell
python main.py --dataset CIFAR100 `
  --dataset_protocol cifar100_fc100 `
  --cv_k 4 `
  --cv_mode superclass_group_kfold
```

## Fine-Class Disjoint

Use:

```text
dataset_protocol=cifar100_fine_class_disjoint
```

This is the closer analogue to Cars196/SOP: development and final test use
different fine classes, but semantically related fine classes from every
superclass occur on both sides. For every superclass, the first three sorted
fine-class IDs go to development and the remaining two go to final test. This
produces exactly 60/40 fine classes.

| Superclass ID | Superclass | Development fine IDs | Test fine IDs |
|---:|---|---|---|
| 0 | aquatic_mammals | 4, 30, 55 | 72, 95 |
| 1 | fish | 1, 32, 67 | 73, 91 |
| 2 | flowers | 54, 62, 70 | 82, 92 |
| 3 | food_containers | 9, 10, 16 | 28, 61 |
| 4 | fruit_and_vegetables | 0, 51, 53 | 57, 83 |
| 5 | household_electrical_devices | 22, 39, 40 | 86, 87 |
| 6 | household_furniture | 5, 20, 25 | 84, 94 |
| 7 | insects | 6, 7, 14 | 18, 24 |
| 8 | large_carnivores | 3, 42, 43 | 88, 97 |
| 9 | large_man-made_outdoor_things | 12, 17, 37 | 68, 76 |
| 10 | large_natural_outdoor_scenes | 23, 33, 49 | 60, 71 |
| 11 | large_omnivores_and_herbivores | 15, 19, 21 | 31, 38 |
| 12 | medium_mammals | 34, 63, 64 | 66, 75 |
| 13 | non-insect_invertebrates | 26, 45, 77 | 79, 99 |
| 14 | people | 2, 11, 35 | 46, 98 |
| 15 | reptiles | 27, 29, 44 | 78, 93 |
| 16 | small_mammals | 36, 50, 65 | 74, 80 |
| 17 | trees | 47, 52, 56 | 59, 96 |
| 18 | vehicles_1 | 8, 13, 48 | 58, 90 |
| 19 | vehicles_2 | 41, 69, 81 | 85, 89 |

## Superclass Disjoint

Use:

```text
dataset_protocol=cifar100_superclass_disjoint
```

This is the harder, cleaner open-set protocol. Complete superclasses are kept
together, so no superclass appears on both sides:

```text
development superclass IDs: 0, 2, 4, 6, 8, 10, 12, 14, 16, 18
final-test superclass IDs:   1, 3, 5, 7, 9, 11, 13, 15, 17, 19
```

Each side therefore contains 10 superclasses, 50 fine classes, and 30,000
pooled images.

## Legacy Protocol

`cifar100_unseen_classes` remains available for reproducibility. It uses fine
class IDs `0..49` for development and `50..99` for final test. It is not
superclass-aware and should not be used as the default for new CIFAR-100 DML
experiments.
