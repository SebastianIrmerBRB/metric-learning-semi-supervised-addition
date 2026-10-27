# Long-Tailed CIFAR

Set `--cifar_imbalance_factor` to create a long-tailed CIFAR training or
development source. The final test set remains balanced.

```powershell
python main.py --dataset CIFAR100 `
  --dataset_protocol cifar100_fine_class_disjoint `
  --cifar_imbalance_factor 0.01
```

The factor is `img_min / img_max`, matching the
[richardaecn/class-balanced-loss](https://github.com/richardaecn/class-balanced-loss)
implementation. For example, `0.01` produces approximately 100:1
head-to-tail imbalance. `None`, the default, keeps the original balanced
dataset.

For class index `i` among `C` sorted classes, the retained count is:

```text
int(img_max * factor ** (i / (C - 1)))
```

The experiment seed controls which examples are retained. The realized counts,
seed, factor, and upstream attribution are written to the dataset protocol
section of `split_info.json`.

Please cite the source paper when using this dataset construction:

```bibtex
@inproceedings{cui2019classbalancedloss,
  title={Class-Balanced Loss Based on Effective Number of Samples},
  author={Cui, Yin and Jia, Menglin and Lin, Tsung-Yi and Song, Yang and Belongie, Serge},
  booktitle={CVPR},
  year={2019}
}
```
