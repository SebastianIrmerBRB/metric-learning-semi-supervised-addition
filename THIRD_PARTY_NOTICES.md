# Third-Party Notices

## Long-Tailed CIFAR Dataset Generation

The CIFAR long-tail class-count schedule and per-class subsampling approach in
`utils.apply_cifar_long_tail` are adapted from:

- Repository: [richardaecn/class-balanced-loss](https://github.com/richardaecn/class-balanced-loss)
- Paper: *Class-Balanced Loss Based on Effective Number of Samples*, Yin Cui,
  Menglin Jia, Tsung-Yi Lin, Yang Song, and Serge Belongie, CVPR 2019.
- Upstream files: `src/data_utils.py` and `src/generate_cifar_tfrecords_im.py`

The upstream repository is distributed under the following MIT License:

> MIT License
>
> Copyright (c) 2018 Yin Cui
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.

## Self-Taught Metric Learning Without Labels (STML)

The two-head student model, contextual teacher-similarity, relaxed contrastive
objective, KL self-distillation, nearest-neighbor batching, and
exponential-moving-average teacher used by `metric_losses.STMLLoss` and the
STML training path in `experiment_training.py` are adapted from:

- Repository: [kdwonn/STML](https://github.com/kdwonn/STML)
- Bundled upstream source: `STML-CVPR22-main/STML-CVPR22-main`
- Paper: *Self-Taught Metric Learning without Labels*, Sungyeon Kim, Dongwon
  Kim, Minsu Cho, and Suha Kwak, CVPR 2022.
- Upstream files: `code/loss.py` and `code/main.py`

The implementation is integrated with this repository's DINO backbone and
optional supervised warm-up workflow while retaining STML's two student heads
and full loss objective.

Citation:

```bibtex
@inproceedings{kim2022self,
  title={Self-Taught Metric Learning without Labels},
  author={Kim, Sungyeon and Kim, Dongwon and Cho, Minsu and Kwak, Suha},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

The upstream repository is distributed under the following MIT License:

> MIT License
>
> Copyright (c) 2022 Sungyeon Kim
>
> Permission is hereby granted, free of charge, to any person obtaining a copy
> of this software and associated documentation files (the "Software"), to deal
> in the Software without restriction, including without limitation the rights
> to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
> copies of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all
> copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
> IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
> FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
> AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
> LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
> OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
> SOFTWARE.
