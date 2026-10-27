# SLADE CompCars Unlabeled Pool

SLADE ([Duan et al., CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Duan_SLADE_A_Self-Training_Framework_for_Distance_Metric_Learning_CVPR_2021_paper.pdf))
pairs labeled Cars-196 with CompCars as unlabeled data and describes the pool in
one sentence:

> We use CompCars as the unlabeled data. It is collected at model level, so we
> filter out unbalanced categories to avoid being biased towards minority
> classes, resulting in 16,537 images categorized into 145 classes.

`external_unlabeled_filter="compcars_slade_paper"` builds that pool.

## Configuration

```json
{
  "dataset": "Cars196",
  "mode": "ssl",
  "ssl_config": "configs/ssl_slade.json",
  "unlabeled_source": "external",
  "external_unlabeled_dir": "data/CompCars",
  "external_unlabeled_filter": "compcars_slade_paper",
  "compcars_min_model_images": 100,
  "compcars_strict_paper_counts": true
}
```

[configs/experiments/slade_cars196_compcars.json](../configs/experiments/slade_cars196_compcars.json)
is a runnable version of this. `compcars_strict_paper_counts` turns a pool that
does not match the published counts into an error instead of a warning, which is
what you want for a paper reproduction.

CompCars is distributed by CUHK MMLab for non-commercial research
([dataset page](https://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html)). A
run whose `external_unlabeled_dir` holds no images downloads the web-nature
archive there first: 16.5 GB from the Google Drive copy and with the password
that the page's `instruction.txt` links, rejoined with `zip -F` and extracted
with `unzip`, so both must be installed. Drive refuses a file for up to a day
once its download quota is used up; such parts come from the Kaggle mirror
`duythinguyn/compcars`, whose `data.zip` (holding every file's CRC) is
byte-identical to Drive's (see
[`utils/external_pool_download.py`](../utils/external_pool_download.py)). The
archive extracts to the layout below. `external_unlabeled_dir` may point at the
release root or at its parent:

```text
data/CompCars/data/
  image/<make_id>/<model_id>/<released_year>/<image>.jpg
  label/
  misc/
  part/
  train_test_split/
    classification/
      train.txt
      test.txt
```

## What the filter does

1. **Candidates are all whole-car web-nature images.** Every image below the
   release's `image/` directory enters: 136,726 images of 1,716 models. Car
   parts (`part/`) and the surveillance-nature data never do, and the
   classification split files are not read.
2. **Categories are make/model/released-year directories.** CompCars files each
   image under `image/<make>/<model>/<year>/`; that directory is the category,
   4,446 of them.
3. **Unbalanced categories are dropped.** Categories holding fewer than
   `compcars_min_model_images` (default 100) images are removed. That keeps 145
   categories (of 144 distinct models; one model has two qualifying years) and
   16,537 images, 100 to 167 per category: the paper's numbers.
4. **The result is verified.** The realized counts are compared against
   16,537 images / 145 classes and written to
   `split/split_info.json` under `external_unlabeled_pool.filter_info`, together
   with the candidate-pool size (expected 136,726 / 4,446), the effective
   threshold, and a table of nearby thresholds. Retained images are appended
   after the Cars-196 train/validation split and carry the hidden label `-1`.
   `external_unlabeled_fraction` keeps a share of every category.

### Threshold calibration

The paper states the resulting pool but not the threshold that produced it, so
the 100-image default is an inference. If a local copy of CompCars lands beside
the published counts at that threshold, the filter searches for the threshold
that reproduces them exactly and records `calibrated_min_images_per_model: true`
plus the requested and effective values. Set
`"compcars_paper_threshold_calibration": "off"` to always use
`compcars_min_model_images` verbatim.

### Checking a local copy

```powershell
python scripts/check_compcars_slade_subset.py data/CompCars
```

The script runs the same discovery, category inference, and calibration as
training without reading image data, prints the candidate pool, the effective
threshold, the kept pool, and the nearest thresholds, and exits non-zero when
the pool does not match the paper. `--json report.json` dumps the per-category
counts for inspection.

## Inferred versus quoted

Quoted by SLADE: CompCars as the unlabeled data for Cars-196, model-level
categories, "filter out unbalanced categories", 16,537 images, 145 classes.

Inferred here, because SLADE released no code and specifies neither the source
subset, the category, nor the threshold. Checked on 2026-10-09 against the full
release's file list (the central directory in `data.zip`), exactly one reading
reproduces both numbers: all whole-car web-nature images, make/model/year
categories, at least 100 images. Nothing else tried reaches them: no count
threshold on CompCars Part-I (the classification split, 30,955 images of 431
models; 100 gives 13,565 / 100, 82 gives 17,538 / 144), on its train or test
half, on make/model or make categories of the full set, nor any min–max range.
An earlier version of this filter used Part-I with make/model categories, which
the published counts were assumed to imply before the data was on disk; that
version fails the count check on the real release.

## STML

STML ([Kim et al., CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Kim_Self-Taught_Metric_Learning_Without_Labels_CVPR_2022_paper.pdf))
states that it "directly adopt[s] the evaluation protocol of SLADE", including
the {Cars, CompCars} and {CUB, NABirds} pairings. `compcars_stml_paper` therefore
builds the identical pool and differs only in the paper label recorded in
`filter_info`. See [stml.md](stml.md) for the STML training setup and the NABirds
pool used for CUB.
