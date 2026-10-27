# Dataset Protocols

`--dataset_protocol` (or `"dataset_protocol"` in an experiment config) selects
how a dataset's samples are divided into the development pool that HPO and
cross-validation see and the final-test pool that is only touched after tuning.
Every run records the resolved protocol under `dataset_protocol` in
`split/split_info.json`.

The authoritative list lives in `utils/dataset_constants.py`:

```text
DATASET_PROTOCOLS      the names accepted by --dataset_protocol
DATASET_PROTOCOL_INFO  one entry per protocol: datasets, disjointness, summary
```

`--help` renders the same registry, and
`tests/test_dataset_protocol_registry.py` fails if this document, the registry,
and `validate_dataset_protocol` ever disagree.

## All protocols

| Protocol | Datasets | Class-disjoint test | What it does |
| --- | --- | --- | --- |
| `official` | all | depends on the dataset | Keeps each dataset class's own train/test boundary. See [What `official` means per dataset](#what-official-means-per-dataset). |
| `cifar_balanced_fraction` | CIFAR10, CIFAR100 | no | Pools the official splits and draws sample-disjoint, class-balanced development and test subsets sized by `cifar_train_fraction` / `cifar_test_fraction`. |
| `cifar10_unseen_classes` | CIFAR10 | yes | Develops on classes 0-7, holds out 8-9. |
| `cifar100_unseen_classes` | CIFAR100 | yes | Develops on fine classes 0-49, holds out 50-99. The contiguous cut splits superclasses, so both halves share coarse labels. |
| `cifar100_fine_class_disjoint` | CIFAR100 | yes | Takes 3 of the 5 fine classes of every superclass, holds out the other 2. All 20 superclasses appear on both sides. |
| `cifar100_superclass_disjoint` | CIFAR100 | yes | Develops on the 10 even superclasses, holds out the 10 odd ones. No held-out class has a coarse relative in development. |
| `cifar100_fc100` | CIFAR100 | yes | Reproduces the TADAM FC100 12/4/4 superclass split. Requires `cv_mode=superclass_group_kfold` when `cv_k > 1`. |
| `semi_aves_oracle_500_500` | SemiAves | yes | Pools every released label and alternates all 1,000 classes into a fixed 500/500 split, so the most-to-least-frequent id ordering is halved evenly. |
| `semi_aves_oracle_hash_500_500` | SemiAves | yes | The same oracle pool split 500/500 by SHA-256 hash rank instead of alternation, as a check that a result does not depend on the id ordering. |
| `semi_aves_known_100_100` | SemiAves | yes | Splits the 200 known classes 100/100 by alternating class id and attaches the out-of-class pool as native unlabeled data (`--semi_aves_ood_fraction`). |
| `semi_aves_known_hash_100_100` | SemiAves | yes | The same known pool split 100/100 by SHA-256 hash rank instead of alternation. Same unlabeled pool; the image balance is left to the hash. |
| `semi_inat_oracle_50_50` | iNat | yes | Pools every released label and alternates all 2,439 species into two halves of roughly equal size. |
| `semi_inat_known_810` | iNat | yes | Splits the 810 labeled species and attaches the 1,629 out-of-class species as native unlabeled data (`--semi_inat_ood_fraction`). |
| `cub_alternating_100_100` | CUB | yes | Pools CUB's official halves and alternates class ids 100/100, so each genus appears on both sides instead of only one. |
| `nabirds_alternating_277_278` | NABirds | yes | Alternates whole hierarchy-parent groups instead of taking a prefix of them, which evens the images at 50.5/49.5 instead of the 46.4/53.6 `official` gives. |
| `nabirds_puma_278_277` | NABirds | yes | Reproduces the PUMA benchmark's cut at the class id midpoint for comparability with its published numbers. Ignores the hierarchy, so 12 species keep categories on both sides. |
| `stanford_dogs_puma_60_60` | StanfordDogs | yes | Reproduces the PUMA benchmark's Stanford Dogs cut, the first 60 breed directories against the last 60, instead of the default hash-ranked 72/48 split. |

The three CIFAR-100 class-disjoint protocols are described in detail in
[`docs/cifar100_class_disjoint_protocols.md`](cifar100_class_disjoint_protocols.md).

## What `official` means per dataset

`official` never invents a split: it asks the dataset class for its `train` and
`test` splits. What that returns differs, because only some providers ship a
class-disjoint benchmark split. Datasets whose provider split shares classes
between the halves pool it first and reassign complete classes, which is
recorded as `official_image_level_split_used: false` in `split_info.json`.

| Dataset | `official` gives | Class-disjoint test |
| --- | --- | --- |
| Cars196 | Provider split: classes 1-98 develop, 99-196 held out. | yes |
| CUB | Provider split: classes 1-100 develop, 101-200 held out. | yes |
| StanfordOnlineProducts | Provider `Ebay_train.txt` / `Ebay_test.txt`. | yes |
| DeepFashionInShop | Official train partition; test is query-to-gallery retrieval. | yes |
| CIFAR10, CIFAR100 | Provider train/test batches, which share every class. | no |
| iNat2018 | The metric-learning `Inat_dataset_splits/Inaturalist_{train,test}_set1.txt` lists over the pooled train2018+val2018 images. | yes |
| iNat2019 | Pooled official train+validation images; HPL's first 656 species (48 genera) develop and the remaining 354 species (24 genera) are held out. | yes |
| iNat2021Mini | Official 500,000-image `train_mini`; the Contextual Similarity authors' released 5,003-species development / 4,997-species test partition. | yes |
| Food101 | Pooled official train+test, complete classes to a fixed 71/30 split. | yes |
| Flowers102 | Pooled official train+val+test, contiguous class ids 51/51. | yes |
| FGVCAircraft | Pooled official train+val+test, first 50 variants develop, last 50 held out. | yes |
| FGVCFungi | Meta-Dataset train+validation species pooled for CV (1,194); its 200 test species remain held out. | yes |
| StanfordDogs | Pooled official lists, hash-ranked 72/48 breed split. | yes |
| NABirds | Pooled official train+test, a prefix of the hierarchy-parent groups develops, 277/278. | yes |
| Pittsburgh30k, Pittsburgh250k | Official geographic splits: `train`+`val` pooled develop, `test` is query-to-gallery retrieval. | yes |
| SPED | Pooled camera observations; SHA-256-ranked camera ids split 50/50, with every frame from one camera kept together. | yes |
| SemiAves | The `known` 100/100 split, i.e. the default mode of the dataset class. Prefer naming `semi_aves_known_100_100` or `semi_aves_oracle_500_500` explicitly. | yes |
| VehicleID | The released identity lists: one training list against the 2,400-identity held-out list, the largest of the three published evaluation sizes. Name `vehicleid_test_800` or `vehicleid_test_1600` for the smaller two. | yes |
| PUMA, PUMAStandard | The union of the member datasets, each on its own `official` split. See [PUMA unified benchmark](#puma-unified-benchmark). | yes |
| iNat | The `oracle` 50/50 split, i.e. the default mode of the dataset class. Prefer naming `semi_inat_oracle_50_50` or `semi_inat_known_810` explicitly. | yes |

`official` asserts class-disjointness only for the datasets that advertise it
through `class_disjoint_split`, which are exactly the ones that repartition. For
the provider splits above, the property comes from the benchmark itself and is
not re-checked at load time.

## iNaturalist 2019

`iNat2019` reproduces the retrieval split from
[Hierarchical Proxy-Based Loss](https://openaccess.thecvf.com/content/WACV2022/html/Yang_Hierarchical_Proxy-Based_Loss_for_Deep_Metric_Learning_WACV_2022_paper.html).
It pools all 268,243 labeled competition images, then uses category ids 0--655
(656 species, 48 complete genera, 189,625 images) as development and ids
656--1009 (354 species, 24 complete genera, 78,618 images) as the final test.
Cross-validation operates only on the development species. Both classes and
genera are disjoint across the final boundary.

The official 74 GB image archive is streamed and each image is immediately
resized, so only atomic 224x224 JPEGs are retained:

```bash
python scripts/download_inaturalist2019_224.py
```

The annotation archives and an atomic compact index remain beside the images.
An interrupted run reuses already-complete 224x224 files. A manually downloaded
`train_val2019.tar.gz` placed in `data/iNat2019` is streamed locally instead.

## mini-iNaturalist-2021

`iNat2021Mini` is the mini-iNaturalist-2021 benchmark introduced for metric
learning by [Contextual Similarity Optimization](https://proceedings.mlr.press/v202/liao23b.html).
It uses the complete official `2021_train_mini` collection: 50 images for each
of 10,000 species, or 500,000 images total. The loader reproduces the exact
species lists from the [authors' released code](https://github.com/Chris210634/metric-learning-using-contextual-similarity/tree/8433dcb67c2205c0e30ec07ed1e5b2fb92da016d/data),
not a newly seeded random split.

The paper describes the partition as 5,000 training and 5,000 test species. Its
released notebook actually shuffles and halves each of 11 supercategories
separately. Six supercategories have odd sizes, so the released manifests are
5,003 development species (250,150 images) and 4,997 held-out test species
(249,850 images). This implementation treats the published files as canonical;
cross-validation operates only on the 5,003 development species.

Download and resize it with:

```bash
python scripts/download_mini_inaturalist2021_224.py
```

The official archive is streamed. Each source image is immediately converted
to an atomic 224x224 JPEG, and the source archive is never stored. The temporary
199 MB annotation JSON is replaced by a compact gzip index and taxonomy file.
Interrupted downloads reuse already-complete images. If
`data/iNat2021Mini/train_mini.tar.gz` was downloaded manually, it is streamed
locally instead (and is not deleted).

## SPED

The full Specific PlacEs Dataset (SPED) is a training collection, not the
607-query/607-reference `SPEDTest` benchmark commonly used in visual-place
recognition evaluation. The [SPED paper](https://arxiv.org/abs/1701.05105)
selected 2,543 static outdoor cameras from the Archive of Many Outdoor Scenes,
collected their February and August 2014 frames, and produced roughly 2.5
million images. A camera id is the place label, so one class has hundreds of
observations across day/night, weather, illumination, and season changes.

That is the structure exposed by this loader. One accepted layout is a native
camera tree below `data/SPED`:

```text
data/SPED/images/<camera_id>/<optional year/month folders>/<image>
```

The equivalent `data/SPED/<train|val|test>/<camera_id>/...` layout is also
accepted; those sample-level directories are pooled before cameras are
partitioned, so a camera is never mistaken for several classes or split across
the final boundary.

Alternatively, create `data/SPED/sped_manifest.csv` with these required
columns:

```csv
path,camera_id
images/000001/2014/02/20140201_120000.jpg,000001
images/000001/2014/08/20140801_120000.jpg,000001
```

Paths are relative to the directory holding the manifest. The aliases
`image_path`/`filepath` and `place_id`/`location_id`/`class_id` are accepted.
On the first load of an unindexed camera tree, the loader writes
`.sped_index_v1.csv` atomically; subsequent train/test constructions read that
index instead of walking approximately 2.5 million files again.

SPED did not release a metric-learning class split. Here, `official` pools the
prepared observations, ranks camera ids by
`sha256("sped_camera_sha256_half_v1:<camera_id>")`, and gives the first
`ceil(number_of_cameras / 2)` cameras to development. The rest are held out for
testing. Thus the paper's 2,543 cameras become 1,272 development and 1,271 test
classes. The split is independent of directory enumeration order, and all
February/August observations of a camera stay on one side. This leaves the
development classes dense enough for experiments such as five labelled
observations plus hundreds of unlabelled observations per location while the
final metric-learning test still contains unseen places. The exact split
version, class counts, sample counts, and SHA-256 digests of both camera-id
lists are recorded in `split_info.json`.

Automatic download is intentionally disabled. The paper says that the authors
randomly selected the 2,543 cameras and then randomly selected 1,272,000
training images plus roughly 120,000 validation images, but it publishes
neither the camera ids, image filenames, nor a sampling seed. The authors'
archived release page still said that the dataset would go online and exposed
model files only. Consequently, the exact training collection cannot be
reconstructed from AMOS, even if the much larger source archive is available.
Supplying a new random AMOS subset is not silently presented as the paper's
SPED selection; point the loader at an author-provided tree or exact manifest.

The publicly downloadable **SPEDTest** collection is different: it contains
607 matched query/reference pairs (1,214 images total) for VPR evaluation. It
is manageable and reproducible, but two observations per place provide no
meaningful within-class unlabeled pool, so this loader does not substitute it
for the paper's 2,543-class SPED training collection.

## NABirds

NABirds ships 48,562 images over 555 visual categories, and its official
`train_test_split.txt` puts every category on both sides, so `official` pools it
and repartitions by class like CUB.

The complication is that NABirds separates sexes and age classes of one species
into different visual categories -- for example `Adult male`, `Adult female` and
`Immature` of the same bird -- and lists them as children of a shared hierarchy
node in `hierarchy.txt`. Cutting the ordered class ids at the midpoint -- or
alternating them the way CUB does -- can put one plumage in development and its
sibling in the final test set, which leaks a near-identical training class into
the evaluation. Both NABirds protocols therefore deal out complete
hierarchy-parent groups, ordered by their lowest class id, and differ only in
how they deal them:

```text
official                     hierarchy_parent_groups_in_class_id_order
nabirds_alternating_277_278  alternating_hierarchy_parent_groups_in_class_id_order
```

`official` takes groups from the front until the development half holds at
least half of the categories. `nabirds_alternating_277_278` takes every other
group. The 555 categories sit in 404 parent groups -- 267 hold a single
category and the other 137 hold the two to five plumages of one species -- and
both rules split them 277/278:

| | development | held out | development share | straddling species |
| --- | --- | --- | --- | --- |
| `official` | 277 categories, 22,534 images, 186 parents | 278 categories, 26,028 images, 218 parents | 46.4% | 0 |
| `nabirds_alternating_277_278` | 277 categories, 24,548 images, 202 parents | 278 categories, 24,014 images, 202 parents | 50.5% | 0 |
| `nabirds_puma_278_277` | 278 categories, 22,909 images | 277 categories, 25,653 images | 47.2% | 12 |

The difference is worth knowing before you pick one: the low class ids are the
sparser taxa, so the prefix that `official` takes leaves development holding
less than half the images -- the smaller half is the one that trains, and the
larger one becomes a harder retrieval gallery. Striding evens both. `official`
stays the default so that runs already logged against
`nabirds_parent_group_class_disjoint_v1` remain reproducible; the alternating
rule records `nabirds_alternating_parent_groups_v1` instead. The two are not
comparable to each other, since they train and test on different categories.

### The PUMA split

`nabirds_puma_278_277` is the exception to all of the above: it ignores
`hierarchy.txt` and cuts the sorted class ids at their midpoint, giving
development the first 278 categories and holding out the last 277. That is the
split PUMA uses in its unified metric learning benchmark
([arXiv:2309.08944](https://arxiv.org/abs/2309.08944), WACV 2025), whose Table 1
reports 278 train classes over 22.9K images and 277 test classes over 25.6K --
matched here exactly at 22,909 and 25,653.

The paper itself publishes no class ids -- neither the WACV version nor the
arXiv one states a class-selection rule -- but the authors' implementation does,
and this rule is theirs verbatim
([PUMA-WACV25/dataset.py](https://github.com/sung-yeon-kim/PUMA-WACV25/blob/main/dataset.py)):

```python
label_list = list(set(image_class_labels['target']))
label_list = sorted(label_list)
label_map = {k: i for i, k in enumerate(label_list)}
...
classes = range(0, 278)      # train
classes = range(278, 555)    # eval / query / gallery
```

So the split is fully determined and carries no randomness: sort the 555 unique
raw targets, number them `0..554`, and cut at 278. `docs/nabirds_puma_class_split.json`
holds the resulting class lists -- raw NABirds id, densified label, class name
and image count for all 555 categories -- so the partition can be checked
without the images. The loader here is verified against it end to end and
reproduces the paper's Table 1 exactly, at 22,909 and 25,653 images.

Two details are easy to get wrong. First, `0..554` are **remapped** labels, not
raw NABirds ids: the labeled categories run from **295 to 1010** inside a
1,011-node `classes.txt` that also names interior taxa like "Finches and
Allies", so taking raw ids `0..277` gives nothing at all. Label 277 is class 674
(*Anna's Hummingbird, female/immature*) and label 278 is class 675 (*Costa's
Hummingbird, female/immature*) -- that is where the cut falls.

Second, the official `train_test_split.txt` is **not used**. PUMA's loader reads
it into its dataframe but never tests `is_training_img`; membership is decided
by the remapped class id alone. That is visible in the totals rather than only
in the code: the official image-level split is 23,929 / 24,633 with all 555
categories on both sides, whereas repartitioning all 48,562 images by class
gives the 22,909 / 25,653 the paper reports. Had the image-level split been
applied as a filter, no class partition could recover those numbers.

For the record, the ordering rule is also the only plausible one that lands on
the published counts, which is how this split was pinned down before the
implementation surfaced:

| class ordering, same 278/277 cut | train / test images |
| --- | --- |
| **sorted class id (the official rule)** | **22,909 / 25,653** |
| alphabetical by class name | 24,060 / 24,502 |
| descending class id | 25,762 / 22,800 |
| by hierarchy parent, then class id | 22,719 / 25,843 |
| by images per class, descending | 30,879 / 17,683 |

It is provided for comparability, not because it is the better split. NABirds
does not number the plumages of a species consecutively -- not one of the 137
multi-category groups is contiguous -- so the midpoint cut leaves 12 species
with categories on both sides:

```text
[dev]  Red-tailed Hawk (Light morph adult)     [test] Red-tailed Hawk (Light morph immature)
[dev]  Northern Flicker (Yellow-shafted)       [test] Northern Flicker (Red-shafted)
[dev]  Common Eider (Adult male, Female/juv.)  [test] Common Eider (Immature/Eclipse male)
```

...and five hummingbirds where development gets the adult male and the held-out
set gets the female/immature of the same species. The halves remain disjoint in
visual categories, which is what `class_disjoint_split` asserts; it is the
species that leak. Every NABirds split therefore records
`straddling_parent_count` in `split_info.json` -- zero for the two group-based
rules, twelve for this one -- so a run cannot quietly inherit the leak.

`development_parent_count` and `test_parent_count` in `split_info.json` record
how many parent nodes ended up on each side.

### Getting the data

Cornell releases NABirds only after its terms are accepted, so the automatic
path reads a third-party Kaggle mirror of the 2015 release. The images stay
Cornell's and remain under the terms at
<https://dl.allaboutbirds.org/nabirds>; the mirror only removes the
click-through.

```bash
python scripts/download_nabirds_224.py                      # via kagglehub
python scripts/download_nabirds_224.py --source data/archive.zip
```

Like the other downloaders here, this stores 224x224 JPEGs rather than the
10 GB of originals, and it writes them in the official layout -- `images/` plus
the four metadata files, under their official relative paths -- so nothing
downstream has to know the data came from Kaggle. It is resumable: images
already at the target size are left alone. `--source` takes an archive or an
already extracted official directory; without it, an `archive.zip` in or beside
`data/NABirds` is used before anything is downloaded, and an archive that is
not NABirds is skipped rather than reported as broken.

`.nabirds_224_complete.json` marks a finished resize and is what
`NABirds.is_224_complete` checks. `NABirds.is_ready` is the weaker test -- it
only asks whether an official layout is present -- so extracting the full
resolution archive under `data/NABirds` by hand still works and is preferred if
you accepted the terms at the source. The downloader refuses to resize a
directory into itself, so a hand-extracted copy cannot be silently downsized.

The same directory doubles as the unlabeled bird pool for CUB
(`--external_unlabeled_filter nabirds`), which reads the identical metadata but
discards the labels.

Note that NABirds keys `images.txt`, `image_class_labels.txt` and
`train_test_split.txt` by UUID rather than by a running integer, while
`classes.txt` and `hierarchy.txt` use integer node ids. Both readers pass the
key type per file for that reason.

## Pittsburgh

`Pittsburgh30k` and `Pittsburgh250k` are the two NetVLAD visual place
recognition benchmarks, and they are the only datasets here whose classes are
not a provided annotation. A *place* is one Google Street View panorama, and the
released images are its 24 perspective crops -- 12 yaw directions at two
pitches. Queries are crops of panoramas photographed at the same locations years
later, which is where the temporal variation comes from; they live in
`queries_real/` and never appear in the database.

The dataset class turns that into labels:

```text
database crop   label = its own panorama id
query crop      label = the nearest database panorama within posDistThr (25 m)
```

`posDistThr` is read from the split's `dbStruct`, not hard-coded, and
`--place_radius_m` has no CLI equivalent -- pass `place_radius_m` to the dataset
class directly if you want a different radius. Queries with no database
panorama inside the radius cannot be answered by a class-based retrieval
protocol and are dropped; `unmatched_query_image_count` in `split_info.json`
records how many.

The panorama id doubles as the integer class label. That keeps labels stable
across splits without reading every split's metadata, and it is checked: all
crops sharing a panorama id must sit within one metre of each other, so an id
namespace collision fails the load instead of quietly merging two streets into
one class.

### This is not the published Recall@N

Published Pittsburgh numbers count a retrieval correct when *any* database image
within 25 m of the query is returned. Here only the crops of the single nearest
panorama carry the query's label, so `precision_at_1` is strictly harsher and
the two numbers are not comparable. `geographic_positive_labels()` on the
dataset returns the full radius-based positive list per query for anyone who
wants to score the published way instead.

### Splits

Pittsburgh ships its own validation split, but this project carves validation
out of the development pool itself, so `official` pools `train`+`val` for
development and holds out `test`; leaving `val` alone would waste a third of the
data. The dataset class still accepts `train`, `val`, `test`, `train+val` and
`train+val+test` directly. The three official splits are cut geographically, so
no panorama appears in more than one and `class_disjoint_split` holds.

| | database | queries | | database | queries |
| --- | --- | --- | --- | --- | --- |
| **Pitts30k** train | 10,000 | 7,416 | **Pitts250k** train | 91,464 | 7,824 |
| val | 10,000 | 7,608 | val | 78,648 | 7,608 |
| test | 10,000 | 6,816 | test | 83,952 | 8,280 |
| total | 30,000 | 21,840 | total | 254,064 | 23,712 |

`test` is scored query-to-gallery: `num_test_queries` queries against
`num_test_gallery` database crops, with the database crops no query matched left
in as distractors. Loaded counts are compared against this table and a mismatch
is logged as a warning; pass `strict_expected_counts=True` to raise instead. The
Pitts250k `train` and `val` rows are derived from NetVLAD's published totals
rather than quoted from it, which is the other reason the check warns by
default.

### Preparation

The collection is released only on request, so it is never downloaded
automatically. Extract the raw NetVLAD distribution under `data/Pittsburgh250k`
(or `data/Pittsburgh30k`) so that it holds:

```text
000/ ... 010/    database crops, <pano_id>_pitch<p>_yaw<y>.jpg
queries_real/    query crops, same naming
datasets/        pitts30k_{train,val,test}.mat
                 pitts250k_{train,val,test}.mat
```

Pitts30k is a geographic subsample of the same image tree, so one extracted copy
serves both dataset names: when `data/Pittsburgh30k` holds no metadata, the
sibling directories `Pittsburgh`, `pittsburgh`, `Pittsburgh250k` and `pitts250k`
are searched too.

One caveat worth knowing before running: `--image_resize_mode dinov2` adds a
horizontal flip to the training transform. Mirroring a street scene is a more
aggressive augmentation for place recognition than it is for object retrieval,
since it reverses the layout that identifies the place.

## Semi-Aves

Semi-Aves classes carry no taxonomy, only ids, and those ids are numbered from
most to least frequent. That ordering is what every Semi-Aves split has to
handle: a contiguous cut is badly unbalanced, since known ids 0-99 carry 26,348
of the 40,599 pooled known-class images.

All three protocols therefore alternate by default, taking every other id so the
frequency ordering is halved evenly rather than letting per-class counts drive
the partition:

| Protocol | Pool | Rule | Recorded version |
| --- | --- | --- | --- |
| `semi_aves_oracle_500_500` | all 1,000 classes | alternating ids | `semi_aves_oracle_alternating_class_ids_500_500_v2` |
| `semi_aves_oracle_hash_500_500` | all 1,000 classes | SHA-256 rank | `semi_aves_oracle_sha256_500_500_v1` |
| `semi_aves_known_100_100` | the 200 known classes | alternating ids | `semi_aves_known_alternating_class_ids_100_100_v2` |
| `semi_aves_known_hash_100_100` | the 200 known classes | SHA-256 rank | `semi_aves_known_sha256_100_100_v1` |

The two `_hash_` protocols rank the class ids by
`sha256("<version>:<class_id>")` and take the first half. They are arbitrary but
reproducible, and exist as a robustness check: if a result holds under both the
alternating and the hashed split of the same pool, it does not depend on the id
ordering. Neither is the default, because hashing leaves the frequency balance
to chance where alternation fixes it.

That matters more for the known pool than for the oracle pool, because it has
only 200 frequency-ordered classes to average over. The alternating rule hands
development exactly 50 of the 100 most frequent ids; the hash rule as shipped
hands it 47. Both pools record `class_split` and `class_split_version` in
`split_info.json`, so which rule ran is never in doubt.

Both known protocols attach the out-of-class unlabeled pool and accept
`--semi_aves_ood_fraction`; neither oracle protocol does.

Note that the split version string is itself a free parameter. Choosing one
after seeing results would be p-hacking, so treat the recorded version as fixed
and never search over it.

The oracle pool uses every released label, including the out-of-class species;
the known pool uses only the 200 challenge classes and leaves the rest as native
unlabeled data (`--semi_aves_ood_fraction`). Only the known protocol attaches
that pool.

## Adding a protocol

1. Add the `DATASET_PROTOCOL_*` constant in `utils/dataset_constants.py` and
   include it in `DATASET_PROTOCOLS`.
2. Add its `DATASET_PROTOCOL_INFO` entry. `--help` and the drift test read it.
3. Handle it in `load_dataset_protocol_sources` and restrict it to its datasets
   in `validate_dataset_protocol`, both in `utils/dataset_protocols.py`.
4. Add a row to the table above.

A protocol that only repartitions the classes of a single dataset does not need
its own entry: giving the dataset class a `class_disjoint_split` attribute and a
`class_split_info` dict is enough for `official` to pick it up, which is how
Flowers102, StanfordDogs, FGVCFungi and NABirds work.
