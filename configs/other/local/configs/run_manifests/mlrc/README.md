# Reality-check evaluation of the Cars196 studies

These manifests measure finished HPO studies under the evaluation protocol of
*A Metric Learning Reality Check* (Musgrave et al., ECCV 2020), across five
seeds and three per-class label budgets.

```powershell
python launch_runs.py --manifest configs/run_manifests/mlrc/cars_iscen.json
python launch_runs.py --manifest configs/run_manifests/mlrc/cars_lrml.json
```

## What each run does

One run replays one study. It takes that study's single best trial by
validation MAP@R (`final_test_top_n: 1`) and re-measures it under every
combination of the settings below:

| setting | value | why |
|---|---|---|
| `study_dir_mode` | `final_train` | replay the winning trial itself, rather than re-ranking trials |
| `final_fit_mode` | `cross_validation_folds` | the reality-check protocol: no separate final fit, each of the study's 4 folds early-stops on its own partition and that checkpoint is tested |
| `comparison_seeds` | `2, 3, 4, 8, 9` | five comparison points, none of them the tuning seed 7 |
| `comparison_seed_targets` | `seed`, `support_seed` | only the runtime seed and the labeled support draw vary |
| `k_shot_grid` | `4, 8, 16` | three label budgets per class, the tuned `k=16` among them |

`data_split_seed` deliberately stays at the study's 7, so every scenario shares
one train/test class split and one set of folds — the spread they produce is
the spread of the label draw and of training itself, not of the test set.
`hparam_seed` is irrelevant here because no new trials are sampled.

The `k` grid and the seeds cross into 15 scenarios per run, each writing to its
own `class_subset_k_shot_label_1_k_<k>_comparison_seed_<seed>/final/ssl`
directory. Every scenario trains the study's 4 folds and reports both of the
paper's columns: the per-fold mean with its sample standard deviation, and the
concatenated `4 x 768`-dim evaluation. That is 60 fold trainings per run, 300
per manifest.

`k` cannot go below 4 here: all ten study winners use a `<batch>:4`
batch_sampler, and `sampler_m=4` needs four examples in every labeled class.
The class budget is left at the tuned `labeled_fraction=1.0`, so every
development class carries labels at each `k`.

## The fitted heads are kept

Every run passes `--save_head_state`, so each fold writes the checkpoint its own
validation selected to `head_state.pt` next to its logs. With the frozen DINO
backbone these studies use, that head is the whole fitted model, and

```powershell
python main.py --evaluate_saved_heads logs/ssl/mlrc/iscen/iscen_nt/<scenario>/final/ssl
```

re-runs the complete D_test evaluation — per fold and concatenated — without
retraining anything. See `docs/cross_validation.md` for the details.

## Notes

- `cars_lrml.json` keeps the CPU-with-GPU-above-threshold placement of
  `configs/run_manifests/cars_lrml.json`; `iscen.json` trains on the GPU
  like `configs/run_manifests/cars_iscen.json`. Change `device` if the machine
  this runs on wants the other arrangement.
- The `study_dir` paths point at the completed studies under
  `logs/ssl/cars/{iscen,lrml}/`, each with 67-70 finished trials. The scheduler
  fails immediately if one is missing, so a wrong path never costs a training
  run.
