"""Does NTXentLoss need FP32, or is the trainer's BF16 autocast good enough?

``engine._compute_batch_loss`` wraps the forward *and* the loss in
``torch.autocast(bfloat16)`` (``training/engine.py:1560``), so every supervised
run in ``logs/cpu_supervised`` computed NTXent in BF16. Angular classifier losses
already opt out of that through ``losses.metric_losses.classification_loss_float32``
because ``acos`` is unstable at BF16's mantissa. NTXent has a similar shape - a
cosine-similarity matrix divided by a small temperature, then ``logsumexp`` - so
the question is whether it deserves the same promotion.

Two measurements, because they answer different questions
---------------------------------------------------------

Stage 1, *sensitivity at matched weights*: from one set of weights and one
batch, compute the loss and its gradient in each precision mode and compare
against an FP64 evaluation of the same batch. Nothing here has diverged yet, so
this is the loss's own numerical error, uncontaminated by trajectory chaos.

Stage 2, *40-epoch trajectory*: train the same head from the same
initialization on the same batch sequence under each mode, and track how far the
runs drift apart in weights, embeddings, and retrieval metrics. Stage 1 error is
non-zero by construction; Stage 2 says whether it compounds into anything a run
would notice.

Because two trajectories separated by any perturbation eventually decorrelate,
a single-seed gap cannot distinguish "BF16 hurts" from "runs differ". Every
condition is therefore repeated over ``--seeds``, and the between-mode spread is
reported next to the between-seed spread of the FP32 control. Only a mode gap
that clears seed noise is evidence of a precision effect.

Precision modes
---------------
``bf16_autocast``
    What the trainer does today: forward and loss both inside
    ``torch.autocast(bfloat16)``.
``loss_fp32``
    Forward under autocast, loss promoted outside it - the treatment
    ``classification_loss_float32`` gives ArcFace. This is the candidate fix.
``fp32``
    No autocast anywhere; the numerical reference trajectory.

Parameters stay FP32 in every mode (autocast does not touch them), so gradients
are comparable across modes without any rescaling. Evaluation always runs in
FP32, matching the trainer's default (``--eval_amp`` off), so a metric gap comes
from the weights a mode produced rather than from how they were measured.

Setup mirrors trial 0031 of the tuned Cars196 NTXent study
(``logs/cpu_supervised/nt-xent/.../cars_imb_search_supervised``): real frozen
DINOv2 ViT-B/14 features from ``backbone_cache``, a ``Linear + L2 normalize``
head standing in for ``DinoWrapper.fc`` / ``project_features``, an m-per-class
sampler at 256:4, Adam, and that trial's tuned temperature and learning rate.
For reference, its BF16 fold-0 run reached train loss 2.108 / P@1 0.9405 /
MAP@R 0.5833 at epoch 39.

Examples::

    python scripts/ntxent_precision_check.py
    python scripts/ntxent_precision_check.py --epochs 40 --seeds 7 8 9
    python scripts/ntxent_precision_check.py --dataset CUB --output results.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytorch_metric_learning.losses as pml_losses  # noqa: E402
import pytorch_metric_learning.samplers as pml_samplers  # noqa: E402

import utils  # noqa: E402

# Importing this installs the device-aware ``c_f.to_dtype`` guard that every
# real run gets. NTXent has no learnable weights so the guard cannot change its
# result, but importing keeps this script on the same pml behaviour as training.
import losses.metric_losses  # noqa: E402,F401
from utils.dataset_protocols import get_dataset_class  # noqa: E402


PRECISION_MODES = ("bf16_autocast", "loss_fp32", "fp32")
REFERENCE_MODE = "fp32"

# Trial 0031 of the tuned Cars196 NTXent study, read from its ``run_config.json``.
BASELINE_BATCH_SIZE = 256
BASELINE_SAMPLER_M = 4
BASELINE_LR = 3.569626998023795e-05
BASELINE_TEMPERATURE = 0.16218358449843803
BASELINE_FEAT_DIM = 768


class Head(nn.Module):
    """``DinoWrapper.fc`` plus ``project_features``' L2 normalization.

    The runs this mirrors freeze the backbone and read precomputed features, so
    this linear projection is the entire trainable model.
    """

    def __init__(self, backbone_dim, feat_dim):
        super().__init__()
        self.fc = nn.Linear(int(backbone_dim), int(feat_dim))

    def forward(self, features):
        return F.normalize(self.fc(features), p=2.0, dim=1)


# --------------------------------------------------------------------------- #
# Data: real frozen features, position-aligned to their labels
# --------------------------------------------------------------------------- #


def find_cached_feature_matrix(dataset_name, num_rows, data_root):
    """Return the cached feature matrix whose row count matches the dataset.

    The cache is named by a hash of the producing run's key, so it cannot be
    found by dataset name alone. Row count identifies it: the matrix is indexed
    by dataset position, so a matching count means row ``i`` is sample ``i``.
    That assumption is verified rather than trusted - see
    ``nearest_neighbor_label_agreement``.
    """

    cache_dir = Path(data_root) / dataset_name / "backbone_cache"
    for path in sorted(cache_dir.glob("v*/*/*.features.npy"), reverse=True):
        if np.load(path, mmap_mode="r").shape[0] == num_rows:
            return path
    return None


def nearest_neighbor_label_agreement(features, labels, sample_size=4096, seed=0):
    """Return 1-NN label agreement, which detects a feature/label misalignment.

    Frozen DINOv2 features of a retrieval dataset land far above chance; a value
    near ``1 / num_classes`` means the cache rows do not correspond to the
    dataset positions the labels came from, which would invalidate every number
    this script prints.
    """

    generator = np.random.default_rng(seed)
    if len(features) > sample_size:
        rows = np.sort(generator.choice(len(features), size=sample_size, replace=False))
    else:
        rows = np.arange(len(features))
    rows = torch.as_tensor(rows)
    sampled = F.normalize(features[rows].float(), dim=1)
    similarity = sampled @ sampled.t()
    similarity.fill_diagonal_(-2.0)
    neighbors = similarity.argmax(dim=1)
    sampled_labels = labels[rows]
    return float((sampled_labels[neighbors] == sampled_labels).float().mean())


def load_split(dataset_name, split, data_root, device):
    """Load a split's labels and its cached frozen features, position-aligned."""

    dataset_class = get_dataset_class(dataset_name)
    dataset = dataset_class(
        root=str(Path(data_root) / dataset_name),
        split=split,
        transform=None,
        download=False,
    )
    labels = torch.as_tensor(np.asarray(dataset.labels), dtype=torch.long)
    cache_path = find_cached_feature_matrix(dataset_name, len(labels), data_root)
    if cache_path is None:
        raise FileNotFoundError(
            f"No cached backbone features with {len(labels)} rows under "
            f"{Path(data_root) / dataset_name / 'backbone_cache'}; run a "
            f"--use_cache training job on {dataset_name} first"
        )
    features = torch.from_numpy(np.load(cache_path)).float()
    agreement = nearest_neighbor_label_agreement(features, labels)
    chance = 1.0 / max(1, len(torch.unique(labels)))
    if agreement < 10 * chance:
        raise RuntimeError(
            f"{dataset_name}/{split}: 1-NN label agreement {agreement:.3f} is near "
            f"chance {chance:.3f}; the cache at {cache_path} is not aligned with "
            "these labels"
        )
    return {
        "name": f"{dataset_name}/{split}",
        "features": features.to(device),
        "labels": labels.to(device),
        "cache_path": str(cache_path),
        "nn_label_agreement": agreement,
    }


def make_epoch_batches(labels, batch_size, sampler_m, epochs, seed):
    """Materialize the full batch sequence once so every mode sees the same data.

    Reproduces the trainer's sampler exactly: ``utils.make_train_loader`` builds
    an ``MPerClassSampler`` over ``utils.make_sampler_epoch_length`` indices per
    epoch, which the loader then cuts into ``batch_size`` chunks. Drawing it up
    front and replaying it makes data order a constant across precision modes,
    so any divergence they show is arithmetic rather than sampling.
    """

    sampler_length = utils.make_sampler_epoch_length(len(labels), batch_size)
    utils.validate_m_per_class_sampler_capacity(labels, batch_size, sampler_m)
    sampler = pml_samplers.MPerClassSampler(
        labels.cpu().numpy(),
        m=sampler_m,
        batch_size=batch_size,
        length_before_new_iter=sampler_length,
    )
    generator = torch.Generator().manual_seed(int(seed))
    state = torch.random.get_rng_state()
    try:
        # MPerClassSampler draws from the global torch RNG; seed it so the whole
        # sequence is a pure function of ``seed``.
        torch.random.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item()))
        epoch_batches = []
        for _ in range(int(epochs)):
            indices = torch.as_tensor(list(sampler), dtype=torch.long)
            usable = (len(indices) // batch_size) * batch_size
            epoch_batches.append(indices[:usable].view(-1, batch_size))
    finally:
        torch.random.set_rng_state(state)
    return epoch_batches


# --------------------------------------------------------------------------- #
# The three precision modes
# --------------------------------------------------------------------------- #


def compute_loss(mode, head, features, labels, criterion, device_type):
    """Run one forward+loss under ``mode``, matching how the trainer would."""

    if mode == "fp32":
        return criterion(head(features), labels)

    if mode == "bf16_autocast":
        # engine._compute_batch_loss: forward and loss share one autocast region.
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            return criterion(head(features), labels)

    if mode == "loss_fp32":
        # losses.metric_losses.classification_loss_float32: keep the model
        # forward under autocast, promote the embeddings for the objective.
        with torch.autocast(device_type=device_type, dtype=torch.bfloat16):
            embeddings = head(features)
        with torch.autocast(device_type=device_type, enabled=False):
            return criterion(embeddings.float(), labels)

    raise ValueError(f"unknown precision mode: {mode}")


def make_criterion(temperature):
    return pml_losses.NTXentLoss(temperature=float(temperature))


def gradient_vector(head):
    """Flatten the head's gradients into one FP64 vector for comparison."""

    return torch.cat(
        [
            parameter.grad.detach().reshape(-1).double()
            for parameter in head.parameters()
            if parameter.grad is not None
        ]
    )


def parameter_vector(head):
    return torch.cat(
        [parameter.detach().reshape(-1).double() for parameter in head.parameters()]
    )


# --------------------------------------------------------------------------- #
# Stage 1: sensitivity at matched weights
# --------------------------------------------------------------------------- #


def stage1_sensitivity(source, args, device, temperature=None):
    """Per-mode loss and gradient error against an FP64 evaluation of the same batch.

    The head is re-initialized per batch draw and every mode is measured from
    those identical weights, so the only thing that varies is the arithmetic.
    """

    device_type = torch.device(device).type
    features, labels = source["features"], source["labels"]
    batches = make_epoch_batches(
        labels, args.batch_size, args.sampler_m, args.sensitivity_batches, args.seeds[0]
    )
    criterion = make_criterion(args.temperature if temperature is None else temperature)

    records = {mode: {"loss_rel_err": [], "grad_rel_err": [], "grad_cos": []} for mode in PRECISION_MODES}
    losses_seen = []

    for draw, epoch_batch in enumerate(batches):
        indices = epoch_batch[0].to(device)
        batch_features = features[indices]
        batch_labels = labels[indices]

        torch.manual_seed(args.seeds[0] + draw)
        head = Head(source["features"].shape[1], args.feat_dim).to(device)
        initial_state = {key: value.clone() for key, value in head.state_dict().items()}

        # FP64 reference: same weights, same batch, no mixed precision anywhere.
        reference_head = Head(source["features"].shape[1], args.feat_dim).to(device).double()
        reference_head.load_state_dict({k: v.double() for k, v in initial_state.items()})
        reference_loss = criterion(reference_head(batch_features.double()), batch_labels)
        reference_head.zero_grad(set_to_none=True)
        reference_loss.backward()
        reference_gradient = gradient_vector(reference_head)
        reference_value = float(reference_loss.detach())
        losses_seen.append(reference_value)

        for mode in PRECISION_MODES:
            head.load_state_dict(initial_state)
            head.zero_grad(set_to_none=True)
            loss = compute_loss(mode, head, batch_features, batch_labels, criterion, device_type)
            loss.backward()
            gradient = gradient_vector(head)

            records[mode]["loss_rel_err"].append(
                abs(float(loss.detach().double()) - reference_value) / abs(reference_value)
            )
            records[mode]["grad_rel_err"].append(
                float(torch.linalg.vector_norm(gradient - reference_gradient))
                / float(torch.linalg.vector_norm(reference_gradient))
            )
            records[mode]["grad_cos"].append(
                float(F.cosine_similarity(gradient, reference_gradient, dim=0))
            )

    return {
        "temperature": float(args.temperature if temperature is None else temperature),
        "reference_loss_mean": statistics.fmean(losses_seen),
        "batches": len(batches),
        "modes": {
            mode: {
                "loss_rel_err_mean": statistics.fmean(values["loss_rel_err"]),
                "loss_rel_err_max": max(values["loss_rel_err"]),
                "grad_rel_err_mean": statistics.fmean(values["grad_rel_err"]),
                "grad_rel_err_max": max(values["grad_rel_err"]),
                "grad_cos_min": min(values["grad_cos"]),
            }
            for mode, values in records.items()
        },
    }


# --------------------------------------------------------------------------- #
# Stage 2: the 40-epoch trajectory
# --------------------------------------------------------------------------- #


def evaluate_head(head, evaluation_source, device, batch_size=4096):
    """Embed the evaluation split in FP32 and return retrieval metrics.

    Evaluation precision is fixed across modes on purpose - the trainer's
    default (``--eval_amp`` off) evaluates in FP32 - so a metric gap reflects the
    weights a mode produced, not how they were measured.
    """

    head.eval()
    features = evaluation_source["features"]
    with torch.no_grad():
        embeddings = torch.cat(
            [head(features[start : start + batch_size]) for start in range(0, len(features), batch_size)]
        )
    head.train()
    precision_at_1, mean_average_precision_at_r = utils.evaluate_embeddings(
        embeddings,
        evaluation_source["labels"],
        name="precision check",
        device=device,
    )
    return embeddings, {
        "precision_at_1": float(precision_at_1),
        "mean_average_precision_at_r": float(mean_average_precision_at_r),
    }


def train_one(mode, seed, train_source, evaluation_source, args, device):
    """Train the head for ``args.epochs`` under one precision mode."""

    device_type = torch.device(device).type
    features, labels = train_source["features"], train_source["labels"]
    epoch_batches = make_epoch_batches(labels, args.batch_size, args.sampler_m, args.epochs, seed)

    torch.manual_seed(seed)
    head = Head(features.shape[1], args.feat_dim).to(device)
    criterion = make_criterion(args.temperature)
    optimizer = torch.optim.Adam(head.parameters(), lr=args.lr)

    history = []
    for epoch, batches in enumerate(epoch_batches):
        epoch_losses = []
        for row in batches:
            indices = row.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(
                mode, head, features[indices], labels[indices], criterion, device_type
            )
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach()))

        record = {"epoch": epoch, "train_loss": statistics.fmean(epoch_losses)}
        if (epoch + 1) % args.eval_interval == 0 or epoch == len(epoch_batches) - 1:
            embeddings, metrics = evaluate_head(head, evaluation_source, device)
            record.update(metrics)
            record["_embeddings"] = embeddings
        history.append(record)

    return {"history": history, "parameters": parameter_vector(head)}


def compare_trajectories(runs, args):
    """Turn per-run histories into mode-vs-reference and seed-noise summaries."""

    final_epoch = args.epochs - 1
    summary = {"modes": {}, "seed_noise": {}, "divergence": {}}

    for mode in PRECISION_MODES:
        per_seed = {
            key: [runs[(mode, seed)]["history"][final_epoch][key] for seed in args.seeds]
            for key in ("train_loss", "precision_at_1", "mean_average_precision_at_r")
        }
        summary["modes"][mode] = {
            f"{key}_mean": statistics.fmean(values) for key, values in per_seed.items()
        } | {
            f"{key}_values": values for key, values in per_seed.items()
        }

    # The control: how much the reference mode alone moves when only the seed
    # changes. A mode gap smaller than this is not evidence of a precision effect.
    for key in ("train_loss", "precision_at_1", "mean_average_precision_at_r"):
        values = summary["modes"][REFERENCE_MODE][f"{key}_values"]
        summary["seed_noise"][key] = {
            "min": min(values),
            "max": max(values),
            "spread": max(values) - min(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }

    for mode in PRECISION_MODES:
        if mode == REFERENCE_MODE:
            continue
        parameter_drift, embedding_cosine, metric_gaps = [], [], []
        for seed in args.seeds:
            reference = runs[(REFERENCE_MODE, seed)]
            candidate = runs[(mode, seed)]
            parameter_drift.append(
                float(torch.linalg.vector_norm(candidate["parameters"] - reference["parameters"]))
                / float(torch.linalg.vector_norm(reference["parameters"]))
            )
            embedding_cosine.append(
                float(
                    F.cosine_similarity(
                        candidate["history"][final_epoch]["_embeddings"].float(),
                        reference["history"][final_epoch]["_embeddings"].float(),
                        dim=1,
                    ).mean()
                )
            )
            metric_gaps.append(
                candidate["history"][final_epoch]["precision_at_1"]
                - reference["history"][final_epoch]["precision_at_1"]
            )
        summary["divergence"][mode] = {
            "parameter_rel_l2_mean": statistics.fmean(parameter_drift),
            "embedding_cosine_mean": statistics.fmean(embedding_cosine),
            "precision_at_1_delta_mean": statistics.fmean(metric_gaps),
            "precision_at_1_delta_values": metric_gaps,
        }

    return summary


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def report(stage1, stage2, args, sources, sweep=None):
    print()
    print("=" * 78)
    print("NTXentLoss BF16 vs FP32")
    print("=" * 78)
    print(f"train split      : {sources['train']['name']} ({len(sources['train']['labels'])} samples)")
    print(f"eval split       : {sources['eval']['name']} ({len(sources['eval']['labels'])} samples)")
    print(f"1-NN agreement   : train {sources['train']['nn_label_agreement']:.3f}, "
          f"eval {sources['eval']['nn_label_agreement']:.3f}")
    print(f"operating point  : batch {args.batch_size}:{args.sampler_m}, lr {args.lr:.4g}, "
          f"temperature {args.temperature:.4g}, feat_dim {args.feat_dim}")
    print(f"epochs           : {args.epochs}   seeds: {args.seeds}")

    print()
    print("-" * 78)
    print(f"Stage 1  loss and gradient error at matched weights, vs FP64")
    print(f"         {stage1['batches']} batches, reference loss {stage1['reference_loss_mean']:.6f}")
    print("-" * 78)
    print(f"{'mode':<18}{'loss rel err':>16}{'grad rel err':>16}{'grad cos':>14}")
    for mode, values in stage1["modes"].items():
        print(
            f"{mode:<18}"
            f"{values['loss_rel_err_mean']:>16.3e}"
            f"{values['grad_rel_err_mean']:>16.3e}"
            f"{values['grad_cos_min']:>14.8f}"
        )

    if sweep:
        print()
        print("-" * 78)
        print("Stage 1b  gradient error across NTXent temperatures, vs FP64")
        print("          Logits are cosines / temperature, so a fixed relative mantissa")
        print("          error lands on a wider range as temperature falls. Read the FP32")
        print("          column first: where it also degrades, the objective itself has")
        print("          become ill-conditioned and the BF16 gap is no longer the story.")
        print("-" * 78)
        print(f"{'temperature':<13}{'fp64 loss':>12}{'fp32':>13}{'loss_fp32':>13}{'bf16':>13}{'bf16/fp32':>12}")
        for entry in sweep:
            reference_error = entry["modes"]["fp32"]["grad_rel_err_mean"]
            bf16_error = entry["modes"]["bf16_autocast"]["grad_rel_err_mean"]
            print(
                f"{entry['temperature']:<13.4f}"
                f"{entry['reference_loss_mean']:>12.3f}"
                f"{reference_error:>13.3e}"
                f"{entry['modes']['loss_fp32']['grad_rel_err_mean']:>13.3e}"
                f"{bf16_error:>13.3e}"
                f"{bf16_error / reference_error:>12.1f}x"
            )

    if stage2 is None:
        print()
        return

    print()
    print("-" * 78)
    print(f"Stage 2  after {args.epochs} epochs, mean over {len(args.seeds)} seeds")
    print("-" * 78)
    print(f"{'mode':<18}{'train loss':>14}{'P@1':>12}{'MAP@R':>12}")
    for mode, values in stage2["modes"].items():
        print(
            f"{mode:<18}"
            f"{values['train_loss_mean']:>14.5f}"
            f"{values['precision_at_1_mean']:>12.5f}"
            f"{values['mean_average_precision_at_r_mean']:>12.5f}"
        )

    print()
    print(f"seed noise in {REFERENCE_MODE} alone (max - min over seeds):")
    for key, values in stage2["seed_noise"].items():
        print(f"  {key:<32}{values['spread']:.5f}")

    print()
    print(f"drift from the {REFERENCE_MODE} trajectory:")
    print(f"{'mode':<18}{'param rel L2':>16}{'embed cos':>14}{'P@1 delta':>14}")
    for mode, values in stage2["divergence"].items():
        print(
            f"{mode:<18}"
            f"{values['parameter_rel_l2_mean']:>16.3e}"
            f"{values['embedding_cosine_mean']:>14.6f}"
            f"{values['precision_at_1_delta_mean']:>+14.5f}"
        )
    print()


def strip_embeddings(runs):
    """Drop the cached embedding tensors so the JSON dump stays small."""

    return {
        f"{mode}/seed{seed}": [
            {key: value for key, value in record.items() if key != "_embeddings"}
            for record in run["history"]
        ]
        for (mode, seed), run in runs.items()
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset", default="Cars196")
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="test")
    parser.add_argument("--data-root", default=str(PROJECT_ROOT / "data"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 8, 9])
    parser.add_argument("--batch-size", type=int, default=BASELINE_BATCH_SIZE)
    parser.add_argument("--sampler-m", type=int, default=BASELINE_SAMPLER_M)
    parser.add_argument("--lr", type=float, default=BASELINE_LR)
    parser.add_argument("--temperature", type=float, default=BASELINE_TEMPERATURE)
    parser.add_argument("--feat-dim", type=int, default=BASELINE_FEAT_DIM)
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument(
        "--sensitivity-batches",
        type=int,
        default=16,
        help="batches drawn for the Stage 1 matched-weight comparison",
    )
    parser.add_argument(
        "--sensitivity-temperatures",
        type=float,
        nargs="+",
        default=None,
        help="also repeat Stage 1 at these NTXent temperatures",
    )
    parser.add_argument(
        "--stage1-only",
        action="store_true",
        help="skip the 40-epoch trajectory and report only the matched-weight error",
    )
    parser.add_argument("--output", default=None, help="write the full results as JSON")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    sources = {
        "train": load_split(args.dataset, args.train_split, args.data_root, device),
        "eval": load_split(args.dataset, args.eval_split, args.data_root, device),
    }

    stage1 = stage1_sensitivity(sources["train"], args, device)
    sweep = [
        stage1_sensitivity(sources["train"], args, device, temperature=temperature)
        for temperature in (args.sensitivity_temperatures or [])
    ]

    runs = {}
    stage2 = None
    if not args.stage1_only:
        for mode in PRECISION_MODES:
            for seed in args.seeds:
                print(f"training {mode} seed {seed} for {args.epochs} epochs ...", flush=True)
                runs[(mode, seed)] = train_one(
                    mode, seed, sources["train"], sources["eval"], args, device
                )
        stage2 = compare_trajectories(runs, args)

    report(stage1, stage2, args, sources, sweep=sweep)

    if args.output:
        payload = {
            "config": vars(args),
            "sources": {
                key: {
                    "name": source["name"],
                    "cache_path": source["cache_path"],
                    "samples": len(source["labels"]),
                    "nn_label_agreement": source["nn_label_agreement"],
                }
                for key, source in sources.items()
            },
            "stage1_sensitivity": stage1,
            "stage1_temperature_sweep": sweep,
            "stage2_summary": stage2,
            "stage2_histories": strip_embeddings(runs),
        }
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
