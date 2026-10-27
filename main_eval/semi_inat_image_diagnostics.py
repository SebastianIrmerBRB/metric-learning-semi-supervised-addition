"""Image-level diagnostics: are the Semi-iNat species too similar, are the organisms too small for DINOv2 at 224?

``extract`` runs the frozen DINOv2 ViT-B/14 over a test split the way the pipeline does (squash resize,
ImageNet normalization, final-norm CLS token, full float32) and keeps per image:

* ``cls``: the pipeline's feature; ``analyze`` checks that it reproduces the recorded DINOv2 baseline.
* ``gap``: mean of the patch tokens, an alternative global descriptor.
* ``attn``: last-block CLS->patch attention, mean over heads.
* ``norm``: pre-norm patch-token norms. DINOv2 without registers writes global information into a few
  high-norm background tokens (Darcet et al. 2024); those are excluded from the attention and PCA summaries.
* ``pc1``: first component of a per-image PCA over the patch tokens, signed so the border ring is lower
  than the interior, as in the foreground masks of the DINOv2 paper (Fig. 1). ``mask`` thresholds it
  per image with Otsu's method; splitting at zero kept most of the background.
* ``fg``: foreground fraction of the patch grid, ``attn50``: share of patches holding half the attention,
  ``box``: square foreground box with one patch of margin, as image fractions (x0, y0, x1, y1).

The last-block CLS attention is kept but not used for conclusions: on Semi-iNat the top-left patch alone
receives ~22 % of it at an ordinary token norm (an attention sink), so it does not locate the organism.

Variants: ``base`` (224, the pipeline), ``up448`` (the image squashed to 224 and upsampled to 448, 32x32 tokens,
more tokens per object but no new pixels; Semi-iNat is stored at 224 already), ``low112`` (224 -> 112 -> 224, information removed), ``zoom`` (crop to the
``base`` foreground box, resized to 224), ``fg_only`` / ``bg_only`` (everything outside / inside the ``base``
mask, dilated by one patch, replaced by the ImageNet mean colour).

``analyze`` computes per-query P@1 and AP@R per variant, the taxonomic rank at which the wrong neighbours
agree with the query, and AP@R over foreground-size bins. ``sheet`` draws image | attention | PC1 | mask.

Examples::

    venv/bin/python main_eval/semi_inat_image_diagnostics.py extract --dataset semi_inat --variant base
    venv/bin/python main_eval/semi_inat_image_diagnostics.py analyze --dataset semi_inat
    venv/bin/python main_eval/semi_inat_image_diagnostics.py sheet --dataset semi_inat --pick small_fail
"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "logs" / "analysis" / "semi_inat_image"
VARIANTS = ("base", "up448", "low112", "zoom", "fg_only", "bg_only")
# Variants that read the foreground box or mask of ``base``.
MASKED_VARIANTS = ("zoom", "fg_only", "bg_only")
IMAGENET_MEAN_RGB = (124, 116, 104)
RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")
# Recorded DINOv2 test baseline (scripts/reports/untrained_baseline/semi_inat_raw_dinov2.json).
SEMI_INAT_BASELINE = {"p_at_1": 0.8645987772576588, "map_at_r": 0.33950242859018925}
CARS_TEST_CACHE = ROOT / (
    "data/Cars196/backbone_cache/v3/dinov2_vitb14/"
    "0beabb254a556ce0bda74bdca94fde62c1944fba076b19f1904224c08f7fd2fd.features.npy"
)
MULTI_WORD_MAKES = ("Land Rover", "Aston Martin", "AM General")


# ------------------------------------------------------------------ data
def load_split(name):
    """Test-split paths, integer labels, and a {rank: array of names} grouping per image."""

    data = ROOT / "data"
    if name == "semi_inat":
        from utils.local_datasets import SemiINaturalist2021

        ds = SemiINaturalist2021(data / "iNat", split="test", mode="known")
        labels = np.asarray(ds.labels)
        groups = {r: np.array([ds.taxonomy[int(l)][r] for l in labels]) for r in RANKS}
        return np.asarray(ds.paths), labels, groups
    from utils.dataset_protocols import get_dataset_class

    dir_name = {"cars": "Cars196", "cub": "CUB"}[name]
    ds = get_dataset_class(dir_name)(root=str(data / dir_name), split="test", download=False)
    paths, labels = np.asarray([str(p) for p in ds.paths]), np.asarray(ds.labels)
    groups = {}
    if name == "cars":
        models = np.array([Path(p).parent.name for p in paths])
        groups["make"] = np.array([next((m for m in MULTI_WORD_MAKES if n.startswith(m)), n.split(" ")[0])
                                   for n in models])
        groups["species"] = models
    else:  # CUB: the last word of the class name (Warbler, Sparrow, ...) as a crude bird group
        names = np.array([Path(p).parent.name.split(".", 1)[1] for p in paths])
        groups["group"] = np.array([n.split("_")[-1] for n in names])
        groups["species"] = names
    return paths, labels, groups


def dilate(masks):
    """One-patch 8-neighbour dilation of (N, G, G) boolean masks."""

    padded = np.pad(masks, ((0, 0), (1, 1), (1, 1)))
    g = masks.shape[1]
    out = np.zeros_like(masks)
    for dy in range(3):
        for dx in range(3):
            out |= padded[:, dy:dy + g, dx:dx + g]
    return out


class ImageSet:
    def __init__(self, paths, transform, boxes=None, degrade=False, masks=None, keep_inside=True, squash224=False):
        self.paths, self.transform, self.boxes, self.degrade = paths, transform, boxes, degrade
        self.masks, self.keep_inside, self.squash224 = masks, keep_inside, squash224

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image

        im = Image.open(self.paths[i]).convert("RGB")
        if self.squash224:  # Semi-iNat is stored at 224; Cars/CUB get the same pixel budget before upsampling
            im = im.resize((224, 224), Image.Resampling.BILINEAR)
        if self.boxes is not None:
            w, h = im.size
            x0, y0, x1, y1 = self.boxes[i]
            im = im.crop((round(x0 * w), round(y0 * h), round(x1 * w), round(y1 * h)))
        if self.degrade:
            im = im.resize((112, 112), Image.Resampling.BILINEAR).resize((224, 224), Image.Resampling.BILINEAR)
        if self.masks is not None:
            inside = Image.fromarray((self.masks[i] == self.keep_inside).astype(np.uint8) * 255).resize(
                im.size, Image.Resampling.NEAREST)
            im = Image.composite(im, Image.new("RGB", im.size, IMAGENET_MEAN_RGB), inside)
        return self.transform(im)


# ------------------------------------------------------------------ patch summaries
def border_mask(g):
    m = np.zeros((g, g), dtype=bool)
    m[0], m[-1], m[:, 0], m[:, -1] = True, True, True, True
    return m.reshape(-1)


def otsu_threshold(values, live):
    """Per-row Otsu threshold over the live entries of (B, P) values; entries above it are the upper class."""

    import torch

    v = torch.where(live, values, torch.full_like(values, float("inf")))
    s, order = v.sort(1)
    w = live.gather(1, order).float()
    s = torch.where(w > 0, s, torch.zeros_like(s))
    n = w.sum(1, keepdim=True)
    cw, cs = w.cumsum(1), (s * w).cumsum(1)
    mu0 = cs / cw.clamp_min(1)
    mu1 = (cs[:, -1:] - cs) / (n - cw).clamp_min(1)
    between = (cw / n) * (1 - cw / n) * (mu0 - mu1) ** 2
    between = torch.where((w > 0) & (cw < n), between, torch.full_like(between, -1.0))
    return s.gather(1, between.argmax(1, keepdim=True))


def patch_summaries(tokens, attn, norms, grid, artifact_threshold):
    """Per-image PCA foreground, cleaned attention and foreground box for one batch (torch, on device)."""

    import torch

    artifact = norms > artifact_threshold  # (B, P)
    keep = (~artifact).float().unsqueeze(-1)
    n_keep = keep.sum(1).clamp_min(1)
    mean = (tokens * keep).sum(1, keepdim=True) / n_keep.unsqueeze(1)
    xc = (tokens - mean) * keep  # artifact rows are zero, so they score zero on every component
    gram = xc @ xc.transpose(1, 2)
    _, vecs = torch.linalg.eigh(gram)
    pc1 = vecs[..., -1] * (gram.diagonal(dim1=1, dim2=2).sum(1, keepdim=True).sqrt())
    border = torch.as_tensor(border_mask(grid), device=tokens.device)
    live = ~artifact
    b_mean = (pc1 * (border & live)).sum(1) / (border & live).sum(1).clamp_min(1)
    i_mean = (pc1 * (~border & live)).sum(1) / (~border & live).sum(1).clamp_min(1)
    pc1 = pc1 * torch.where(b_mean > i_mean, -1.0, 1.0).unsqueeze(1)
    fg_mask = (pc1 > otsu_threshold(pc1, live)) & live

    clean = attn * live
    clean = clean / clean.sum(1, keepdim=True).clamp_min(1e-12)
    srt = clean.sort(1, descending=True).values.cumsum(1)
    attn50 = ((srt < 0.5).sum(1) + 1) / live.sum(1).clamp_min(1)
    return pc1, clean, attn50, artifact.sum(1), fg_mask


def largest_component(masks):
    """Keep the largest 8-connected component of each (G, G) mask; stray background patches go."""

    from scipy import ndimage

    out = np.zeros_like(masks)
    structure = np.ones((3, 3), dtype=bool)
    for i, m in enumerate(masks):
        lab, n = ndimage.label(m, structure=structure)
        if n:
            out[i] = lab == np.bincount(lab.ravel())[1:].argmax() + 1
    return out


def square_boxes(masks):
    """Square box around each mask with one patch of margin, at least 4 patches, as image fractions."""

    n, g, _ = masks.shape
    boxes = np.tile(np.array([0.0, 0.0, 1.0, 1.0]), (n, 1))
    for i, m in enumerate(masks):
        rows, cols = np.where(m.any(1))[0], np.where(m.any(0))[0]
        if not len(rows):
            continue
        side = min(max(rows[-1] + 1 - rows[0], cols[-1] + 1 - cols[0]) + 2, g)
        side = max(side, 4)
        cy, cx = (rows[0] + rows[-1] + 1) / 2, (cols[0] + cols[-1] + 1) / 2
        y0 = min(max(cy - side / 2, 0), g - side)
        x0 = min(max(cx - side / 2, 0), g - side)
        boxes[i] = np.array([x0, y0, x0 + side, y0 + side]) / g
    return boxes


# ------------------------------------------------------------------ extract
def extract(options):
    import torch
    from torch.utils.data import DataLoader

    import main as entry
    from models.retrieval_model import load_dinov2_with_retry
    from utils.image_transforms import make_test_transform

    entry.configure_float32_precision()
    device = torch.device(options.device)
    paths, labels, _ = load_split(options.dataset)
    if options.limit:
        paths, labels = paths[: options.limit], labels[: options.limit]
    size = 448 if options.variant == "up448" else 224
    boxes = masks = None
    if options.variant in MASKED_VARIANTS:
        base = np.load(OUT / f"{options.dataset}_base{'_limit' if options.limit else ''}.npz")
        assert (base["labels"][: len(paths)] == labels).all()
        if options.variant == "zoom":
            boxes = base["box"][: len(paths)]
        else:
            masks = dilate(base["mask"][: len(paths)])
    loader = DataLoader(
        ImageSet(paths, make_test_transform("squash", size), boxes, degrade=options.variant == "low112",
                 masks=masks, keep_inside=options.variant == "fg_only", squash224=options.variant == "up448"),
        batch_size=options.batch_size, num_workers=options.workers, pin_memory=device.type == "cuda",
    )
    model = load_dinov2_with_retry("b").to(device).eval()
    grid = size // 14
    captured = {}
    last = model.blocks[-1]
    hook = last.norm1.register_forward_hook(lambda _m, _i, out: captured.__setitem__("x", out))
    names = ("cls", "gap") if options.global_only else (
        "cls", "gap", "attn", "norm", "pc1", "attn50", "artifacts", "mask")
    keep = {k: [] for k in names}
    done = 0
    with torch.no_grad():
        for images in loader:
            images = images.to(device, non_blocking=True)
            feats = model.forward_features(images)
            keep["cls"].append(feats["x_norm_clstoken"].float().cpu().numpy())
            keep["gap"].append(feats["x_norm_patchtokens"].mean(1).half().cpu().numpy())
            done += len(images)
            if done // options.batch_size % 50 == 0:
                print(f"{done}/{len(paths)}", flush=True)
            if options.global_only:
                continue
            x = captured["x"]
            b, n, c = x.shape
            heads = last.attn.num_heads
            qkv = last.attn.qkv(x).reshape(b, n, 3, heads, c // heads).permute(2, 0, 3, 1, 4)
            att = ((qkv[0] * last.attn.scale) @ qkv[1].transpose(-2, -1)).softmax(-1)
            cls_att = att[:, :, 0, 1:].mean(1)  # (B, P); DINOv2-B has no register tokens
            patches = feats["x_norm_patchtokens"]
            norms = feats["x_prenorm"][:, 1:].norm(dim=-1)
            pc1, clean, attn50, n_art, fg_mask = patch_summaries(
                patches, cls_att, norms, grid, options.artifact_threshold)
            keep["attn"].append(clean.view(b, grid, grid).half().cpu().numpy())
            keep["norm"].append(norms.view(b, grid, grid).half().cpu().numpy())
            keep["pc1"].append(pc1.view(b, grid, grid).half().cpu().numpy())
            keep["attn50"].append(attn50.float().cpu().numpy())
            keep["artifacts"].append(n_art.cpu().numpy())
            keep["mask"].append(fg_mask.view(b, grid, grid).cpu().numpy())
    hook.remove()
    out = {k: np.concatenate(v) for k, v in keep.items()}
    if not options.global_only:
        out["mask"] = largest_component(out["mask"])
        out["fg"] = out["mask"].sum((1, 2)) / (grid * grid - out["artifacts"])
        out["box"] = square_boxes(out["mask"])
    OUT.mkdir(parents=True, exist_ok=True)
    target = OUT / f"{options.dataset}_{options.variant}{'_limit' if options.limit else ''}.npz"
    np.savez(target, labels=labels, image_size=size, artifact_threshold=options.artifact_threshold, **out)
    print("wrote", target, {k: v.shape for k, v in out.items()})


# ------------------------------------------------------------------ retrieval
def per_query(embeddings, labels, keep_top=10, device="cuda", chunk=2048):
    """Leave-one-out P@1 and AP@R for every query (MAP@R as in Musgrave et al.), plus the top neighbours."""

    import torch

    e = torch.nn.functional.normalize(torch.as_tensor(np.asarray(embeddings, dtype=np.float32)), dim=1).to(device)
    _, y = np.unique(labels, return_inverse=True)
    y = torch.as_tensor(y, device=device)
    r = torch.bincount(y)[y] - 1
    depth = int(max(r.max().item(), keep_top))
    ranks = torch.arange(1, depth + 1, device=device, dtype=torch.float32)
    ap, p1, top, sim_top = [], [], [], []
    for s in range(0, len(e), chunk):
        q = e[s:s + chunk]
        sim = q @ e.T
        rows = torch.arange(len(q), device=device)
        sim[rows, s + rows] = -float("inf")
        v, idx = sim.topk(depth, dim=1)
        rel = (y[idx] == y[s:s + chunk, None]).float()
        rq = r[s:s + chunk]
        rel_r = rel * (ranks[None] <= rq[:, None])
        ap.append(((rel_r.cumsum(1) / ranks) * rel_r).sum(1) / rq)
        p1.append(rel[:, 0])
        top.append(idx[:, :keep_top])
        sim_top.append(v[:, :keep_top])
    cat = lambda xs: torch.cat(xs).cpu().numpy()
    return {"ap": cat(ap), "p1": cat(p1), "top": cat(top), "sim": cat(sim_top), "R": r.cpu().numpy()}


def rank_agreement(top, groups, order):
    """Share of the wrong top-k neighbours whose deepest shared rank is each entry of ``order``, against chance."""

    species = groups["species"]
    q = np.repeat(np.arange(len(species)), top.shape[1])
    n = top.reshape(-1)
    wrong = species[q] != species[n]
    q, n = q[wrong], n[wrong]
    deepest = np.full(len(q), "none", dtype=object)
    for rank in order:  # coarse to fine, so the last match wins
        deepest[groups[rank][q] == groups[rank][n]] = rank
    rng = np.random.default_rng(0)
    cq = rng.integers(0, len(species), 400_000)
    cn = rng.integers(0, len(species), 400_000)
    other = species[cq] != species[cn]
    cq, cn = cq[other], cn[other]
    chance = np.full(len(cq), "none", dtype=object)
    for rank in order:
        chance[groups[rank][cq] == groups[rank][cn]] = rank
    return {rank: {"neighbours": float(np.mean(deepest == rank)), "chance": float(np.mean(chance == rank))}
            for rank in ("none", *order)}


def bins_table(values, per_variant, n_bins=5):
    edges = np.quantile(values, np.linspace(0, 1, n_bins + 1))
    which = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        sel = which == b
        row = {"bin": b, "lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(sel.sum())}
        for name, res in per_variant.items():
            row[f"{name}_map_at_r"] = float(res["ap"][sel].mean())
            row[f"{name}_p_at_1"] = float(res["p1"][sel].mean())
        rows.append(row)
    return rows


def analyze(options):
    import scipy.stats as st

    paths, labels, groups = load_split(options.dataset)
    report = {"dataset": options.dataset, "n": int(len(labels)), "classes": int(len(np.unique(labels)))}
    stores = {v: np.load(OUT / f"{options.dataset}_{v}.npz") for v in VARIANTS
              if (OUT / f"{options.dataset}_{v}.npz").exists()}
    base = stores["base"]
    assert (base["labels"] == labels).all()
    results = {}
    for variant, store in stores.items():
        results[variant] = per_query(store["cls"], labels, device=options.device)
        report[f"{variant}_map_at_r"] = float(results[variant]["ap"].mean())
        report[f"{variant}_p_at_1"] = float(results[variant]["p1"].mean())
    # ~20 % of the Semi-iNat files are byte-identical copies: u_train_in repeats most l_train/val/test
    # images. A copy is a perfect rank-1 hit, so the deduplicated scores keep one file per MD5.
    md5_path = OUT / f"{options.dataset}_test_md5.npy"
    if md5_path.exists():
        hashes = np.load(md5_path)
        unique = np.sort(np.unique(hashes, return_index=True)[1])
        report["dedup_n"] = int(len(unique))
        for variant, store in stores.items():
            res = per_query(store["cls"][unique], labels[unique], device=options.device)
            report[f"{variant}_dedup_map_at_r"] = float(res["ap"].mean())
            report[f"{variant}_dedup_p_at_1"] = float(res["p1"].mean())
    gap = per_query(base["gap"].astype(np.float32), labels, device=options.device)
    report["base_gap_map_at_r"], report["base_gap_p_at_1"] = float(gap["ap"].mean()), float(gap["p1"].mean())
    if options.dataset == "semi_inat":
        report["baseline_check"] = {k: [report[f"base_{k}"], v] for k, v in SEMI_INAT_BASELINE.items()}
    if options.dataset == "cars" and CARS_TEST_CACHE.exists():
        cached = np.load(CARS_TEST_CACHE).astype(np.float32)
        a = base["cls"] / np.linalg.norm(base["cls"], axis=1, keepdims=True)
        b = cached / np.linalg.norm(cached, axis=1, keepdims=True)
        report["cache_cosine_min"] = float((a * b).sum(1).min())

    order = [r for r in (*RANKS[:-1], "make", "group") if r in groups]
    report["wrong_neighbour_rank_top10"] = rank_agreement(results["base"]["top"], groups, order)
    fg, attn50 = base["fg"], base["attn50"]
    report["fg_quantiles"] = {q: float(np.quantile(fg, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    report["attn50_quantiles"] = {q: float(np.quantile(attn50, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)}
    report["artifacts_per_image_mean"] = float(base["artifacts"].mean())
    report["spearman_fg_vs_ap"] = float(st.spearmanr(fg, results["base"]["ap"]).statistic)
    report["spearman_attn50_vs_ap"] = float(st.spearmanr(attn50, results["base"]["ap"]).statistic)
    report["by_fg_bin"] = bins_table(fg, results)
    report["top1_similarity_quantiles"] = {
        q: float(np.quantile(results["base"]["sim"][:, 0], q)) for q in (0.1, 0.5, 0.9, 0.99)}
    if options.dataset == "semi_inat":
        within = {}
        for tax_class in np.unique(groups["class"]):
            sel = groups["class"] == tax_class
            if sel.sum() < 2000:
                continue
            row = {"images": int(sel.sum()), "species": int(len(np.unique(labels[sel]))),
                   "fg_median": float(np.median(fg[sel])),
                   "spearman_fg_vs_ap": float(st.spearmanr(fg[sel], results["base"]["ap"][sel]).statistic)}
            for variant, res in results.items():
                row[f"{variant}_map_at_r"] = float(res["ap"][sel].mean())
            within[str(tax_class)] = row
        report["by_taxonomic_class"] = within
        sp = groups["species"]
        congeneric = {s for s in np.unique(sp)
                      if len(np.unique(sp[groups["genus"] == groups["genus"][sp == s][0]])) > 1}
        has = np.isin(sp, list(congeneric))
        report["congeneric_test_species"] = {
            "species": len(congeneric),
            "map_at_r_with_congener": float(results["base"]["ap"][has].mean()),
            "map_at_r_without_congener": float(results["base"]["ap"][~has].mean()),
        }
        # Where the neighbours of each taxonomic class come from: an insect photographed on a flower
        # retrieves the flower's species, so animal queries lose neighbours to Plantae.
        top = results["base"]["top"]
        cross = {}
        for tax_class, row in within.items():
            sel = groups["class"] == tax_class
            nb = top[sel].reshape(-1)
            same_species = np.repeat(labels[sel], top.shape[1]) == labels[nb]
            cross[tax_class] = {
                "same_species": float(same_species.mean()),
                "same_class_other_species": float(((groups["class"][nb] == tax_class) & ~same_species).mean()),
                **{f"kingdom_{k}": float((groups["kingdom"][nb] == k).mean()) for k in ("Animalia", "Plantae", "Fungi")},
                "kingdom_share_of_test_images": {k: float((groups["kingdom"] == k).mean())
                                                 for k in ("Animalia", "Plantae", "Fungi")},
            }
        report["top10_neighbour_origin_by_class"] = cross
    np.savez(OUT / f"{options.dataset}_per_query.npz", fg=fg, attn50=attn50,
             **{f"{v}_{k}": r[k] for v, r in results.items() for k in ("ap", "p1", "top", "sim")},
             R=results["base"]["R"])
    (OUT / f"{options.dataset}_report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


# ------------------------------------------------------------------ occlusion
def occlusion(options):
    """Which image regions carry the evidence for the true species and for the wrongly retrieved one.

    A grey square of ``--occluder`` patches slides over the 224 image at a stride of one patch. For each position
    the drop of cos(embedding, prototype) is recorded, for the prototype of the query's species (mean base CLS
    of its other, non-duplicate images) and of the species most frequent among its wrong top-6 neighbours. Each
    patch gets the mean drop of the positions covering it: large values mark regions the embedding relies on.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from PIL import Image

    import main as entry
    from models.retrieval_model import load_dinov2_with_retry
    from utils.image_transforms import make_test_transform

    entry.configure_float32_precision()
    paths, labels, groups = load_split(options.dataset)
    base = np.load(OUT / f"{options.dataset}_base.npz")
    pq = np.load(OUT / f"{options.dataset}_per_query.npz")
    hashes = np.load(OUT / f"{options.dataset}_test_md5.npy")
    cls = base["cls"] / np.linalg.norm(base["cls"], axis=1, keepdims=True)
    device = torch.device(options.device)
    model = load_dinov2_with_retry("b").to(device).eval()
    transform = make_test_transform("squash", 224)
    g, k = 16, options.occluder
    positions = [(y, x) for y in range(g - k + 1) for x in range(g - k + 1)]

    def prototype(species_label, exclude_hash):
        sel = (labels == species_label) & (hashes != exclude_hash)
        v = cls[sel].mean(0)
        return v / np.linalg.norm(v)

    def embed(batch):
        with torch.no_grad():
            out = torch.cat([model(batch[i:i + 49].to(device)) for i in range(0, len(batch), 49)])
        return torch.nn.functional.normalize(out, dim=1).cpu().numpy()

    picks = [int(i) for i in options.indices.split(",")]
    fig, axes = plt.subplots(len(picks), 4, figsize=(4 * 1.9, len(picks) * 2.05))
    axes = np.atleast_2d(axes)
    for r, q in enumerate(picks):
        im = Image.open(paths[q]).convert("RGB").resize((224, 224))
        x = transform(im)
        grey = torch.zeros(3)  # the ImageNet mean after normalization
        batch = x.unsqueeze(0).repeat(len(positions) + 1, 1, 1, 1)
        for j, (py, px) in enumerate(positions, start=1):
            batch[j, :, py * 14:(py + k) * 14, px * 14:(px + k) * 14] = grey[:, None, None]
        e = embed(batch)
        wrong = [labels[i] for i in pq["base_top"][q][:6] if labels[i] != labels[q]]
        wrong_label = max(set(wrong), key=wrong.count) if wrong else None
        maps = {}
        for name, lab in (("true", labels[q]), ("wrong", wrong_label)):
            if lab is None:
                continue
            p = prototype(lab, hashes[q])
            drop = float(e[0] @ p) - e[1:] @ p
            acc, cnt = np.zeros((g, g)), np.zeros((g, g))
            for d, (py, px) in zip(drop, positions):
                acc[py:py + k, px:px + k] += d
                cnt[py:py + k, px:px + k] += 1
            maps[name] = (acc / cnt, float(e[0] @ p), groups["species"][labels == lab][0])
        tokens = None
        with torch.no_grad():
            tokens = model.forward_features(x.unsqueeze(0).to(device))["x_norm_patchtokens"][0].cpu().numpy()
        tc = tokens - tokens.mean(0)
        _, _, vt = np.linalg.svd(tc, full_matrices=False)
        rgb = tc @ vt[:3].T
        rgb = (rgb - rgb.min(0)) / (rgb.max(0) - rgb.min(0) + 1e-9)
        axes[r, 0].imshow(im)
        axes[r, 0].set_title(f"#{q} {groups['species'][q][:26]}\nAP@R {pq['base_ap'][q]:.2f}", fontsize=6)
        axes[r, 1].imshow(rgb.reshape(g, g, 3), interpolation="nearest")
        axes[r, 1].set_title("patch tokens, PCA 1-3", fontsize=6)
        lim = max(max(np.abs(m[0]).max() for m in maps.values()), 1e-6)
        for c, name in ((2, "true"), (3, "wrong")):
            if name not in maps:
                axes[r, c].axis("off")
                continue
            m, sim, species = maps[name]
            axes[r, c].imshow(im)
            up = np.asarray(Image.fromarray(m.astype(np.float32)).resize((224, 224), Image.Resampling.BILINEAR))
            axes[r, c].imshow(up, cmap="RdBu_r", vmin=-lim, vmax=lim, alpha=0.6)
            label = "evidence: own species" if name == "true" else "evidence: retrieved"
            axes[r, c].set_title(f"{label}\n{species[:24]} (cos {sim:.2f})", fontsize=6)
        for a in axes[r]:
            a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    target = OUT / f"occlusion_{options.dataset}_{options.name}.png"
    fig.savefig(target, dpi=options.dpi)
    print("wrote", target)


# ------------------------------------------------------------------ contact sheet
def sheet(options):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    paths, labels, groups = load_split(options.dataset)
    base = np.load(OUT / f"{options.dataset}_base{options.store_suffix}.npz")
    pq = np.load(OUT / f"{options.dataset}_per_query.npz") if (OUT / f"{options.dataset}_per_query.npz").exists() else None
    rng = np.random.default_rng(options.seed)
    fg = base["fg"]
    if options.indices:
        pick = np.array([int(i) for i in options.indices.split(",")])
    elif options.pick == "random" or pq is None:
        pick = rng.choice(len(paths), options.n, replace=False)
    else:
        ap = pq["base_ap"]
        small, large = fg <= np.quantile(fg, 0.2), fg >= np.quantile(fg, 0.8)
        low, high = ap <= np.quantile(ap, 0.2), ap >= np.quantile(ap, 0.8)
        pool = {"small_fail": small & low, "small_ok": small & high,
                "large_fail": large & low, "large_ok": large & high}[options.pick]
        pick = rng.choice(np.where(pool)[0], min(options.n, pool.sum()), replace=False)
    cols = 4
    fig, axes = plt.subplots(len(pick), cols, figsize=(cols * 1.9, len(pick) * 1.95))
    axes = np.atleast_2d(axes)
    for r, i in enumerate(pick):
        im = np.asarray(Image.open(paths[i]).convert("RGB").resize((224, 224)))
        g = base["attn"].shape[1]
        up = lambda m: np.asarray(Image.fromarray(m.astype(np.float32)).resize((224, 224), Image.Resampling.NEAREST))
        attn = base["attn"][i].astype(np.float32)
        pc1 = base["pc1"][i].astype(np.float32)
        art = base["norm"][i].astype(np.float32) > float(base["artifact_threshold"])
        axes[r, 0].imshow(im)
        name = groups["species"][i]
        extra = f"\nAP@R {pq['base_ap'][i]:.2f}  P@1 {int(pq['base_p1'][i])}" if pq is not None else ""
        axes[r, 0].set_title(f"#{i} {name[:28]}{extra}", fontsize=6)
        axes[r, 1].imshow(im)
        axes[r, 1].imshow(up(attn / max(attn.max(), 1e-9)), cmap="inferno", alpha=0.6)
        axes[r, 1].set_title("CLS attention (sink-dominated)", fontsize=6)
        lim = np.abs(pc1).max() or 1
        axes[r, 2].imshow(up(pc1), cmap="coolwarm", vmin=-lim, vmax=lim)
        ay, ax_ = np.where(art)
        axes[r, 2].scatter((ax_ + 0.5) * 224 / g, (ay + 0.5) * 224 / g, marker="x", s=10, c="k")
        axes[r, 2].set_title("per-image PC1 (x = artifact token)", fontsize=6)
        axes[r, 3].imshow(im)
        axes[r, 3].imshow(up(base["mask"][i]), cmap="Greens", alpha=0.45, vmin=0, vmax=1)
        x0, y0, x1, y1 = base["box"][i] * 224
        axes[r, 3].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="yellow", lw=1))
        axes[r, 3].set_title(f"foreground {fg[i]:.2f} + zoom box", fontsize=6)
        for a in axes[r]:
            a.set_xticks([]), a.set_yticks([])
    fig.tight_layout()
    target = OUT / f"sheet_{options.dataset}_{options.pick if not options.indices else 'custom'}.png"
    fig.savefig(target, dpi=options.dpi)
    print("wrote", target, "indices", ",".join(map(str, pick)))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--dataset", choices=("semi_inat", "cars", "cub"), default="semi_inat")
    e.add_argument("--variant", choices=VARIANTS, default="base")
    e.add_argument("--device", default="cuda")
    e.add_argument("--batch-size", type=int, default=128)
    e.add_argument("--workers", type=int, default=6)
    e.add_argument("--limit", type=int, default=0, help="first N images only (debugging)")
    e.add_argument("--global-only", action="store_true",
                   help="keep only cls/gap; the per-image eigendecomposition dominates at 448 (32x32 patches)")
    e.add_argument("--artifact-threshold", type=float, default=100.0,
                   help="pre-norm patch-token norm above which a token counts as a high-norm artifact")
    a = sub.add_parser("analyze")
    a.add_argument("--dataset", choices=("semi_inat", "cars", "cub"), default="semi_inat")
    a.add_argument("--device", default="cuda")
    s = sub.add_parser("sheet")
    s.add_argument("--dataset", choices=("semi_inat", "cars", "cub"), default="semi_inat")
    s.add_argument("--pick", choices=("random", "small_fail", "small_ok", "large_fail", "large_ok"), default="random")
    s.add_argument("--indices", default="")
    s.add_argument("--n", type=int, default=8)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dpi", type=int, default=110)
    s.add_argument("--store-suffix", default="", help="'_limit' reads a debugging extract")
    o = sub.add_parser("occlusion")
    o.add_argument("--dataset", choices=("semi_inat", "cars", "cub"), default="semi_inat")
    o.add_argument("--indices", required=True)
    o.add_argument("--occluder", type=int, default=3, help="side of the grey square in patches")
    o.add_argument("--name", default="custom")
    o.add_argument("--device", default="cuda")
    o.add_argument("--dpi", type=int, default=110)
    options = parser.parse_args()
    {"extract": extract, "analyze": analyze, "sheet": sheet, "occlusion": occlusion}[options.command](options)


if __name__ == "__main__":
    main()
