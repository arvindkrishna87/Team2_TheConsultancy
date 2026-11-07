#!/usr/bin/env python3
"""
check_augmentation_similarity.py

Sanity check before building augmented embeddings (Part 3):
does a given image transform actually change the KimiaNet pooled embedding?

For a random sample of patches, embed the original and several transformed
versions with exactly the same pipeline as precompute_pooled_features.py
(no resize, ImageNet normalisation, DenseNet121 features -> AdaptiveAvgPool2d(2,2)
-> 4096-d vector), then compare:

  * cos(orig, aug)                 how close each patch is to its own augmented copy
  * cos(orig_i, orig_j)            baseline: two different patches from the same slice
                                   and two patches from different cases
  * cos(aug, quadrant-permuted orig)  for flips/rotations: is the augmented embedding
                                   just the original with its 2x2 pooled cells reordered?

All cosines are reported both raw and "centered" (sample-mean embedding subtracted),
because raw DenseNet features share a large common component that pushes every
cosine towards 1 and hides differences.

Rule of thumb for reading the output: an augmentation is useful when its
"aug vs orig" distance is a meaningful fraction of the "same-slice neighbour"
distance (relative_shift column). If it is ~0, the model sees essentially the
same vector and the augmentation will do little against memorisation.

Nothing is written to the shared patch/embedding folders; outputs go to --out_dir.
"""

import os
import re
import csv
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from precompute_pooled_features import load_kimianet_densenet121, MIN_PATCH_SIZE


# ----------------------------
# Augmentations (operate on PIL RGB images)
# ----------------------------
def hed_jitter(img: Image.Image, rng: np.random.RandomState, sigma: float = 0.05, bias: float = 0.05) -> Image.Image:
    """
    H&E stain jitter (Tellez et al. 2018 style): convert RGB -> HED stain space,
    scale and shift each stain channel by a small random amount, convert back.
    """
    from skimage.color import rgb2hed, hed2rgb

    arr = np.asarray(img).astype(np.float64) / 255.0
    hed = rgb2hed(arr)
    alpha = rng.uniform(1 - sigma, 1 + sigma, size=3)
    beta = rng.uniform(-bias, bias, size=3)
    hed = hed * alpha + beta
    rgb = np.clip(hed2rgb(hed), 0, 1)
    return Image.fromarray((rgb * 255).astype(np.uint8))


def build_augmentations(seed: int, hed_sigma: float, hed_bias: float):
    rng = np.random.RandomState(seed)
    color_jitter = T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.03)
    return {
        "hflip": TF.hflip,
        "vflip": TF.vflip,
        "rot90": lambda im: im.transpose(Image.ROTATE_90),
        "rot180": lambda im: im.transpose(Image.ROTATE_180),
        "color_jitter": color_jitter,
        "hed_jitter": lambda im: hed_jitter(im, rng, sigma=hed_sigma, bias=hed_bias),
    }


# Which reordering of the 2x2 pooled cells a geometric transform should roughly
# correspond to, if the CNN were perfectly equivariant. Cells are indexed
# (row, col) -> r*2 + c, i.e. [TL, TR, BL, BR].
CELL_PERMUTATIONS = {
    "hflip": [1, 0, 3, 2],   # left <-> right
    "vflip": [2, 3, 0, 1],   # top <-> bottom
    "rot180": [3, 2, 1, 0],
    # PIL ROTATE_90 is counter-clockwise: new TL = old TR, new TR = old BR,
    # new BL = old TL, new BR = old BL
    "rot90": [1, 3, 0, 2],
}


def permute_cells(v: torch.Tensor, perm) -> torch.Tensor:
    """v is (4*F,) flattened from (F, 2, 2); reorder the 4 spatial cells."""
    F = v.numel() // 4
    cells = v.view(F, 4)
    return cells[:, perm].reshape(-1)


# ----------------------------
# Sampling
# ----------------------------
NAME_RE = re.compile(r"case_(\d+)_([a-z]+_?\d+)_(.+?)_patch\d+\.png$", re.IGNORECASE)


def sample_patches(patches_dir: str, n: int, seed: int):
    """Random sample of patch filenames, spread over slices, with parsed metadata."""
    files = sorted(f for f in os.listdir(patches_dir) if f.lower().endswith(".png"))
    by_slice = defaultdict(list)
    for f in files:
        m = NAME_RE.match(f)
        if not m:
            continue
        case_id, slice_id, stain = int(m.group(1)), m.group(2), m.group(3).lower()
        by_slice[(case_id, slice_id, stain)].append(f)

    rnd = random.Random(seed)
    slices = list(by_slice.keys())
    rnd.shuffle(slices)

    # Take 2 patches per slice so every sampled patch has a same-slice neighbour.
    chosen = []
    for key in slices:
        if len(by_slice[key]) < 2:
            continue
        for f in rnd.sample(by_slice[key], 2):
            chosen.append((f, key))
        if len(chosen) >= n:
            break
    return chosen[:n]


# ----------------------------
# Main
# ----------------------------
def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches_dir", default="/projects/e32998/patches_benign_split")
    ap.add_argument("--kimianet_ckpt", default="KimiaNet.pth")
    ap.add_argument("--out_dir", default="aug_similarity_check")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--reference_embeddings_dir", default="/projects/e32998/patches_varsize_pooled4096",
                    help="Existing embeddings, used only to verify this script reproduces them")
    ap.add_argument("--hed_sigma", type=float, default=0.05, help="HED jitter: max relative scale per stain")
    ap.add_argument("--hed_bias", type=float, default=0.005, help="HED jitter: max additive shift per stain")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    to_tensor = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model = load_kimianet_densenet121(args.kimianet_ckpt, device="cpu", verbose=True)
    features = model.features.eval().to(args.device)
    pool = nn.AdaptiveAvgPool2d((2, 2)).to(args.device)

    @torch.inference_mode()
    def embed(img: Image.Image) -> torch.Tensor:
        x = to_tensor(img).unsqueeze(0).to(args.device)
        return pool(features(x)).flatten(1)[0].float().cpu()

    augs = build_augmentations(args.seed, args.hed_sigma, args.hed_bias)
    sample = sample_patches(args.patches_dir, args.n, args.seed)
    print(f"[INFO] sampled {len(sample)} patches from "
          f"{len({k for _, k in sample})} slices / {len({k[0] for _, k in sample})} cases")

    names, keys = [], []
    orig = []
    aug_vecs = {a: [] for a in augs}
    repro = []

    for i, (fname, key) in enumerate(sample, 1):
        img = Image.open(os.path.join(args.patches_dir, fname)).convert("RGB")
        if min(img.size) < MIN_PATCH_SIZE:
            continue
        v = embed(img)
        names.append(fname)
        keys.append(key)
        orig.append(v)
        for a, fn in augs.items():
            aug_vecs[a].append(embed(fn(img)))

        ref = os.path.join(args.reference_embeddings_dir, os.path.splitext(fname)[0] + ".pt")
        if os.path.exists(ref):
            repro.append(cos(v, torch.load(ref, map_location="cpu").float().view(-1)))

        if i % 50 == 0:
            print(f"[PROGRESS] {i}/{len(sample)}")

    O = torch.stack(orig)                       # (N, 4096)
    A = {a: torch.stack(v) for a, v in aug_vecs.items()}
    mu = O.mean(0)
    N = len(names)

    # Same-slice neighbour baseline: patches were sampled in pairs from the same slice.
    same_slice_pairs = [(i, i + 1) for i in range(0, N - 1, 2) if keys[i] == keys[i + 1]]
    # Different-case baseline: random pairs from different cases.
    rnd = random.Random(args.seed + 1)
    diff_case_pairs = []
    while len(diff_case_pairs) < N:
        i, j = rnd.randrange(N), rnd.randrange(N)
        if keys[i][0] != keys[j][0]:
            diff_case_pairs.append((i, j))

    def stats(xs):
        xs = np.asarray(xs)
        return xs.mean(), np.percentile(xs, 5), np.median(xs), np.percentile(xs, 95)

    def pair_cos(pairs, centered):
        c = mu if centered else 0
        return [cos(O[i] - c, O[j] - c) for i, j in pairs]

    rows = []
    summary_lines = []

    def add(label, raw, cen):
        r, c = stats(raw), stats(cen)
        summary_lines.append(
            f"{label:<34} raw mean={r[0]:.4f} (p5={r[1]:.4f}, p95={r[3]:.4f}) | "
            f"centered mean={c[0]:.4f} (p5={c[1]:.4f}, p95={c[3]:.4f})"
        )
        return c[0]

    ss_cen_mean = add("baseline: same-slice neighbour", pair_cos(same_slice_pairs, False), pair_cos(same_slice_pairs, True))
    add("baseline: different case", pair_cos(diff_case_pairs, False), pair_cos(diff_case_pairs, True))
    summary_lines.append("")

    # Distance-based "how far does the augmentation move a patch, relative to a
    # same-slice neighbour" -- 0 = no change, 1 = as different as another patch.
    ss_dist = np.mean([float((O[i] - O[j]).norm()) for i, j in same_slice_pairs])

    for a in augs:
        raw = [cos(O[i], A[a][i]) for i in range(N)]
        cen = [cos(O[i] - mu, A[a][i] - mu) for i in range(N)]
        add(f"{a}: aug vs orig", raw, cen)
        dist = [float((O[i] - A[a][i]).norm()) for i in range(N)]
        rel = np.mean(dist) / ss_dist
        summary_lines.append(f"{'':<34} relative_shift = {rel:.3f}  (aug distance / same-slice neighbour distance)")

        if a in CELL_PERMUTATIONS:
            perm = CELL_PERMUTATIONS[a]
            pc = [cos(permute_cells(O[i], perm) - permute_cells(mu, perm), A[a][i] - permute_cells(mu, perm))
                  for i in range(N)]
            summary_lines.append(
                f"{'':<34} centered cos(aug, orig with 2x2 cells reordered) mean={np.mean(pc):.4f}"
                f"  (close to 1 => augmentation is mostly a cell reshuffle)"
            )
        summary_lines.append("")

        for i in range(N):
            rows.append({
                "filename": names[i], "case_id": keys[i][0], "slice_id": keys[i][1], "stain": keys[i][2],
                "augmentation": a, "cos_raw": raw[i], "cos_centered": cen[i], "l2_dist": dist[i],
            })

    header = [
        f"Patches: {N}  (slices: {len(set(keys))}, cases: {len({k[0] for k in keys})})",
        f"Same-slice pairs: {len(same_slice_pairs)}, different-case pairs: {len(diff_case_pairs)}",
    ]
    if repro:
        header.append(
            f"Reproduction check vs {args.reference_embeddings_dir}: "
            f"{len(repro)} matched, mean cos={np.mean(repro):.5f}, min={np.min(repro):.5f} "
            f"(should be ~1.0; fp16 storage explains tiny gaps)"
        )
    header.append("")

    text = "\n".join(header + summary_lines)
    print("\n" + text)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    with open(os.path.join(args.out_dir, "per_patch_similarity.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[DONE] wrote {args.out_dir}/summary.txt and per_patch_similarity.csv")


if __name__ == "__main__":
    main()
