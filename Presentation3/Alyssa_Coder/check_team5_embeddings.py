#!/usr/bin/env python3
"""Which UNI2 method (native size or 224 resize) made an existing embeddings folder?"""
import argparse
import os
import random

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from precompute_pooled_features import UNI2_INPUT_SIZE, load_uni2

NORM = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb_dir", required=True)
    ap.add_argument("--patches_dir", default="/projects/e32998/patch_images")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    names = sorted(f[:-3] for f in os.listdir(args.emb_dir) if f.endswith(".pt"))
    random.Random(args.seed).shuffle(names)
    picked = []
    for n in names:
        p = os.path.join(args.patches_dir, n + ".png")
        if os.path.isfile(p):
            w, h = Image.open(p).size
            if (w, h) != (224, 224) and max(w, h) <= 700:
                picked.append((n, w, h))
        if len(picked) == args.n:
            break
    if not picked:
        raise SystemExit("No matching PNGs found for these embeddings")

    model = load_uni2(pretrained=True, native_size=True).eval()
    tf = {"native": T.Compose([T.ToTensor(), NORM]),
          "224": T.Compose([T.Resize((UNI2_INPUT_SIZE, UNI2_INPUT_SIZE)), T.ToTensor(), NORM])}

    votes = {"native": 0, "224": 0}
    print(f"{'patch':48s} {'size':>9s} {'cos native':>11s} {'cos 224':>9s}")
    with torch.inference_mode():
        for n, w, h in picked:
            saved = torch.load(os.path.join(args.emb_dir, n + ".pt"), map_location="cpu").float().view(-1)
            img = Image.open(os.path.join(args.patches_dir, n + ".png")).convert("RGB")
            cos = {}
            for mode in ("native", "224"):
                e = model(tf[mode](img).unsqueeze(0))[0].float()
                cos[mode] = F.cosine_similarity(e, saved, dim=0).item()
            votes[max(cos, key=cos.get)] += 1
            print(f"{n[:48]:48s} {w:4d}x{h:<4d} {cos['native']:11.4f} {cos['224']:9.4f}")
    print(f"\nBest match: native {votes['native']} / 224 {votes['224']} of {len(picked)} patches")
    print("A method matches if its cosines are ~0.999-1.000.")


if __name__ == "__main__":
    main()
