"""
Outline the top-5 patches by patch attention on each of 6 slices (cases 1 and 4 x H&E, Melan-A, Sox10),
one picture per slice. Run from folder that also has find_patch.py:
    /projects/e32998/env/bin/python patch_locations.py
Saves patch_locations/<slice>_top5.png. Prints one line per slice, plus any patch that fails to match.
"""
import os
import cv2
import numpy as np
import pandas as pd
from find_patch import load_rgb, gray, build_sift, find_patch

RUN_DIR   = "/projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/Presentation2/Lily_Presentation2_code/run_20261105_131639"
PATCH_DIR = "/projects/e32998/patches_benign_split"
OUT = "patch_locations"
TOP_K = 5
RED = (0, 0, 255)  
LINE_SCALE = 1.5   # bigger means larger lines
TEXT_SCALE = 1.5   # bigger means larger text
os.makedirs(OUT, exist_ok=True)

BASE = "/home/pnx4472/Presentation2_base_code_edited/patches"
SLICES = {
    "case_4_match_1_sox10": f"{BASE}/case_4_tif_files/h2114156 sox10_ROI_1.tif",
    "case_4_match_1_h&e":   f"{BASE}/case_4_tif_files/h2114156  h&e_ROI_3.tif",
    "case_4_match_1_melan": f"{BASE}/case_4_tif_files/h2114156  melan_ROI_2.tif",
    "case_1_match_1_h&e":   f"{BASE}/case_1_tif_files/h2114153  h&e_4.tif",
    "case_1_match_1_melan": f"{BASE}/case_1_tif_files/h2114153  melan_2.tif",
    "case_1_match_1_sox10": f"{BASE}/case_1_tif_files/h2114153 sox10_1.tif",
}

for p in SLICES.values():
    if not os.path.isfile(p):
        raise FileNotFoundError(p)

df = pd.read_csv(os.path.join(RUN_DIR, "attention_analysis", "patch_attention_by_filename.csv"))
df["slice_name"] = "case_" + df.case_id.astype(str) + "_" + df.slice_id + "_" + df.stain


def box_corners(res, patch_shape):
    """4 corners of the matched patch on the slice."""
    if "corners" in res:  # use find_patch's own corners if it returns them
        return np.asarray(res["corners"], dtype=np.float32).reshape(-1, 2)
    h, w = patch_shape[:2]
    rect = (tuple(res["center"]), (w * res["scale"], h * res["scale"]), res["rotation_deg"])
    return cv2.boxPoints(rect)


sift = build_sift()
for slice_name, tif_path in SLICES.items():
    s = df[df.slice_name == slice_name]
    if s.empty:
        raise ValueError(f"{slice_name} not in this run's test set")
    top = s.nlargest(TOP_K, "patch_attn_weight")

    slice_rgb = load_rgb(tif_path)                       # ~1 min per slice for SIFT
    skp, sdesc = sift.detectAndCompute(gray(slice_rgb), None)

    canvas = cv2.cvtColor(np.ascontiguousarray(slice_rgb), cv2.COLOR_RGB2BGR)
    base_thick = max(3, canvas.shape[1] // 600)
    line_thick = int(round(base_thick * LINE_SCALE))
    font = canvas.shape[1] / 1500 * TEXT_SCALE
    text_thick = max(1, int(round(font * 1.5)))

    found = 0
    for rank, (_, r) in enumerate(top.iterrows(), start=1):
        patch_path = os.path.join(PATCH_DIR, r.patch_file)
        if not os.path.isfile(patch_path):
            raise FileNotFoundError(patch_path)
        patch = load_rgb(patch_path)

        res = find_patch(gray(slice_rgb), skp, sdesc, patch, sift)
        if res is None:
            print(f"  {slice_name} #{rank} {r.patch_file}: NOT FOUND")
            continue

        pts = box_corners(res, patch.shape).astype(np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=RED, thickness=line_thick)
        x, y = pts.min(axis=0)
        cv2.putText(canvas, f"#{rank}  {r.patch_attn_weight:.4f}", (int(x), int(y) - 2 * line_thick),
                    cv2.FONT_HERSHEY_SIMPLEX, font, RED, text_thick)
        found += 1

    out_path = os.path.join(OUT, f"{slice_name}_top{TOP_K}.png")
    cv2.imwrite(out_path, canvas)
    print(f"{slice_name}: {found}/{len(top)} outlined -> {out_path}")