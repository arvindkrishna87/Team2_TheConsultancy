#!/usr/bin/env python3
"""
build_patch_features_v2.py  --  Data engineers, choice 2 (patch features)

Writes three tables to --out_dir:
  patches_v2.parquet                       one row per patch: Team 5's columns + patch size
  case_features_v2.parquet / .csv          one row per case: patch features + flags

Built on Team 5's build_metadata.py (same columns, same fold logic, rows never dropped:
anything unusable gets an exclude_reason). Differences:
  * filenames are parsed the way training parses them (data_utils.group_patches_by_slice),
    so the 1,151 "-labels" / "sox10." patches that Team 5 excluded are kept;
  * slice_id is normalised like training (unmatched2 -> unmatched_2);
  * label CSV row index 64 is dropped, as data_utils.load_labels does;
  * embeddings are matched by (case, slice, stain, patch number), so zero-padded names
    (case_038_... vs case_38_...) still match.
"""
import argparse
import os
import re
import struct
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

MIN_PATCH_SIZE = 32       # embedding script skips patches smaller than this
UNI2_MAX_SIDE = 518       # UNI2 embeddings: native size unless a side is > 518 px (then resized)
PATCH_CAP_SIDE = 2000     # patching code stops searching at 2,048 px; >= 2,000 px = at the limit
PATIENT_OF = {107: 22, 108: 22, 109: 26, 110: 26, 111: 26, 118: 24, 119: 25}
CLASS_TO_LABEL = {1.0: 0, 3.0: 1, 4.0: 1}
STAINS = ("h&e", "melan", "sox10")

# training's regexes (data_utils.group_patches_by_slice) and Team 5's stricter one
TRAIN_RE_1 = re.compile(r"case_(\d+)_([a-z]+_\d+)_")
TRAIN_RE_2 = re.compile(r"case_(\d+)_([a-z]+\d+)_")
STRICT_RE = re.compile(r"^case_\d+_.+?_(h&e|melan|sox10)_patch\d+\.png$", re.IGNORECASE)
PATCH_NUM_RE = re.compile(r"_patch0*(\d+)$", re.IGNORECASE)

# flagging: modified z-score |z| > 3.5 (Iglewicz & Hoaglin); log scale for counts and sizes;
# minimum spread so near-constant features don't flag tiny differences
Z_CUTOFF, LOG_FLOOR, SHARE_FLOOR = 3.5, 0.05, 0.02
FLAG_FEATURES = {
    "n_patches": True, "patches_per_slice": True, "median_side_px": True, "side_iqr_px": True,
    "share_lt32px": False, "share_gt518px": False, "share_at_size_cap": False,
    "he_share": False, "melan_share": False, "sox10_share": False,
    "top_slice_share": False, "share_seen_by_model": False,
}
CASE_COLUMNS = [
    "case_id", "grade", "fold",
    "n_patches", "n_slices", "patches_per_slice", "top_slice_share",
    "median_side_px", "side_iqr_px", "share_lt32px", "share_gt518px", "share_at_size_cap",
    "he_share", "melan_share", "sox10_share",
    "missing_embeddings", "share_seen_by_model",
    "data_issues", "flag_reasons",
]


# =============================================================================
# Helpers
# =============================================================================
def stain_from_filename(name):
    """Copy of data_utils.extract_stain_from_filename."""
    s = name.lower()
    if "h&e" in s or "_he_" in s:
        return "h&e"
    if "melan" in s:
        return "melan"
    if "sox10" in s:
        return "sox10"
    return None


def parse_name(name):
    """(case, slice_raw, slice_id, stain, patch number) as training reads it, or None."""
    stem = os.path.splitext(name)[0]
    m = TRAIN_RE_1.match(stem) or TRAIN_RE_2.match(stem)
    stain = stain_from_filename(stem)
    if not m or stain is None:
        return None
    slice_raw = m.group(2)
    slice_id = slice_raw if "_" in slice_raw else re.sub(r"([A-Za-z])(\d)", r"\1_\2", slice_raw)
    num = PATCH_NUM_RE.search(stem)
    return int(m.group(1)), slice_raw, slice_id, stain, int(num.group(1)) if num else None


def png_size(path):
    """(width, height) from the PNG header; PIL for anything else."""
    try:
        with open(path, "rb") as f:
            head = f.read(24)
        if head[:8] == b"\x89PNG\r\n\x1a\n" and head[12:16] == b"IHDR":
            return struct.unpack(">II", head[16:24])
        from PIL import Image
        with Image.open(path) as im:
            return im.size
    except Exception:
        return (None, None)


def list_files(folder, suffix):
    with os.scandir(folder) as it:
        return sorted(e.name for e in it if e.is_file() and e.name.endswith(suffix))


def load_labels(path, case_col, class_col):
    raw = pd.read_csv(path).drop(index=64, errors="ignore")   # as data_utils.load_labels
    df = raw[[case_col, class_col]].rename(columns={case_col: "case_id", class_col: "grade_raw"})
    df["case_id"] = pd.to_numeric(df["case_id"], errors="coerce")
    df["grade_raw"] = pd.to_numeric(df["grade_raw"], errors="coerce")
    df["label"] = df["grade_raw"].map(CLASS_TO_LABEL)
    df = df.dropna(subset=["case_id", "label"]).drop_duplicates("case_id")
    return df.astype({"case_id": "Int64", "label": "Int64"})


def load_folds(splits_dir):
    """fold Fk = test_cases of data_splits_new_0k.npz."""
    fold_of = {}
    for k in range(1, 6):
        f = Path(splits_dir) / f"data_splits_new_0{k}.npz"
        if f.exists():
            for c in np.load(f, allow_pickle=True)["test_cases"]:
                fold_of[int(c)] = f"F{k}"
    return fold_of


def iqr(s):
    s = s.dropna()
    return float(s.quantile(0.75) - s.quantile(0.25)) if len(s) else np.nan


def robust_z(x, floor):
    """Modified z-score 0.6745 * (x - median) / MAD, MAD floored at `floor`."""
    med = x.median()
    return 0.6745 * (x - med) / max((x - med).abs().median(), floor)


# =============================================================================
# 1  Patch table
# =============================================================================
def build_patch_table(args):
    pdir, edir = Path(args.patches_dir), Path(args.emb_dir)
    pts = list_files(edir, ".pt") if edir.exists() else []
    pt_set = set(pts)
    pt_by_key = {}
    for n in pts:
        k = parse_name(n)
        if k:
            pt_by_key.setdefault((k[0], k[2], k[3], k[4]), []).append(n)

    rows = []
    for name in list_files(pdir, ".png"):
        row = dict(patch_id=name[:-4], png_name=name, png_path=str(pdir / name),
                   name_irregular=STRICT_RE.match(name) is None)
        p = parse_name(name)
        if p is None:
            row["exclude_reason"] = "filename did not parse"
        else:
            case, slice_raw, slice_id, stain, idx = p
            row.update(case_id=case, patient_id=PATIENT_OF.get(case, case), slice_raw=slice_raw,
                       slice_id=slice_id, stain=stain, patch_idx=idx, exclude_reason=None)
            same = name[:-4] + ".pt"
            cands = pt_by_key.get((case, slice_id, stain, idx), [])
            row["emb_path"] = (str(edir / same) if same in pt_set
                               else str(edir / cands[0]) if len(cands) == 1 else None)
        rows.append(row)
    df = pd.DataFrame(rows).astype({"case_id": "Int64", "patient_id": "Int64", "patch_idx": "Int64"})

    labels = load_labels(args.labels_csv, args.case_col, args.class_col)
    df = df.merge(labels[["case_id", "grade_raw", "label"]], on="case_id", how="left")
    df.loc[df["exclude_reason"].isna() & df["label"].isna(), "exclude_reason"] = "case has no valid label"
    df["fold"] = df["case_id"].map(load_folds(args.splits_dir)) if args.splits_dir else None

    # patch size
    wh = [png_size(p) for p in df["png_path"]]
    df["width_px"] = pd.array([w for w, _ in wh], dtype="Int64")
    df["height_px"] = pd.array([h for _, h in wh], dtype="Int64")
    w, h = df["width_px"].astype(float), df["height_px"].astype(float)
    df["size_wxh"] = df["width_px"].astype(str) + "x" + df["height_px"].astype(str)
    df["area_px"] = (w * h).astype("Int64")
    df["side_px"] = np.sqrt(w * h).round(1)
    df["aspect_ratio"] = (np.maximum(w, h) / np.minimum(w, h)).round(3)
    df["size_status"] = np.where(w.isna(), "unreadable",
                                 np.where((w < MIN_PATCH_SIZE) | (h < MIN_PATCH_SIZE), "too_small", "ok"))
    df["resized_for_embedding"] = np.maximum(w, h) > UNI2_MAX_SIDE

    # embedding status
    has = df["emb_path"].notna()
    embedded_cases = set(df.loc[has, "case_id"].dropna().astype(int))
    df["emb_status"] = np.select(
        [has, ~df["case_id"].isin(embedded_cases), df["size_status"].eq("too_small")],
        ["ok", "case_not_embedded", "too_small"], default="missing")
    df.loc[df["case_id"].isna(), "emb_status"] = None
    df["emb_version"] = np.where(has, args.emb_version, None)
    df["dataset_version"] = args.version
    df["built_at"] = datetime.now().isoformat(timespec="seconds")
    return df.sort_values(["case_id", "stain", "slice_id", "patch_idx"], na_position="last").reset_index(drop=True), labels


# =============================================================================
# 2  Case table
# =============================================================================
def patches_seen_by_model(inc, per_slice_cap, max_slices_per_stain):
    """Patches the model reads (dataset.py): per stain, the first max_slices_per_stain slices
    by slice_id; per slice, embedded patches capped at per_slice_cap."""
    seen = {}
    for (cid, _), g in inc[inc["emb_status"].eq("ok")].groupby(["case_id", "stain"]):
        counts = g.groupby("slice_id").size().sort_index(key=lambda i: i.astype(str))
        if max_slices_per_stain:
            counts = counts.iloc[:max_slices_per_stain]
        if per_slice_cap:
            counts = counts.clip(upper=per_slice_cap)
        seen[cid] = seen.get(cid, 0) + int(counts.sum())
    return pd.Series(seen, dtype=float)


def build_case_table(df, labels, args):
    inc = df[df["exclude_reason"].isna()].copy()
    inc["side"] = inc["side_px"].astype(float)
    g = inc.groupby("case_id")
    t = pd.DataFrame({
        "grade": g["label"].first().map({0: "Benign", 1: "High-grade"}),
        "fold": g["fold"].first(),
        "n_patches": g.size(),
        "n_slices": g["slice_id"].nunique(),
        "median_side_px": g["side"].median(),
        "side_iqr_px": g["side"].apply(iqr),
        "share_lt32px": g["side"].apply(lambda s: (s < MIN_PATCH_SIZE).mean()),
        "share_gt518px": g["resized_for_embedding"].mean(),
        "share_at_size_cap": g["side"].apply(lambda s: (s >= PATCH_CAP_SIDE).mean()),
        "missing_embeddings": g["emb_status"].apply(lambda s: int(s.ne("ok").sum())),
    })
    t["patches_per_slice"] = t["n_patches"] / t["n_slices"]
    t["top_slice_share"] = inc.groupby(["case_id", "slice_id"]).size().groupby(level=0).max() / t["n_patches"]
    for s in STAINS:
        t[f"{s.replace('&', '')}_share"] = (inc[inc.stain == s].groupby("case_id").size()
                                            .reindex(t.index).fillna(0) / t["n_patches"])
    embedded = g["emb_status"].apply(lambda s: not s.eq("case_not_embedded").all())
    seen = patches_seen_by_model(inc, args.per_slice_cap, args.max_slices_per_stain)
    t["share_seen_by_model"] = np.where(embedded, seen.reindex(t.index).fillna(0) / t["n_patches"], np.nan)

    # data problems, kept apart from feature outliers
    issues = {}
    for c in t.index:
        i = ["no embeddings yet"] if not embedded[c] else []
        if pd.isna(t.at[c, "fold"]):
            i.append("not in any split")
        miss = [s for s in STAINS if t.at[c, f"{s.replace('&', '')}_share"] == 0]
        if miss:
            i.append("missing stain: " + ",".join(miss))
        issues[c] = "; ".join(i)
    t["data_issues"] = pd.Series(issues)
    t = t.reset_index()

    none = labels[~labels["case_id"].isin(t["case_id"])]
    if len(none):
        t = pd.concat([t, pd.DataFrame({
            "case_id": none["case_id"], "grade": none["label"].map({0: "Benign", 1: "High-grade"}),
            "n_patches": 0, "data_issues": "no patches"})], ignore_index=True)
    return t


# =============================================================================
# 3  Flags
# =============================================================================
def flag_cases(t):
    why = {i: [] for i in t.index}
    has = t["n_patches"].fillna(0) > 0
    for feat, use_log in FLAG_FEATURES.items():
        x = t.loc[has & t[feat].notna(), feat].astype(float)
        z = robust_z(np.log1p(x) if use_log else x, LOG_FLOOR if use_log else SHARE_FLOOR)
        for i in z.index[z.abs() > Z_CUTOFF]:
            why[i].append(f"{feat} {'high' if z[i] > 0 else 'low'} ({t.at[i, feat]:.3g}, z={z[i]:+.1f})")
    t["flag_reasons"] = [", ".join(why[i]) for i in t.index]
    return t


# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patches_dir", default="/projects/e32998/patch_images")
    ap.add_argument("--emb_dir", default="/projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/"
                                         "Presentation2/patch_embeddings_UNI2")
    ap.add_argument("--labels_csv", default="/projects/e32998/STAT390_Krish/Code/Code4_reduce_runtime/"
                                            "extra_benign_case_grade_match.csv")
    ap.add_argument("--splits_dir", default=None, help="folder with data_splits_new_01..05.npz")
    ap.add_argument("--case_col", default="Case")
    ap.add_argument("--class_col", default="Class")
    ap.add_argument("--per_slice_cap", type=int, default=500, help="as in run_training_custom_split.sbatch")
    ap.add_argument("--max_slices_per_stain", type=int, default=5, help="as in run_training_custom_split.sbatch")
    ap.add_argument("--emb_version", default="uni2_h_native_cap518")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--out_dir", default="./patch_features_v2")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p_path = out / f"patches_{args.version}.parquet"
    if p_path.exists() and not args.overwrite:
        raise SystemExit(f"{p_path} exists. Use a new --version (or --overwrite).")

    df, labels = build_patch_table(args)
    df.to_parquet(p_path, index=False)

    t = flag_cases(build_case_table(df, labels, args))
    t = t[CASE_COLUMNS].sort_values("case_id")
    t.to_parquet(out / f"case_features_{args.version}.parquet", index=False)
    t.to_csv(out / f"case_features_{args.version}.csv", index=False)


if __name__ == "__main__":
    main()
