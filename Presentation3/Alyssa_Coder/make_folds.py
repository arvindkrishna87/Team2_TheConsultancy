#!/usr/bin/env python3
"""
make_folds.py - the 5 outer folds A-E for nested cross-validation.

Writes folds.json:
    {"A": [case ids], "B": [...], ..., "E": [...],
     "meta": {...counts, seed, source...}}

Two ways to build the folds:
  1) from scratch (default): the same StratifiedGroupKFold logic as make_splits.py
     (grouped cases stay together, seed relabels the groups, shuffle stays off),
     with n_splits = 5 instead of the 3:1:1 rotation;
  2) --from_splits_dir DIR: reuse 5 existing data_splits_new_0*.npz files, taking
     each file's test_cases as one fold (they rotate, so the 5 test sets partition the cases).

Either way the script checks that the folds are disjoint, cover every labelled case
with patches, and keep grouped cases in one fold.
"""
import argparse
import glob
import json
import os
from collections import Counter

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from config import DATA_PATHS, GROUPED_CASES, TRAINING_CONFIG
from data_utils import (build_slice_to_class_map, get_all_patch_files,
                        group_patches_by_slice, load_labels)

FOLD_NAMES = ["A", "B", "C", "D", "E"]


def case_labels(labels_csv, patches_dir):
    labels = load_labels(labels_csv)
    patches = group_patches_by_slice(get_all_patch_files(patches_dir), patches_dir)
    s2c = build_slice_to_class_map(patches, labels)
    case_to_label = {}
    for (cid, _sid), y in s2c.items():
        case_to_label.setdefault(int(cid), int(y))
    return case_to_label


def folds_from_scratch(case_to_label, seed):
    case_ids = sorted(case_to_label)
    groups = np.array(case_ids)
    for g in GROUPED_CASES:
        groups[np.isin(groups, g)] = g[0]
    uniq = np.unique(groups)
    new_id = dict(zip(uniq, np.random.RandomState(seed).permutation(len(uniq))))
    groups = np.array([new_id[g] for g in groups])
    y = np.array([case_to_label[c] for c in case_ids])
    sgkf = StratifiedGroupKFold(n_splits=5)  # shuffle off, as in make_splits.py
    folds = {}
    for name, (_, test_idx) in zip(FOLD_NAMES, sgkf.split(np.array(case_ids), y, groups)):
        folds[name] = sorted(int(case_ids[i]) for i in test_idx)
    return folds


def folds_from_splits(splits_dir, pattern):
    paths = sorted(glob.glob(os.path.join(splits_dir, pattern)))
    if len(paths) != 5:
        raise SystemExit(f"Expected 5 split files matching {pattern} in {splits_dir}, found {len(paths)}")
    folds = {}
    for name, p in zip(FOLD_NAMES, paths):
        d = np.load(p, allow_pickle=True)
        folds[name] = sorted(int(c) for c in d["test_cases"])
        print(f"Fold {name} <- test_cases of {os.path.basename(p)}")
    return folds


def check_folds(folds, case_to_label):
    seen = Counter(c for f in FOLD_NAMES for c in folds[f])
    dup = [c for c, n in seen.items() if n > 1]
    if dup:
        raise SystemExit(f"Cases in more than one fold: {dup}")
    all_cases = set(case_to_label)
    missing = sorted(all_cases - set(seen))
    extra = sorted(set(seen) - all_cases)
    if missing:
        raise SystemExit(f"Cases with patches and a label but in no fold: {missing}")
    if extra:
        print(f"[WARN] fold cases with no patches/label here (ignored by training): {extra}")
    where = {c: f for f in FOLD_NAMES for c in folds[f]}
    for g in GROUPED_CASES:
        fs = {where[c] for c in g if c in where}
        if len(fs) > 1:
            raise SystemExit(f"Grouped cases {g} are split across folds {fs}")
    print("\nFold   n   benign  high-grade")
    for f in FOLD_NAMES:
        ys = [case_to_label[c] for c in folds[f] if c in case_to_label]
        print(f"  {f}  {len(ys):3d}   {ys.count(0):5d}   {ys.count(1):5d}")
    n = sum(len(folds[f]) for f in FOLD_NAMES)
    print(f"Total: {n} cases | disjoint: yes | grouped cases kept together: yes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels_csv", default=DATA_PATHS["labels_csv"])
    ap.add_argument("--patches_dir", default=DATA_PATHS["patches_dir"])
    ap.add_argument("--seed", type=int, default=TRAINING_CONFIG["random_state"])
    ap.add_argument("--from_splits_dir", default=None)
    ap.add_argument("--splits_pattern", default="data_splits_new_0*.npz")
    ap.add_argument("--out", default="folds.json")
    args = ap.parse_args()

    case_to_label = case_labels(args.labels_csv, args.patches_dir)
    print(f"Labelled cases with patches: {len(case_to_label)} "
          f"(benign {sum(v == 0 for v in case_to_label.values())}, "
          f"high-grade {sum(v == 1 for v in case_to_label.values())})")

    if args.from_splits_dir:
        folds = folds_from_splits(args.from_splits_dir, args.splits_pattern)
        source = f"test_cases of {args.from_splits_dir}/{args.splits_pattern}"
    else:
        folds = folds_from_scratch(case_to_label, args.seed)
        source = f"StratifiedGroupKFold(5), seed {args.seed}"

    check_folds(folds, case_to_label)
    out = dict(folds)
    out["meta"] = {"source": source, "labels_csv": args.labels_csv, "patches_dir": args.patches_dir,
                   "grouped_cases": [list(g) for g in GROUPED_CASES],
                   "sizes": {f: len(folds[f]) for f in FOLD_NAMES}}
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
