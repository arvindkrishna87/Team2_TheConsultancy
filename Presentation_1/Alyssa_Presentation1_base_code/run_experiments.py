#!/usr/bin/env python3
"""
run_experiments.py

Runs main.py once per saved split (data_splits_new_*.npz), times each run, then
builds the deliverables for the "Analyze results" step.

Usage (from the folder containing main.py):

  # 1) create splits with YOUR seed (needs the shuffle patch to make_splits.py)
  python make_splits.py --seed 2026 --save_dir my_splits

  # 2) train + test on every split, then analyse
  python run_experiments.py --splits_glob "my_splits/data_splits_new_*.npz" \
         --seed 2026 --out_dir exp_seed2026

  # 3) re-do only the analysis later (no training)
  python run_experiments.py --out_dir exp_seed2026 --analyze_only

Outputs in --out_dir:
  runs.json                  split file, run dir, runtime, return code per split
  logs/split_XX.log          full stdout of each main.py run
  per_split_metrics.csv      accuracy, recall (benign / high-grade), runtime per split
  all_test_probabilities.csv predicted probability for every test case of every split
  confusion_matrices.png     one matrix per split + pooled over all splits
  loss_curves.png            train/val loss per split
  summary.md                 everything above in readable form

Notes:
- main.py evaluates the model as it is at the END of training (not best.pth).
- Train curve plotted is the cross-entropy part only (train_ce_losses), so it is
  comparable with validation loss, which also excludes the entropy term.
"""
import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN_DIR_RE = re.compile(r"Created run directory:\s*(\S+)")


def run_split(split_path, seed, extra_args, log_path):
    cmd = [sys.executable, "main.py", "--load_splits", split_path, "--seed", str(seed)] + extra_args
    print("Running:", " ".join(cmd))
    run_dir = None
    t0 = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            m = RUN_DIR_RE.search(line)
            if m:
                run_dir = m.group(1)
        rc = proc.wait()
    return run_dir, time.time() - t0, rc


def load_history(run_dir):
    """Loss history is stored inside every checkpoint; read the latest epoch."""
    ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "*_epoch*.pth"))
    if not ckpts:
        return None
    last = max(ckpts, key=lambda p: int(re.search(r"_epoch(\d+)\.pth$", p).group(1)))
    ck = torch.load(last, map_location="cpu", weights_only=False)
    train = ck.get("train_ce_losses") or ck["train_losses"]
    return {"train": list(train), "val": list(ck["val_losses"])}


def split_metrics(pred_csv):
    df = pd.read_csv(pred_csv)
    y, p = df["true_label"].to_numpy(), df["predicted_label"].to_numpy()
    cm = np.zeros((2, 2), dtype=int)  # rows = true, cols = predicted; 0 benign, 1 high-grade
    for t, q in zip(y, p):
        cm[int(t), int(q)] += 1
    tn, fp, fn, tp = cm.ravel()
    metrics = {
        "n_cases": len(df),
        "accuracy": (tn + tp) / max(cm.sum(), 1),
        "recall_benign": tn / (tn + fp) if (tn + fp) else np.nan,
        "recall_high_grade": tp / (tp + fn) if (tp + fn) else np.nan,
    }
    return df, metrics, cm


def plot_confusions(cms, pooled, path):
    mats = list(cms.items()) + [("pooled", pooled)]
    fig, axes = plt.subplots(1, len(mats), figsize=(3.2 * len(mats), 3.4), squeeze=False)
    for ax, (name, cm) in zip(axes[0], mats):
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13)
        ax.set_xticks([0, 1], ["benign", "high"])
        ax.set_yticks([0, 1], ["benign", "high"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Split {name}" if name != "pooled" else "All splits")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_losses(histories, path):
    n = len(histories)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.5), squeeze=False, sharey=True)
    for ax, (split, h) in zip(axes[0], histories.items()):
        ep = np.arange(1, len(h["val"]) + 1)
        ax.plot(ep[: len(h["train"])], h["train"], label="train (CE)")
        ax.plot(ep, h["val"], label="val")
        best = int(np.argmin(h["val"])) + 1
        ax.axvline(best, ls="--", lw=0.8, color="gray")
        ax.set_title(f"Split {split} (best val epoch {best})")
        ax.set_xlabel("Epoch")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Loss")
    axes[0][0].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def analyze(manifest, out_dir):
    rows, probs, cms, hists = [], [], {}, {}
    for m in manifest:
        sid = m["split"]
        pred = os.path.join(m["run_dir"] or "", "predictions.csv")
        if m["returncode"] != 0 or not os.path.exists(pred):
            print(f"[WARN] split {sid}: no usable predictions ({pred}); skipping")
            continue
        df, met, cm = split_metrics(pred)
        h = load_history(m["run_dir"])
        met.update(split=sid, runtime_min=round(m["runtime_sec"] / 60, 2),
                   epochs_run=len(h["val"]) if h else np.nan)
        rows.append(met)
        cms[sid] = cm
        if h:
            hists[sid] = h
        df.insert(0, "split", sid)
        probs.append(df)

    if not rows:
        raise SystemExit("Nothing to analyse.")

    metrics = pd.DataFrame(rows)[["split", "n_cases", "accuracy", "recall_benign",
                                  "recall_high_grade", "epochs_run", "runtime_min"]]
    metrics.to_csv(os.path.join(out_dir, "per_split_metrics.csv"), index=False)
    pd.concat(probs, ignore_index=True).to_csv(
        os.path.join(out_dir, "all_test_probabilities.csv"), index=False)

    pooled = sum(cms.values())
    plot_confusions(cms, pooled, os.path.join(out_dir, "confusion_matrices.png"))
    if hists:
        plot_losses(hists, os.path.join(out_dir, "loss_curves.png"))

    agg = metrics[["accuracy", "recall_benign", "recall_high_grade", "runtime_min"]].agg(["mean", "std"])
    with open(os.path.join(out_dir, "summary.md"), "w") as f:
        f.write("# Results per split\n\n```\n" + metrics.round(4).to_string(index=False) + "\n```\n\n")
        f.write("# Mean / std across splits\n\n```\n" + agg.round(4).to_string() + "\n```\n\n")
        f.write("# Pooled confusion matrix (rows = true, cols = predicted; 0 benign, 1 high-grade)\n\n```\n")
        f.write(str(pooled) + "\n```\n")
    print(metrics.round(4).to_string(index=False))
    print(agg.round(4).to_string())
    print(f"Saved results to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits_glob", default="data_splits_new_*.npz")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_dir", default="experiment_results")
    ap.add_argument("--main_args", default="", help='extra args for main.py, e.g. "--epochs 30"')
    ap.add_argument("--analyze_only", action="store_true")
    args = ap.parse_args()

    os.makedirs(os.path.join(args.out_dir, "logs"), exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "runs.json")

    if args.analyze_only:
        with open(manifest_path) as f:
            manifest = json.load(f)
    else:
        splits = sorted(glob.glob(args.splits_glob))
        if not splits:
            raise SystemExit(f"No split files match {args.splits_glob}")
        manifest = []
        for i, sp in enumerate(splits, start=1):
            log_path = os.path.join(args.out_dir, "logs", f"split_{i:02d}.log")
            run_dir, secs, rc = run_split(os.path.abspath(sp), args.seed,
                                          shlex.split(args.main_args), log_path)
            manifest.append({"split": i, "split_file": os.path.abspath(sp), "run_dir": run_dir,
                             "runtime_sec": round(secs, 1), "returncode": rc})
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)
            print(f"Split {i}: rc={rc}, {secs / 60:.1f} min, run_dir={run_dir}")

    analyze(manifest, args.out_dir)


if __name__ == "__main__":
    main()