#!/usr/bin/env python3
"""
Presentation 3, Analyst choice 1: results table + loss curves for every experiment.

Expects the folder layout made by run_p3_analyst1.sbatch:
    <runs_root>/split01/<experiment>/run_YYYYMMDD_HHMMSS/
        predictions_best.csv
        checkpoints/<timestamp>_HierarchicalAttnMIL_epoch<N>.pth   (one per epoch)
        checkpoints/best.pth

Usage:
    python3 summarize_analyst1.py /projects/e32998/STAT390_Fall2026/Team2_TheConsultancy/P3_analyst1/runs/split01

Outputs (written into the split folder):
    results_table.csv   accuracy, benign recall, high-grade recall, best epoch,
                        epochs trained, first epoch with train loss < threshold
    loss_curves.png     train and validation loss per epoch, one panel per experiment

Train loss plotted is the cross-entropy part only (train_ce_losses). The trainer's
total train loss also subtracts the patch-entropy term, so CE is the fair curve to
compare with validation loss.
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch

NEAR_ZERO = 0.05   # "training loss has reached ~0" threshold; state it on the slide


def load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # older torch without weights_only
        return torch.load(path, map_location="cpu")


def latest_run(exp_dir):
    runs = sorted(glob.glob(os.path.join(exp_dir, "run_*")))
    return runs[-1] if runs else None


def last_epoch_ckpt(ckpt_dir):
    best_n, best_p = -1, None
    for p in glob.glob(os.path.join(ckpt_dir, "*_epoch*.pth")):
        m = re.search(r"_epoch(\d+)\.pth$", p)
        if m and int(m.group(1)) > best_n:
            best_n, best_p = int(m.group(1)), p
    return best_p


def recall(df, cls):
    sub = df[df["true_label"] == cls]
    return (sub["predicted_label"] == cls).mean() if len(sub) else float("nan")


def main(split_dir):
    rows, curves = [], {}
    for exp_dir in sorted(d for d in glob.glob(os.path.join(split_dir, "*")) if os.path.isdir(d)):
        exp = os.path.basename(exp_dir)
        run = latest_run(exp_dir)
        if run is None:
            print(f"[skip] {exp}: no run_* folder yet")
            continue

        preds = glob.glob(os.path.join(run, "predictions*.csv"))
        if not preds:
            print(f"[skip] {exp}: no predictions CSV in {run} (job still running or failed?)")
            continue
        df = pd.read_csv(preds[0])

        ckpt_dir = os.path.join(run, "checkpoints")
        last = last_epoch_ckpt(ckpt_dir)
        best = os.path.join(ckpt_dir, "best.pth")
        hist = load_ckpt(last) if last else {}
        train_ce = hist.get("train_ce_losses", [])
        val = hist.get("val_losses", [])
        best_epoch = load_ckpt(best)["epoch"] if os.path.isfile(best) else None
        near_zero = next((i + 1 for i, v in enumerate(train_ce) if v < NEAR_ZERO), None)

        rows.append({
            "experiment": exp,
            "n_test_cases": len(df),
            "accuracy": df["correct"].mean(),
            "benign_recall": recall(df, 0),
            "high_grade_recall": recall(df, 1),
            "best_epoch": best_epoch,
            "epochs_trained": len(train_ce),
            f"first_epoch_train_loss_below_{NEAR_ZERO}": near_zero,
            "run_folder": run,
        })
        curves[exp] = (train_ce, val, best_epoch)

    if not rows:
        print("No finished experiments found.")
        return

    table = pd.DataFrame(rows)
    out_csv = os.path.join(split_dir, "results_table.csv")
    table.to_csv(out_csv, index=False)
    print(table.drop(columns="run_folder").round(3).to_string(index=False))
    print(f"\nSaved {out_csv}")

    n = len(curves)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 3.4), sharey=True, squeeze=False)
    for ax, (exp, (tr, va, be)) in zip(axes[0], curves.items()):
        ax.plot(range(1, len(tr) + 1), tr, label="train (CE)")
        ax.plot(range(1, len(va) + 1), va, label="validation")
        if be:
            ax.axvline(be, color="gray", ls="--", lw=1, label=f"best epoch {be}")
        ax.set_title(exp)
        ax.set_xlabel("epoch")
    axes[0][0].set_ylabel("loss")
    axes[0][0].legend(fontsize=8)
    fig.tight_layout()
    out_png = os.path.join(split_dir, "loss_curves.png")
    fig.savefig(out_png, dpi=200)
    print(f"Saved {out_png}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])