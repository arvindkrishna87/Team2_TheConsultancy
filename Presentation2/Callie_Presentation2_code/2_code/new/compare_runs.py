#!/usr/bin/env python3
"""
compare_runs.py

Compare the baseline model with the augmented (flipped-patch) model across the
5 cross-validation splits, producing everything Part 4 of the assignment asks for.

Usage (run folders are paired by position: 1st baseline <-> 1st flip = split 01, ...):

    python3 compare_runs.py \
        --baseline_runs RUN_b1 RUN_b2 RUN_b3 RUN_b4 RUN_b5 \
        --aug_runs      RUN_f1 RUN_f2 RUN_f3 RUN_f4 RUN_f5 \
        --out_dir comparison_baseline_vs_flip

Each run folder is a main.py output folder containing epoch_losses.csv,
step_losses.csv, metrics.json and predictions*.csv.

Outputs (in --out_dir):
  loss_per_epoch.png         train CE + val loss per epoch, one panel per split
  loss_per_update.png        train CE per update (raw + 1-epoch moving average) and
                             val loss at each epoch end, one panel per split
  confusion_matrices.png     pooled over all test cases, baseline vs augmented
  per_case_predictions.png   P(high-grade) for every test case, both models
  metrics_per_split.csv      accuracy / recalls / loss / memorization stats per split
  per_case_predictions.csv   both models' predictions for every test case
  summary.txt                pooled metrics, mean +/- sd, paired case-level test,
                             memorization summary
"""

import os
import glob
import json
import argparse

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

# Validated categorical pair (blue / orange) + neutral inks on a light surface.
COLORS = {"baseline": "#2a78d6", "aug": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlesize": 11, "axes.labelsize": 10, "font.size": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False,
})


# ----------------------------
# Loading
# ----------------------------
def load_run(run_dir):
    def need(name):
        p = os.path.join(run_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(f"{p} not found - was this run trained with the updated main.py/trainer.py?")
        return p

    preds = sorted(glob.glob(os.path.join(run_dir, "predictions*.csv")))
    if not preds:
        raise FileNotFoundError(f"No predictions*.csv in {run_dir}")
    cfg_path = os.path.join(run_dir, "run_config.json")
    return {
        "dir": run_dir,
        "epochs": pd.read_csv(need("epoch_losses.csv")),
        "steps": pd.read_csv(need("step_losses.csv")),
        "metrics": json.load(open(need("metrics.json"))),
        "preds": pd.read_csv(preds[0]),
        "config": json.load(open(cfg_path)) if os.path.exists(cfg_path) else {},
    }


def check_pairing(base, aug, split_idx):
    b, a = set(base["preds"]["case_id"]), set(aug["preds"]["case_id"])
    if b != a:
        raise ValueError(f"Split {split_idx}: baseline and augmented runs have different test cases "
                         f"(only baseline: {sorted(b - a)}, only augmented: {sorted(a - b)}). "
                         f"Pass runs in the same split order and use the same --load_splits.")
    bs = base["config"].get("args", {}).get("load_splits")
    as_ = aug["config"].get("args", {}).get("load_splits")
    if bs and as_ and os.path.basename(bs) != os.path.basename(as_):
        print(f"[WARN] split {split_idx}: different split files ({bs} vs {as_})")


# ----------------------------
# Metrics
# ----------------------------
def cm_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    n = tn + fp + fn + tp
    br = tn / (tn + fp) if tn + fp else np.nan
    hr = tp / (tp + fn) if tp + fn else np.nan
    return {"cm": np.array([[tn, fp], [fn, tp]]), "n": n, "accuracy": (tn + tp) / n if n else np.nan,
            "benign_recall": br, "high_grade_recall": hr, "balanced_accuracy": np.nanmean([br, hr])}


def memorization_stats(run):
    """Train-loss based memorization indicators (CE part only; total loss includes the entropy term)."""
    e = run["epochs"]
    best = run["metrics"].get("best_epoch") or int(e.loc[e["val_loss"].idxmin(), "epoch"])
    row_best = e[e["epoch"] == best].iloc[0]
    last = e.iloc[-1]
    below = e[e["train_ce_loss"] < 0.05]
    return {
        "epochs_trained": int(len(e)),
        "best_epoch": int(best),
        "train_ce_at_best": float(row_best["train_ce_loss"]),
        "val_loss_at_best": float(row_best["val_loss"]),
        "gap_at_best": float(row_best["val_loss"] - row_best["train_ce_loss"]),
        "train_ce_final": float(last["train_ce_loss"]),
        "val_loss_final": float(last["val_loss"]),
        "gap_final": float(last["val_loss"] - last["train_ce_loss"]),
        "min_train_ce": float(e["train_ce_loss"].min()),
        "first_epoch_train_ce_below_0.05": int(below["epoch"].iloc[0]) if len(below) else None,
        # Val loss rise after its minimum: how much it climbs while train keeps falling
        "val_rise_after_best": float(e["val_loss"].iloc[e["val_loss"].idxmin():].max() - e["val_loss"].min()),
    }


def mcnemar_exact(b_only, a_only):
    """Exact two-sided McNemar test on discordant pairs."""
    n = b_only + a_only
    if n == 0:
        return 1.0
    try:
        from scipy.stats import binomtest
        return float(binomtest(min(b_only, a_only), n, 0.5).pvalue)
    except Exception:
        from math import comb
        k = min(b_only, a_only)
        return float(min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n))


# ----------------------------
# Plots
# ----------------------------
def _legend_handles(names):
    from matplotlib.lines import Line2D
    return [
        Line2D([], [], color=COLORS["baseline"], lw=2, label=names["baseline"]),
        Line2D([], [], color=COLORS["aug"], lw=2, label=names["aug"]),
        Line2D([], [], color=INK_2, lw=2, ls="-", label="train CE"),
        Line2D([], [], color=INK_2, lw=2, ls="--", label="validation loss"),
        Line2D([], [], color=INK_2, lw=0, marker="o", markerfacecolor=SURFACE, markersize=8,
               markeredgewidth=2, label="best epoch (checkpoint used)"),
    ]


def plot_loss_per_epoch(pairs, names, path):
    n = len(pairs)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.8), sharey=True, squeeze=False)
    for k, (base, aug) in enumerate(pairs):
        ax = axes[0, k]
        for key, run in (("baseline", base), ("aug", aug)):
            e = run["epochs"]
            c = COLORS[key]
            ax.plot(e["epoch"], e["train_ce_loss"], color=c, lw=2, ls="-")
            ax.plot(e["epoch"], e["val_loss"], color=c, lw=2, ls="--")
            best = run["metrics"].get("best_epoch")
            if best is not None:
                r = e[e["epoch"] == best]
                ax.plot(r["epoch"], r["val_loss"], "o", ms=8, mfc=SURFACE, mec=c, mew=2, zorder=5)
        ax.set_title(f"Split {k + 1:02d}")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        if k == 0:
            ax.set_ylabel("Loss (cross-entropy)")
    fig.legend(handles=_legend_handles(names), loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Train and validation loss per epoch", y=1.13, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_loss_per_update(pairs, names, path):
    n = len(pairs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.6 * n), squeeze=False)
    for k, (base, aug) in enumerate(pairs):
        ax = axes[k, 0]
        ymax = 0
        for key, run in (("baseline", base), ("aug", aug)):
            s, e = run["steps"], run["epochs"]
            c = COLORS[key]
            steps_per_epoch = max(int(s.groupby("epoch").size().median()), 1)
            smooth = s["ce_loss"].rolling(steps_per_epoch, min_periods=1).mean()
            ax.plot(s["global_step"], s["ce_loss"], color=c, lw=0.6, alpha=0.18)
            ax.plot(s["global_step"], smooth, color=c, lw=2)
            # Validation is computed once per epoch -> place it at that epoch's last update.
            end_step = s.groupby("epoch")["global_step"].max()
            vx = [end_step.get(ep, np.nan) for ep in e["epoch"]]
            ax.plot(vx, e["val_loss"], color=c, lw=2, ls="--", marker="o", ms=4)
            ymax = max(ymax, smooth.max(), e["val_loss"].max())
        ax.set_ylim(0, ymax * 1.15)  # raw per-update spikes are clipped; the moving average is the readable signal
        ax.set_title(f"Split {k + 1:02d}", loc="left")
        ax.set_ylabel("Loss")
        if k == n - 1:
            ax.set_xlabel("Update (one training case = one update)")
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], color=COLORS["baseline"], lw=2, label=names["baseline"]),
        Line2D([], [], color=COLORS["aug"], lw=2, label=names["aug"]),
        Line2D([], [], color=INK_2, lw=2, label="train CE, moving avg over 1 epoch (faint: each update)"),
        Line2D([], [], color=INK_2, lw=2, ls="--", marker="o", ms=4, label="validation loss (end of each epoch)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Train and validation loss per update", y=1.06, fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion(m_base, m_aug, names, path, n_splits):
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue_seq", ["#f0efec", "#9ec5f4", "#2a78d6", "#104281"])
    vmax = max(m_base["cm"].max(), m_aug["cm"].max())
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    labels = ["benign", "high-grade"]
    for ax, key, m in ((axes[0], "baseline", m_base), (axes[1], "aug", m_aug)):
        cm = m["cm"]
        ax.imshow(cm, cmap=cmap, vmin=0, vmax=vmax)
        ax.grid(False)
        for i in range(2):
            for j in range(2):
                row = cm[i].sum()
                pct = 100 * cm[i, j] / row if row else 0
                dark = cm[i, j] > 0.55 * vmax
                ax.text(j, i, f"{cm[i, j]}\n({pct:.0f}% of row)", ha="center", va="center",
                        color="#ffffff" if dark else INK, fontsize=11)
        ax.set_xticks([0, 1], labels)
        ax.set_yticks([0, 1], labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(f"{names[key]}\nacc {m['accuracy']:.3f} · benign recall {m['benign_recall']:.3f} · "
                     f"HG recall {m['high_grade_recall']:.3f}", fontsize=10)
    fig.suptitle(f"Confusion matrix, all {m_base['n']} test cases ({n_splits} split(s) pooled)", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_per_case(df, names, path):
    """
    One row per test case (all splits). x = predicted P(high-grade).
    Large gray ring = true label (0 or 1); blue/orange dots = the two models;
    thin line joins the two models' predictions to show how far each case moved.
    """
    d = df.sort_values(["true_label", "case_id"]).reset_index(drop=True)
    n = len(d)
    fig, ax = plt.subplots(figsize=(7.5, max(6, 0.19 * n + 1.5)))
    y = np.arange(n)
    ax.axvline(0.5, color=INK_2, lw=1, ls=":", zorder=1)
    ax.scatter(d["true_label"], y, s=110, facecolors="none", edgecolors="#b5b3ad", linewidths=1.5,
               zorder=2, label="true label")
    ax.hlines(y, d["p_hg_baseline"], d["p_hg_aug"], color="#b5b3ad", lw=1, zorder=2)
    ax.scatter(d["p_hg_baseline"], y, s=34, color=COLORS["baseline"], edgecolors=SURFACE, linewidths=1,
               zorder=4, label=names["baseline"])
    ax.scatter(d["p_hg_aug"], y, s=34, color=COLORS["aug"], edgecolors=SURFACE, linewidths=1,
               zorder=5, label=names["aug"])

    # Separate benign / high-grade blocks
    n_benign = int((d["true_label"] == 0).sum())
    if 0 < n_benign < n:
        ax.axhline(n_benign - 0.5, color=INK_2, lw=1)
    ax.set_yticks(y, [f"{c} (s{s:02d})" for c, s in zip(d["case_id"], d["split"])], fontsize=7)
    ax.set_ylim(n - 0.5, -0.5)
    ax.set_xlim(-0.08, 1.08)
    ax.set_xlabel("Predicted probability of high-grade")
    ax.grid(axis="y", visible=False)
    ax.text(0.25, -1.2, "← predicted benign", ha="center", color=INK_2, fontsize=9)
    ax.text(0.75, -1.2, "predicted high-grade →", ha="center", color=INK_2, fontsize=9)
    ax.set_title(f"Per-case predictions, all {n} test cases\n"
                 f"(top block: true benign · bottom block: true high-grade; s = split)", fontsize=11, pad=22)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.tight_layout(rect=(0, 0.035 * 17 / max(n, 17) + 0.01, 1, 1))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_runs", nargs="+", required=True)
    ap.add_argument("--aug_runs", nargs="+", required=True)
    ap.add_argument("--baseline_name", default="baseline")
    ap.add_argument("--aug_name", default="flip-augmented")
    ap.add_argument("--out_dir", default="comparison_baseline_vs_flip")
    args = ap.parse_args()

    if len(args.baseline_runs) != len(args.aug_runs):
        raise SystemExit("Give the same number of baseline and augmented runs (one per split, same order).")
    os.makedirs(args.out_dir, exist_ok=True)
    names = {"baseline": args.baseline_name, "aug": args.aug_name}

    pairs = []
    for k, (b, a) in enumerate(zip(args.baseline_runs, args.aug_runs), 1):
        base, aug = load_run(b), load_run(a)
        check_pairing(base, aug, k)
        pairs.append((base, aug))

    # ---- per-split table
    rows, case_rows = [], []
    for k, (base, aug) in enumerate(pairs, 1):
        for key, run in (("baseline", base), ("aug", aug)):
            p = run["preds"]
            m = cm_metrics(p["true_label"], p["predicted_label"])
            rows.append({
                "split": k, "model": names[key], "run_dir": run["dir"], "n_test": m["n"],
                "accuracy": m["accuracy"], "benign_recall": m["benign_recall"],
                "high_grade_recall": m["high_grade_recall"], "balanced_accuracy": m["balanced_accuracy"],
                "test_loss": run["metrics"].get("test_loss"),
                "TN": m["cm"][0, 0], "FP": m["cm"][0, 1], "FN": m["cm"][1, 0], "TP": m["cm"][1, 1],
                **memorization_stats(run),
            })
        pb = base["preds"].set_index("case_id")
        pa = aug["preds"].set_index("case_id")
        for cid in pb.index:
            case_rows.append({
                "split": k, "case_id": cid, "true_label": int(pb.loc[cid, "true_label"]),
                "p_hg_baseline": float(pb.loc[cid, "prob_class1"]),
                "p_hg_aug": float(pa.loc[cid, "prob_class1"]),
                "pred_baseline": int(pb.loc[cid, "predicted_label"]),
                "pred_aug": int(pa.loc[cid, "predicted_label"]),
            })
    per_split = pd.DataFrame(rows)
    per_case = pd.DataFrame(case_rows)
    per_case["correct_baseline"] = per_case["pred_baseline"] == per_case["true_label"]
    per_case["correct_aug"] = per_case["pred_aug"] == per_case["true_label"]
    per_split.to_csv(os.path.join(args.out_dir, "metrics_per_split.csv"), index=False)
    per_case.to_csv(os.path.join(args.out_dir, "per_case_predictions.csv"), index=False)

    dup = per_case["case_id"].duplicated().sum()

    # ---- pooled metrics
    m_base = cm_metrics(per_case["true_label"], per_case["pred_baseline"])
    m_aug = cm_metrics(per_case["true_label"], per_case["pred_aug"])
    b_only = int((per_case["correct_baseline"] & ~per_case["correct_aug"]).sum())
    a_only = int((~per_case["correct_baseline"] & per_case["correct_aug"]).sum())
    p_mc = mcnemar_exact(b_only, a_only)

    # ---- plots
    plot_loss_per_epoch(pairs, names, os.path.join(args.out_dir, "loss_per_epoch.png"))
    plot_loss_per_update(pairs, names, os.path.join(args.out_dir, "loss_per_update.png"))
    plot_confusion(m_base, m_aug, names, os.path.join(args.out_dir, "confusion_matrices.png"), len(pairs))
    plot_per_case(per_case, names, os.path.join(args.out_dir, "per_case_predictions.png"))

    # ---- summary
    L = []
    L.append(f"Baseline vs augmented - {len(pairs)} splits, {len(per_case)} test cases"
             + (f"  [WARN: {dup} case ids appear in more than one test set]" if dup else ""))
    L.append("")
    L.append("POOLED TEST METRICS (all test cases together)")
    for key, m in (("baseline", m_base), ("aug", m_aug)):
        L.append(f"  {names[key]:<16} acc={m['accuracy']:.3f}  benign recall={m['benign_recall']:.3f}  "
                 f"high-grade recall={m['high_grade_recall']:.3f}  balanced acc={m['balanced_accuracy']:.3f}  "
                 f"CM [[TN,FP],[FN,TP]]={m['cm'].tolist()}")
    L.append(f"  Paired case-level comparison: baseline-only correct={b_only}, augmented-only correct={a_only}, "
             f"exact McNemar p={p_mc:.3f}")
    L.append("  (With ~90 cases, a difference of 1-3 cases is within noise; p<0.05 needs a clear gap.)")
    L.append("")
    L.append("PER-SPLIT MEAN +/- SD")
    cols = ["accuracy", "benign_recall", "high_grade_recall", "test_loss"]
    for model, g in per_split.groupby("model", sort=False):
        L.append(f"  {model:<16} " + "  ".join(f"{c}={g[c].mean():.3f}±{g[c].std():.3f}" for c in cols))
    L.append("")
    L.append("MEMORIZATION (train CE = classification part of training loss)")
    mcols = ["train_ce_final", "min_train_ce", "train_ce_at_best", "val_loss_at_best", "gap_at_best",
             "gap_final", "val_rise_after_best", "best_epoch", "epochs_trained"]
    for model, g in per_split.groupby("model", sort=False):
        L.append(f"  {model:<16} " + "  ".join(f"{c}={g[c].mean():.3f}" for c in mcols))
        firsts = g["first_epoch_train_ce_below_0.05"].tolist()
        L.append(f"  {'':<16} first epoch with train CE < 0.05 per split: {firsts}")
    L.append("")
    L.append("How to read: memorization is reduced if the augmented model's train CE stays higher / falls")
    L.append("more slowly AND its gap (val loss - train CE) and val_rise_after_best are smaller.")
    L.append("Performance improved only if pooled accuracy / recalls are higher; check the McNemar p and")
    L.append("the per-split spread before calling a difference real.")
    text = "\n".join(L)
    print(text)
    with open(os.path.join(args.out_dir, "summary.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\n[DONE] outputs in {args.out_dir}")


if __name__ == "__main__":
    main()
