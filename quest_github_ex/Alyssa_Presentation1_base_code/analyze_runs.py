#!/usr/bin/env python3
"""analyze_runs.py -- combine the 5 split runs into the Step-3 results."""
import argparse, glob, os, re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score

CLASS_NAMES = {0: "Benign", 1: "High-grade"}


def parse_log(log_path):
    txt = open(log_path, errors="replace").read()
    m = re.search(r"Results saved to:\s*(\S+)", txt) or re.search(r"Created run directory:\s*(\S+)", txt)
    run_dir = m.group(1) if m else None
    m = re.search(r"Total execution time:\s*([\d.]+)s", txt)
    main_time = float(m.group(1)) if m else np.nan
    early = re.search(r"Early stopping triggered at epoch (\d+)", txt)
    return run_dir, main_time, (int(early.group(1)) if early else None)


def load_history(run_dir):
    ckpts = glob.glob(os.path.join(run_dir, "checkpoints", "*_epoch*.pth"))
    if not ckpts:
        return None
    last = max(ckpts, key=lambda p: int(re.search(r"_epoch(\d+)\.pth$", p).group(1)))
    try:
        ck = torch.load(last, map_location="cpu", weights_only=False)
    except TypeError:
        ck = torch.load(last, map_location="cpu")
    return {k: list(ck.get(src, [])) for k, src in
            [("train_loss", "train_losses"), ("train_ce", "train_ce_losses"),
             ("val_loss", "val_losses"), ("val_acc", "val_accuracies"), ("lr", "learning_rates")]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--seed", type=str, default="?")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    metric_rows, prob_frames, hist_rows, runtime_rows = [], [], [], []
    cms, hists = {}, {}

    for i in range(1, args.n_splits + 1):
        tag = f"{i:02d}"
        log = os.path.join(args.logs_dir, f"split_{tag}.log")
        if not os.path.exists(log):
            print(f"[WARN] missing {log}; skipping split {tag}")
            continue
        run_dir, main_time, early_epoch = parse_log(log)
        wall = np.nan
        rt = os.path.join(args.logs_dir, f"runtime_{tag}.txt")
        if os.path.exists(rt):
            wall = float(open(rt).read().strip() or "nan")

        pred_csv = os.path.join(run_dir, "predictions.csv") if run_dir else None
        if not pred_csv or not os.path.exists(pred_csv):
            print(f"[WARN] no predictions.csv for split {tag} (run_dir={run_dir})")
            continue
        df = pd.read_csv(pred_csv)
        y, yhat = df["true_label"].astype(int), df["predicted_label"].astype(int)
        cm = confusion_matrix(y, yhat, labels=[0, 1])
        cms[tag] = cm
        metric_rows.append({
            "split": tag, "n_test_cases": len(df),
            "n_benign": int((y == 0).sum()), "n_high_grade": int((y == 1).sum()),
            "accuracy": accuracy_score(y, yhat),
            "recall_benign": recall_score(y, yhat, pos_label=0, zero_division=0),
            "recall_high_grade": recall_score(y, yhat, pos_label=1, zero_division=0),
            "TN_benign_as_benign": int(cm[0, 0]), "FP_benign_as_high": int(cm[0, 1]),
            "FN_high_as_benign": int(cm[1, 0]), "TP_high_as_high": int(cm[1, 1]),
            "runtime_sec": wall if not np.isnan(wall) else main_time,
        })

        p = df.rename(columns={"prob_class0": "prob_benign", "prob_class1": "prob_high_grade"}).copy()
        p.insert(0, "split", tag)
        prob_frames.append(p)

        h = load_history(run_dir)
        n_ep = len(h["val_loss"]) if h else np.nan
        best_ep = int(np.argmin(h["val_loss"]) + 1) if h and h["val_loss"] else np.nan
        if h:
            hists[tag] = h
            for e in range(len(h["val_loss"])):
                g = lambda k: h[k][e] if e < len(h[k]) else np.nan
                hist_rows.append({"split": tag, "epoch": e + 1, "train_loss_total": g("train_loss"),
                                  "train_ce": g("train_ce"), "val_loss": g("val_loss"),
                                  "val_acc": g("val_acc"), "lr": g("lr")})
        runtime_rows.append({
            "split": tag, "wall_clock_sec": wall, "main_py_reported_sec": main_time,
            "wall_clock_min": wall / 60 if not np.isnan(wall) else np.nan,
            "epochs_run": n_ep, "best_val_epoch": best_ep, "early_stopped_at": early_epoch,
            "sec_per_epoch": (main_time / n_ep) if n_ep and not np.isnan(main_time) else np.nan,
            "run_dir": run_dir,
        })

    if not metric_rows:
        raise SystemExit("No completed splits found.")

    metrics = pd.DataFrame(metric_rows)
    mean = metrics.select_dtypes("number").mean()
    std = metrics.select_dtypes("number").std()
    pd.concat([metrics, pd.DataFrame([{"split": "mean", **mean.to_dict()},
                                      {"split": "std", **std.to_dict()}])], ignore_index=True) \
        .to_csv(os.path.join(args.out_dir, "metrics_by_split.csv"), index=False)
    probs = pd.concat(prob_frames, ignore_index=True)
    probs.to_csv(os.path.join(args.out_dir, "test_probabilities.csv"), index=False)
    pd.DataFrame(hist_rows).to_csv(os.path.join(args.out_dir, "loss_history.csv"), index=False)
    runtime = pd.DataFrame(runtime_rows)
    runtime.to_csv(os.path.join(args.out_dir, "runtime_by_split.csv"), index=False)

    # confusion matrices
    tags = sorted(cms)
    fig, axes = plt.subplots(1, len(tags), figsize=(3.3 * len(tags), 3.4))
    axes = np.atleast_1d(axes)
    for ax, tag in zip(axes, tags):
        cm = cms[tag]
        ax.imshow(cm, cmap="Blues", vmin=0)
        for r in range(2):
            for c in range(2):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                        color="white" if cm[r, c] > cm.max() / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Benign", "High"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Benign", "High"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("True" if tag == tags[0] else "")
        acc = metrics.loc[metrics.split == tag, "accuracy"].item()
        ax.set_title(f"Split {tag}  (acc {acc:.2f})", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out_dir, "confusion_matrices.png"), dpi=200)
    plt.close(fig)

    # loss curves
    if hists:
        tags = sorted(hists)
        fig, axes = plt.subplots(1, len(tags), figsize=(3.6 * len(tags), 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, tag in zip(axes, tags):
            h = hists[tag]
            ep = np.arange(1, len(h["val_loss"]) + 1)
            ax.plot(ep, h["train_loss"][:len(ep)], label="Train", color="#2a6fdb", lw=2)
            ax.plot(ep, h["val_loss"], label="Validation", color="#e07a1f", lw=2)
            b = int(np.argmin(h["val_loss"])) + 1
            ax.axvline(b, color="gray", ls="--", lw=1)
            ax.text(b, max(h["val_loss"]), f" best={b}", fontsize=8, color="gray", va="top")
            ax.set_title(f"Split {tag}", fontsize=11); ax.set_xlabel("Epoch")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.grid(alpha=0.25)
        axes[0].set_ylabel("Loss"); axes[0].legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, "loss_curves.png"), dpi=200)
        plt.close(fig)

    # one-page summary
    L = [f"# 5-split results (seed {args.seed})", "",
         "| Split | n | Accuracy | Recall benign | Recall high-grade | CM [[TN FP],[FN TP]] | Runtime (min) |",
         "|---|---|---|---|---|---|---|"]
    for r in metric_rows:
        L.append(f"| {r['split']} | {r['n_test_cases']} | {r['accuracy']:.3f} | {r['recall_benign']:.3f} | "
                 f"{r['recall_high_grade']:.3f} | [[{r['TN_benign_as_benign']} {r['FP_benign_as_high']}],"
                 f"[{r['FN_high_as_benign']} {r['TP_high_as_high']}]] | {r['runtime_sec']/60:.1f} |")
    L.append(f"| mean +/- sd | | {mean['accuracy']:.3f} +/- {std['accuracy']:.3f} | "
             f"{mean['recall_benign']:.3f} +/- {std['recall_benign']:.3f} | "
             f"{mean['recall_high_grade']:.3f} +/- {std['recall_high_grade']:.3f} | | "
             f"{mean['runtime_sec']/60:.1f} +/- {std['runtime_sec']/60:.1f} |")
    L += ["", "## Runtime", "", "```",
          runtime.drop(columns=["run_dir"]).to_string(index=False, float_format=lambda v: f"{v:.1f}"), "```",
          "", "## Predicted probability for each test case", ""]
    for tag, g in probs.groupby("split"):
        L += [f"### Split {tag}", "", "| Case | True | Pred | P(benign) | P(high-grade) | Correct |",
              "|---|---|---|---|---|---|"]
        for _, r in g.sort_values(["true_label", "case_id"]).iterrows():
            L.append(f"| {r['case_id']} | {CLASS_NAMES[int(r['true_label'])]} | "
                     f"{CLASS_NAMES[int(r['predicted_label'])]} | {r['prob_benign']:.3f} | "
                     f"{r['prob_high_grade']:.3f} | {'yes' if r['correct'] else 'NO'} |")
        L.append("")
    with open(os.path.join(args.out_dir, "results_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(open(os.path.join(args.out_dir, "results_summary.md"), encoding="utf-8").read().split("## Predicted")[0])
    print(f"All outputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
