#!/usr/bin/env python3
"""
compare_encoders.py - KimiaNet vs UNI2 on the same 5 splits.

Input: a text file with one line per run:   <encoder> <label containing split number> <run folder>
    kimianet kimia_split1 /projects/e32998/MIL_training/final_runs/run_..._kimianet_split01
    uni2     uni2_split1  /projects/e32998/MIL_training/final_runs/run_..._uni2_split01
    ...
Each run folder must contain predictions.csv, results.txt, loss_per_epoch.csv, loss_per_update.csv
(written by main.py).

Outputs (in --out_dir):
    summary_per_split.csv       accuracy, benign / high-grade recall, AUC, best epoch per encoder per split
    summary_overall.csv         pooled over all test cases + mean/sd over splits
    per_case_predictions.csv    P(high-grade) per case, both encoders side by side
    confusion_matrices.png      pooled confusion matrix per encoder
    per_case_probabilities.png  P(high-grade) for every test case, both encoders
    loss_per_epoch.png          train CE + validation loss per epoch, one panel per split
    loss_per_update.png         train CE per gradient update (+ validation at epoch ends), one panel per split
    report.txt                  tables + paired (McNemar) test

Usage:
    python3 compare_encoders.py --runs_file ~/run_folders.txt --out_dir ~/uni2_vs_kimianet
"""
import argparse
import os
import re
from math import comb

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score

COLORS = ["#1f5fa8", "#d9822b", "#2a9d8f", "#8e5bb5"]
CLASS_NAMES = ["benign", "high-grade"]


def read_runs(path):
    runs = []
    for line in open(os.path.expanduser(path)):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        enc, label, folder = line.split()[:3]
        split = int(label.rsplit("split", 1)[1])
        need = ["predictions_best.csv", "loss_per_epoch.csv", "loss_per_update.csv"]
        missing = [f for f in need if not os.path.isfile(os.path.join(folder, f))]
        if missing:
            raise SystemExit(f"{folder} is missing {missing}")
        res = {}
        if os.path.isfile(os.path.join(folder, "results.txt")):
            for l in open(os.path.join(folder, "results.txt")):
                if ":" in l:
                    k, v = l.split(":", 1)
                    res[k.strip()] = v.strip()
        pred = pd.read_csv(os.path.join(folder, "predictions_best.csv"))
        pred["encoder"], pred["split"] = enc, split
        runs.append(dict(encoder=enc, split=split, folder=folder, results=res, pred=pred,
                         epochs=pd.read_csv(os.path.join(folder, "loss_per_epoch.csv")),
                         updates=pd.read_csv(os.path.join(folder, "loss_per_update.csv"))))
    seen = {}
    for r in runs:
        key = (r["encoder"], r["split"])
        if key in seen:
            raise SystemExit(f"Two runs for {key}: {seen[key]} and {r['folder']}")
        seen[key] = r["folder"]
    folders = [r["folder"] for r in runs]
    if len(set(folders)) != len(folders):
        raise SystemExit("The same run folder is listed twice (collided runs?) - fix run_folders.txt")
    return runs


def metrics(df):
    y = df["true_label"].to_numpy()
    yhat = df["predicted_label"].to_numpy()
    p = df["prob_class1"].to_numpy()
    cm = confusion_matrix(y, yhat, labels=[0, 1])
    rb = cm[0, 0] / cm[0].sum() if cm[0].sum() else np.nan
    rh = cm[1, 1] / cm[1].sum() if cm[1].sum() else np.nan
    auc = roc_auc_score(y, p) if len(set(y)) == 2 else np.nan
    return dict(n=len(y), accuracy=(y == yhat).mean(), benign_recall=rb, high_grade_recall=rh,
                balanced_accuracy=np.nanmean([rb, rh]), auc=auc,
                tn=int(cm[0, 0]), fp=int(cm[0, 1]), fn=int(cm[1, 0]), tp=int(cm[1, 1]))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_file", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    out = os.path.expanduser(args.out_dir)
    os.makedirs(out, exist_ok=True)

    runs = read_runs(args.runs_file)
    encoders = list(dict.fromkeys(r["encoder"] for r in runs))
    splits = sorted({r["split"] for r in runs})
    print(f"Encoders: {encoders} | splits: {splits} | runs: {len(runs)}")

    # ---------- per split ----------
    rows = []
    for r in runs:
        m = metrics(r["pred"])
        ep = r["epochs"]
        best = int(ep["val_loss"].idxmin()) + 1
        m.update(encoder=r["encoder"], split=r["split"], best_epoch=best,
                 epochs_trained=len(ep), best_val_loss=ep["val_loss"].min(),
                 test_loss=r["results"].get("Test Loss"), folder=r["folder"])
        rows.append(m)
    per_split = pd.DataFrame(rows).sort_values(["split", "encoder"])
    cols = ["encoder", "split", "n", "accuracy", "benign_recall", "high_grade_recall", "auc",
            "best_epoch", "epochs_trained", "best_val_loss", "test_loss", "tn", "fp", "fn", "tp", "folder"]
    per_split = per_split[cols]
    per_split.to_csv(os.path.join(out, "summary_per_split.csv"), index=False)

    # ---------- pooled ----------
    preds = pd.concat([r["pred"] for r in runs], ignore_index=True)
    overall = []
    for e in encoders:
        m = metrics(preds[preds.encoder == e])
        ps = per_split[per_split.encoder == e]
        for k in ["accuracy", "benign_recall", "high_grade_recall", "auc"]:
            m[k + "_mean_splits"] = ps[k].mean()
            m[k + "_sd_splits"] = ps[k].std(ddof=1)
        m["encoder"] = e
        overall.append(m)
    overall = pd.DataFrame(overall)
    overall = overall[["encoder"] + [c for c in overall.columns if c != "encoder"]]
    overall.to_csv(os.path.join(out, "summary_overall.csv"), index=False)

    # ---------- per case ----------
    wide = None
    for e in encoders:
        s = preds[preds.encoder == e][["case_id", "true_label", "split", "prob_class1", "predicted_label", "correct"]]
        s = s.rename(columns={"split": f"split_{e}", "prob_class1": f"p_hg_{e}",
                              "predicted_label": f"pred_{e}", "correct": f"correct_{e}"})
        wide = s if wide is None else wide.merge(s, on=["case_id", "true_label"], how="outer")
    wide = wide.sort_values(["true_label", "case_id"]).reset_index(drop=True)
    wide.to_csv(os.path.join(out, "per_case_predictions.csv"), index=False)

    report = []
    if len(encoders) == 2:
        a, b = encoders
        both = wide.dropna(subset=[f"correct_{a}", f"correct_{b}"])
        ca = both[f"correct_{a}"].astype(bool)
        cb = both[f"correct_{b}"].astype(bool)
        oa, ob = int((ca & ~cb).sum()), int((~ca & cb).sum())
        report += [f"Paired comparison over {len(both)} test cases",
                   f"  correct by both: {int((ca & cb).sum())}, neither: {int((~ca & ~cb).sum())}",
                   f"  correct by {a} only: {oa}, by {b} only: {ob}",
                   f"  exact McNemar p-value (two-sided): {mcnemar_exact(oa, ob):.3f}"]
        diff = both[ca != cb][["case_id", "true_label", f"p_hg_{a}", f"p_hg_{b}"]]
        if len(diff):
            report += ["  cases where only one encoder is correct:", diff.to_string(index=False)]

    # ---------- plots: confusion matrices ----------
    fig, axes = plt.subplots(1, len(encoders), figsize=(4.3 * len(encoders), 4))
    axes = np.atleast_1d(axes)
    for ax, e in zip(axes, encoders):
        m = overall[overall.encoder == e].iloc[0]
        cm = np.array([[m.tn, m.fp], [m.fn, m.tp]])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=16,
                        color="white" if cm[i, j] > cm.sum() / 2 else "black")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(CLASS_NAMES)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_title(f"{e}: accuracy {m.accuracy:.3f}\nbenign recall {m.benign_recall:.3f} | "
                     f"high-grade recall {m.high_grade_recall:.3f}", fontsize=9)
    fig.suptitle(f"All test cases, pooled over {len(splits)} splits (n = {int(overall.n.iloc[0])})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrices.png"), dpi=200)
    plt.close(fig)

    # ---------- plot: per-case probabilities ----------
    x = np.arange(len(wide))
    fig, ax = plt.subplots(figsize=(max(10, 0.17 * len(wide)), 4.3))
    if len(encoders) == 2:
        ax.vlines(x, wide[f"p_hg_{encoders[0]}"], wide[f"p_hg_{encoders[1]}"], color="#c8c8c8", lw=1, zorder=1)
    for i, e in enumerate(encoders):
        col = wide[f"p_hg_{e}"]
        wrong = ~wide[f"correct_{e}"].fillna(True).astype(bool)
        mk = ["o", "D", "s", "^"][i]
        ax.scatter(x[~wrong], col[~wrong], s=20, marker=mk, color=COLORS[i], label=e, zorder=2)
        ax.scatter(x[wrong], col[wrong], s=40, marker=mk, facecolors="none", edgecolors=COLORS[i],
                   linewidths=1.5, label=f"{e} (misclassified)", zorder=3)
    ax.axhline(0.5, color="#555", ls="--", lw=1)
    nb = int((wide.true_label == 0).sum())
    ax.axvline(nb - 0.5, color="#999", lw=1)
    ax.text(nb / 2, 1.06, "true benign", ha="center", fontsize=9)
    ax.text(nb + (len(wide) - nb) / 2, 1.06, "true high-grade", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(wide.case_id.astype(int).astype(str), rotation=90, fontsize=6)
    ax.set_xlim(-1, len(wide))
    ax.set_ylim(-0.03, 1.12)
    ax.set_xlabel("case id (each case is tested once, in its split's test fold)")
    ax.set_ylabel("P(high-grade)")
    ax.legend(fontsize=7, ncol=4, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "per_case_probabilities.png"), dpi=200)
    plt.close(fig)

    # ---------- plots: loss curves ----------
    for kind in ["epoch", "update"]:
        fig, axes = plt.subplots(1, len(splits), figsize=(3.6 * len(splits), 3.5), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, s in zip(axes, splits):
            for i, e in enumerate(encoders):
                r = next((r for r in runs if r["encoder"] == e and r["split"] == s), None)
                if r is None:
                    continue
                ep, up = r["epochs"], r["updates"]
                best = int(ep["val_loss"].idxmin()) + 1
                if kind == "epoch":
                    ax.plot(ep["epoch"], ep["train_ce"], "--", color=COLORS[i], lw=1.2)
                    ax.plot(ep["epoch"], ep["val_loss"], "-", color=COLORS[i], lw=1.7, label=e)
                    ax.plot(best, ep["val_loss"].iloc[best - 1], "*", color=COLORS[i], ms=11)
                else:
                    upe = int(up.groupby("epoch").size().iloc[0])
                    ax.plot(up["update"], up["train_ce"], color=COLORS[i], lw=0.4, alpha=0.25)
                    ax.plot(up["update"], up["train_ce"].rolling(upe, min_periods=1).mean(), "--",
                            color=COLORS[i], lw=1.2)
                    ax.plot(ep["epoch"] * upe, ep["val_loss"], "-o", color=COLORS[i], lw=1.5, ms=2.5, label=e)
            ax.set_title(f"split {s}", fontsize=9)
            ax.set_xlabel("epoch" if kind == "epoch" else "gradient update (1 case each)")
        axes[0].set_ylabel("weighted cross-entropy")
        axes[0].plot([], [], "k--", lw=1, label="train CE" + (" (rolling 1-epoch mean)" if kind == "update" else ""))
        axes[0].plot([], [], "k-", lw=1, label="validation" + (" (end of each epoch)" if kind == "update" else ""))
        if kind == "epoch":
            axes[0].plot([], [], "k*", label="best epoch (evaluated)")
        axes[0].legend(fontsize=7, frameon=False)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"loss_per_{kind}.png"), dpi=200)
        plt.close(fig)

    # ---------- report ----------
    pd.set_option("display.width", 200)
    show = per_split.drop(columns=["folder", "tn", "fp", "fn", "tp"])
    text = ["PER SPLIT", show.to_string(index=False, float_format=lambda v: f"{v:.3f}"), "",
            "POOLED OVER ALL TEST CASES",
            overall[["encoder", "n", "accuracy", "benign_recall", "high_grade_recall", "auc",
                     "accuracy_mean_splits", "accuracy_sd_splits", "tn", "fp", "fn", "tp"]]
            .to_string(index=False, float_format=lambda v: f"{v:.3f}"), "",
            "Note: pooled AUC mixes 5 different models; per-split AUCs are cleaner.", ""] + report
    open(os.path.join(out, "report.txt"), "w").write("\n".join(text) + "\n")
    print("\n".join(text))
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()