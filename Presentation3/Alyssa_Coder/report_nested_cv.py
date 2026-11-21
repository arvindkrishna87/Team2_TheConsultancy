#!/usr/bin/env python3
"""
report_nested_cv.py - the slide-32 table and plots from a finished nested CV.

Reads <runs_dir>/outer_{A..E}/{selected.json, trials.csv, refit/*, inner_histories/*} and writes to --out_dir:

  results_per_fold.csv        one row per outer test fold: selected configuration, refit epochs,
                              accuracy, benign recall, high-grade recall (slide 32 table)
  results_pooled.csv          pooled over all outer folds (every case tested once) + range across folds
  predictions_all_cases.csv   every case's P(high-grade), with its outer fold
  report.md                   the two tables in Markdown, plus edge warnings
  fold_<X>_curves.png         per refitted model: train loss per update and per epoch, with the
                              selected trial's 4 inner validation curves (5 sets of plots)
  fold_<X>_cases.png          P(high-grade) of the fold's test cases against their true label
  all_cases.png               the same for all cases, coloured by outer fold
  confusion_pooled.png        pooled confusion matrix
  search_history.png          f(x) per trial and best-so-far, one panel per outer fold
  selected_hparams.png        the configuration each outer fold selected, on each range's scale
"""
import argparse
import glob
import json
import math
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FOLD_NAMES = ["A", "B", "C", "D", "E"]
BLUE, ORANGE, GREY, DARK = "#1f5fa8", "#d9822b", "#9aa3ad", "#14213d"
FOLD_COLORS = ["#1f5fa8", "#d9822b", "#2a9d8f", "#8e5bb5", "#c0392b"]
HP_ORDER = ["learning_rate", "weight_decay", "entropy_lambda", "classifier_dropout", "projector_dropout"]


def metrics(df):
    y, yh = df.true_label.to_numpy(), df.predicted_label.to_numpy()
    tn = int(((y == 0) & (yh == 0)).sum()); fp = int(((y == 0) & (yh == 1)).sum())
    fn = int(((y == 1) & (yh == 0)).sum()); tp = int(((y == 1) & (yh == 1)).sum())
    return {"n": len(y), "accuracy": (tn + tp) / len(y),
            "benign_recall": tn / max(tn + fp, 1), "high_grade_recall": tp / max(tp + fn, 1),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def fmt_hp(hp):
    return (f"lr {hp['learning_rate']:.2g}, wd {hp['weight_decay']:.2g}, "
            f"λ_ent {hp['entropy_lambda']:.2g}, drop cls {hp['classifier_dropout']:.2f} / "
            f"proj {hp['projector_dropout']:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    folds = {}
    for f in FOLD_NAMES:
        d = os.path.join(args.runs_dir, f"outer_{f}")
        if os.path.isfile(os.path.join(d, "selected.json")):
            folds[f] = {"dir": d, "sel": json.load(open(os.path.join(d, "selected.json"))),
                        "pred": pd.read_csv(os.path.join(d, "refit", "predictions.csv")),
                        "ep": pd.read_csv(os.path.join(d, "refit", "loss_per_epoch.csv")),
                        "up": pd.read_csv(os.path.join(d, "refit", "loss_per_update.csv")),
                        "trials": pd.read_csv(os.path.join(d, "trials.csv"))}
        else:
            print(f"[skip] outer fold {f}: not finished ({d}/selected.json missing)")
    if not folds:
        raise SystemExit("No finished outer folds")

    # ---------------- tables ----------------
    rows = []
    for f, F in folds.items():
        s, m = F["sel"], metrics(F["pred"])
        rows.append({"outer_fold": f, "selected_trial": s["selected_trial"], "kind": s["selected_kind"],
                     **{k: s["hparams"][k] for k in HP_ORDER}, "inner_f": s["objective"],
                     "refit_epochs": s["refit_epochs"], **m, "edge_flags": "; ".join(s["edge_flags"])})
    per_fold = pd.DataFrame(rows)
    per_fold.to_csv(os.path.join(args.out_dir, "results_per_fold.csv"), index=False)

    preds = pd.concat([F["pred"] for F in folds.values()], ignore_index=True)
    preds.to_csv(os.path.join(args.out_dir, "predictions_all_cases.csv"), index=False)
    pooled = metrics(preds)
    pooled_row = {"folds": "+".join(folds), **pooled}
    for k in ["accuracy", "benign_recall", "high_grade_recall"]:
        pooled_row[f"{k}_min"] = per_fold[k].min()
        pooled_row[f"{k}_max"] = per_fold[k].max()
    pd.DataFrame([pooled_row]).to_csv(os.path.join(args.out_dir, "results_pooled.csv"), index=False)

    lines = ["# Nested cross-validation results", "",
             "| Outer test fold | Selected configuration | Refit epochs | Accuracy | Benign recall | High-grade recall |",
             "|---|---|---|---|---|---|"]
    for r, (f, F) in zip(rows, folds.items()):
        lines.append(f"| {f} (n={r['n']}) | {fmt_hp(F['sel']['hparams'])} | {r['refit_epochs']} | "
                     f"{r['accuracy']:.3f} | {r['benign_recall']:.3f} | {r['high_grade_recall']:.3f} |")
    lines.append(f"| **All {len(folds)}, pooled** (n={pooled['n']}) | Differs by fold | | "
                 f"**{pooled['accuracy']:.3f}** ({per_fold.accuracy.min():.3f}–{per_fold.accuracy.max():.3f}) | "
                 f"**{pooled['benign_recall']:.3f}** ({per_fold.benign_recall.min():.3f}–{per_fold.benign_recall.max():.3f}) | "
                 f"**{pooled['high_grade_recall']:.3f}** ({per_fold.high_grade_recall.min():.3f}–{per_fold.high_grade_recall.max():.3f}) |")
    lines += ["", f"Pooled confusion matrix: TN {pooled['tn']}, FP {pooled['fp']}, FN {pooled['fn']}, TP {pooled['tp']}",
              "Ranges in brackets are min–max across the outer folds.", ""]
    edges = [f"- Fold {r['outer_fold']}: {r['edge_flags']}" for r in rows if r["edge_flags"]]
    if edges:
        lines += ["## Selected values at a range edge (widen and rerun)", *edges, ""]
    with open(os.path.join(args.out_dir, "report.md"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))

    # ---------------- per-fold curves (5 sets) ----------------
    for f, F in folds.items():
        ep, up, s = F["ep"], F["up"], F["sel"]
        inner = sorted(glob.glob(os.path.join(F["dir"], "inner_histories", f"trial{s['selected_trial']:03d}_val*_epochs.csv")))
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
        ax = axes[0]
        upe = int(up.groupby("epoch").size().iloc[0])
        ax.plot(up["update"], up["train_ce"], color=BLUE, lw=0.5, alpha=0.3, label="refit train CE (1 case per update)")
        ax.plot(up["update"], up["train_ce"].rolling(upe, min_periods=1).mean(), color=BLUE, lw=1.8,
                label=f"refit train CE, rolling mean ({upe} updates = 1 epoch)")
        for i, path in enumerate(inner):
            h = pd.read_csv(path)
            ax.plot(h["updates_end"], h["val_balanced_ce"], color=ORANGE, lw=1, alpha=0.8, marker=".", ms=3,
                    label="inner validation (balanced CE), selected trial, at its own update count" if i == 0 else None)
        ax.set_xlabel("gradient update"); ax.set_ylabel("loss"); ax.set_title("Per gradient update")
        ax.legend(fontsize=7, frameon=False)
        ax = axes[1]
        ax.plot(ep["epoch"], ep["train_ce"], "-o", ms=3, color=BLUE, lw=1.8, label="refit train CE")
        for i, path in enumerate(inner):
            h = pd.read_csv(path)
            vf = os.path.basename(path).split("_val")[1][0]
            ax.plot(h["epoch"], h["train_ce"], ":", color=GREY, lw=1, label="inner train CE" if i == 0 else None)
            ax.plot(h["epoch"], h["val_balanced_ce"], "-", color=ORANGE, lw=1.2, alpha=0.85,
                    label="inner validation (balanced CE)" if i == 0 else None)
            ax.annotate(vf, (h["epoch"].iloc[-1], h["val_balanced_ce"].iloc[-1]), fontsize=7, color=ORANGE)
        ax.axvline(s["refit_epochs"], color=DARK, ls="--", lw=1, label=f"refit epochs = {s['refit_epochs']}")
        ax.set_xlabel("epoch"); ax.set_title("Per epoch"); ax.legend(fontsize=7, frameon=False)
        fig.suptitle(f"Outer fold {f}: refitted model on {'+'.join(s['search_folds'])}; "
                     f"{fmt_hp(s['hparams'])}", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"fold_{f}_curves.png"), dpi=170)
        plt.close(fig)

        p = F["pred"].sort_values(["true_label", "prob_class1"])
        fig, ax = plt.subplots(figsize=(max(6, 0.32 * len(p)), 3.8))
        x = np.arange(len(p))
        cols = np.where(p.true_label == 1, ORANGE, BLUE)
        wrong = ~p.correct.astype(bool).to_numpy()
        ax.scatter(x[~wrong], p.prob_class1[~wrong], c=cols[~wrong], s=34)
        ax.scatter(x[wrong], p.prob_class1[wrong], facecolors="none", edgecolors=cols[wrong], s=60, lw=1.6)
        ax.axhline(0.5, color=DARK, ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(p.case_id.astype(str), rotation=90, fontsize=7)
        ax.set_ylim(-0.03, 1.03); ax.set_ylabel("P(high-grade)"); ax.set_xlabel("case id")
        m = metrics(F["pred"])
        ax.set_title(f"Outer fold {f}: blue = true benign, orange = true high-grade, hollow = misclassified | "
                     f"acc {m['accuracy']:.3f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out_dir, f"fold_{f}_cases.png"), dpi=170)
        plt.close(fig)

    # ---------------- all cases ----------------
    p = preds.sort_values(["true_label", "case_id"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(max(10, 0.17 * len(p)), 4.2))
    x = np.arange(len(p))
    for i, f in enumerate(folds):
        m = (p.outer_fold == f).to_numpy()
        w = ~p.correct.astype(bool).to_numpy()
        ax.scatter(x[m & ~w], p.prob_class1[m & ~w], color=FOLD_COLORS[i], s=20, label=f"outer fold {f}")
        ax.scatter(x[m & w], p.prob_class1[m & w], facecolors="none", edgecolors=FOLD_COLORS[i], s=44, lw=1.5)
    nb = int((p.true_label == 0).sum())
    ax.axvline(nb - 0.5, color=GREY); ax.axhline(0.5, color=DARK, ls="--", lw=1)
    ax.text(nb / 2, 1.06, "true benign", ha="center", fontsize=9)
    ax.text(nb + (len(p) - nb) / 2, 1.06, "true high-grade", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(p.case_id.astype(int).astype(str), rotation=90, fontsize=6)
    ax.set_xlim(-1, len(p)); ax.set_ylim(-0.03, 1.12); ax.set_ylabel("P(high-grade)")
    ax.set_xlabel("case id (each case tested once, by a model tuned without its labels; hollow = misclassified)")
    ax.legend(fontsize=7, ncol=5, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "all_cases.png"), dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 4))
    cm = np.array([[pooled["tn"], pooled["fp"]], [pooled["fn"], pooled["tp"]]])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.sum())
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=16,
                    color="white" if cm[i, j] > cm.sum() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["benign", "high-grade"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["benign", "high-grade"])
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Pooled, n={pooled['n']}: acc {pooled['accuracy']:.3f}\nbenign rec {pooled['benign_recall']:.3f} | "
                 f"HG rec {pooled['high_grade_recall']:.3f}", fontsize=9)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "confusion_pooled.png"), dpi=200); plt.close(fig)

    # ---------------- search history + selected values ----------------
    fig, axes = plt.subplots(1, len(folds), figsize=(3.6 * len(folds), 3.3), sharey=True, squeeze=False)
    for ax, (f, F) in zip(axes[0], folds.items()):
        t = F["trials"][F["trials"].state == "COMPLETE"].sort_values("number")
        kind = t.get("user_attrs_kind", pd.Series(["tpe"] * len(t), index=t.index)).fillna("tpe")
        for k, c in [("sobol", GREY), ("tpe", BLUE), ("zero", ORANGE)]:
            m = kind == k
            ax.scatter(t.number[m], t.value[m], s=14, color=c, label=k)
        ax.plot(t.number, t.value.cummin(), color=DARK, lw=1.2, label="best so far")
        ax.set_title(f"outer fold {f}", fontsize=9); ax.set_xlabel("trial")
    axes[0][0].set_ylabel("f(x): mean inner balanced CE"); axes[0][0].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "search_history.png"), dpi=170); plt.close(fig)

    any_sel = next(iter(folds.values()))["sel"]
    space = any_sel["settings"].get("space") or {
        "learning_rate": [1e-5, 1e-3, "log"], "weight_decay": [1e-5, 1e3, "log"],
        "entropy_lambda": [1e-3, 1.0, "log"], "classifier_dropout": [0, 0.5, "linear"],
        "projector_dropout": [0, 0.5, "linear"]}
    fig, axes = plt.subplots(1, len(HP_ORDER), figsize=(3.2 * len(HP_ORDER), 3), squeeze=False)
    for ax, name in zip(axes[0], HP_ORDER):
        lo, hi, scale = space[name]
        for i, (f, F) in enumerate(folds.items()):
            v = F["sel"]["hparams"][name]
            if scale == "log" and v == 0:
                ax.scatter(i, lo, marker="v", color=FOLD_COLORS[i], s=60, clip_on=False)
                ax.annotate("0", (i, lo), textcoords="offset points", xytext=(0, 6), ha="center",
                            color=FOLD_COLORS[i], fontsize=10, fontweight="bold")
                continue
            ax.scatter(i, v, color=FOLD_COLORS[i], s=40)
        if scale == "log":
            ax.set_yscale("log")
        ax.set_ylim(lo, hi) if scale == "log" else ax.set_ylim(lo - 0.02, hi + 0.02)
        ax.set_xticks(range(len(folds))); ax.set_xticklabels(list(folds))
        ax.set_title(name, fontsize=9); ax.set_xlabel("outer fold")
    fig.suptitle("Selected configuration per outer fold (axis = full search range; ▼0 = zero selected)", fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(args.out_dir, "selected_hparams.png"), dpi=170); plt.close(fig)
    print(f"\nOutputs in {args.out_dir}")


if __name__ == "__main__":
    main()
