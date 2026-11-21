#!/usr/bin/env python3
"""
nested_cv.py - nested cross-validation for the hierarchical attention MIL model (UNI2 features).

Presentation 3, Coders objective (slides 25-33): for each outer test fold, choose the
hyperparameters WITHOUT that fold's labels, refit, and test it once.

  Outer loop  (one job per test fold A..E, run as a Slurm array):
    1 Hold out   the outer test fold; its cases are never loaded into the search.
    2 Search     N Optuna trials on the other 4 folds. Each trial trains 4 inner models
                 (train on 3 folds, validate on the 4th) and returns
                 f(x) = mean of the 4 inner models' best balanced validation loss.
    3 Select     x_hat = the trial with the lowest f(x); values at a range edge are flagged.
    4 Refit      one model on all 4 folds with x_hat, for the inner models' mean best epoch
                 count, replaying their median learning-rate schedule (no validation set).
    5 Test once  on the outer fold; every case's predicted probability is saved.

  Losses (slide 30):
    training   weighted CE, reduction="sum", weights from THIS training set's own class counts
               w_B = N/(2 n_B), w_H = N/(2 n_H), minus lambda_ent * (normalized patch-attention entropy)
    validation balanced CE = 1/2 (benign mean CE + high-grade mean CE), no weights, no entropy;
               used for model selection, the LR scheduler and early stopping.

  Search (slides 28, 29, 31):
    learning rate 1e-5..1e-3 (log), weight decay 1e-5..1e3 (log), entropy weight 1e-3..1 (log),
    classifier and projector dropout 0..0.5 (linear). Zero weight decay and zero entropy weight are
    tested separately: after the search, the best configuration is re-run with each set to 0, and with
    both set to 0 (3 extra trials inside the same budget).
    16 scrambled Sobol startup points, then TPE with good-set size n_good = max(6, ceil(0.15 n)).

  Speed: every UNI2 embedding is loaded ONCE into a single cache file (build it with --build_cache),
  then kept in memory (GPU if available) for all ~5 x (4N+1) trainings.

Usage:
  python nested_cv.py --build_cache --emb_dir DIR --cache FILE          # once
  python nested_cv.py --outer_fold A --folds folds.json --cache FILE --out_dir DIR   # per fold
"""
import argparse
import copy
import json
import math
import os
import random
import sys
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DATA_PATHS, MODEL_CONFIG, TRAINING_CONFIG
from data_utils import (build_case_dict, build_slice_to_class_map,
                        get_all_patch_files, group_patches_by_slice, load_labels)
from models import create_model

FOLD_NAMES = ["A", "B", "C", "D", "E"]
STAINS = list(MODEL_CONFIG["stains"])

# name: (low, high, scale)
DEFAULT_SPACE = {
    "learning_rate": (1e-5, 1e-3, "log"),
    "weight_decay": (1e-5, 1e3, "log"),
    "entropy_lambda": (1e-3, 1.0, "log"),
    "classifier_dropout": (0.0, 0.5, "linear"),
    "projector_dropout": (0.0, 0.5, "linear"),
}
ZERO_CHECKS = [["weight_decay"], ["entropy_lambda"], ["weight_decay", "entropy_lambda"]]


def log(*a):
    print(*a, flush=True)


# =============================================================================
# 1. Embedding cache: load every patch embedding once
# =============================================================================
def build_cache(args):
    labels = load_labels(args.labels_csv)
    files = get_all_patch_files(args.patches_dir)
    patches = group_patches_by_slice(files, args.patches_dir)
    s2c = build_slice_to_class_map(patches, labels)
    case_dict, label_map = build_case_dict(list(s2c.keys()), patches, s2c)
    have = {os.path.splitext(f)[0] for f in os.listdir(args.emb_dir) if f.endswith(".pt")}
    log(f"Cases: {len(case_dict)} | embeddings in {args.emb_dir}: {len(have)}")

    cache = {"emb_dir": args.emb_dir, "patches_dir": args.patches_dir,
             "labels_csv": args.labels_csv, "cases": {}}
    n_loaded = n_missing = n_empty_slices = 0
    missing_examples, dims = [], Counter()
    t0 = time.time()
    for i, cid in enumerate(sorted(case_dict), 1):
        stains = {}
        for stain in STAINS:
            slices = []
            for paths in case_dict[cid].get(stain, []):
                keep = []
                for p in paths:
                    stem = os.path.splitext(os.path.basename(p))[0]
                    if stem in have:
                        keep.append(p)
                    else:
                        n_missing += 1
                        if len(missing_examples) < 10:
                            missing_examples.append(os.path.basename(p))
                if not keep:
                    n_empty_slices += 1
                    continue
                vecs = []
                for p in keep:
                    v = torch.load(os.path.join(args.emb_dir, os.path.splitext(os.path.basename(p))[0] + ".pt"),
                                   map_location="cpu")
                    vecs.append(v.reshape(-1).to(torch.float16))
                x = torch.stack(vecs)
                dims[x.shape[1]] += 1
                slices.append({"x": x, "files": [os.path.basename(p) for p in keep]})
                n_loaded += len(keep)
            stains[stain] = slices
        if not any(stains.values()):
            raise SystemExit(f"Case {cid} has no embedded patches at all")
        cache["cases"][int(cid)] = {"label": int(label_map[cid]), "stains": stains}
        if i % 10 == 0 or i == len(case_dict):
            log(f"  {i}/{len(case_dict)} cases, {n_loaded} embeddings loaded ({time.time() - t0:.0f}s)")

    if len(dims) != 1:
        raise SystemExit(f"Embeddings have different lengths: {dict(dims)}")
    cache["dim"] = next(iter(dims))
    cache["stats"] = {"embeddings_loaded": n_loaded, "patches_without_embedding": n_missing,
                      "slices_without_any_embedding": n_empty_slices,
                      "missing_examples": missing_examples}
    os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
    torch.save(cache, args.cache)
    labs = Counter(c["label"] for c in cache["cases"].values())
    log(f"Saved cache {args.cache}: {len(cache['cases'])} cases (benign {labs[0]}, high-grade {labs[1]}), "
        f"{n_loaded} embeddings of length {cache['dim']}, {n_missing} patches without an embedding "
        f"(expected: patches < 32 px), {n_empty_slices} slices dropped as empty")
    if missing_examples:
        log(f"  missing examples: {missing_examples}")


# =============================================================================
# 2. Model inputs from the cache
# =============================================================================
class CaseStore:
    def __init__(self, cache, device, per_slice_cap, max_slices_per_stain):
        self.cases = cache["cases"]
        self.dim = cache["dim"]
        self.cap = per_slice_cap
        self.max_slices = max_slices_per_stain
        for c in self.cases.values():          # move all embeddings to the device once
            for slices in c["stains"].values():
                for s in slices:
                    s["x"] = s["x"].to(device)

    def label(self, cid):
        return self.cases[cid]["label"]

    def inputs(self, cid, train, gen=None):
        """Same rules as dataset.py: first max_slices slices per stain; training shuffles the
        patches of each slice before capping, evaluation keeps the sorted order."""
        out = {}
        for stain in STAINS:
            slices = self.cases[cid]["stains"].get(stain, [])
            if self.max_slices:
                slices = slices[: self.max_slices]
            tensors = []
            for s in slices:
                x = s["x"]
                n = x.shape[0]
                if train:
                    idx = torch.randperm(n, generator=gen)
                    if self.cap and n > self.cap:
                        idx = idx[: self.cap]
                    x = x[idx.to(x.device)]
                elif self.cap and n > self.cap:
                    x = x[: self.cap]
                tensors.append(x.float())
            out[stain] = tensors
        return out


def make_model(hp, dim, device):
    model = create_model(num_classes=MODEL_CONFIG["num_classes"], embed_dim=MODEL_CONFIG["embed_dim"],
                         dropout=0.0, pooled_dim=dim)
    # models.py has one dropout argument; the search tunes the two dropout layers separately
    assert isinstance(model.patch_projector[2], nn.Dropout) and isinstance(model.classifier[0], nn.Dropout)
    model.patch_projector[2].p = float(hp["projector_dropout"])
    model.classifier[0].p = float(hp["classifier_dropout"])
    return model.to(device)


def patch_entropy(attn, eps=1e-8):
    """Mean normalized entropy of the patch-attention vectors (as in trainer.py, normalize=True)."""
    ents = []
    for sd in attn.get("stain_weights", {}).values():
        for w in sd.get("patch_weights", []):
            if w is None or w.numel() == 0:
                continue
            p = w.clamp_min(eps)
            e = -(p * torch.log(p)).sum()
            if p.numel() > 1:
                e = e / math.log(p.numel())
            ents.append(e)
    if not ents:
        return None
    return torch.stack(ents).mean()


@torch.inference_mode()
def evaluate(model, store, cases):
    """Balanced CE (slide 30) + per-case probabilities."""
    model.eval()
    rows, ce = [], {0: [], 1: []}
    for cid in cases:
        y = store.label(cid)
        logits = model(store.inputs(cid, train=False)).view(1, -1)
        loss = F.cross_entropy(logits, torch.tensor([y], device=logits.device)).item()
        p = torch.softmax(logits, dim=1)[0]
        ce[y].append(loss)
        rows.append({"case_id": cid, "true_label": y, "predicted_label": int(p.argmax().item()),
                     "prob_class0": float(p[0]), "prob_class1": float(p[1]), "ce": loss})
    means = [np.mean(v) for v in ce.values() if v]
    return float(np.mean(means)), rows


# =============================================================================
# 3. Train one model
# =============================================================================
def train_model(store, train_cases, val_cases, hp, args, device, seed,
                max_epochs, lr_schedule=None, log_updates=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    model = make_model(hp, store.dim, device)
    opt = torch.optim.AdamW(model.parameters(), lr=hp["learning_rate"], weight_decay=hp["weight_decay"])

    ys = [store.label(c) for c in train_cases]
    n, n_b, n_h = len(ys), ys.count(0), ys.count(1)
    if n_b == 0 or n_h == 0:
        raise ValueError(f"Training set has only one class (benign {n_b}, high-grade {n_h})")
    weights = torch.tensor([n / (2 * n_b), n / (2 * n_h)], dtype=torch.float32, device=device)
    crit = nn.CrossEntropyLoss(weight=weights, reduction="sum")   # sum: the weight is not divided out
    lam = float(hp["entropy_lambda"])

    sched = None
    if val_cases and lr_schedule is None:
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=TRAINING_CONFIG["scheduler_factor"],
            patience=TRAINING_CONFIG["scheduler_patience"], min_lr=TRAINING_CONFIG["scheduler_min_lr"])

    hist_epochs, hist_updates = [], []
    best = {"val": float("inf"), "epoch": 0}
    es_best, es_bad = float("inf"), 0
    order_rng = np.random.RandomState(seed)
    step = 0
    for epoch in range(1, max_epochs + 1):
        if lr_schedule is not None:
            for g in opt.param_groups:
                g["lr"] = lr_schedule[min(epoch - 1, len(lr_schedule) - 1)]
        lr = opt.param_groups[0]["lr"]
        model.train()
        t0 = time.time()
        sums = {"loss": 0.0, "ce": 0.0}
        for cid in order_rng.permutation(train_cases):
            y = torch.tensor([store.label(int(cid))], device=device)
            inp = store.inputs(int(cid), train=True, gen=gen)
            if lam > 0:
                logits, attn = model(inp, return_attn_weights=True)
                ent = patch_entropy(attn)
            else:
                logits, ent = model(inp), None
            ce = crit(logits.view(1, -1), y)
            loss = ce - lam * ent if ent is not None else ce
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            step += 1
            sums["loss"] += loss.item(); sums["ce"] += ce.item()
            if log_updates:
                hist_updates.append({"update": step, "epoch": epoch, "case_id": int(cid),
                                     "label": int(y.item()), "train_loss": loss.item(),
                                     "train_ce": ce.item(),
                                     "patch_entropy": float(ent.item()) if ent is not None else 0.0})
        rec = {"epoch": epoch, "lr": lr, "train_loss": sums["loss"] / n, "train_ce": sums["ce"] / n,
               "updates_end": step, "seconds": time.time() - t0}
        if val_cases:
            val, _ = evaluate(model, store, val_cases)
            rec["val_balanced_ce"] = val
            if sched is not None:
                sched.step(val)
            if val < best["val"]:
                best = {"val": val, "epoch": epoch}
            if val < es_best - TRAINING_CONFIG["early_stopping_min_delta"]:
                es_best, es_bad = val, 0
            else:
                es_bad += 1
        hist_epochs.append(rec)
        if (val_cases and epoch >= TRAINING_CONFIG["early_stopping_min_epochs"]
                and es_bad >= TRAINING_CONFIG["early_stopping_patience"]):
            break
    return {"model": model, "best_val": best["val"], "best_epoch": best["epoch"],
            "epochs": hist_epochs, "updates": hist_updates, "class_weights": weights.tolist()}


# =============================================================================
# 4. The search inside one outer fold
# =============================================================================
def sobol_points(space, n, seed):
    from torch.quasirandom import SobolEngine
    names = list(space)
    pts = SobolEngine(dimension=len(names), scramble=True, seed=seed).draw(n).numpy()
    out = []
    for row in pts:
        d = {}
        for u, name in zip(row, names):
            lo, hi, scale = space[name]
            d[name] = float(10 ** (math.log10(lo) + u * (math.log10(hi) - math.log10(lo)))) if scale == "log" \
                else float(lo + u * (hi - lo))
        out.append(d)
    return out


def good_set_size(n):
    """Slide 31: n_good = max(6, 0.15 n) (Optuna's default is 10%, at most 25)."""
    return min(n, max(6, math.ceil(0.15 * n)))


def edge_flags(hp, space, tol=0.05):
    flags = []
    for name, (lo, hi, scale) in space.items():
        v = hp[name]
        if v == 0:
            continue
        if scale == "log":
            pos = (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo))
        else:
            pos = (v - lo) / (hi - lo)
        if pos <= tol:
            flags.append(f"{name} at lower edge")
        elif pos >= 1 - tol:
            flags.append(f"{name} at upper edge")
    return flags


def run_outer_fold(args):
    import optuna
    from optuna.trial import TrialState

    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    if device == "cpu":
        torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", os.environ.get("SLURM_NTASKS", os.cpu_count() or 1))))
    space = dict(DEFAULT_SPACE)
    if args.space_json:
        space.update({k: tuple(v) for k, v in json.load(open(args.space_json)).items()})

    folds = json.load(open(args.folds))
    test_name = args.outer_fold
    inner_names = [f for f in FOLD_NAMES if f != test_name]
    out = os.path.join(args.out_dir, f"outer_{test_name}")
    os.makedirs(os.path.join(out, "inner_histories"), exist_ok=True)

    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    known = set(cache["cases"])
    fold_cases = {f: [int(c) for c in folds[f] if int(c) in known] for f in FOLD_NAMES}
    dropped = {f: [c for c in folds[f] if int(c) not in known] for f in FOLD_NAMES}
    if any(dropped.values()):
        log(f"[WARN] fold cases not in the cache (no patches/label): {dropped}")
    test_cases = fold_cases[test_name]
    search_cases = [c for f in inner_names for c in fold_cases[f]]
    assert not set(test_cases) & set(search_cases), "outer test cases leak into the search"

    # keep the outer test fold out of memory until the final evaluation
    test_cache = {c: cache["cases"].pop(c) for c in test_cases}
    store = CaseStore(cache, device, args.per_slice_cap, args.max_slices_per_stain)

    log("=" * 78)
    log(f"NESTED CV - outer test fold {test_name} ({len(test_cases)} cases), search on "
        f"{'+'.join(inner_names)} ({len(search_cases)} cases) | device {device}")
    for f in FOLD_NAMES:
        ys = [cache['cases'][c]['label'] if c in cache['cases'] else test_cache[c]['label'] for c in fold_cases[f]]
        log(f"  fold {f}: {len(ys)} cases (benign {ys.count(0)}, high-grade {ys.count(1)})"
            + ("  <- outer test" if f == test_name else ""))
    log(f"Trials: {args.n_trials} ({args.n_startup} Sobol + TPE + {len(ZERO_CHECKS)} zero checks) | "
        f"models to train: 4 x {args.n_trials} + 1 = {4 * args.n_trials + 1}")
    log(f"Search space: {space}")
    log("=" * 78)

    def inner_cv(hp, tag, seed_base):
        losses, best_epochs, lr_hists = [], [], []
        for k, val_fold in enumerate(inner_names):
            tr = [c for f in inner_names if f != val_fold for c in fold_cases[f]]
            va = fold_cases[val_fold]
            r = train_model(store, tr, va, hp, args, device, seed=seed_base + k,
                            max_epochs=args.max_epochs, log_updates=args.save_inner_updates)
            losses.append(r["best_val"]); best_epochs.append(r["best_epoch"])
            lr_hists.append([e["lr"] for e in r["epochs"]])
            pd.DataFrame(r["epochs"]).to_csv(
                os.path.join(out, "inner_histories", f"{tag}_val{val_fold}_epochs.csv"), index=False)
            if args.save_inner_updates:
                pd.DataFrame(r["updates"]).to_csv(
                    os.path.join(out, "inner_histories", f"{tag}_val{val_fold}_updates.csv"), index=False)
            log(f"    inner val {val_fold}: best balanced CE {r['best_val']:.4f} at epoch {r['best_epoch']} "
                f"({len(r['epochs'])} epochs, weights {[round(w, 3) for w in r['class_weights']]})")
            del r
        return losses, best_epochs, lr_hists

    def objective(trial):
        hp = {}
        for name, (lo, hi, scale) in space.items():
            hp[name] = trial.suggest_float(name, lo, hi, log=(scale == "log"))
        for z in trial.user_attrs.get("zero_params", []):
            hp[z] = 0.0                          # zero tested separately (a log scale cannot reach 0)
        trial.set_user_attr("effective_hparams", hp)
        log(f"\nTrial {trial.number} {trial.user_attrs.get('kind', 'tpe')}: "
            + ", ".join(f"{k}={v:.3g}" for k, v in hp.items()))
        t0 = time.time()
        losses, best_epochs, lr_hists = inner_cv(hp, f"trial{trial.number:03d}", 1000 * trial.number + args.seed)
        f = float(np.mean(losses))
        trial.set_user_attr("inner_losses", losses)
        trial.set_user_attr("inner_best_epochs", best_epochs)
        trial.set_user_attr("inner_lr_histories", lr_hists)
        trial.set_user_attr("minutes", (time.time() - t0) / 60)
        log(f"  f_{test_name}(x) = {f:.4f}  ({(time.time() - t0) / 60:.1f} min)")
        return f

    optuna.logging.set_verbosity(optuna.logging.WARNING)   # our own log prints the effective values
    sampler = optuna.samplers.TPESampler(n_startup_trials=args.n_startup, gamma=good_set_size, seed=args.seed)
    storage = f"sqlite:///{os.path.join(out, 'optuna.db')}"
    study = optuna.create_study(study_name=f"nested_outer_{test_name}", storage=storage,
                                sampler=sampler, direction="minimize", load_if_exists=True)
    # a job that hit its time limit leaves RUNNING trials behind: mark them failed and carry on
    for t in study.get_trials(deepcopy=False, states=(TrialState.RUNNING,)):
        study._storage.set_trial_user_attr(t._trial_id, "interrupted", True)
        study._storage.set_trial_state_values(t._trial_id, state=TrialState.FAIL)
        log(f"[resume] trial {t.number} was interrupted; marked FAIL")
    if len(study.trials) == 0:
        for params in sobol_points(space, args.n_startup, args.seed):
            study.enqueue_trial(params, user_attrs={"kind": "sobol"})
        log(f"Enqueued {args.n_startup} scrambled Sobol startup points")

    def n_done(kind=None):
        return sum(1 for t in study.trials if t.state in (TrialState.COMPLETE, TrialState.FAIL)
                   and not t.user_attrs.get("interrupted")      # a time-limit kill does not use up budget
                   and (kind is None or (t.user_attrs.get("kind") == "zero") == (kind == "zero")))

    search_budget = args.n_trials - len(ZERO_CHECKS)
    while n_done("search") < search_budget:
        study.optimize(objective, n_trials=1, catch=(RuntimeError, ValueError))

    if n_done("zero") == 0:
        best = study.best_trial
        for zp in ZERO_CHECKS:
            study.enqueue_trial(best.params, user_attrs={"kind": "zero", "zero_params": zp})
        log(f"\nZero checks around trial {best.number}: {ZERO_CHECKS}")
    while n_done("zero") < len(ZERO_CHECKS):
        study.optimize(objective, n_trials=1, catch=(RuntimeError, ValueError))

    # ---- select ----
    done = [t for t in study.trials if t.state == TrialState.COMPLETE]
    sel = min(done, key=lambda t: t.value)
    hp = sel.user_attrs["effective_hparams"]
    refit_epochs = max(1, int(round(np.mean(sel.user_attrs["inner_best_epochs"]))))
    lr_hists = sel.user_attrs["inner_lr_histories"]
    lr_schedule = [float(np.median([h[min(e, len(h) - 1)] for h in lr_hists])) for e in range(refit_epochs)]
    flags = edge_flags(hp, space)
    trials_df = study.trials_dataframe(attrs=("number", "value", "state", "params", "user_attrs"))
    trials_df.to_csv(os.path.join(out, "trials.csv"), index=False)
    log("\n" + "=" * 78)
    log(f"SELECTED trial {sel.number} ({sel.user_attrs.get('kind', 'tpe')}): f = {sel.value:.4f}")
    log("  " + ", ".join(f"{k}={v:.4g}" for k, v in hp.items()))
    log(f"  refit epochs = round(mean{sel.user_attrs['inner_best_epochs']}) = {refit_epochs}")
    for fl in flags:
        log(f"  [EDGE] {fl}: widen this range (--space_json) and rerun this outer fold")

    # ---- refit on all 4 folds, then test once ----
    t0 = time.time()
    r = train_model(store, search_cases, None, hp, args, device, seed=args.seed + 777,
                    max_epochs=refit_epochs, lr_schedule=lr_schedule, log_updates=True)
    refit_dir = os.path.join(out, "refit")
    os.makedirs(refit_dir, exist_ok=True)
    pd.DataFrame(r["epochs"]).to_csv(os.path.join(refit_dir, "loss_per_epoch.csv"), index=False)
    pd.DataFrame(r["updates"]).to_csv(os.path.join(refit_dir, "loss_per_update.csv"), index=False)
    torch.save(r["model"].state_dict(), os.path.join(refit_dir, "model.pt"))

    # the outer test fold enters memory only now
    for c in test_cases:
        for slices in test_cache[c]["stains"].values():
            for s in slices:
                s["x"] = s["x"].to(device)
        store.cases[c] = test_cache[c]
    test_loss, rows = evaluate(r["model"], store, test_cases)
    pred = pd.DataFrame(rows)
    pred["correct"] = pred.true_label == pred.predicted_label
    pred["outer_fold"] = test_name
    pred.to_csv(os.path.join(refit_dir, "predictions.csv"), index=False)
    y, yh = pred.true_label.to_numpy(), pred.predicted_label.to_numpy()
    acc = float((y == yh).mean())
    rb = float(((y == 0) & (yh == 0)).sum() / max((y == 0).sum(), 1))
    rh = float(((y == 1) & (yh == 1)).sum() / max((y == 1).sum(), 1))

    summary = {"outer_fold": test_name, "test_cases": test_cases, "search_folds": inner_names,
               "n_trials_complete": len(done), "selected_trial": sel.number,
               "selected_kind": sel.user_attrs.get("kind", "tpe"), "objective": sel.value,
               "inner_losses": sel.user_attrs["inner_losses"],
               "inner_best_epochs": sel.user_attrs["inner_best_epochs"],
               "hparams": hp, "refit_epochs": refit_epochs, "refit_lr_schedule": lr_schedule,
               "refit_class_weights": r["class_weights"], "edge_flags": flags,
               "test_balanced_ce": test_loss, "accuracy": acc, "benign_recall": rb,
               "high_grade_recall": rh, "n_test": len(test_cases),
               "refit_minutes": (time.time() - t0) / 60,
               "settings": {**{k: v for k, v in vars(args).items()}, "space": {k: list(v) for k, v in space.items()}}}
    with open(os.path.join(out, "selected.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    log(f"\nOUTER FOLD {test_name}: accuracy {acc:.3f} | benign recall {rb:.3f} | "
        f"high-grade recall {rh:.3f} | balanced CE {test_loss:.4f} | n={len(test_cases)}")
    log(f"Saved {out}")


def parse_args():
    ap = argparse.ArgumentParser(description="Nested CV for the MIL model")
    ap.add_argument("--build_cache", action="store_true", help="Only build the embedding cache and exit")
    ap.add_argument("--labels_csv", default=DATA_PATHS["labels_csv"])
    ap.add_argument("--patches_dir", default=DATA_PATHS["patches_dir"])
    ap.add_argument("--emb_dir", default=None, help="UNI2 embeddings folder (with --build_cache)")
    ap.add_argument("--cache", required=True, help="Embedding cache file (.pt)")
    ap.add_argument("--folds", default="folds.json")
    ap.add_argument("--outer_fold", choices=FOLD_NAMES,
                    default=FOLD_NAMES[int(os.environ["SLURM_ARRAY_TASK_ID"])] if "SLURM_ARRAY_TASK_ID" in os.environ else None)
    ap.add_argument("--out_dir", default="nested_cv_runs")
    ap.add_argument("--n_trials", type=int, default=40, help="Total trials per outer fold, incl. 3 zero checks")
    ap.add_argument("--n_startup", type=int, default=16, help="Sobol startup trials")
    ap.add_argument("--max_epochs", type=int, default=TRAINING_CONFIG["epochs"])
    ap.add_argument("--per_slice_cap", type=int, default=500)
    ap.add_argument("--max_slices_per_stain", type=int, default=5)
    ap.add_argument("--space_json", default=None, help='Override ranges, e.g. {"learning_rate": [1e-6, 1e-3, "log"]}')
    ap.add_argument("--save_inner_updates", action="store_true", help="Also save per-update losses of every inner model")
    ap.add_argument("--seed", type=int, default=TRAINING_CONFIG["random_state"])
    ap.add_argument("--cpu", action="store_true")
    return ap.parse_args()


if __name__ == "__main__":
    a = parse_args()
    if a.build_cache:
        if not a.emb_dir:
            sys.exit("--build_cache needs --emb_dir")
        build_cache(a)
    else:
        if a.outer_fold is None:
            sys.exit("Give --outer_fold (A-E) or run as a Slurm array 0-4")
        run_outer_fold(a)
