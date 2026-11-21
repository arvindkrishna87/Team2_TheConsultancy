# Nested cross-validation (Presentation 3, Coders)

Question: what test accuracy do we get when each test fold's hyperparameters are chosen without its labels?

## Files (add to your copy of `Presentation3_base_code`; no base file is changed)

| File | What it does |
|---|---|
| `make_folds.py` | Outer folds A–E → `folds.json` (grouped cases kept together; checks disjoint + full coverage) |
| `nested_cv.py` | `--build_cache`: loads every UNI2 embedding once. Otherwise: search, select, refit, test for one outer fold |
| `report_nested_cv.py` | Slide-32 table, pooled result with ranges, 5 sets of plots |
| `nested_settings.sh` | All paths in one place: **edit this first** |
| `01_make_folds.sbatch` … `04_report.sbatch` | Run in order |

## Run

```bash
cd /home/<netid>/Presentation3_nested_cv      # the folder with the code and these files
mkdir -p logs
sbatch 01_make_folds.sbatch      # ~1 min   -> folds.json (check the printed table: 5 folds, disjoint)
sbatch 02_build_cache.sbatch     # ~10-30 min, once -> uni2_cache.pt (check: 99 cases, length 1536)
sbatch 03_nested_cv.sbatch       # 5 jobs in parallel, one per outer fold
sbatch 04_report.sbatch          # after all 5 finish -> report/
```
Submit from the code folder: the jobs read `nested_settings.sh` from there.

## What one outer-fold job does (slides 25–31)

1. **Hold out** the outer fold. Its cases are removed from memory before the search starts.
2. **Search**: 40 trials. Each trains 4 inner models (3 folds train, 1 validates) and returns
   f(x) = mean of their best balanced validation losses.
   - 16 scrambled Sobol startup points, then TPE with good set n_good = max(6, ⌈0.15 n⌉).
   - Ranges: lr 1e-5–1e-3 (log), weight decay 1e-5–1e3 (log), entropy weight 1e-3–1 (log),
     classifier and projector dropout 0–0.5 (linear).
   - **Zero tested separately**: the last 3 trials re-run the best configuration with weight decay = 0,
     entropy weight = 0, and both = 0.
3. **Select** the trial with the lowest f(x). A value within 5% of a range edge is flagged `[EDGE]`
   in the log and report: widen that range with `--space_json` and rerun that fold.
4. **Refit** on all 4 folds for round(mean of the 4 inner best epochs), replaying the inner models'
   median learning-rate schedule (the refit has no validation set to drive the scheduler).
5. **Test once** on the outer fold; predictions saved.

Losses (slide 30): training = weighted CE with `reduction="sum"`, weights N/(2 n_class) from that
training set's own counts, minus λ_ent × normalized patch-attention entropy. Validation = balanced CE
½(benign mean CE + high-grade mean CE), no weights, no entropy; it drives selection, the LR scheduler
and early stopping (patience 8, min 10 epochs, max 30).

Models trained: 5 × (4 × 40 + 1) = 805. Patches: first 5 slices per stain, 500 patches per slice
(same as earlier presentations).

## Outputs

```
runs/outer_A/ optuna.db  trials.csv  selected.json  inner_histories/  refit/{predictions.csv, loss_per_*.csv, model.pt}
report/       report.md  results_per_fold.csv  results_pooled.csv  predictions_all_cases.csv
              fold_<X>_curves.png  fold_<X>_cases.png  all_cases.png  confusion_pooled.png
              search_history.png  selected_hparams.png
```

## Checks before trusting it
- `logs/ncv_outer_*_0.out`: the first trial prints its minutes. Multiply by 40 to check it fits the 24 h limit
  (if not: resubmit — progress is kept — or lower `N_TRIALS`).
- Every log prints the fold sizes and "search on B+C+D+E" for outer fold A, etc.
- The cache log reports patches without an embedding: those should only be patches under 32 px.

## Slide footnotes (required on every slide)
Code: `Team2_TheConsultancy/<path>/nested_cv.py` (search/refit/test), `report_nested_cv.py` (tables, plots),
`make_folds.py` (folds).
