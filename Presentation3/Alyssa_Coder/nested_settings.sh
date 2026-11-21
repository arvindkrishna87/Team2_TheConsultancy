#!/bin/bash
# nested_settings.sh - edit these paths once; every nested-CV sbatch file reads them.
CODE_DIR="/home/npg2887/Alyssa_Presentation3_base_code"
PATCHES_DIR="/projects/e32998/patches_benign_split"
LABELS_CSV="/projects/e32998/STAT390_Krish/Code/Code4_reduce_runtime/extra_benign_case_grade_match.csv"
UNI2_EMB_DIR="/home/npg2887/patch_embeddings_UNI2"
WORK_DIR="/home/npg2887/nested_cv"                             # folds.json, cache, runs, report
FOLDS="${WORK_DIR}/folds.json"
CACHE="${WORK_DIR}/uni2_cache_93.pt"
RUNS_DIR="${WORK_DIR}/runs"
REPORT_DIR="${WORK_DIR}/report"
N_TRIALS=40          # per outer fold: 16 Sobol + 21 TPE + 3 zero checks

activate_env () {
    module purge
    module load python-anaconda3/2019.10
    eval "$(conda shell.bash hook)"
    conda activate /projects/e32998/env
    python3 -m pip install -q "optuna>=3.4" scikit-learn pandas matplotlib
    export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
    mkdir -p "${WORK_DIR}"
    cd "${CODE_DIR}"
}
