# 5-split results (seed 1805)

| Split | n | Accuracy | Recall benign | Recall high-grade | CM [[TN FP],[FN TP]] | Runtime (min) |
|---|---|---|---|---|---|---|
| 01 | 18 | 0.778 | 0.857 | 0.727 | [[6 1],[3 8]] | 26.4 |
| 02 | 19 | 0.789 | 0.571 | 0.917 | [[4 3],[1 11]] | 26.0 |
| 03 | 19 | 0.895 | 0.857 | 0.917 | [[6 1],[1 11]] | 18.5 |
| 04 | 19 | 0.789 | 0.714 | 0.833 | [[5 2],[2 10]] | 26.4 |
| 05 | 18 | 0.944 | 1.000 | 0.909 | [[7 0],[1 10]] | 47.0 |
| mean +/- sd | | 0.839 +/- 0.076 | 0.800 +/- 0.163 | 0.861 +/- 0.082 | | 28.9 +/- 10.7 |

## Runtime

```
split  wall_clock_sec  main_py_reported_sec  wall_clock_min  epochs_run  best_val_epoch  early_stopped_at  sec_per_epoch
   01          1581.0                1537.2            26.4          16               8                16           96.1
   02          1562.0                1531.3            26.0          15               7                15          102.1
   03          1109.0                1072.6            18.5          19              11                19           56.5
   04          1585.0                1552.4            26.4          10               2                10          155.2
   05          2818.0                2781.3            47.0          23              15                23          120.9
```

## Predicted probability for each test case

### Split 01

| Case | True | Pred | P(benign) | P(high-grade) | Correct |
|---|---|---|---|---|---|
| 2 | Benign | Benign | 1.000 | 0.000 | yes |
| 24 | Benign | Benign | 1.000 | 0.000 | yes |
| 53 | Benign | Benign | 0.997 | 0.003 | yes |
| 78 | Benign | High-grade | 0.264 | 0.736 | NO |
| 85 | Benign | Benign | 0.999 | 0.001 | yes |
| 96 | Benign | Benign | 0.996 | 0.004 | yes |
| 118 | Benign | Benign | 0.999 | 0.001 | yes |
| 9 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 14 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 15 | High-grade | Benign | 0.978 | 0.022 | NO |
| 17 | High-grade | High-grade | 0.001 | 0.999 | yes |
| 44 | High-grade | High-grade | 0.010 | 0.990 | yes |
| 48 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 63 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 66 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 83 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 84 | High-grade | Benign | 0.644 | 0.356 | NO |
| 97 | High-grade | Benign | 0.999 | 0.001 | NO |

### Split 02

| Case | True | Pred | P(benign) | P(high-grade) | Correct |
|---|---|---|---|---|---|
| 26 | Benign | Benign | 0.935 | 0.065 | yes |
| 36 | Benign | Benign | 0.999 | 0.001 | yes |
| 86 | Benign | High-grade | 0.025 | 0.975 | NO |
| 94 | Benign | High-grade | 0.291 | 0.709 | NO |
| 109 | Benign | Benign | 0.986 | 0.014 | yes |
| 110 | Benign | Benign | 1.000 | 0.000 | yes |
| 111 | Benign | High-grade | 0.093 | 0.907 | NO |
| 1 | High-grade | High-grade | 0.098 | 0.902 | yes |
| 12 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 18 | High-grade | Benign | 0.700 | 0.300 | NO |
| 40 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 42 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 60 | High-grade | High-grade | 0.007 | 0.993 | yes |
| 64 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 67 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 73 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 75 | High-grade | High-grade | 0.001 | 0.999 | yes |
| 87 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 88 | High-grade | High-grade | 0.000 | 1.000 | yes |

### Split 03

| Case | True | Pred | P(benign) | P(high-grade) | Correct |
|---|---|---|---|---|---|
| 4 | Benign | Benign | 1.000 | 0.000 | yes |
| 22 | Benign | Benign | 1.000 | 0.000 | yes |
| 23 | Benign | Benign | 0.992 | 0.008 | yes |
| 56 | Benign | High-grade | 0.000 | 1.000 | NO |
| 93 | Benign | Benign | 1.000 | 0.000 | yes |
| 107 | Benign | Benign | 1.000 | 0.000 | yes |
| 108 | Benign | Benign | 1.000 | 0.000 | yes |
| 3 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 19 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 38 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 47 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 49 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 55 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 59 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 61 | High-grade | High-grade | 0.073 | 0.927 | yes |
| 91 | High-grade | High-grade | 0.004 | 0.996 | yes |
| 92 | High-grade | Benign | 0.664 | 0.336 | NO |
| 100 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 106 | High-grade | High-grade | 0.000 | 1.000 | yes |

### Split 04

| Case | True | Pred | P(benign) | P(high-grade) | Correct |
|---|---|---|---|---|---|
| 25 | Benign | Benign | 0.992 | 0.008 | yes |
| 34 | Benign | Benign | 1.000 | 0.000 | yes |
| 45 | Benign | High-grade | 0.172 | 0.828 | NO |
| 46 | Benign | High-grade | 0.004 | 0.996 | NO |
| 79 | Benign | Benign | 0.981 | 0.019 | yes |
| 82 | Benign | Benign | 1.000 | 0.000 | yes |
| 119 | Benign | Benign | 1.000 | 0.000 | yes |
| 5 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 7 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 11 | High-grade | Benign | 0.970 | 0.030 | NO |
| 32 | High-grade | Benign | 0.890 | 0.110 | NO |
| 41 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 50 | High-grade | High-grade | 0.022 | 0.978 | yes |
| 51 | High-grade | High-grade | 0.051 | 0.949 | yes |
| 57 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 58 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 80 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 89 | High-grade | High-grade | 0.465 | 0.535 | yes |
| 99 | High-grade | High-grade | 0.000 | 1.000 | yes |

### Split 05

| Case | True | Pred | P(benign) | P(high-grade) | Correct |
|---|---|---|---|---|---|
| 21 | Benign | Benign | 0.997 | 0.003 | yes |
| 27 | Benign | Benign | 1.000 | 0.000 | yes |
| 28 | Benign | Benign | 1.000 | 0.000 | yes |
| 30 | Benign | Benign | 0.999 | 0.001 | yes |
| 43 | Benign | Benign | 0.999 | 0.001 | yes |
| 77 | Benign | Benign | 1.000 | 0.000 | yes |
| 90 | Benign | Benign | 1.000 | 0.000 | yes |
| 16 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 52 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 54 | High-grade | High-grade | 0.003 | 0.997 | yes |
| 68 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 69 | High-grade | Benign | 0.741 | 0.259 | NO |
| 70 | High-grade | High-grade | 0.195 | 0.805 | yes |
| 72 | High-grade | High-grade | 0.004 | 0.996 | yes |
| 95 | High-grade | High-grade | 0.001 | 0.999 | yes |
| 98 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 101 | High-grade | High-grade | 0.000 | 1.000 | yes |
| 104 | High-grade | High-grade | 0.000 | 1.000 | yes |
