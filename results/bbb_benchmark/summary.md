### Random split — 10 seeds (test n ≈ 780, mean ± SD)

| Model | ROC-AUC | PR-AUC | F1 | MCC | BALANCED-ACCURACY |
|---|---|---|---|---|---|
| Rule (MW≤500, TPSA≤90, HBD≤3) | 0.733 ± 0.015 | 0.774 ± 0.012 | 0.793 ± 0.011 | 0.458 ± 0.030 | 0.733 ± 0.015 |
| Logistic regression (6 descriptors) | 0.826 ± 0.018 | 0.867 ± 0.014 | 0.850 ± 0.012 | 0.543 ± 0.042 | 0.747 ± 0.019 |
| Random forest (Morgan FP) | 0.958 ± 0.005 | 0.975 ± 0.005 | 0.905 ± 0.009 | 0.726 ± 0.024 | 0.849 ± 0.014 |
| Random forest (RDKit descriptors) | 0.960 ± 0.006 | 0.977 ± 0.005 | 0.915 ± 0.009 | 0.756 ± 0.025 | 0.869 ± 0.014 |
| XGBoost (descriptors + FP) | 0.962 ± 0.006 | 0.975 ± 0.006 | 0.919 ± 0.010 | 0.772 ± 0.027 | 0.878 ± 0.013 |
| GINE graph network | 0.945 ± 0.010 | 0.964 ± 0.009 | 0.902 ± 0.008 | 0.728 ± 0.020 | 0.861 ± 0.011 |

Paired difference in ROC-AUC, GNN minus other model (same splits):

| Other model | mean diff | 95% CI | Wilcoxon p |
|---|---|---|---|
| Rule (MW≤500, TPSA≤90, HBD≤3) | +0.212 | [+0.200, +0.225] | 0.00195 |
| Logistic regression (6 descriptors) | +0.119 | [+0.106, +0.132] | 0.00195 |
| Random forest (Morgan FP) | -0.012 | [-0.018, -0.007] | 0.00391 |
| Random forest (RDKit descriptors) | -0.015 | [-0.021, -0.010] | 0.00195 |
| XGBoost (descriptors + FP) | -0.016 | [-0.021, -0.012] | 0.00195 |

### Scaffold split — 10 seeds (test n ≈ 767, mean ± SD)

| Model | ROC-AUC | PR-AUC | F1 | MCC | BALANCED-ACCURACY |
|---|---|---|---|---|---|
| Rule (MW≤500, TPSA≤90, HBD≤3) | 0.735 ± 0.048 | 0.746 ± 0.033 | 0.780 ± 0.049 | 0.474 ± 0.097 | 0.735 ± 0.048 |
| Logistic regression (6 descriptors) | 0.833 ± 0.037 | 0.855 ± 0.027 | 0.837 ± 0.025 | 0.552 ± 0.101 | 0.753 ± 0.053 |
| Random forest (Morgan FP) | 0.910 ± 0.029 | 0.935 ± 0.021 | 0.860 ± 0.017 | 0.622 ± 0.075 | 0.793 ± 0.048 |
| Random forest (RDKit descriptors) | 0.914 ± 0.027 | 0.940 ± 0.014 | 0.862 ± 0.019 | 0.639 ± 0.075 | 0.811 ± 0.044 |
| XGBoost (descriptors + FP) | 0.917 ± 0.026 | 0.941 ± 0.014 | 0.853 ± 0.058 | 0.589 ± 0.220 | 0.785 ± 0.110 |
| GINE graph network | 0.893 ± 0.032 | 0.918 ± 0.024 | 0.855 ± 0.026 | 0.627 ± 0.078 | 0.803 ± 0.046 |

Paired difference in ROC-AUC, GNN minus other model (same splits):

| Other model | mean diff | 95% CI | Wilcoxon p |
|---|---|---|---|
| Rule (MW≤500, TPSA≤90, HBD≤3) | +0.158 | [+0.122, +0.194] | 0.00195 |
| Logistic regression (6 descriptors) | +0.061 | [+0.043, +0.078] | 0.00195 |
| Random forest (Morgan FP) | -0.017 | [-0.031, -0.003] | 0.0137 |
| Random forest (RDKit descriptors) | -0.021 | [-0.034, -0.007] | 0.00586 |
| XGBoost (descriptors + FP) | -0.024 | [-0.041, -0.006] | 0.0137 |
