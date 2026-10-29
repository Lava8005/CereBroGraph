# BBB ensemble + conformal prediction (scaffold split, seed 0)

Members: 5; val n=657, test n=695.

Single-member test ROC-AUC: 0.922 ± 0.004; ensemble: 0.932.
ECE: single 0.072 vs ensemble 0.075.

| alpha | coverage | cov BBB+ | cov BBB- | singleton frac | acc on singletons |
|---|---|---|---|---|---|
| 0.05 | 0.938 | 0.937 | 0.939 | 0.807 | 0.923 |
| 0.1 | 0.891 | 0.924 | 0.850 | 0.957 | 0.886 |
| 0.2 | 0.801 | 0.874 | 0.712 | 0.871 | 0.921 |

| kept fraction | acc (lowest ensemble std) | acc (highest confidence) |
|---|---|---|
| 1.0 | 0.883 | 0.883 |
| 0.8 | 0.894 | 0.917 |
| 0.6 | 0.923 | 0.940 |
| 0.4 | 0.935 | 0.946 |