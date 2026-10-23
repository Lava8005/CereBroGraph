# Pilot docking analysis (AChE, MAO-B)

Library 171; successfully docked on both targets: 159 (failed/timeouts: 12).

## Step 2 - Docking validation: known actives vs property-matched decoys

| Target | n actives | n decoys | AUC (-Vina) | 95% CI | AUC raw size-only baseline (heavy atoms) |
|---|---|---|---|---|---|
| AChE | 6 | 58 | 0.62 | 0.40-0.82 | 0.63 |
| MAOB | 6 | 58 | 0.68 | 0.43-0.90 | 0.34 |

Actives per target: AChE = donepezil, galantamine, rivastigmine, tacrine, huperzine A, physostigmine, ladostigil; MAO-B = selegiline, rasagiline, safinamide, lazabemide, pargyline, ladostigil. Decoys are 5 property-matched (MW, logP, heavy atoms) B3DB test molecules per active, not DUD-E decoys; this is a small pilot, not a benchmark.

Correlation between heavy-atom count and Vina score (random+decoy molecules): AChE r=-0.50, MAOB r=0.07 (negative = larger molecules score better, the known Vina size bias).

| Active | Target | Vina | Rank among its matched decoys |
|---|---|---|---|
| rivastigmine | AChE | -7.7 | 6/6 |
| tacrine | AChE | -9.6 | 1/6 |
| huperzine A | AChE | -11.1 | 1/6 |
| donepezil | AChE | -9.7 | 3/6 |
| selegiline | MAOB | -7.1 | 3/6 |
| physostigmine | AChE | -9.5 | 2/5 |
| rasagiline | MAOB | -8.2 | 1/6 |
| lazabemide | MAOB | -6.5 | 4/5 |
| pargyline | MAOB | -6.7 | 5/6 |
| safinamide | MAOB | -9.4 | 1/6 |
| ladostigil | AChE | -8.8 | 5/6 |
| ladostigil | MAOB | -8.4 | 4/6 |

## Step 3 - Ablation of the selection scheme (random B3DB test molecules, true BBB labels)

Top-20 selected from 87 molecules (base rate of true BBB+ = 0.62). 95% bootstrap CIs over molecules.

| Scheme | n selected | true BBB+ rate | CI | mean Vina | CI |
|---|---|---|---|---|---|
| docking only | 20 | 0.60 | 0.35-0.80 | -9.25 | -9.56--8.92 |
| docking + physchem rule | 20 | 0.70 | 0.50-0.90 | -9.09 | -9.46--8.72 |
| docking + GNN gate (p>=0.5) | 20 | 0.85 | 0.70-1.00 | -8.80 | -9.30--8.33 |
| docking + rule + GNN gate | 20 | 0.85 | 0.70-1.00 | -8.80 | -9.29--8.34 |
| weighted sum (dock, BBB prob, QED) | 20 | 0.90 | 0.75-1.00 | -8.59 | -9.05--8.18 |
| Pareto (AChE, MAO-B, BBB prob) | 20 | 0.85 | 0.62-0.95 | -8.53 | -8.96--8.11 |

## Step 4 - Retrospective ranking of known CNS drugs among the pilot library

Pool size 159. Pareto objectives: Vina AChE, Vina MAO-B, BBB probability.

| Compound | set | Vina AChE | Vina MAO-B | GNN BBB p | rule pass | Pareto rank | rank by mean Vina |
|---|---|---|---|---|---|---|---|
| tacrine | active | -9.6 | -8.8 | 0.87 | True | 1 | 12 |
| rasagiline | active | -8.3 | -8.2 | 0.97 | True | 1 | 56 |
| risperidone | control | -11.0 | -9.8 | 0.74 | True | 1 | 2 |
| donepezil | active | -9.7 | -7.1 | 0.84 | True | 3 | 48 |
| physostigmine | active | -9.5 | -5.3 | 0.95 | True | 3 | 108 |
| ladostigil | active | -8.8 | -8.4 | 0.88 | True | 3 | 38 |
| safinamide | active | -8.7 | -9.4 | 0.61 | True | 3 | 20 |
| huperzine A | active | -11.1 | -5.9 | 0.62 | True | 3 | 42 |
| pimavanserin | control | -10.0 | -6.5 | 0.65 | True | 3 | 56 |
| pargyline | active | -7.1 | -6.7 | 0.93 | True | 5 | 129 |
| selegiline | active | -8.2 | -7.1 | 0.93 | True | 5 | 93 |
| loperamide | control | -9.7 | -5.2 | 0.82 | True | 5 | 105 |
| rivastigmine | active | -7.7 | -6.7 | 0.87 | True | 7 | 118 |
| lazabemide | active | -7.0 | -6.5 | 0.66 | True | 8 | 133 |

Fraction of pool on Pareto front 1: 0.13.