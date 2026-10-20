# EvoNets: controlled sparse NILM classification

This repository retains the original DNN/CNN, LASSO, group-LASSO (L21), SET, and LASSO+SET experiment families, with additional adaptive and structured sparse comparisons.

Start with [the executed research notebook](SET_with_UKDALE_research.ipynb), [setup instructions](README.txt), and [the experimental protocol](PROTOCOL.txt). Historical original and repaired notebooks are in [legacy/](legacy/).

## Notebooks

- [Repaired original experiments](SET_with_UKDALE_fixed.ipynb)
- [Adaptive SET](appliance_adaptive_SET/SET_with_UKDALE_adaptive.ipynb)
- [Research comparison](SET_with_UKDALE_research.ipynb)

## Reproduce

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python test_research_set.py
.venv/bin/python research_set.py --mode smoke --data-dir . --out fresh_smoke
```

Open the notebook from the repository root to inspect the delivered pilot. `RUN_TRAINING=False` reads saved results; setting it to `True` runs or resumes the configured experiment. Generated models and individual run artifacts are excluded from Git and can be regenerated. The published results folder contains summary tables, figures, and configuration only.

## Delivered evidence

The developmental pilot completed **288 runs: 12 methods × 2 backbones × 4 label columns × 3 seeds**, ten epochs each, using 9,107 unique windows. Fourteen correctness tests passed. See [all results](pilot_results/metrics.csv), [the summary](pilot_results/RESULTS_SUMMARY.txt), and [paired uncertainty](pilot_results/paired_uncertainty.csv).

| Backbone | Fixed SET macro F1 | Adaptive structured + L21 macro F1 | Exported weight reduction |
|---|---:|---:|---:|
| DNN | 0.7971 | 0.7934 | 23.3% |
| CNN | 0.8949 | 0.8917 | 21.4% |

The pilot shows compression with small average F1 losses; it does not establish consistent accuracy superiority. All targets and negative comparisons are retained.

## Scientific limits

These CSVs describe synthetic-window appliance-presence classification. Source-window and household independence are unverified. Full tuning, independent-house evaluation, and sensitivity sweeps have not been completed. The proposed combination has substantial prior art; methodological novelty, state-of-the-art performance, and Q1 publication readiness are **not established**. Read [related work](RELATED_WORK.txt), [statistical-power limitations](POWER_NOTE.txt), and [validation records](VALIDATION.txt).

Training uses dense tensors. Native compact export removes inactive neurons and verifies prediction equivalence; logical nonzero connection counts alone do not imply runtime acceleration.
