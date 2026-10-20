# EvoNets

Two folders share the same data and dependencies.

- [original/](original/SET_with_UKDALE_fixed.ipynb): repaired original experiments, with the original model families.
- [research/](research/SET_with_UKDALE_research.ipynb): adaptive sparsity, comparisons, tests, and saved pilot results.
- `samX.csv`, `samY.csv`: shared inputs.
- `requirements.txt`: dependencies.

Ten tracked files. Generated outputs stay in ignored `runs/`. Open either notebook from the repository root or its own folder.

## Run

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python research/test_research_set.py
.venv/bin/python research/research_set.py --mode smoke --data-dir . --out runs/smoke
```

The research notebook reads saved results by default. Set `RUN_TRAINING=True` to train. Modes: smoke (3 epochs, 1 seed), pilot (10 epochs, 3 seeds), full (50 epochs, 10 seeds and validation-only tuning).

The original notebook defaults to a short smoke run. Set `SET_RUN_MODE=full` for its full experiment.

## Methods

All twelve methods remain: dense, LASSO, L21, static sparse, fixed SET, LASSO+SET, gradient SET, adaptive SET, adaptive gradient SET, adaptive structured SET, fixed group SET, and adaptive group SET. Both DNN and CNN backbones are supported. L21 groups incoming neuron weights and bias. Training uses dense tensors; compact export removes zero neurons and checks prediction equivalence.

## Pilot results

The saved pilot used 9,107 unique synthetic windows: 5,770 train, 1,474 validation, and 1,863 test. It completed 288 runs, ten epochs each, with seeds 11, 23 and 37. Settings, thresholds and computation measures are recorded in `research/results.csv`; detailed [original configuration](https://github.com/jinsongwang/EvoNets/blob/c7d672fe6fc9255c0b01b624d6280da3a939a946/pilot_results/config.json) remains in history. Consolidating files changes packaging, not those measured scores.

| Backbone | Fixed SET macro F1 | Adaptive group SET macro F1 | Weight reduction |
|---|---:|---:|---:|
| DNN | 0.7971 | 0.7934 | 23.3% |
| CNN | 0.8949 | 0.8917 | 21.4% |

Scores show compression with a small average F1 loss. Source-window and household independence are unverified. Three seeds provide limited statistical evidence. Training sparsity does not establish speedups. Full tuning, independent-house evaluation, and sensitivity studies remain incomplete; originality and Q1 readiness are not established.

SET, gradient regrowth and structured pruning have prior art: [SET](https://www.nature.com/articles/s41467-018-04316-3), [RigL](https://proceedings.mlr.press/v119/evci20a.html), and [structured sparse training](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8c5f30296296d2ae402ebbd09aaa9c12-Abstract-Conference.html).
