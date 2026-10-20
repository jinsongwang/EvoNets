# EvoNets

Open [the notebook](SET_with_UKDALE_research.ipynb). Edit settings there and models in `research_set.py`.

## Files

- `SET_with_UKDALE_research.ipynb`: run and view experiments.
- `research_set.py`: all models, training, and evaluation.
- `test_research_set.py`: correctness tests.
- `samX.csv`, `samY.csv`: input data and labels.
- `results.csv`: all 288 saved pilot scores.
- `requirements.txt`: Python dependencies.

`README.md` and `.gitignore` complete the nine-file repository. Generated files stay in the ignored `runs/` folder. Older notebooks and detailed records remain in Git history.

## Run

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python test_research_set.py
.venv/bin/python research_set.py --mode smoke --data-dir . --out runs/smoke
```

The notebook reads saved results by default. Set `RUN_TRAINING=True` to train. Modes: smoke (3 epochs, 1 seed), pilot (10 epochs, 3 seeds), full (50 epochs, 10 seeds and validation-only tuning).

## Methods

All twelve methods remain: dense, LASSO, L21, static sparse, fixed SET, LASSO+SET, gradient SET, adaptive SET, adaptive gradient SET, adaptive structured SET, fixed group SET, and adaptive group SET. Both DNN and CNN backbones are supported. L21 groups incoming neuron weights and bias. Training uses dense tensors; compact export removes zero neurons and checks prediction equivalence.

## Pilot results

The saved pilot used 9,107 unique synthetic windows: 5,770 train, 1,474 validation, and 1,863 test. It completed 288 runs, ten epochs each, with seeds 11, 23 and 37. Settings, thresholds and computation measures are recorded in `results.csv`; detailed [original configuration](https://github.com/jinsongwang/EvoNets/blob/c7d672fe6fc9255c0b01b624d6280da3a939a946/pilot_results/config.json) remains in history. Consolidating files changes packaging, not those measured scores.

| Backbone | Fixed SET macro F1 | Adaptive group SET macro F1 | Weight reduction |
|---|---:|---:|---:|
| DNN | 0.7971 | 0.7934 | 23.3% |
| CNN | 0.8949 | 0.8917 | 21.4% |

Scores show compression with a small average F1 loss. Source-window and household independence are unverified. Three seeds provide limited statistical evidence. Training sparsity does not establish speedups. Full tuning, independent-house evaluation, and sensitivity studies remain incomplete; originality and Q1 readiness are not established.

SET, gradient regrowth and structured pruning have prior art: [SET](https://www.nature.com/articles/s41467-018-04316-3), [RigL](https://proceedings.mlr.press/v119/evci20a.html), and [structured sparse training](https://proceedings.iclr.cc/paper_files/paper/2024/hash/8c5f30296296d2ae402ebbd09aaa9c12-Abstract-Conference.html).
