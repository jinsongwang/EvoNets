# EvoNets

Evolutionary Deep Nets for appliance-presence classification, adaptive sparsity, and joint F1–sparsity improvement.

## Three code folders

| Folder | Purpose |
| --- | --- |
| `Evolutionary Deep Nets` | Original baseline notebook. |
| `adaptive_sparsity` | Adaptive SET engine, protocol, notebook, tests, and historical pilot scores. |
| `accuracy_improvement` | Gradual pruning, distillation, warm-start refinement, evaluation, and measured result summaries. |

The accuracy scripts import the shared engine from `adaptive_sparsity`; it is stored once. `samX.csv`, `samY.csv`, and `label_schema.json` are shared inputs. The Word paper is `paper.docx`.

## Results in the paper

The two experimental objectives analyze the same completed CNN cohort: four targets, three seeds, 120 trained models. Recipe, checkpoint, and threshold selection use validation data.

| Metric | New fixed SET | Selected compressed policy |
| --- | ---: | ---: |
| Binary accuracy (%) | 91.5325 | 91.5280 |
| Macro positive-class F1 (%) | 92.4065 | 92.5896 |
| Compact weight bytes | 485,940 | 401,590 |
| Active first-layer connections | 25,680 | 19,536 |
| Compact dense kernel MACs | 257,924 | 236,854 |

1. **Compression with approximately preserved accuracy:** both accuracies round to 91.53%; weight storage falls 17.36%. Exact equality and statistical equivalence are not established.
2. **Joint F1 and sparsity improvement:** macro F1 rises 0.1831 percentage points and active connections fall 23.93%. Binary accuracy does not increase.

The policy uses width-80 refinement for T0–T2 and width-90 refinement for T3. Refinement costs 50 source epochs plus 25 additional epochs; this is not an equal-compute comparison. Three seeds and unverified source-window independence limit the conclusions. Appliance names remain unverified; T0–T3 are the primary identifiers. Dense arrays retain unstructured zeros, so sparsity does not imply measured speed or energy savings.

`adaptive_sparsity/results.csv` is the separate ten-epoch historical pilot. New summary tables and figures are in `accuracy_improvement/results`. The repository omits saved model binaries, per-run histories, prediction arrays, caches, duplicate documents, and validation logs to keep the file set small. The complete experiment archive is retained separately.

## Run

Use Python 3.11 or later. Run these commands from the repository root in a virtual environment:

```sh
pip install -r requirements.txt
python -m unittest discover -s adaptive_sparsity -p 'test_*.py'
python -m unittest discover -s accuracy_improvement -p 'test_*.py'
```

The two research notebooks show saved results by default. The original baseline retains its historical training workflow; its original split is not an independent-source benchmark.

To reproduce the full CNN study in a fresh output directory:

```sh
python accuracy_improvement/train_accuracy.py --data-dir . --out accuracy_improvement/runs/reproduction --epochs 50
python accuracy_improvement/train_refine.py --data-dir . --out accuracy_improvement/runs/reproduction
python accuracy_improvement/train_conservative.py --data-dir . --out accuracy_improvement/runs/reproduction
python accuracy_improvement/evaluate_accuracy.py --data-dir . --out accuracy_improvement/runs/reproduction
python accuracy_improvement/present_accuracy.py --run-dir accuracy_improvement/runs/reproduction --out accuracy_improvement/runs/reproduction/figures
```

Training uses seeds 11, 23, 37 and all four targets by default. Complete training and validation-only selection before test evaluation. Training/evaluation scripts record source hashes and reject incompatible output folders. The import-path packaging changes alter script hashes; use fresh outputs rather than resuming the separately archived study. The supplied summaries remain the measured results of that completed study.
