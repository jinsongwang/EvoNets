SET research comparison — start here

OPEN
SET_with_UKDALE_research.ipynb shows the delivered pilot and its limitations.
Keep research_set.py and base_set.py beside the notebook. RUN_TRAINING=False
opens the saved results; set it True to run/resume completed model runs.

ORIGINAL METHODS
Both the original and the earlier repaired notebooks are in legacy/.
The new controlled study retains dense, LASSO, group LASSO, SET, and LASSO+SET
families and adds static sparse, gradient-regrowth, adaptive and structured
ablations. Architecture/regularization changes are detailed in PROTOCOL.txt.
Do not mix historical embedded legacy outputs with the new pilot findings.

DELIVERED EXPERIMENT
Pilot: 12 methods x 2 backbones x 4 target columns x 3 seeds = 288 runs,
10 epochs each, on all 9,107 unique windows. Train 5,770; validation 1,474;
test 1,863. No test-based tuning. Pilot uses fixed default hyperparameters.
See pilot_results/metrics.csv and aggregate_metrics.csv for all findings.
This is a developmental synthetic-window classification experiment. It does
not verify source-independent houses or actual appliance-power disaggregation.

RUN
Use Python 3.12 with the pinned requirements.txt.
Create an environment with: python -m venv .venv
Then install: .venv/bin/python -m pip install -r requirements.txt
Set SET_DATA_DIR to the CSV directory or pass --data-dir explicitly.

python research_set.py --mode smoke --data-dir /path/to/CSVs --out fresh_smoke
python research_set.py --mode pilot --data-dir /path/to/CSVs --out fresh_pilot
python research_set.py --mode full --data-dir /path/to/CSVs --out fresh_full
python test_research_set.py

Optional --metadata /path/to/provenance.csv requires one record per input row:
row_index (zero based), split (train/validation/test), group_id.
Groups must be justified from actual houses/time/source windows. The loader
rejects groups shared across splits. See PROTOCOL.txt before generating groups.
A new data split or code/configuration change requires a fresh output directory.
Resume is at completed-run level; interrupted runs are restarted.

Full mode is NOT executed in this delivery. It configures 50 epochs, ten seeds
and three validation-only tuning trials per method/target/backbone, which is
substantially more expensive. These settings do not themselves confer publication
readiness. Plan statistical power and independent evaluation before a confirmatory
study; pilot inference with three seeds is particularly coarse.

WHAT IS MEASURED
Classification F1, accuracy, AUROC, average precision, MCC, Brier/ECE, confusion
counts, validation-selected and fixed-0.5 thresholds, perturbation robustness,
paired uncertainty, logical MACs, actual compact dense MACs, weight storage and
warmed native-model median/p95 latency. Training remains dense; gradient/regrowth
overhead is measured, not presented as a training acceleration.

Selected native compact models are saved as *_compact.keras and load without
custom sparse-layer objects. Every exported prediction is checked against its
pre-export counterpart. Model bytes exclude framework/runtime/activation memory;
no peak-memory claim is made. Timing needs independent hardware replication.

IMPORTANT SCIENTIFIC LIMITS
Read PROTOCOL.txt, RELATED_WORK.txt and pilot_results/research_status.json.
The adaptive/structured combination is a research candidate, not an established
new algorithm. Dynamic regrowth and structured pruning have substantial prior art.
Q1 publication cannot be guaranteed by code changes or improved pilot scores.
Raw UK-DALE source data and row provenance are needed for the next independent-
house/time validation stage. A verified row-to-source mapping is not included.

GITHUB CONTENTS
The repository includes code, notebooks, summary tables, figures and configuration.
Per-run histories, predictions, run records, split manifests and model files are
excluded from the published branch. Generate them locally by running the experiment.
