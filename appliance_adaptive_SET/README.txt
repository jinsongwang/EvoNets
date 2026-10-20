Appliance-aware adaptive SET — hypothesis-testing experiment

Open SET_with_UKDALE_adaptive.ipynb in a Python 3.12 / TensorFlow kernel.
The notebook is self-contained. adaptive_set.py is an equivalent command-line
implementation. The earlier notebook and source files are unchanged.

Question
Can independently adapting sparsity to each appliance's validation behavior
retain classification quality with fewer active connections than fixed SET?
"Appliance-aware" here means separate validation feedback and budget control
for each target; it does not use appliance names or hand-engineered features.
This is a proposed experimental method, not a claim of literature novelty.

Methods
Dense connectivity, fixed-budget SET, adaptive-budget SET, on DNN and CNN.
Four binary targets are trained independently. Their appliance-name mapping
cannot be verified from the supplied CSV column names.

Fixed/adaptive SET have identical initial weights, masks, optimizer settings,
training examples and minibatch permutations for each paired seed. Their
initial-state hashes are checked. All methods use Adam, binary cross-entropy,
and the same validation checkpoint criterion (BCE + 0.02 * first-layer density).
DNN: flatten, dropout 0.2, masked dense 100 ReLU, dense 20 ReLU, sigmoid output.
CNN: two Conv1D(64,2,ReLU), batch normalization, max pool 2, then the DNN head.
These controlled architectures differ from the original notebook. L1/L21
experiments remain in the earlier version and are omitted here to isolate SET.

Adaptive rule
Start at the same epsilon=20 connectivity as fixed SET. Maintain an independent
controller for every appliance/model/seed. After warmup, shrink the budget by
20% if validation BCE is within 2% of its best prior value; otherwise grow it
by 25%. Bound density to 2%-60%. SET removes the smallest-magnitude weights by
sign and randomly regrows connections to reach the new budget. Regrown weights
receive Glorot-scale uniform values. Surviving weights and Adam moments persist;
moments on removed/regrown connections are cleared. The controller never sees
test labels or test metrics. These settings are configurable research choices.

Data controls
Use first 2,000 rows as test, next 1,600 as validation, remaining 6,400 as train.
Remove exact input duplicates across partitions (test takes priority, then
validation). Standardize with training-only statistics. The saved config.json
records retained counts and duplicates. Source-window overlap or independent
households cannot be established from these already-aggregated CSVs.

Run modes
Default smoke mode: four epochs, seed 42, up to 128 train / 64 validation /
64 test rows before duplicate filtering. 24 model/target runs. This validates
execution and adaptation, not the scientific hypothesis.
Full mode: change MODE to "full" in the notebook. Uses 50 epochs, seeds 42/43/44,
a five-epoch warmup, both architectures and all targets (72 runs). Requires
substantially more computation. Selected models are saved as .keras files.
Full mode has NOT been run in this delivery.

Command line (from this folder, with an appropriate Python environment)
python adaptive_set.py --mode smoke --data-dir /path/to/csv/folder --output-dir smoke_results
python adaptive_set.py --mode full --data-dir /path/to/csv/folder --output-dir full_results
python test_adaptive_set.py

Dependencies
Use requirements.txt (copied from the validated local environment).
Use the repository Python environment.
The notebook locates the original iCloud CSV directory as a fallback; for
portability, set SET_DATA_DIR or edit DATA_DIR.

Interpretation
The proposed full-run descriptive criterion is mean F1 no more than 0.01 below fixed
SET and at least 5% fewer whole-model active MACs, for each target/architecture.
Report every comparison and seed variability. A positive descriptive comparison
is not statistical confirmation. Additional seeds, suitable uncertainty
analysis, unseen-house data and sensitivity experiments are needed for a paper.
Do not tune controller parameters using test results.

All TensorFlow operations execute as dense tensors. Active MACs count enabled
convolution/dense connections and are ONLY a theoretical connectivity proxy;
BN/activation/pooling costs are excluded. Lower MAC proxies do not establish
real speedups or memory savings. Dense connectivity also uses the same masked
layer with an all-one mask, so this is not a native Dense performance benchmark.
Weights/masks occupy dense storage in all cases.
Training times include graph tracing but exclude validation/data loading;
topology time is reported separately. Inference latency is the median of 10
warmed calls on identical batch sizes and includes call/materialization overhead.
Timing is noisy and requires controlled repeated hardware measurements.

Outputs
metrics.csv: per-run held-out scores, budget, active MACs and measured timings.
aggregate_metrics.csv: per-target/method seed means and standard deviations.
paired_comparisons.csv: matched-seed adaptive-versus-fixed differences/ratios.
hypothesis_summary.csv: the descriptive criterion, labeled by run mode.
*_history.csv: validation losses, budget trajectory and shrink/grow decisions.
*_test_predictions.npz: held-out predictions and labels for later analysis.
tradeoff.png and budget_trajectories.png: visual summaries.
config.json: exact settings, data audit and runtime/device information.

Tests cover mask gradients, exact rewiring budgets, controller shrink/grow and
bounds, preservation/reset of optimizer moments, and model serialization.
