# Energy-Aware Early Stopping: Supporting Material

Supporting material for the dissertation project on deciding, from the training logs of a neural network, when further training is no longer worth its energy cost.

The project has two steps:

1. **Step 1, training logs.** Small neural networks are trained under many configurations. After every epoch the loss, accuracy, time and energy are recorded.
2. **Step 2, decision model.** Every epoch of every run is labelled *stop* or *continue*. Classifiers (Random Forest, Gradient Boosting, Logistic Regression, AdaBoost) learn to predict that label from signals available during training. Each run is then replayed under the learned stopper and under patience-based early stopping, and the two are compared.

An original MNIST-only version (`step1_baseline_training.py`, `step2_decision_model.py`) and an extended four-dataset version (`step1_v2.py`, `step2_v2.py`) are included. All four scripts are the fixed versions; the scripts that produced the first set of logs are kept in `archive/scripts_before_fixes/`.

---

## 1. Contents

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── code/
│   ├── common.py                    shared helpers (seeds, config design, energy meter, labelling, evaluation)
│   ├── step1_baseline_training.py   Step 1, MNIST only: 40 MLPs
│   ├── step1_v2.py                  Step 1, four datasets: 160 runs (--datasets selects which)
│   ├── step2_decision_model.py      Step 2, MNIST only
│   └── step2_v2.py                  Step 2, four datasets, 4 meta-models, full evaluation
├── training_logs_v2_rerun/          NEW: MNIST + FashionMNIST, fixed Step 1, real power measurement (80 runs)
├── results_v2_rerun/                NEW: Step 2 on those logs (primary results, section 5)
├── training_logs/                   first-run logs: 40 MNIST runs (old Step 1)
├── training_logs_v2/                first-run logs: 160 runs on four datasets (old Step 1)
├── results/                         Step 2 (fixed) on training_logs/
├── results_v2/                      Step 2 (fixed) on training_logs_v2/
├── energy_measurement/
│   └── powermetrics_log.txt         short powermetrics capture (see section 6)
└── archive/
    ├── scripts_before_fixes/        the scripts exactly as they were when the first logs were produced
    └── results_before_fixes/        the results those scripts produced
```

`venv/` and `data/` are not part of the repository (see `.gitignore`); `data/` is downloaded automatically.

## 2. Environment

| | |
|---|---|
| Machine | MacBook Pro, Apple M2 (model identifier `Mac14,7`) |
| Python | 3.13.9 |
| Packages | pinned in `requirements.txt` (torch 2.12.1, torchvision 0.27.1, numpy 2.5.0, pandas 3.0.3, scikit-learn 1.9.0, scipy 1.18.0, matplotlib 3.11.0, codecarbon 3.2.8) |
| Compute | CPU. The scripts use CUDA when available; the logged epoch times (about 3 s per MNIST epoch, 30 s for CIFAR-10, 68 s for SVHN) are consistent with the laptop CPU, not a GPU. |
| Duration | first run: the 160 extended runs took about 70 hours (the 80 MLP runs about 2 hours, the 80 CNN runs the rest). Re-run of the 80 MLP runs: about 2 hours |

```bash
git clone <repository-url>
cd <repository>
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Running the code

Run everything from the repository root. The scripts import `common.py`, which sits next to them in `code/`.

```bash
# Step 1 (long). On a Mac, sudo gives real power measurement via powermetrics (see section 6).
python code/step1_baseline_training.py                       # 40 MNIST runs  -> training_logs/
python code/step1_v2.py                                      # 160 runs       -> training_logs_v2/
sudo -E venv/bin/python code/step1_v2.py                     # same, with real power measurement

# Only some datasets, into their own folder (this is how training_logs_v2_rerun/ was made)
caffeinate -i sudo -E venv/bin/python code/step1_v2.py --datasets mnist fashionmnist --log-dir training_logs_v2_rerun

# Step 2 (minutes)
python code/step2_decision_model.py                          # -> results/
python code/step2_v2.py                                      # -> results_v2/   (add --skip-loro to skip the slow 160-fold scheme)
python code/step2_v2.py --logs training_logs_v2_rerun/all_runs_combined.csv --out results_v2_rerun
```

Step 1 skips any run whose CSV already exists, so an interrupted experiment can be restarted, and running further datasets into the same `--log-dir` (for example `--datasets cifar10 svhn`) adds them to the same `all_runs_combined.csv`. Every run is seeded (`42 + run_id`), and the train/validation split is seeded, so a restart is reproducible. Step 2 options: `--logs`, `--out`, `--acc-tol`, `--gain-per-wh`, `--max-drop`, `--skip-loro`, `--jobs`.

## 4. Method

**Experiment design.** Each dataset gets 40 runs: 5 architectures × 4 learning rates × 4 batch sizes (80 combinations), of which 40 are chosen so that every learning rate and every batch size is used and learning rate is not tied to batch size (`balanced_configs` in `common.py`). MLPs for MNIST and FashionMNIST, small CNNs for CIFAR-10 and SVHN. Training: Adam, 30 epochs, 10,000-image validation set per dataset, official test set logged every epoch.

**Labels.** At epoch *t* of a run, with *remaining gain* = (best validation accuracy from *t* onwards) − (accuracy at *t*), in percentage points, and *remaining energy* in Wh:

- *stop* if the remaining gain is below 0.5 pp, or
- *stop* if the remaining gain per remaining Wh is below 1 pp/Wh,
- otherwise *continue*.

**Features** (17, all observable while training): epoch, losses, accuracy, one-epoch and three-epoch improvements, loss-worsening streak, train-validation gap, gap to the best accuracy so far, time and energy per epoch, cumulative energy, accuracy per energy, recent gain per Wh.

**Evaluation.**

- *Leave-one-dataset-out* (primary): the meta-model is trained on three datasets and judged on the fourth. *Leave-one-run-out* (secondary) leaves other runs of the same dataset in training, so it is an optimistic, in-distribution number.
- Results are paired per run: accuracy drop (percentage points) and energy saved (%) against training all 30 epochs, with 95% bootstrap confidence intervals.
- Patience baselines are swept from 1 to 10 epochs. The strongest baseline is the one that saves the most energy within a 0.5 pp mean accuracy drop.
- Paired Wilcoxon signed-rank tests with Holm correction; ablation (with or without energy features); sensitivity to both labelling thresholds.

## 5. Results

### 5.1 Re-run with the fixed Step 1 (MNIST + FashionMNIST, 80 runs)

`training_logs_v2_rerun/`, `results_v2_rerun/`. Every epoch of every run has a real `powermetrics` energy reading (no zero-energy epochs; 4 to 10 W, 0.095 to 0.159 Wh per run); all four learning rates and all four batch sizes appear in 20 runs each; no run failed to learn (best test accuracy 98.3% MNIST, 89.3% FashionMNIST). Accuracy is **test** accuracy (the stopping decisions use validation accuracy).

Leave-one-dataset-out (train the stopper on one dataset, judge it on the other; **two folds only**), paired per run (`results_table.csv`):

| Method | Accuracy drop, pp (95% CI) | Energy saved, % | Epochs used |
|---|---|---|---|
| Patience-3 | 0.31 (0.16, 0.47) | 66.9 | 9.9 |
| Patience-5 | 0.15 (0.02, 0.27) | 58.2 | 12.5 |
| Patience-10 | 0.03 (−0.08, 0.15) | 40.2 | 17.9 |
| Random Forest (ours) | 0.03 (−0.08, 0.13) | 63.4 | 10.9 |
| AdaBoost (ours) | 0.10 (−0.02, 0.23) | 67.6 | 9.7 |
| Logistic Regression (ours) | 0.18 (0.02, 0.33) | 51.1 | 14.6 |
| Gradient Boosting (ours) | 1.38 (1.07, 1.70) | 80.1 | 6.0 |

- The **Random Forest keeps the accuracy of Patience-10 and saves 23 points more energy** (Holm-corrected p < 0.001). Against Patience-3 it loses 0.28 pp less accuracy (Holm p = 0.0016) and saves 3.5 points less energy (not significant after correction). It sits at or slightly better than the trade-off line traced by the patience baselines; it does not dominate them.
- **Gradient Boosting fails on unseen MNIST**: trained on FashionMNIST alone it stops after about one epoch (2.5 pp drop, 96% saved on MNIST); its overall figures are a failure of that fold and not a usable trade-off (`per_dataset_results.csv`).
- With other runs of the same dataset in the training data (leave-one-run-out, `results_table_loro.csv`) all four meta-models match the patience trade-off closely (for example Random Forest −0.03 pp at 63.2% saved), an optimistic, in-distribution number.
- **The energy rule never fired.** All stop labels (719 MNIST, 465 FashionMNIST) come from the accuracy rule (`label_summary.csv`): with real measurements a run costs only about 0.1 Wh, so the remaining gain per remaining Wh is never below 1 pp/Wh. Energy enters only as input features, and removing them changes nothing (classification accuracy 0.828 with, 0.831 without; `ablation_results.csv`). On these runs, the result is that accuracy-based stopping is what is being learned and the energy signal adds nothing beyond time.

### 5.2 First-run logs (four datasets, older Step 1)

Extended experiment, leave-one-dataset-out (four folds), 140 runs that learned, **validation** accuracy (`results_v2/results_table.csv`). These logs come from the earlier Step 1 (see section 7), so treat them as less reliable than 5.1.

| Method | Accuracy drop, pp (95% CI) | Energy saved, % | Epochs used |
|---|---|---|---|
| Patience-2 (strongest baseline) | 0.07 (−0.19, 0.34) | 74.4 | 7.7 |
| Patience-3 | −0.18 (−0.37, 0.00) | 68.6 | 9.4 |
| Patience-5 | −0.32 (−0.49, −0.16) | 59.4 | 12.1 |
| Patience-10 | −0.22 (−0.34, −0.08) | 41.1 | 17.6 |
| Random Forest (ours) | 0.37 (−0.10, 0.90) | 69.5 | 9.0 |
| Gradient Boosting (ours) | 0.52 (0.05, 1.05) | 65.0 | 10.4 |
| Logistic Regression (ours) | 0.61 (0.09, 1.21) | 67.0 | 9.8 |
| AdaBoost (ours) | 1.89 (1.13, 2.63) | 72.6 | 8.1 |

- Across datasets the learned stopper **does not beat simple patience.** The Random Forest saves about as much energy as Patience-3 but is not more accurate, and it saves 4.9 points less than Patience-2 (Holm-corrected p = 0.024). A negative drop means that stopping earlier gave a *higher* validation accuracy than the final epoch.
- With other runs of the same dataset in training (`results_v2/results_table_loro.csv`) the Random Forest reaches −0.16 pp at 69.2% energy saved, matching Patience-3.
- On MNIST alone (`results/results_table.csv`, leave-one-architecture-out) the Random Forest saves 67.3% energy at a 0.14 pp drop, compared with Patience-10 at 44.2% for 0.15 pp.
- In these logs the energy rule produced 290 of 4,800 stop labels, because per-epoch energy was derived from time (section 6).

Earlier drafts of these results reported about 90% energy saved for the original MNIST experiment. That figure came from a labelling rule that only used the final-epoch accuracy, from averaging absolute kWh and from unpaired comparisons. It is replaced by the numbers above.

## 6. Energy measurement

- **Existing logs.** The per-epoch energy was read from CodeCarbon's offline tracker, which only refreshes about every 15 seconds. As a result 68% of MNIST epochs and 65% of FashionMNIST epochs show exactly 0 kWh. The run totals are reliable (about 7 to 8 W on average), so `step2_*.py` re-splits each run's total over its epochs in proportion to epoch time (`repair_legacy_energy` in `common.py`). Consequently, in the existing logs energy is effectively *time × constant power*, and the energy features carry little information beyond time.
- **New logs (`training_logs_v2_rerun/`).** `EnergyMeter` in `common.py` integrates `powermetrics` CPU+GPU+ANE package power (1 s samples) over each epoch when Step 1 is started with `sudo` (all 2,400 epochs of the re-run were measured this way, `energy_method = powermetrics_1s`); otherwise it uses CodeCarbon's run total split by time. The method is saved in the `energy_method` column. Package power excludes DRAM, display and the rest of the machine.
- **`energy_measurement/powermetrics_log.txt`.** A short `powermetrics` capture from the laptop (`Mac14,7`), taken on 3 July 2026 at about 23:44, when the extended experiment had just finished: 10 consecutive samples of about 0.1 s each (about 1.1 s in total). Combined CPU+GPU+ANE power was 227 to 684 mW (mean 343 mW), which is an almost idle machine. It documents the sampling tool and the machine; it is not a continuous trace of the training runs.

## 7. Limitations

- **The first-run logs (`training_logs/`, `training_logs_v2/`) come from the earlier Step 1 scripts** (kept in `archive/scripts_before_fixes/`). In them: only batch sizes 32 and 128 were used (`configs[::2]` never selected 64 or 256); nothing was seeded; no test accuracy was logged, so the results in 5.2 use **validation** accuracy, which is also what the labels use, and are therefore optimistic.
- **Only MNIST and FashionMNIST have been re-run** with the fixed Step 1 (section 5.1). Re-running CIFAR-10 and SVHN takes about 68 hours (`python code/step1_v2.py --datasets cifar10 svhn --log-dir training_logs_v2_rerun`) and has not been done yet; until then the CNN results come from the first-run logs, and the re-run's unseen-dataset test has only two folds, and both are MLP datasets.
- **20 of the 160 first-run extended runs never learned** (7 CIFAR-10 runs at about 10% accuracy, 13 SVHN runs at 18.7%, the majority class), all at learning rates 0.005 or 0.01. They are excluded from the 5.2 tables (`results_table_all_runs.csv` includes them). The current CNN learning-rate grid is lower for this reason.
- One run per configuration, one laptop, one train/validation split: there is no measure of run-to-run noise in energy or accuracy.
- Energy is package power (CPU+GPU+ANE) or modelled power, not wall-socket power.
- In the re-run the energy label rule never fires (section 5.1), so the stop labels are accuracy-based.
- The stopping point of the learned model is the first epoch at which the classifier says *stop*; no smoothing or minimum number of epochs is applied.
- Only small MLPs and CNNs with Adam and a fixed 30-epoch budget were studied.

## 8. Executable software

No standalone executable is provided. The project is Python source code; the dependencies and commands are given above.
