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
├── training_logs_v2/                four datasets, fixed Step 1, real power measurement (160 runs)
├── results_v2/                      Step 2 on those logs (primary results, section 5.1)
├── training_logs/                   first-run logs: 40 MNIST runs (old Step 1)
├── results/                         Step 2 (fixed) on training_logs/ (MNIST only)
├── energy_measurement/
│   └── powermetrics_log.txt         short powermetrics capture (see section 6)
└── archive/
    └── scripts_before_fixes/        the scripts exactly as they were when the first logs were produced
```

`venv/` and `data/` are not part of the repository (see `.gitignore`); `data/` is downloaded automatically.

## 2. Environment

| | |
|---|---|
| Machine | MacBook Pro, Apple M2 (model identifier `Mac14,7`) |
| Python | 3.13.9 |
| Packages | pinned in `requirements.txt` (torch 2.12.1, torchvision 0.27.1, numpy 2.5.0, pandas 3.0.3, scikit-learn 1.9.0, scipy 1.18.0, matplotlib 3.11.0, codecarbon 3.2.8) |
| Compute | CPU. The scripts use CUDA when available; the logged epoch times (about 3 s per MNIST epoch, 30 s for CIFAR-10, 68 s for SVHN) are consistent with the laptop CPU, not a GPU. |
| Duration | first run: the 160 extended runs took about 70 hours. Re-run with the fixed Step 1: about 2 hours for the 80 MLP runs and about 16 hours for the 80 CNN runs (6.2 h CIFAR-10, 9.5 h SVHN of training time) |

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

# Only some datasets (training_logs_v2/ was made in two sessions into the same folder):
caffeinate -i sudo -E venv/bin/python code/step1_v2.py --datasets mnist fashionmnist
caffeinate -i sudo -E venv/bin/python code/step1_v2.py --datasets cifar10 svhn

# Step 2 (minutes)
python code/step2_decision_model.py                          # -> results/
python code/step2_v2.py                                      # -> results_v2/   (add --skip-loro to skip the slow 160-fold scheme)
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

### 5.1 Re-run with the fixed Step 1 (all four datasets, 160 runs)

`training_logs_v2/`, `results_v2/`. Every epoch of every run (4,800) has a real `powermetrics` energy reading (no zero-energy epochs; MLP runs 4 to 10 W, CNN runs 9 to 12 W). All four learning rates and all four batch sizes appear in 20 runs of each model family, 10 per dataset. 159 of 160 runs learned; the one that did not (SVHN run 137, learning rate 0.002, stuck at 19.0%, the majority class) is excluded from the main tables (`results_table_all_runs.csv` includes it). Accuracy is **test** accuracy (the stopping decisions use validation accuracy).

Leave-one-dataset-out (the stopper is trained on three datasets and judged on the fourth; 4 folds), paired per run, 159 runs (`results_table.csv`):

| Method | Accuracy drop, pp (95% CI) | Energy saved, % | Epochs used |
|---|---|---|---|
| Patience-2 (strongest baseline) | −0.13 (−0.33, 0.07) | 67.3 | 9.8 |
| Patience-3 | −0.40 (−0.59, −0.21) | 61.3 | 11.6 |
| Patience-5 | −0.42 (−0.57, −0.27) | 53.3 | 14.0 |
| Patience-10 | −0.23 (−0.33, −0.14) | 37.0 | 18.9 |
| Random Forest (ours) | 0.05 (−0.15, 0.26) | 62.7 | 11.2 |
| Gradient Boosting (ours) | 0.29 (0.06, 0.51) | 66.6 | 10.0 |
| AdaBoost (ours) | 0.22 (−0.07, 0.55) | 66.6 | 10.0 |
| Logistic Regression (ours) | 0.58 (0.21, 0.98) | 67.1 | 9.8 |

A negative drop means that stopping early gave a *higher* test accuracy than training for all 30 epochs, which happens because the CNNs overfit (CIFAR-10: −1.4 pp for Patience-3).

- **Across datasets the learned stopper does not beat simple patience.** The Random Forest saves 4.6 points less energy than Patience-2 (Holm-corrected p = 0.021) at no significant difference in accuracy. Against Patience-3 it saves about the same energy (+1.5 points, not significant) and loses 0.45 pp more accuracy (Holm p = 0.042). Against Patience-5 it saves 9.5 points more energy but loses 0.47 pp more accuracy (both p < 0.001); against Patience-10 it saves 25.7 points more at no significant difference in accuracy.
- **With other runs of the same dataset in training** (leave-one-run-out, `results_table_loro.csv`) the Random Forest reaches −0.17 pp at 66.5% energy saved, which matches Patience-2 (−0.13 pp, 67.3%). The gap to the leave-one-dataset-out result is the cost of an unseen dataset.
- **The stopper transfers unevenly** (`per_dataset_results.csv`). On MNIST and FashionMNIST the Random Forest saves 62.7% and 60.2% at 0.02 and 0.04 pp. On CIFAR-10 it is too cautious (43.7% saved, against 55.2% for Patience-3). On SVHN it stops too early (85.0% saved, but a 0.93 pp drop where Patience-3 has −0.79 pp). Gradient Boosting stops after about 2.6 epochs on MNIST (1.83 pp drop) and Logistic Regression after about 2.3 epochs on SVHN (3.06 pp drop).
- **The energy label rule matters on the CNNs.** It produced 297 of the 2,718 stop labels, all on CIFAR-10 (101) and SVHN (196) and none on MNIST or FashionMNIST, where a run costs only about 0.1 Wh (`label_summary.csv`). Removing the energy features lowers the stop/continue classification accuracy (0.792 to 0.695) and makes the stopper stop earlier and riskier (77.4% saved, 0.49 pp drop, against 62.7% and 0.05 pp; `ablation_results.csv`).
- Sensitivity to both labelling thresholds is in `sensitivity_results.csv`; for example an accuracy tolerance of 0.1 pp saves 45.7% energy and 2 pp saves 85.8% at a 1.15 pp drop.

### 5.2 First-run logs (four datasets, older Step 1)

Extended experiment, leave-one-dataset-out (four folds), 140 runs that learned, **validation** accuracy (first-run results, not included in this repository). These logs come from the earlier Step 1 (see section 7), so treat them as less reliable than 5.1.

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
- With other runs of the same dataset in training (leave-one-run-out) the Random Forest reaches −0.16 pp at 69.2% energy saved, matching Patience-3.
- On MNIST alone (`results/results_table.csv`, leave-one-architecture-out) the Random Forest saves 67.3% energy at a 0.14 pp drop, compared with Patience-10 at 44.2% for 0.15 pp.
- In these logs the energy rule produced 290 of 4,800 stop labels, because per-epoch energy was derived from time (section 6).

Earlier drafts of these results reported about 90% energy saved for the original MNIST experiment. That figure came from a labelling rule that only used the final-epoch accuracy, from averaging absolute kWh and from unpaired comparisons. It is replaced by the numbers above.

## 6. Energy measurement

- **Existing logs.** The per-epoch energy was read from CodeCarbon's offline tracker, which only refreshes about every 15 seconds. As a result 68% of MNIST epochs and 65% of FashionMNIST epochs show exactly 0 kWh. The run totals are reliable (about 7 to 8 W on average), so `step2_*.py` re-splits each run's total over its epochs in proportion to epoch time (`repair_legacy_energy` in `common.py`). Consequently, in the existing logs energy is effectively *time × constant power*, and the energy features carry little information beyond time.
- **New logs (`training_logs_v2/`).** `EnergyMeter` in `common.py` integrates `powermetrics` CPU+GPU+ANE package power (1 s samples) over each epoch when Step 1 is started with `sudo` (all 4,800 epochs of the re-run were measured this way, `energy_method = powermetrics_1s`); otherwise it uses CodeCarbon's run total split by time. The method is saved in the `energy_method` column. Package power excludes DRAM, display and the rest of the machine.
- **`energy_measurement/powermetrics_log.txt`.** A short `powermetrics` capture from the laptop (`Mac14,7`), taken on 3 July 2026 at about 23:44, when the extended experiment had just finished: 10 consecutive samples of about 0.1 s each (about 1.1 s in total). Combined CPU+GPU+ANE power was 227 to 684 mW (mean 343 mW), which is an almost idle machine. It documents the sampling tool and the machine; it is not a continuous trace of the training runs.

## 7. Limitations

- **The first-run logs (`training_logs/` here; the first-run four-dataset logs are not included) come from the earlier Step 1 scripts** (kept in `archive/scripts_before_fixes/`). In them: only batch sizes 32 and 128 were used (`configs[::2]` never selected 64 or 256); nothing was seeded; no test accuracy was logged, so the results in 5.2 use **validation** accuracy, which is also what the labels use, and are therefore optimistic.
- **Training and energy numbers come from one laptop.** The re-run (5.1) uses the fixed Step 1 for all four datasets, but each configuration was trained once, so there is no estimate of run-to-run noise in accuracy or energy.
- **20 of the 160 first-run extended runs never learned** (7 CIFAR-10 runs at about 10% accuracy, 13 SVHN runs at 18.7%, the majority class), all at learning rates 0.005 or 0.01. They are excluded from the 5.2 tables (`results_table_all_runs.csv` includes them). The current CNN learning-rate grid is lower for this reason.
- One train/validation split per dataset.
- Energy is package power (CPU+GPU+ANE) or modelled power, not wall-socket power.
- In the re-run the energy label rule never fires on MNIST or FashionMNIST (a run costs about 0.1 Wh), so their stop labels are accuracy-based; it fires on CIFAR-10 and SVHN.
- The stopping point of the learned model is the first epoch at which the classifier says *stop*; no smoothing or minimum number of epochs is applied.
- Only small MLPs and CNNs with Adam and a fixed 30-epoch budget were studied.

## 8. Executable software

No standalone executable is provided. The project is Python source code; the dependencies and commands are given above.
