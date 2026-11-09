"""
STEP 2: Decision Model (original, MNIST only)

What this does
  - Loads the Step 1 training logs (training_logs/all_runs_combined.csv)
  - Labels each epoch "stop" or "continue" from the accuracy still to gain and the energy
    still to spend (see label_run in common.py)
  - Trains a Random Forest to predict that label from signals available during training
  - Replays every run under each stopping method and compares with patience-based early stopping
  - Writes tables and plots to results/

Changes compared with the first run of this script
  - Evaluation is leave-one-ARCHITECTURE-out (the model has never seen that network shape);
    leave-one-run-out is also reported as the optimistic comparison.
  - Results are paired per run: accuracy drop (percentage points) and per-run energy saved (%),
    with 95% bootstrap confidence intervals (not means of absolute kWh).
  - The labelling rule was redefined. The old energy condition compared an accuracy gain with
    kWh and could never fire (0 of 1,200 labels). Now: stop if less than 0.5 pp of accuracy
    is still to come, or if the remaining gain is below 1 pp per Wh still to spend.
  - Patience is swept from 1 to 10 epochs; the best patience within the accuracy budget is
    the baseline the learned model is compared with.
  - Legacy logs: per-epoch energy is re-split from the run total in proportion to epoch time
    (the raw per-epoch values were mostly zero); logs from the new Step 1 are used as they are.
  - Accuracy is the TEST accuracy if the logs contain `test_acc`, otherwise validation accuracy.

How to run (from the project root)
  python step2_decision_model.py
  python step2_decision_model.py --logs training_logs/all_runs_combined.csv --out results

Output (results/)
  results_table.csv, patience_sweep.csv, label_summary.csv, comparison.png,
  feature_importance.png, example_curve.png, energy_accuracy_tradeoff.png
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from common import (FEATURES, NICE_NAMES, SEED, add_features, best_by_criterion, cv_predict,
                    label_run, load_logs, make_pipeline, simulate_all, simulate_patience,
                    simulate_stop, summarise)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--logs", default="training_logs/all_runs_combined.csv")
parser.add_argument("--out", default="results")
parser.add_argument("--acc-tol", type=float, default=0.5)
parser.add_argument("--gain-per-wh", type=float, default=1.0)
parser.add_argument("--max-drop", type=float, default=0.5)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)


def save(name):
    plt.savefig(os.path.join(args.out, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {args.out}/{name}")


# STEP A: load
print("Loading training logs...")
if not os.path.exists(args.logs):
    raise SystemExit(f"ERROR: cannot find {args.logs}. Run step1_baseline_training.py first.")
df, repaired = load_logs(args.logs)
print(f"  {len(df)} rows from {df['run_id'].nunique()} runs, {df.groupby('run_id').size().iloc[0]} epochs each")
if repaired:
    print("  NOTE: legacy log: per-epoch energy re-split from each run's total in proportion to epoch time.")
use_test = "test_acc" in df.columns
ACC_BASIS = "test" if use_test else "validation"
if not use_test:
    print("  WARNING: no test_acc column; accuracy is VALIDATION accuracy.")

# STEP B: labels and features
df = pd.concat([label_run(g, args.acc_tol, args.gain_per_wh) for _, g in df.groupby("run_id")], ignore_index=True)
df = pd.concat([add_features(g) for _, g in df.groupby("run_id")], ignore_index=True)
n_acc = int((df["label_rule"] == "accuracy").sum())
n_en = int((df["label_rule"] == "energy").sum())
print(f"\nLabels: {int(df['label'].sum())} stop ({n_acc} by the accuracy rule, {n_en} by the energy rule), "
      f"{int((df['label'] == 0).sum())} continue")
pd.DataFrame([{"stop_by_accuracy_rule": n_acc, "stop_by_energy_rule": n_en,
               "continue": int((df["label"] == 0).sum())}]).to_csv(os.path.join(args.out, "label_summary.csv"), index=False)

X = df[FEATURES].fillna(0)
y = df["label"]
make_rf = lambda: make_pipeline(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED))

# STEP C: cross-validation
arch_groups = df["hidden_sizes"] if "hidden_sizes" in df.columns else df["run_id"]
print(f"\nRandom Forest, leave-one-architecture-out ({arch_groups.nunique()} folds)...")
pred_arch, _ = cv_predict(X, y, arch_groups, make_rf)
print(f"  classification accuracy {accuracy_score(y, pred_arch):.3f}")
print(f"Random Forest, leave-one-run-out ({df['run_id'].nunique()} folds)...")
pred_run, _ = cv_predict(X, y, df["run_id"], make_rf)
print(f"  classification accuracy {accuracy_score(y, pred_run):.3f}")
df["pred_arch"], df["pred_run"] = pred_arch, pred_run

# STEP D: replay every run
OURS_ARCH = "Energy-Aware RF (Ours, unseen architecture)"
OURS_RUN = "Energy-Aware RF (Ours, leave-one-run-out)"
sim = simulate_all(df, {OURS_ARCH: "pred_arch", OURS_RUN: "pred_run"})
sim = sim[~sim["diverged"]]
patience_names = [f"Patience-{p}" for p in range(1, 11)]
table = summarise(sim, order=["No Stopping"] + patience_names + [OURS_ARCH, OURS_RUN])
best_patience = best_by_criterion(table, patience_names, args.max_drop)
shown = list(dict.fromkeys(["No Stopping", "Patience-3", "Patience-5", "Patience-10", best_patience, OURS_ARCH, OURS_RUN]))
table[table["Method"].isin(shown)].to_csv(os.path.join(args.out, "results_table.csv"), index=False)
table[table["Method"].isin(patience_names)].to_csv(os.path.join(args.out, "patience_sweep.csv"), index=False)

print(f"\n{'=' * 90}\nRESULTS ({ACC_BASIS} accuracy; paired per run; 95% bootstrap CI)\n{'=' * 90}")
cols = ["Method", "Acc Drop (pp, mean)", "Acc Drop CI low", "Acc Drop CI high",
        "Energy Saved (%, mean)", "Energy Saved CI low", "Energy Saved CI high", "Avg Epochs"]
print(table[table["Method"].isin(shown)][cols].round(2).to_string(index=False))
print(f"\n  Strongest patience baseline within {args.max_drop} pp: {best_patience}")

# STEP E: plots
C_OURS, C_BASE, C_NONE = "#2E7D32", "#C0504D", "#9E9E9E"
cmp_methods = ["No Stopping", "Patience-3", "Patience-5", "Patience-10", OURS_ARCH]
t = table.set_index("Method").loc[cmp_methods]
names = ["No stop", "Pat-3", "Pat-5", "Pat-10", "Ours (RF)"]
colors = [C_NONE, C_BASE, C_BASE, C_BASE, C_OURS]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
for ax, (col, lo, hi, ttl) in zip(axes, [
        ("Acc Drop (pp, mean)", "Acc Drop CI low", "Acc Drop CI high", "Accuracy drop (pp, lower is better)"),
        ("Energy Saved (%, mean)", "Energy Saved CI low", "Energy Saved CI high", "Energy saved (%, higher is better)"),
        ("Avg Epochs", None, None, "Epochs used")]):
    v = t[col].to_numpy()
    err = None if lo is None else np.vstack([v - t[lo].to_numpy(), t[hi].to_numpy() - v])
    ax.bar(range(len(v)), v, color=colors, yerr=err, capsize=4)
    ax.set_xticks(range(len(v))); ax.set_xticklabels(names, fontsize=9); ax.set_title(ttl)
fig.suptitle(f"Learned stopping vs patience on MNIST ({ACC_BASIS} accuracy, 95% CI)", fontweight="bold")
plt.tight_layout(); save("comparison.png")

rf_all = make_rf().fit(X, y)
imp = rf_all.named_steps["clf"].feature_importances_
order = np.argsort(imp)
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.barh(range(len(order)), imp[order], color=C_OURS)
ax.set_yticks(range(len(order))); ax.set_yticklabels([NICE_NAMES[FEATURES[i]] for i in order])
ax.set_xlabel("Feature importance"); ax.set_title("Which training signals matter most? (Random Forest on all rows)", fontweight="bold")
plt.tight_layout(); save("feature_importance.png")

example_id = df["run_id"].iloc[0]
ex = df[df["run_id"] == example_id]
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(ex["epoch"], ex["val_loss"], "b-", lw=2.5, label="Validation loss")
ax.plot(ex["epoch"], ex["train_loss"], "b--", alpha=0.4, label="Training loss")
for p, c in [(3, "#E8A09E"), (5, "#D9706C"), (10, "#C0504D")]:
    s = simulate_patience(ex, p)["stop_epoch"]
    ax.axvline(s, c=c, ls=":", label=f"Patience-{p} (epoch {s})")
s = simulate_stop(ex, ex["pred_arch"].to_numpy())["stop_epoch"]
ax.axvline(s, c=C_OURS, lw=2.5, label=f"Energy-aware RF (epoch {s})")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend()
ax.set_title(f"Example run {example_id}: where each method stops", fontweight="bold")
plt.tight_layout(); save("example_curve.png")

fig, ax = plt.subplots(figsize=(9, 6))
for m, c, n in zip(cmp_methods, colors, names):
    s = sim[sim["method"] == m]
    ax.scatter(s["energy_kwh"] * 1000, s["accuracy"], s=40, alpha=0.65, c=c, label=n)
ax.set_xlabel("Energy per run (Wh)"); ax.set_ylabel(f"{ACC_BASIS.capitalize()} accuracy")
ax.set_title("Energy-accuracy trade-off: each point is one training run", fontweight="bold"); ax.legend()
plt.tight_layout(); save("energy_accuracy_tradeoff.png")

print(f"\nDone. See '{args.out}/'.")
