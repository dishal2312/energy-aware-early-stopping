"""
STEP 2: Decision Model
What this does:
  - Loads training logs from Step 1
  - Labels each epoch as "stop" or "continue" based on energy-accuracy tradeoff
  - Trains a Random Forest to predict when to stop
  - Compares your model against patience-based early stopping
  - Generates plots and a results table for your dissertation

How to run:
  1. Make sure Step 1 has finished (check training_logs/ folder)
  2. Run this script:  python step2_decision_model.py
  3. Takes about 5-30 seconds

Output:
  - results/comparison.png       (bar chart comparing methods)
  - results/feature_importance.png (what signals matter most)
  - results/example_curve.png    (example learning curve with stop points)
  - results/results_table.csv    (numbers for your dissertation table)
  - Printed results table in the terminal
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")


# STEP A: Load the training logs
print("Loading training logs...")

data_path = "training_logs/all_runs_combined.csv"
if not os.path.exists(data_path):
    print(f"ERROR: Can't find {data_path}")
    print("Make sure you've run step1_baseline_training.py first!")
    exit()

df = pd.read_csv(data_path)
num_runs = df["run_id"].nunique()
print(f"  Loaded {len(df)} data points from {num_runs} runs")
print(f"  Epochs per run: {df.groupby('run_id').size().iloc[0]}")


# STEP B: Label each epoch — "stop" or "continue"
# For each epoch in each run, we ask:
#   "From this point, is the remaining accuracy gain worth the remaining energy cost?"
#
# We say STOP if:
#   - Less than 0.5% accuracy is left to gain, OR
#   - The energy cost per percentage point of accuracy is very high
# Otherwise we say CONTINUE

print("\nLabelling epochs...")

def label_one_run(run_df):
    """Look at one run and label each epoch as stop or continue."""
    run_df = run_df.copy()

    # What's the best accuracy this run eventually achieves?
    best_final_acc = run_df["val_acc"].iloc[-1]
    total_energy = run_df["cumulative_energy_kwh"].iloc[-1]

    labels = []
    for _, row in run_df.iterrows():
        remaining_acc = best_final_acc - row["val_acc"]       # How much accuracy is left to gain
        remaining_energy = total_energy - row["cumulative_energy_kwh"]  # How much energy left to spend

        if remaining_acc < 0.005:
            # Less than 0.5% accuracy left — not worth continuing
            labels.append("stop")
        elif remaining_energy > 0 and (remaining_acc / remaining_energy) < 0.5:
            # Very little accuracy gain per unit of energy — inefficient
            labels.append("stop")
        else:
            labels.append("continue")

    run_df["label"] = labels
    return run_df

df = pd.concat([label_one_run(group) for _, group in df.groupby("run_id")], ignore_index=True)

stop_count = (df["label"] == "stop").sum()
continue_count = (df["label"] == "continue").sum()
print(f"  Labels: {continue_count} continue, {stop_count} stop")


# STEP C: Create features for the decision model
# These are the signals the Random Forest uses to decide stop or continue.
# Each feature is something you could observe during training in real-time.

print("Creating features...")

def add_features(run_df):
    """Calculate extra features for one run."""
    run_df = run_df.copy()

    # Average improvement over the last 3 epochs (smoothed trend)
    run_df["avg_loss_improvement_3"] = (
        run_df["val_loss_improvement"].rolling(window=3, min_periods=1).mean()
    )
    run_df["avg_acc_improvement_3"] = (
        run_df["val_acc_improvement"].rolling(window=3, min_periods=1).mean()
    )

    # Is validation loss going UP? (this suggests overfitting)
    run_df["loss_getting_worse"] = (run_df["val_loss_improvement"] < 0).astype(int)

    # How many epochs in a row has val loss been getting worse?
    streaks = []
    streak = 0
    for val in run_df["val_loss_improvement"]:
        if val < 0:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    run_df["no_improve_streak"] = streaks

    # Gap between training and validation loss (big gap = overfitting)
    run_df["train_val_gap"] = run_df["train_loss"] - run_df["val_loss"]

    # What fraction of training is done? (0.0 = start, 1.0 = end)
    run_df["progress"] = run_df["epoch"] / run_df["epoch"].max()

    # Energy efficiency so far: accuracy gained per kWh spent
    run_df["acc_per_energy"] = run_df["val_acc"] / (run_df["cumulative_energy_kwh"] + 1e-10)

    return run_df

df = pd.concat([add_features(group) for _, group in df.groupby("run_id")], ignore_index=True)

# These are all the columns the model will use as inputs
feature_names = [
    "epoch",                    # How far into training we are
    "train_loss",               # Current training loss
    "val_loss",                 # Current validation loss
    "val_acc",                  # Current validation accuracy
    "val_loss_improvement",     # Did val loss go down this epoch?
    "val_acc_improvement",      # Did accuracy go up this epoch?
    "avg_loss_improvement_3",   # Average loss improvement over last 3 epochs
    "avg_acc_improvement_3",    # Average accuracy improvement over last 3 epochs
    "loss_getting_worse",       # Is val loss going up? (0 or 1)
    "no_improve_streak",        # How many epochs with no improvement
    "train_val_gap",            # Gap between train and val loss
    "progress",                 # How far through training (0 to 1)
    "epoch_time_secs",          # How long this epoch took
    "epoch_energy_kwh",         # Energy used this epoch
    "cumulative_energy_kwh",    # Total energy used so far
    "acc_per_energy",           # Accuracy per unit of energy
]


# STEP D: Train the Random Forest
# We use Leave-One-Run-Out cross-validation:
#   - For each run, train on ALL OTHER runs, test on that one
#   - This proves the model works on unseen training runs
#   - This is important for your dissertation!

print("\nTraining Random Forest (Leave-One-Run-Out CV)...")

X = df[feature_names].fillna(0)
y = (df["label"] == "stop").astype(int)  # 1 = stop, 0 = continue
run_ids = df["run_id"]

# Store predictions for every row
all_predictions = np.zeros(len(y))
all_probabilities = np.zeros(len(y))

splitter = LeaveOneGroupOut()
total_folds = len(df["run_id"].unique())
fold_num = 0

for train_idx, test_idx in splitter.split(X, y, run_ids):
    fold_num += 1
    if fold_num % 10 == 0:
        print(f"  Fold {fold_num}/{total_folds}...")

    # Train a Random Forest on all runs except one
    rf = RandomForestClassifier(
        n_estimators=100,   # 100 decision trees
        max_depth=10,       # Don't let trees get too deep (prevents overfitting)
        random_state=42,    # For reproducibility
    )
    rf.fit(X.iloc[train_idx], y.iloc[train_idx])

    # Predict on the held-out run
    all_predictions[test_idx] = rf.predict(X.iloc[test_idx])
    all_probabilities[test_idx] = rf.predict_proba(X.iloc[test_idx])[:, 1]

df["rf_prediction"] = all_predictions
df["rf_stop_probability"] = all_probabilities

# Print how well the model did
print(f"\n  Overall Accuracy: {accuracy_score(y, all_predictions):.4f}")
print("\n  Classification Report:")
print(classification_report(y, all_predictions, target_names=["Continue", "Stop"]))


# STEP E: Train a final model (for feature importance chart)
final_rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
final_rf.fit(X, y)


# STEP F: Compare against patience-based baselines
print("Comparing against baselines...")

def simulate_patience_stopping(run_df, patience):
    """
    Simulate patience-based early stopping:
    Stop if validation loss hasn't improved for 'patience' epochs in a row.
    Returns the epoch it would have stopped at and the results.
    """
    best_val_loss = float("inf")
    wait = 0
    stop_epoch = run_df["epoch"].max()  # Default: train to the end

    for _, row in run_df.iterrows():
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            wait = 0
        else:
            wait += 1

        if wait >= patience:
            stop_epoch = int(row["epoch"])
            break

    stopped_data = run_df[run_df["epoch"] <= stop_epoch]
    return {
        "stop_epoch": stop_epoch,
        "final_acc": stopped_data["val_acc"].iloc[-1],
        "energy_used": stopped_data["cumulative_energy_kwh"].iloc[-1],
    }


def simulate_rf_stopping(run_df):
    """
    Simulate stopping using the Random Forest predictions:
    Stop at the first epoch where the model predicts "stop".
    """
    stop_epoch = run_df["epoch"].max()

    for _, row in run_df.iterrows():
        if row["rf_prediction"] == 1:
            stop_epoch = int(row["epoch"])
            break

    stopped_data = run_df[run_df["epoch"] <= stop_epoch]
    return {
        "stop_epoch": stop_epoch,
        "final_acc": stopped_data["val_acc"].iloc[-1],
        "energy_used": stopped_data["cumulative_energy_kwh"].iloc[-1],
    }


# Run simulations for each method
methods = {
    "No Stopping": [],
    "Patience-3": [],
    "Patience-5": [],
    "Patience-10": [],
    "Energy-Aware RF (Ours)": [],
}

for run_id, run_df in df.groupby("run_id"):
    # No stopping — always train to the end
    methods["No Stopping"].append({
        "stop_epoch": run_df["epoch"].max(),
        "final_acc": run_df["val_acc"].iloc[-1],
        "energy_used": run_df["cumulative_energy_kwh"].iloc[-1],
    })

    # Patience baselines
    methods["Patience-3"].append(simulate_patience_stopping(run_df, 3))
    methods["Patience-5"].append(simulate_patience_stopping(run_df, 5))
    methods["Patience-10"].append(simulate_patience_stopping(run_df, 10))

    # Your model
    methods["Energy-Aware RF (Ours)"].append(simulate_rf_stopping(run_df))


# STEP G: Print results table
baseline_energy = np.mean([r["energy_used"] for r in methods["No Stopping"]])

print(f"\n{'='*80}")
print("RESULTS COMPARISON — This goes in your dissertation!")
print(f"{'='*80}")
print(f"{'Method':<24} {'Avg Accuracy':>14} {'Avg Energy (kWh)':>18} "
      f"{'Avg Epochs':>12} {'Energy Saved':>14}")
print(f"{'-'*80}")

results_for_csv = []

for method_name, runs in methods.items():
    avg_acc = np.mean([r["final_acc"] for r in runs])
    avg_energy = np.mean([r["energy_used"] for r in runs])
    avg_epochs = np.mean([r["stop_epoch"] for r in runs])
    energy_saved_pct = ((baseline_energy - avg_energy) / baseline_energy) * 100 if baseline_energy > 0 else 0
    acc_drop = np.mean([r["final_acc"] for r in methods["No Stopping"]]) - avg_acc

    print(f"{method_name:<24} {avg_acc:>13.4f} {avg_energy:>17.8f} "
          f"{avg_epochs:>11.1f} {energy_saved_pct:>13.1f}%")

    results_for_csv.append({
        "Method": method_name,
        "Avg Accuracy": round(avg_acc, 4),
        "Avg Energy (kWh)": avg_energy,
        "Avg Epochs": round(avg_epochs, 1),
        "Energy Saved (%)": round(energy_saved_pct, 1),
        "Accuracy Drop": round(acc_drop, 4),
    })

print(f"{'='*80}")

# Save results table as CSV
os.makedirs("results", exist_ok=True)
pd.DataFrame(results_for_csv).to_csv("results/results_table.csv", index=False)
print("\nSaved: results/results_table.csv")


# STEP H: Generate plots for your dissertation
print("\nGenerating plots...")

# Plot 1: Feature Importance
fig, ax = plt.subplots(figsize=(10, 7))

importances = final_rf.feature_importances_
sorted_idx = np.argsort(importances)

# Use nicer names for the chart
nice_names = {
    "epoch": "Epoch number",
    "train_loss": "Training loss",
    "val_loss": "Validation loss",
    "val_acc": "Validation accuracy",
    "val_loss_improvement": "Loss improvement (this epoch)",
    "val_acc_improvement": "Accuracy improvement (this epoch)",
    "avg_loss_improvement_3": "Avg loss improvement (3 epochs)",
    "avg_acc_improvement_3": "Avg accuracy improvement (3 epochs)",
    "loss_getting_worse": "Is loss increasing?",
    "no_improve_streak": "No-improvement streak",
    "train_val_gap": "Train-val loss gap",
    "progress": "Training progress (%)",
    "epoch_time_secs": "Time per epoch",
    "epoch_energy_kwh": "Energy per epoch",
    "cumulative_energy_kwh": "Cumulative energy",
    "acc_per_energy": "Accuracy per unit energy",
}

labels = [nice_names.get(feature_names[i], feature_names[i]) for i in sorted_idx]
ax.barh(range(len(sorted_idx)), importances[sorted_idx], color="#4CAF50")
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(labels)
ax.set_xlabel("Feature Importance (higher = more important)")
ax.set_title("Which Training Signals Matter Most for the Stop/Continue Decision?",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results/feature_importance.png", dpi=150, bbox_inches="tight")
print("  Saved: results/feature_importance.png")


# Plot 2: Comparison Bar Charts
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

method_names = list(methods.keys())
colors = ["#cccccc", "#ff9999", "#ff7777", "#ff5555", "#4CAF50"]

avg_accs = [np.mean([r["final_acc"] for r in methods[m]]) for m in method_names]
avg_energies = [np.mean([r["energy_used"] for r in methods[m]]) for m in method_names]
avg_epochs_vals = [np.mean([r["stop_epoch"] for r in methods[m]]) for m in method_names]

# Accuracy comparison
axes[0].bar(range(len(method_names)), avg_accs, color=colors)
axes[0].set_xticks(range(len(method_names)))
axes[0].set_xticklabels(method_names, rotation=35, ha="right", fontsize=8)
axes[0].set_ylabel("Avg Validation Accuracy")
axes[0].set_title("Accuracy")

# Energy comparison
axes[1].bar(range(len(method_names)), avg_energies, color=colors)
axes[1].set_xticks(range(len(method_names)))
axes[1].set_xticklabels(method_names, rotation=35, ha="right", fontsize=8)
axes[1].set_ylabel("Avg Energy (kWh)")
axes[1].set_title("Energy Consumed")

# Epochs comparison
axes[2].bar(range(len(method_names)), avg_epochs_vals, color=colors)
axes[2].set_xticks(range(len(method_names)))
axes[2].set_xticklabels(method_names, rotation=35, ha="right", fontsize=8)
axes[2].set_ylabel("Avg Epochs Used")
axes[2].set_title("Training Length")

fig.suptitle("Energy-Aware Early Stopping vs Patience-Based Methods",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/comparison.png", dpi=150, bbox_inches="tight")
print("  Saved: results/comparison.png")


# Plot 3: Example Learning Curve with Stop Points
fig, ax = plt.subplots(figsize=(11, 6))

# Pick the first run as an example
example_run_id = df["run_id"].unique()[0]
example = df[df["run_id"] == example_run_id].copy()

# Plot the loss curves
ax.plot(example["epoch"], example["val_loss"], "b-", linewidth=2.5, label="Validation Loss")
ax.plot(example["epoch"], example["train_loss"], "b--", alpha=0.4, label="Training Loss")

# Mark where each method would stop
patience_colors = {"Patience-3": "#ff9999", "Patience-5": "#ff7777", "Patience-10": "#ff5555"}
for method_name, color in patience_colors.items():
    patience_val = int(method_name.split("-")[1])
    result = simulate_patience_stopping(example, patience_val)
    ax.axvline(x=result["stop_epoch"], color=color, linestyle=":", linewidth=1.5,
               label=f"{method_name} (epoch {result['stop_epoch']})")

# Mark where your model would stop
rf_result = simulate_rf_stopping(example)
ax.axvline(x=rf_result["stop_epoch"], color="#4CAF50", linewidth=2.5,
           label=f"Energy-Aware RF (epoch {rf_result['stop_epoch']})")

ax.set_xlabel("Epoch", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title(f"Example Run: Where Each Method Would Stop Training (Run {example_run_id})",
             fontsize=13, fontweight="bold")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig("results/example_curve.png", dpi=150, bbox_inches="tight")
print("  Saved: results/example_curve.png")


# Plot 4: Energy-Accuracy Tradeoff Scatter
fig, ax = plt.subplots(figsize=(10, 7))

for method_name, runs in methods.items():
    energies = [r["energy_used"] for r in runs]
    accs = [r["final_acc"] for r in runs]
    color = colors[list(methods.keys()).index(method_name)]
    ax.scatter(energies, accs, label=method_name, alpha=0.6, s=50, color=color)

ax.set_xlabel("Energy Used (kWh)", fontsize=12)
ax.set_ylabel("Final Validation Accuracy", fontsize=12)
ax.set_title("Energy-Accuracy Tradeoff: Each Point is One Training Run",
             fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("results/energy_accuracy_tradeoff.png", dpi=150, bbox_inches="tight")
print("  Saved: results/energy_accuracy_tradeoff.png")


print(f"\n{'='*60}")
print("  ✅ ALL DONE!")
print("  Check the 'results/' folder for all your plots and tables.")
print("  These go straight into your dissertation.")
print(f"{'='*60}")
