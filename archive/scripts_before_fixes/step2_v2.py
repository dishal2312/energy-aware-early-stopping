"""
STEP 2 (Version 2): Extended Analysis
This is the COMPLETE version that includes everything from V1 plus additions.

What's new in this version:
  - 4 meta-models instead of 1 (Random Forest, Gradient Boosting, Logistic Regression, AdaBoost)
  - Per-dataset results breakdown (MNIST vs FashionMNIST vs CIFAR-10 vs SVHN)
  - Sensitivity analysis (what happens if we change the stop/continue threshold?)
  - Per-fold std dev tracking for statistical reporting
  - Ablation study: with vs without energy features
  - Wilcoxon signed-rank tests for statistical significance
  - More plots

How to run:
  1. Make sure step1_v2.py has finished
  2. Run: python step2_v2.py
  3. Check the results_v2/ folder

Output:
  - results_v2/comparison.png
  - results_v2/feature_importance.png
  - results_v2/per_dataset_comparison.png
  - results_v2/sensitivity_analysis.png
  - results_v2/example_curves.png
  - results_v2/energy_accuracy_tradeoff.png
  - results_v2/results_table.csv          -- includes mean +/- std
  - results_v2/per_dataset_results.csv
  - results_v2/sensitivity_results.csv
  - results_v2/std_dev_results.csv        -- per-fold variability
  - results_v2/ablation_results.csv       -- energy features ablation
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")


# SECTION A: Load Data
print("Loading training logs...")

data_path = "training_logs_v2/all_runs_combined.csv"
if not os.path.exists(data_path):
    print(f"ERROR: Can't find {data_path}")
    print("Make sure you've run step1_v2.py first!")
    exit()

df = pd.read_csv(data_path)
num_runs = df["run_id"].nunique()
print(f"  Loaded {len(df)} data points from {num_runs} runs")

# Show breakdown by dataset
for dataset in df["dataset"].unique():
    count = df[df["dataset"] == dataset]["run_id"].nunique()
    print(f"    {dataset}: {count} runs")


# SECTION B: Label Each Epoch — "Stop" or "Continue"
# Same logic as V1:
#   STOP if less than 0.5% accuracy left to gain
#   STOP if energy cost per accuracy gain is too high
#   Otherwise CONTINUE

print("\nLabelling epochs...")

def label_one_run(run_df, acc_threshold=0.005, efficiency_threshold=0.5):
    """Label each epoch as stop or continue based on energy-accuracy tradeoff."""
    run_df = run_df.copy()
    best_final_acc = run_df["val_acc"].iloc[-1]
    total_energy = run_df["cumulative_energy_kwh"].iloc[-1]

    labels = []
    for _, row in run_df.iterrows():
        remaining_acc = best_final_acc - row["val_acc"]
        remaining_energy = total_energy - row["cumulative_energy_kwh"]

        if remaining_acc < acc_threshold:
            labels.append("stop")
        elif remaining_energy > 0 and (remaining_acc / remaining_energy) < efficiency_threshold:
            labels.append("stop")
        else:
            labels.append("continue")

    run_df["label"] = labels
    return run_df

df = pd.concat([label_one_run(group) for _, group in df.groupby("run_id")], ignore_index=True)

stop_count = (df["label"] == "stop").sum()
continue_count = (df["label"] == "continue").sum()
print(f"  Labels: {continue_count} continue, {stop_count} stop")


# SECTION C: Create Features
print("Creating features...")

def add_features(run_df):
    """Calculate training 'vital signs' for the decision model."""
    run_df = run_df.copy()
    run_df["avg_loss_improvement_3"] = run_df["val_loss_improvement"].rolling(window=3, min_periods=1).mean()
    run_df["avg_acc_improvement_3"] = run_df["val_acc_improvement"].rolling(window=3, min_periods=1).mean()
    run_df["loss_getting_worse"] = (run_df["val_loss_improvement"] < 0).astype(int)

    streaks = []
    streak = 0
    for val in run_df["val_loss_improvement"]:
        if val < 0:
            streak += 1
        else:
            streak = 0
        streaks.append(streak)
    run_df["no_improve_streak"] = streaks

    run_df["train_val_gap"] = run_df["train_loss"] - run_df["val_loss"]
    run_df["progress"] = run_df["epoch"] / run_df["epoch"].max()
    run_df["acc_per_energy"] = run_df["val_acc"] / (run_df["cumulative_energy_kwh"] + 1e-10)
    return run_df

df = pd.concat([add_features(group) for _, group in df.groupby("run_id")], ignore_index=True)

feature_names = [
    "epoch", "train_loss", "val_loss", "val_acc",
    "val_loss_improvement", "val_acc_improvement",
    "avg_loss_improvement_3", "avg_acc_improvement_3",
    "loss_getting_worse", "no_improve_streak",
    "train_val_gap", "progress",
    "epoch_time_secs", "epoch_energy_kwh",
    "cumulative_energy_kwh", "acc_per_energy",
]


# HELPER FUNCTIONS (defined early — used in LOGO loop and Section F)

def simulate_patience_stopping(run_df, patience):
    """Stop if val loss hasn't improved for 'patience' epochs."""
    best_val_loss = float("inf")
    wait = 0
    stop_epoch = run_df["epoch"].max()

    for _, row in run_df.iterrows():
        if row["val_loss"] < best_val_loss:
            best_val_loss = row["val_loss"]
            wait = 0
        else:
            wait += 1
        if wait >= patience:
            stop_epoch = int(row["epoch"])
            break

    stopped = run_df[run_df["epoch"] <= stop_epoch]
    return {
        "stop_epoch": stop_epoch,
        "final_acc": stopped["val_acc"].iloc[-1],
        "energy_used": stopped["cumulative_energy_kwh"].iloc[-1],
    }


def simulate_model_stopping(run_df, pred_column):
    """Stop at the first epoch where the meta-model predicts 'stop'."""
    stop_epoch = run_df["epoch"].max()

    for _, row in run_df.iterrows():
        if row[pred_column] == 1:
            stop_epoch = int(row["epoch"])
            break

    stopped = run_df[run_df["epoch"] <= stop_epoch]
    return {
        "stop_epoch": stop_epoch,
        "final_acc": stopped["val_acc"].iloc[-1],
        "energy_used": stopped["cumulative_energy_kwh"].iloc[-1],
    }


# SECTION D: Train Multiple Meta-Models
# NEW: Instead of just Random Forest, we train 4 different models
# and compare which one makes the best stopping decisions.

print("\nTraining 4 meta-models with Leave-One-Run-Out CV...")

X = df[feature_names].fillna(0)
y = (df["label"] == "stop").astype(int)
run_ids = df["run_id"]

# Define all 4 meta-models
model_definitions = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, max_depth=5, random_state=42
    ),
    "Logistic Regression": LogisticRegression(
        max_iter=1000, random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100, random_state=42
    ),
}

# Store predictions for each meta-model
predictions = {name: np.zeros(len(y)) for name in model_definitions}
probabilities = {name: np.zeros(len(y)) for name in model_definitions}

# NEW: per-fold stopping simulation results (for std dev computation)
# fold_stopping_results[model_name] = list of (final_acc, energy_used) per fold
fold_stopping_results = {name: [] for name in model_definitions}
fold_stopping_results["No Stopping"] = []
for p in [3, 5, 10]:
    fold_stopping_results[f"Patience-{p}"] = []

splitter = LeaveOneGroupOut()
total_folds = len(df["run_id"].unique())
all_run_ids = df["run_id"].unique()
fold_num = 0

for train_idx, test_idx in splitter.split(X, y, run_ids):
    fold_num += 1
    if fold_num % 20 == 0:
        print(f"  Fold {fold_num}/{total_folds}...")

    # Scale features (important for Logistic Regression)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X.iloc[train_idx])
    X_test = scaler.transform(X.iloc[test_idx])
    y_train = y.iloc[train_idx]

    # Get the held-out run for this fold
    held_out_run_id = df.iloc[test_idx]["run_id"].iloc[0]
    run_df_fold = df[df["run_id"] == held_out_run_id]

    # Record No Stopping baseline for this fold
    fold_stopping_results["No Stopping"].append({
        "final_acc": run_df_fold["val_acc"].iloc[-1],
        "energy_used": run_df_fold["cumulative_energy_kwh"].iloc[-1],
        "stop_epoch": int(run_df_fold["epoch"].max()),
    })

    # Record patience baselines for this fold
    for p in [3, 5, 10]:
        result = simulate_patience_stopping(run_df_fold, p)
        fold_stopping_results[f"Patience-{p}"].append(result)

    # Train and predict with each meta-model
    for name, model_template in model_definitions.items():
        model = clone(model_template)
        model.fit(X_train, y_train)
        fold_preds = model.predict(X_test)
        predictions[name][test_idx] = fold_preds
        try:
            probabilities[name][test_idx] = model.predict_proba(X_test)[:, 1]
        except Exception:
            probabilities[name][test_idx] = fold_preds

        # Simulate stopping for this fold using this model's predictions
        # Temporarily assign fold predictions to the run_df slice
        run_df_with_pred = run_df_fold.copy()
        run_df_with_pred["fold_pred"] = fold_preds
        result = simulate_model_stopping(run_df_with_pred, "fold_pred")
        fold_stopping_results[name].append(result)

# Add predictions to dataframe
for name in model_definitions:
    col_name = name.lower().replace(" ", "_") + "_pred"
    df[col_name] = predictions[name]

# Print results for each meta-model
print("\n" + "="*60)
print("META-MODEL CLASSIFICATION RESULTS")
print("="*60)

meta_model_accuracies = {}
for name in model_definitions:
    acc = accuracy_score(y, predictions[name])
    meta_model_accuracies[name] = acc
    print(f"\n  {name}: {acc:.4f} accuracy")
    print(classification_report(y, predictions[name],
          target_names=["Continue", "Stop"], zero_division=0))


# SECTION E: Train Final Models (for feature importance)
scaler_final = StandardScaler()
X_scaled = scaler_final.fit_transform(X)

final_models = {}
for name, model_template in model_definitions.items():
    model = clone(model_template)
    model.fit(X_scaled, y)
    final_models[name] = model


# SECTION F: Compare Against Patience Baselines
print("\nComparing against baselines...")
# (simulate_patience_stopping and simulate_model_stopping are defined above)


# Build methods dictionary
methods = {
    "No Stopping": [],
    "Patience-3": [],
    "Patience-5": [],
    "Patience-10": [],
}
# Add each meta-model as a method
for name in model_definitions:
    methods[f"{name} (Ours)"] = []

# Simulate all methods on all runs
for run_id, run_df in df.groupby("run_id"):
    methods["No Stopping"].append({
        "stop_epoch": run_df["epoch"].max(),
        "final_acc": run_df["val_acc"].iloc[-1],
        "energy_used": run_df["cumulative_energy_kwh"].iloc[-1],
        "dataset": run_df["dataset"].iloc[0],
    })

    for p in [3, 5, 10]:
        result = simulate_patience_stopping(run_df, p)
        result["dataset"] = run_df["dataset"].iloc[0]
        methods[f"Patience-{p}"].append(result)

    for name in model_definitions:
        col = name.lower().replace(" ", "_") + "_pred"
        result = simulate_model_stopping(run_df, col)
        result["dataset"] = run_df["dataset"].iloc[0]
        methods[f"{name} (Ours)"].append(result)


# SECTION G: Print Overall Results Table (with Std Devs)
baseline_energy = np.nanmean([r["energy_used"] for r in methods["No Stopping"]])

print(f"\n{'='*100}")
print("OVERALL RESULTS — All Datasets Combined (mean ± std across 160 LOGO folds)")
print(f"{'='*100}")
print(f"{'Method':<30} {'Accuracy (mean±std)':<22} {'Epochs':<8} {'Energy Saved':<14}")
print(f"{'-'*100}")

results_for_csv = []
for method_name, runs in methods.items():
    avg_acc = np.nanmean([r["final_acc"] for r in runs])
    avg_energy = np.nanmean([r["energy_used"] for r in runs])
    avg_epochs = np.nanmean([r["stop_epoch"] for r in runs])
    energy_saved = ((baseline_energy - avg_energy) / baseline_energy) * 100

    # Std dev from per-fold results (for methods we tracked)
    if method_name in fold_stopping_results and fold_stopping_results[method_name]:
        fold_accs = [r["final_acc"] for r in fold_stopping_results[method_name]]
        fold_epochs = [r["stop_epoch"] for r in fold_stopping_results[method_name]]
        fold_energies = [r["energy_used"] for r in fold_stopping_results[method_name]]
        std_acc = np.nanstd(fold_accs)
        std_epochs = np.nanstd(fold_epochs)
        fold_energy_saved = [((baseline_energy - e) / baseline_energy) * 100
                             for e in fold_energies]
        std_energy_saved = np.nanstd(fold_energy_saved)
    else:
        std_acc = float('nan')
        std_epochs = float('nan')
        std_energy_saved = float('nan')

    acc_str = f"{avg_acc:.3f} ± {std_acc:.3f}" if not np.isnan(std_acc) else f"{avg_acc:.3f}"
    print(f"{method_name:<30} {acc_str:<22} {avg_epochs:>7.1f} {energy_saved:>13.1f}%")

    results_for_csv.append({
        "Method": method_name,
        "Avg Accuracy": round(avg_acc, 3),
        "Std Accuracy": round(std_acc, 3) if not np.isnan(std_acc) else "",
        "Avg Energy (kWh)": avg_energy,
        "Avg Epochs": round(avg_epochs, 1),
        "Std Epochs": round(std_epochs, 1) if not np.isnan(std_epochs) else "",
        "Energy Saved (%)": round(energy_saved, 1),
        "Std Energy Saved (%)": round(std_energy_saved, 1) if not np.isnan(std_energy_saved) else "",
    })

print(f"{'='*100}")

os.makedirs("results_v2", exist_ok=True)
pd.DataFrame(results_for_csv).to_csv("results_v2/results_table.csv", index=False)

# SECTION G2: Statistical Significance Tests
print("\n" + "="*70)
print("STATISTICAL SIGNIFICANCE (Wilcoxon signed-rank test)")
print("Comparing each Ours method vs Patience-5 (strongest baseline)")
print("="*70)

p5_fold_accs = [r["final_acc"] for r in fold_stopping_results["Patience-5"]]
stat_rows = []

for name in model_definitions:
    if fold_stopping_results[name]:
        our_fold_accs = [r["final_acc"] for r in fold_stopping_results[name]]
        # Paired Wilcoxon test (non-parametric, appropriate for small differences)
        try:
            stat, p_val = wilcoxon(our_fold_accs, p5_fold_accs)
            sig = "*" if p_val < 0.05 else "ns"
            print(f"  {name} vs Patience-5: p={p_val:.4f} ({sig})")
        except Exception as e:
            p_val = float('nan')
            sig = "error"
            print(f"  {name} vs Patience-5: {e}")

        stat_rows.append({
            "Comparison": f"{name} vs Patience-5",
            "p_value": round(p_val, 4) if not np.isnan(p_val) else "",
            "Significant (p<0.05)": sig,
        })

pd.DataFrame(stat_rows).to_csv("results_v2/significance_tests.csv", index=False)
print("Saved: results_v2/significance_tests.csv")

# SECTION G3: Ablation Study — Energy Features
print("\n" + "="*70)
print("ABLATION STUDY — With vs Without Energy Features")
print("(Using Random Forest, same LOGO CV)")
print("="*70)

energy_feature_names = ["epoch_time_secs", "epoch_energy_kwh", "cumulative_energy_kwh", "acc_per_energy"]
non_energy_features = [f for f in feature_names if f not in energy_feature_names]

ablation_results = []
for label, feat_set in [("With energy features", feature_names),
                        ("Without energy features", non_energy_features)]:
    X_abl = df[feat_set].fillna(0)
    preds_abl = np.zeros(len(y))
    fold_accs_abl = []

    for train_idx, test_idx in splitter.split(X_abl, y, run_ids):
        scaler_abl = StandardScaler()
        X_tr = scaler_abl.fit_transform(X_abl.iloc[train_idx])
        X_te = scaler_abl.transform(X_abl.iloc[test_idx])
        rf_abl = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_abl.fit(X_tr, y.iloc[train_idx])
        fold_preds_abl = rf_abl.predict(X_te)
        preds_abl[test_idx] = fold_preds_abl

        # Simulate stopping for this fold
        held_run = df.iloc[test_idx]["run_id"].iloc[0]
        run_df_abl = df[df["run_id"] == held_run].copy()
        run_df_abl["abl_pred"] = fold_preds_abl
        res_abl = simulate_model_stopping(run_df_abl, "abl_pred")
        fold_accs_abl.append(res_abl["final_acc"])

    clf_acc = accuracy_score(y, preds_abl)
    avg_stop_acc = np.nanmean(fold_accs_abl)
    std_stop_acc = np.nanstd(fold_accs_abl)
    print(f"  {label}: clf_acc={clf_acc:.3f}, stop_acc={avg_stop_acc:.3f} ± {std_stop_acc:.3f}")
    ablation_results.append({
        "Condition": label,
        "Features Used": len(feat_set),
        "Classification Accuracy": round(clf_acc, 3),
        "Avg Stopping Accuracy (mean)": round(avg_stop_acc, 3),
        "Avg Stopping Accuracy (std)": round(std_stop_acc, 3),
    })

pd.DataFrame(ablation_results).to_csv("results_v2/ablation_results.csv", index=False)
print("Saved: results_v2/ablation_results.csv")


# SECTION H: Per-Dataset Results (NEW)
# Shows how each method performs on each dataset separately.
# This proves your approach generalises across different datasets.

print(f"\n{'='*90}")
print("PER-DATASET RESULTS")
print(f"{'='*90}")

per_dataset_rows = []
datasets_in_data = df["dataset"].unique()

for dataset_name in datasets_in_data:
    print(f"\n  {dataset_name.upper()}")
    ds_baseline_energy = np.nanmean([r["energy_used"] for r in methods["No Stopping"]
                                   if r["dataset"] == dataset_name])

    for method_name, runs in methods.items():
        ds_runs = [r for r in runs if r["dataset"] == dataset_name]
        if not ds_runs:
            continue
        avg_acc = np.nanmean([r["final_acc"] for r in ds_runs])
        avg_energy = np.nanmean([r["energy_used"] for r in ds_runs])
        avg_epochs = np.nanmean([r["stop_epoch"] for r in ds_runs])
        energy_saved = ((ds_baseline_energy - avg_energy) / ds_baseline_energy) * 100 if ds_baseline_energy > 0 else 0

        print(f"    {method_name:<30} Acc: {avg_acc:.4f}  Energy Saved: {energy_saved:.1f}%  Epochs: {avg_epochs:.1f}")

        per_dataset_rows.append({
            "Dataset": dataset_name,
            "Method": method_name,
            "Avg Accuracy": round(avg_acc, 4),
            "Avg Energy (kWh)": avg_energy,
            "Avg Epochs": round(avg_epochs, 1),
            "Energy Saved (%)": round(energy_saved, 1),
        })

pd.DataFrame(per_dataset_rows).to_csv("results_v2/per_dataset_results.csv", index=False)
print(f"\nSaved: results_v2/per_dataset_results.csv")


# SECTION I: Sensitivity Analysis (NEW)
# Tests what happens when we change the accuracy threshold for labelling.
# This shows the approach is robust and not dependent on one magic number.

print(f"\n{'='*90}")
print("SENSITIVITY ANALYSIS — Varying the stop/continue threshold")
print(f"{'='*90}")

thresholds = [0.001, 0.005, 0.01, 0.02]  # 0.1%, 0.5%, 1.0%, 2.0%
threshold_labels = ["0.1%", "0.5% (default)", "1.0%", "2.0%"]
sensitivity_rows = []

for thresh, thresh_label in zip(thresholds, threshold_labels):
    print(f"\n  Testing threshold: {thresh_label} ({thresh})...")

    # Re-label with this threshold
    df_temp = pd.concat([label_one_run(group, acc_threshold=thresh)
                         for _, group in df.groupby("run_id")], ignore_index=True)
    df_temp = pd.concat([add_features(group)
                         for _, group in df_temp.groupby("run_id")], ignore_index=True)

    X_temp = df_temp[feature_names].fillna(0)
    y_temp = (df_temp["label"] == "stop").astype(int)
    run_ids_temp = df_temp["run_id"]

    # Train only Random Forest for sensitivity analysis (keeps it manageable)
    preds = np.zeros(len(y_temp))
    splitter_temp = LeaveOneGroupOut()

    for train_idx, test_idx in splitter_temp.split(X_temp, y_temp, run_ids_temp):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_temp.iloc[train_idx])
        X_test = scaler.transform(X_temp.iloc[test_idx])

        rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf.fit(X_train, y_temp.iloc[train_idx])
        preds[test_idx] = rf.predict(X_test)

    df_temp["rf_pred"] = preds

    # Simulate stopping
    rf_results = []
    no_stop_results = []
    for run_id, run_df in df_temp.groupby("run_id"):
        no_stop_results.append({
            "energy_used": run_df["cumulative_energy_kwh"].iloc[-1],
            "final_acc": run_df["val_acc"].iloc[-1],
        })
        rf_results.append(simulate_model_stopping(run_df, "rf_pred"))

    base_energy = np.nanmean([r["energy_used"] for r in no_stop_results])
    avg_acc = np.nanmean([r["final_acc"] for r in rf_results])
    avg_energy = np.nanmean([r["energy_used"] for r in rf_results])
    avg_epochs = np.nanmean([r["stop_epoch"] for r in rf_results])
    energy_saved = ((base_energy - avg_energy) / base_energy) * 100

    print(f"    Accuracy: {avg_acc:.4f} | Energy Saved: {energy_saved:.1f}% | Avg Epochs: {avg_epochs:.1f}")

    sensitivity_rows.append({
        "Threshold": thresh_label,
        "Threshold Value": thresh,
        "Avg Accuracy": round(avg_acc, 4),
        "Avg Epochs": round(avg_epochs, 1),
        "Energy Saved (%)": round(energy_saved, 1),
    })

pd.DataFrame(sensitivity_rows).to_csv("results_v2/sensitivity_results.csv", index=False)
print(f"\nSaved: results_v2/sensitivity_results.csv")


# SECTION J: Generate All Plots
print("\nGenerating plots...")

# Identify the best meta-model (for plots that need a single "ours" method)
best_meta_model = max(meta_model_accuracies, key=meta_model_accuracies.get)
best_pred_col = best_meta_model.lower().replace(" ", "_") + "_pred"
print(f"  Best meta-model: {best_meta_model}")


# Plot 0a: Pipeline Diagram
fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(0, 5)
ax.axis("off")

# Box style
box_style = dict(boxstyle="round,pad=0.5", facecolor="#E3F2FD", edgecolor="#1565C0", linewidth=2)
arrow_style = dict(arrowstyle="-|>", color="#1565C0", lw=2)
title_style = dict(boxstyle="round,pad=0.4", facecolor="#1565C0", edgecolor="#1565C0")

# Stage 1 boxes
stage1_items = [
    (1.0, "160 Training Runs\n(MNIST, FashionMNIST,\nCIFAR-10, SVHN)"),
    (3.5, "Per-Epoch Logging\n(loss, accuracy,\nenergy via CodeCarbon)"),
    (6.0, "Training Logs CSV\n(4,800 epoch records\nacross 160 runs)"),
]
for x, label in stage1_items:
    ax.text(x, 2.5, label, ha="center", va="center", fontsize=8.5,
            bbox=box_style, transform=ax.transData)

# Stage 2 boxes
stage2_items = [
    (8.5, "Retrospective\nLabelling\n(stop / continue)"),
    (10.5, "Feature\nEngineering\n(16 features)"),
    (12.5, "Meta-Model\nTraining\n(LOGO-CV)"),
]
box_style2 = dict(boxstyle="round,pad=0.5", facecolor="#E8F5E9", edgecolor="#2E7D32", linewidth=2)
for x, label in stage2_items:
    ax.text(x, 2.5, label, ha="center", va="center", fontsize=8.5,
            bbox=box_style2, transform=ax.transData)

# Arrows between Stage 1 boxes
for x_start, x_end in [(1.75, 2.65), (4.25, 5.15), (6.8, 7.7)]:
    ax.annotate("", xy=(x_end, 2.5), xytext=(x_start, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#1565C0", lw=2))

# Arrows within Stage 2
for x_start, x_end in [(9.3, 9.8), (11.3, 11.8)]:
    ax.annotate("", xy=(x_end, 2.5), xytext=(x_start, 2.5),
                arrowprops=dict(arrowstyle="-|>", color="#2E7D32", lw=2))

# Stage labels at top
ax.text(3.5, 4.3, "STAGE 1 — Data Collection", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#1565C0")
ax.text(10.5, 4.3, "STAGE 2 — Meta-Model Training", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#2E7D32")
ax.axvline(x=7.35, color="grey", linestyle="--", alpha=0.5)

# Evaluation box at bottom
eval_box = dict(boxstyle="round,pad=0.5", facecolor="#FFF3E0", edgecolor="#E65100", linewidth=2)
ax.text(7.0, 0.8, "Evaluation: Simulate Stopping → Compare vs Patience Baselines\n"
        "(Accuracy retained, Energy saved, Wilcoxon significance test)",
        ha="center", va="center", fontsize=9, bbox=eval_box)
ax.annotate("", xy=(7.0, 1.3), xytext=(12.5, 1.9),
            arrowprops=dict(arrowstyle="-|>", color="#E65100", lw=2, connectionstyle="arc3,rad=0.3"))

plt.title("System Pipeline: Energy-Aware Early Stopping Framework",
          fontsize=13, fontweight="bold", pad=15)
plt.tight_layout()
plt.savefig("results_v2/pipeline_diagram.png", dpi=200, bbox_inches="tight")
plt.close()
print("  Saved: results_v2/pipeline_diagram.png")


# Plot 0b: Confusion Matrices (all 4 meta-models)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()
cm_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

for ax, (name, color) in zip(axes, zip(model_definitions.keys(), cm_colors)):
    pred_col = name.lower().replace(" ", "_") + "_pred"
    cm = confusion_matrix(y, df[pred_col])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=["Continue", "Stop"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\n(Overall accuracy: {meta_model_accuracies[name]:.3f})",
                 fontsize=11, fontweight="bold", color=color)

fig.suptitle("Confusion Matrices: Meta-Model Stop/Continue Predictions",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results_v2/confusion_matrices.png", dpi=150, bbox_inches="tight")
plt.close()
print("  Saved: results_v2/confusion_matrices.png")


# Plot 1: Feature Importance
fig, ax = plt.subplots(figsize=(10, 7))

# Use Random Forest for feature importance (it's the most interpretable)
rf_importances = final_models["Random Forest"].feature_importances_
sorted_idx = np.argsort(rf_importances)

nice_names = {
    "epoch": "Epoch number", "train_loss": "Training loss",
    "val_loss": "Validation loss", "val_acc": "Validation accuracy",
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
ax.barh(range(len(sorted_idx)), rf_importances[sorted_idx], color="#4CAF50")
ax.set_yticks(range(len(sorted_idx)))
ax.set_yticklabels(labels)
ax.set_xlabel("Feature Importance")
ax.set_title("Which Training Signals Matter Most? (Random Forest)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig("results_v2/feature_importance.png", dpi=150, bbox_inches="tight")
print("  Saved: results_v2/feature_importance.png")


# Plot 2: Overall Comparison (best meta-model vs patience baselines)
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

compare_methods = ["No Stopping", "Patience-3", "Patience-5", "Patience-10",
                   f"{best_meta_model} (Ours)"]
colors = ["#cccccc", "#ff9999", "#ff7777", "#ff5555", "#4CAF50"]

avg_accs = [np.nanmean([r["final_acc"] for r in methods[m]]) for m in compare_methods]
avg_energies = [np.nanmean([r["energy_used"] for r in methods[m]]) for m in compare_methods]
avg_epochs_vals = [np.nanmean([r["stop_epoch"] for r in methods[m]]) for m in compare_methods]

short_names = ["No Stop", "Pat-3", "Pat-5", "Pat-10", "Ours"]

for ax_idx, (values, ylabel, title) in enumerate([
    (avg_accs, "Avg Validation Accuracy", "Accuracy"),
    (avg_energies, "Avg Energy (kWh)", "Energy Consumed"),
    (avg_epochs_vals, "Avg Epochs Used", "Training Length"),
]):
    axes[ax_idx].bar(range(len(short_names)), values, color=colors)
    axes[ax_idx].set_xticks(range(len(short_names)))
    axes[ax_idx].set_xticklabels(short_names, fontsize=9)
    axes[ax_idx].set_ylabel(ylabel)
    axes[ax_idx].set_title(title)

fig.suptitle("Energy-Aware Early Stopping vs Patience-Based Methods (All Datasets)",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results_v2/comparison.png", dpi=150, bbox_inches="tight")
print("  Saved: results_v2/comparison.png")


# Plot 3: Per-Dataset Comparison (NEW)
fig, axes = plt.subplots(1, len(datasets_in_data), figsize=(6 * len(datasets_in_data), 6))
if len(datasets_in_data) == 1:
    axes = [axes]

for ax, dataset_name in zip(axes, datasets_in_data):
    method_names_short = []
    energy_saved_values = []
    acc_values = []

    ds_baseline_energy = np.nanmean([r["energy_used"] for r in methods["No Stopping"]
                                   if r["dataset"] == dataset_name])

    for method_name in compare_methods:
        ds_runs = [r for r in methods[method_name] if r["dataset"] == dataset_name]
        if not ds_runs:
            continue
        avg_energy = np.nanmean([r["energy_used"] for r in ds_runs])
        energy_saved = ((ds_baseline_energy - avg_energy) / ds_baseline_energy) * 100

        method_names_short.append(method_name.replace(" (Ours)", "\n(Ours)"))
        energy_saved_values.append(energy_saved)

    bar_colors = colors[:len(method_names_short)]
    ax.bar(range(len(method_names_short)), energy_saved_values, color=bar_colors)
    ax.set_xticks(range(len(method_names_short)))
    ax.set_xticklabels(method_names_short, fontsize=7)
    ax.set_ylabel("Energy Saved (%)")
    ax.set_title(f"{dataset_name.upper()}", fontsize=12, fontweight="bold")

fig.suptitle("Energy Savings by Dataset", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results_v2/per_dataset_comparison.png", dpi=150, bbox_inches="tight")
print("  Saved: results_v2/per_dataset_comparison.png")


# Plot 4: Meta-Model Comparison (NEW)
fig, ax = plt.subplots(figsize=(10, 6))

meta_names = list(model_definitions.keys())
meta_accs = [meta_model_accuracies[n] for n in meta_names]
meta_colors = ["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"]

bars = ax.bar(range(len(meta_names)), meta_accs, color=meta_colors)
ax.set_xticks(range(len(meta_names)))
ax.set_xticklabels(meta_names, fontsize=10)
ax.set_ylabel("Classification Accuracy")
ax.set_title("Meta-Model Comparison: Which Model Best Predicts When to Stop?",
             fontsize=13, fontweight="bold")
ax.set_ylim(min(meta_accs) - 0.05, 1.0)

# Add accuracy labels on bars
for bar, acc in zip(bars, meta_accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
            f"{acc:.3f}", ha="center", fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("results_v2/meta_model_comparison.png", dpi=150, bbox_inches="tight")
print("  Saved: results_v2/meta_model_comparison.png")


# Plot 5: Sensitivity Analysis (NEW)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

thresh_labels = [r["Threshold"] for r in sensitivity_rows]
thresh_accs = [r["Avg Accuracy"] for r in sensitivity_rows]
thresh_energy = [r["Energy Saved (%)"] for r in sensitivity_rows]

axes[0].bar(range(len(thresh_labels)), thresh_accs, color="#4CAF50")
axes[0].set_xticks(range(len(thresh_labels)))
axes[0].set_xticklabels(thresh_labels, fontsize=9)
axes[0].set_ylabel("Avg Accuracy")
axes[0].set_title("Accuracy at Different Thresholds")

axes[1].bar(range(len(thresh_labels)), thresh_energy, color="#2196F3")
axes[1].set_xticks(range(len(thresh_labels)))
axes[1].set_xticklabels(thresh_labels, fontsize=9)
axes[1].set_ylabel("Energy Saved (%)")
axes[1].set_title("Energy Savings at Different Thresholds")

fig.suptitle("Sensitivity Analysis: How Does the Threshold Affect Results?",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results_v2/sensitivity_analysis.png", dpi=150, bbox_inches="tight")
print("  Saved: results_v2/sensitivity_analysis.png")


# Plot 6: Example Curves from Each Dataset (NEW)
num_datasets = len(datasets_in_data)
fig, axes = plt.subplots(1, num_datasets, figsize=(6 * num_datasets, 5))
if num_datasets == 1:
    axes = [axes]

for ax, dataset_name in zip(axes, datasets_in_data):
    ds_runs = df[df["dataset"] == dataset_name]
    example_run_id = ds_runs["run_id"].unique()[0]
    example = ds_runs[ds_runs["run_id"] == example_run_id]

    ax.plot(example["epoch"], example["val_loss"], "b-", linewidth=2, label="Val Loss")
    ax.plot(example["epoch"], example["train_loss"], "b--", alpha=0.4, label="Train Loss")

    # Patience-3 stop point
    p3_result = simulate_patience_stopping(example, 3)
    ax.axvline(x=p3_result["stop_epoch"], color="#ff9999", linestyle=":", linewidth=1.5,
               label=f"Patience-3 (ep {p3_result['stop_epoch']})")

    # Our model stop point
    rf_result = simulate_model_stopping(example, best_pred_col)
    ax.axvline(x=rf_result["stop_epoch"], color="#4CAF50", linewidth=2.5,
               label=f"Ours (ep {rf_result['stop_epoch']})")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{dataset_name.upper()} — Run {example_run_id}", fontweight="bold")
    ax.legend(fontsize=7)

fig.suptitle("Example Learning Curves with Predicted Stop Points", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results_v2/example_curves.png", dpi=150, bbox_inches="tight")
print("  Saved: results_v2/example_curves.png")


# Plot 7: Energy-Accuracy Scatter
fig, ax = plt.subplots(figsize=(10, 7))

scatter_methods = ["No Stopping", "Patience-3", "Patience-10", f"{best_meta_model} (Ours)"]
scatter_colors = ["#cccccc", "#ff9999", "#ff5555", "#4CAF50"]

for method_name, color in zip(scatter_methods, scatter_colors):
    runs = methods[method_name]
    energies = [r["energy_used"] for r in runs]
    accs = [r["final_acc"] for r in runs]
    ax.scatter(energies, accs, label=method_name, alpha=0.6, s=50, color=color)

ax.set_xlabel("Energy Used (kWh)")
ax.set_ylabel("Final Validation Accuracy")
ax.set_title("Energy-Accuracy Tradeoff", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("results_v2/energy_accuracy_tradeoff.png", dpi=150, bbox_inches="tight")
print("  Saved: results_v2/energy_accuracy_tradeoff.png")


# DONE
print(f"\n{'='*60}")
print("  ✅ ALL DONE!")
print("  Check the 'results_v2/' folder for all plots and tables.")
print(f"")
print("  Files generated:")
print("    results_v2/results_table.csv             — overall comparison with mean±std")
print("    results_v2/per_dataset_results.csv        — per-dataset breakdown")
print("    results_v2/sensitivity_results.csv        — threshold sensitivity")
print("    results_v2/significance_tests.csv         — Wilcoxon p-values")
print("    results_v2/ablation_results.csv           — energy feature ablation")
print("    results_v2/pipeline_diagram.png           — system pipeline flowchart")
print("    results_v2/confusion_matrices.png         — stop/continue confusion matrices")
print("    results_v2/feature_importance.png         — which signals matter")
print("    results_v2/comparison.png                 — main comparison chart")
print("    results_v2/per_dataset_comparison.png     — per-dataset chart")
print("    results_v2/meta_model_comparison.png      — meta-model comparison")
print("    results_v2/sensitivity_analysis.png       — threshold sensitivity")
print("    results_v2/example_curves.png             — example learning curves")
print("    results_v2/energy_accuracy_tradeoff.png   — scatter plot")
print(f"{'='*60}")
