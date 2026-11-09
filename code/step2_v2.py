"""
STEP 2 (Version 2): Energy-aware stopping model, extended analysis

Reads the Step 1 logs, labels every epoch "stop" or "continue", trains four meta-models to
predict that label from signals available during training, replays every run under each
stopping method, and compares them with patience-based early stopping.

How the evaluation works
  - Two cross-validation schemes, both leave-one-group-out:
      LODO  leave-one-DATASET-out (primary): the meta-model never saw the dataset it is
            judged on. This is the test of generalisation.
      LORO  leave-one-RUN-out (secondary): other runs of the same dataset are in training,
            so this is an optimistic, in-distribution number.
  - Results are PAIRED per run: accuracy drop (percentage points) and energy saved (%)
    relative to training all epochs. Means come with 95% bootstrap confidence intervals.
    Accuracy is measured on the TEST set when the logs contain `test_acc`, otherwise on the
    validation set (a warning is printed).
  - Runs that never learned (validation accuracy flat near chance) are flagged. The main
    tables exclude them; results_table_all_runs.csv includes them.
  - Patience is swept from 1 to 10 epochs, so the learned model is compared with the whole
    patience trade-off curve, not three hand-picked points.
  - Significance: paired Wilcoxon signed-rank tests on accuracy drop and energy saved,
    Holm-corrected, against the strongest patience baseline and Patience-3/5/10.
  - Ablation (with / without energy-related features) and sensitivity analysis (labelling
    thresholds) use Random Forest under LODO.

How to run (from the project root)
  python step2_v2.py
  python step2_v2.py --logs training_logs_v2/all_runs_combined.csv --out results_v2
  python step2_v2.py --skip-loro        # faster: skip the 160-fold run-level scheme

Output (in results_v2/)
  results_table.csv, results_table_all_runs.csv, results_table_loro.csv,
  per_dataset_results.csv, patience_sweep.csv, significance_tests.csv, ablation_results.csv,
  sensitivity_results.csv, label_summary.csv, meta_model_results.csv, and the plots (*.png)
"""

import argparse
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix

from common import (ENERGY_FEATURES, FEATURES, NICE_NAMES, SEED, add_features, best_by_criterion,
                    cv_predict, holm, label_run, load_logs, make_pipeline, simulate_all,
                    simulate_patience, simulate_stop, summarise)

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--logs", default="training_logs_v2/all_runs_combined.csv")
parser.add_argument("--out", default="results_v2")
parser.add_argument("--acc-tol", type=float, default=0.5, help="label rule (a): stop if < this many pp left to gain")
parser.add_argument("--gain-per-wh", type=float, default=1.0, help="label rule (b): stop if gain per Wh falls below this")
parser.add_argument("--max-drop", type=float, default=0.5, help="accuracy budget (pp) used to pick the best method")
parser.add_argument("--skip-loro", action="store_true")
parser.add_argument("--jobs", type=int, default=2)
args = parser.parse_args()

OUT = args.out
os.makedirs(OUT, exist_ok=True)


def save(name):
    plt.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {OUT}/{name}")


# ---------------------------------------------------------------------------
# SECTION A: Load
# ---------------------------------------------------------------------------
print("Loading training logs...")
if not os.path.exists(args.logs):
    raise SystemExit(f"ERROR: cannot find {args.logs}. Run step1_v2.py first.")

df, repaired = load_logs(args.logs)
num_runs = df["run_id"].nunique()
epochs_per_run = df.groupby("run_id").size().iloc[0]
use_test = "test_acc" in df.columns
print(f"  {len(df)} rows, {num_runs} runs x {epochs_per_run} epochs")
for ds, g in df.groupby("dataset", sort=False):
    print(f"    {ds}: {g['run_id'].nunique()} runs, "
          f"{g.groupby('run_id')['diverged'].first().sum()} never learned")
if repaired:
    print("  NOTE: legacy log (no energy_method column): per-epoch energy was re-split from each "
          "run's total in proportion to epoch time, because the raw per-epoch values were mostly 0.")
if not use_test:
    print("  WARNING: no test_acc column; accuracy below is VALIDATION accuracy, which is also what "
          "the labels use. Re-run Step 1 to log test accuracy.")
ACC_BASIS = "test" if use_test else "validation"

# ---------------------------------------------------------------------------
# SECTION B: Labels and features
# ---------------------------------------------------------------------------
print("\nLabelling epochs...")
df = pd.concat([label_run(g, args.acc_tol, args.gain_per_wh) for _, g in df.groupby("run_id")],
               ignore_index=True)
df = pd.concat([add_features(g) for _, g in df.groupby("run_id")], ignore_index=True)

label_summary = (df.assign(rule=df["label_rule"].replace("", "continue"))
                 .groupby(["dataset", "rule"]).size().unstack(fill_value=0))
label_summary["stop_share_%"] = 100 * (label_summary.get("accuracy", 0) + label_summary.get("energy", 0)) / label_summary.sum(axis=1)
label_summary.to_csv(os.path.join(OUT, "label_summary.csv"))
print(label_summary.to_string())
print(f"  Stop labels: {int(df['label'].sum())} of {len(df)} "
      f"(accuracy rule {int((df['label_rule'] == 'accuracy').sum())}, "
      f"energy rule {int((df['label_rule'] == 'energy').sum())})")

X = df[FEATURES].fillna(0)
y = df["label"]
groups_dataset = df["dataset"]
groups_run = df["run_id"]

MODELS = {
    "Random Forest": lambda: make_pipeline(RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED)),
    "Gradient Boosting": lambda: make_pipeline(GradientBoostingClassifier(n_estimators=100, max_depth=5, random_state=SEED)),
    "Logistic Regression": lambda: make_pipeline(LogisticRegression(max_iter=1000, random_state=SEED)),
    "AdaBoost": lambda: make_pipeline(AdaBoostClassifier(n_estimators=100, random_state=SEED)),
}
OURS = {name: f"{name} (Ours)" for name in MODELS}

# ---------------------------------------------------------------------------
# SECTION C: Cross-validated meta-models
# ---------------------------------------------------------------------------
n_datasets = df["dataset"].nunique()
schemes = {"LODO": groups_dataset}
if not args.skip_loro:
    schemes["LORO"] = groups_run

clf_rows = []
pred_cols = {s: {} for s in schemes}
for scheme, groups in schemes.items():
    print(f"\nTraining 4 meta-models, leave-one-{'dataset' if scheme == 'LODO' else 'run'}-out "
          f"({groups.nunique()} folds)...")
    for name, factory in MODELS.items():
        pred, proba = cv_predict(X, y, groups, factory, n_jobs=args.jobs if scheme == "LORO" else 1)
        col = f"pred_{scheme}_{name}"
        df[col] = pred
        df[f"proba_{scheme}_{name}"] = proba
        pred_cols[scheme][OURS[name]] = col
        acc = accuracy_score(y, pred)
        stop_recall = ((pred == 1) & (y == 1)).sum() / max((y == 1).sum(), 1)
        stop_prec = ((pred == 1) & (y == 1)).sum() / max((pred == 1).sum(), 1)
        clf_rows.append({"Scheme": scheme, "Model": name, "Classification Accuracy": acc,
                         "Stop Precision": stop_prec, "Stop Recall": stop_recall,
                         "Stop Share of Labels": float(y.mean())})
        print(f"  {scheme} {name:<20} accuracy {acc:.3f}  stop precision {stop_prec:.3f}  stop recall {stop_recall:.3f}")
clf_df = pd.DataFrame(clf_rows)
clf_df.to_csv(os.path.join(OUT, "meta_model_results.csv"), index=False)

# ---------------------------------------------------------------------------
# SECTION D: Replay every run under every stopping method
# ---------------------------------------------------------------------------
print("\nReplaying runs under each stopping method...")
sims = {}
for scheme in schemes:
    sims[scheme] = simulate_all(df, pred_cols[scheme])

patience_names = [f"Patience-{p}" for p in range(1, 11)]
display_order = ["No Stopping", "Patience-3", "Patience-5", "Patience-10"] + list(OURS.values())


def table(sim, only_learned=True):
    s = sim[~sim["diverged"]] if only_learned else sim
    return summarise(s, order=["No Stopping"] + patience_names + list(OURS.values()))


primary = table(sims["LODO"])
primary_all = table(sims["LODO"], only_learned=False)
best_patience = best_by_criterion(primary, patience_names, args.max_drop)
best_model = best_by_criterion(primary, list(OURS.values()), args.max_drop)
print(f"  Strongest patience baseline within {args.max_drop} pp: {best_patience}")
print(f"  Best meta-model within {args.max_drop} pp (LODO): {best_model}")

shown = ["No Stopping", "Patience-3", "Patience-5", "Patience-10", best_patience] + list(OURS.values())
shown = list(dict.fromkeys(shown))
primary[primary["Method"].isin(shown)].to_csv(os.path.join(OUT, "results_table.csv"), index=False)
primary_all[primary_all["Method"].isin(shown)].to_csv(os.path.join(OUT, "results_table_all_runs.csv"), index=False)
primary[primary["Method"].isin(patience_names)].to_csv(os.path.join(OUT, "patience_sweep.csv"), index=False)
if "LORO" in sims:
    t = table(sims["LORO"])
    t[t["Method"].isin(shown)].to_csv(os.path.join(OUT, "results_table_loro.csv"), index=False)

print(f"\n{'=' * 100}\nOVERALL ({ACC_BASIS} accuracy, runs that learned, leave-one-dataset-out)\n{'=' * 100}")
cols = ["Method", "Runs", "Acc Drop (pp, mean)", "Acc Drop CI low", "Acc Drop CI high",
        "Energy Saved (%, mean)", "Energy Saved CI low", "Energy Saved CI high", "Avg Epochs"]
print(primary[primary["Method"].isin(shown)][cols].round(2).to_string(index=False))

# per-dataset
rows = []
for ds, g in sims["LODO"].groupby("dataset", sort=False):
    t = summarise(g[~g["diverged"]], order=["No Stopping"] + patience_names + list(OURS.values()))
    t = t[t["Method"].isin(shown)]
    t.insert(0, "Dataset", ds)
    rows.append(t)
per_dataset = pd.concat(rows, ignore_index=True)
per_dataset.to_csv(os.path.join(OUT, "per_dataset_results.csv"), index=False)

# ---------------------------------------------------------------------------
# SECTION E: Significance (paired Wilcoxon, Holm-corrected)
# ---------------------------------------------------------------------------
print("\nSignificance tests (paired, per run, runs that learned)...")
sim = sims["LODO"]
sim = sim[~sim["diverged"]]
comparators = list(dict.fromkeys([best_patience, "Patience-3", "Patience-5", "Patience-10"]))
tests = []
for model_name in OURS.values():
    mine = sim[sim["method"] == model_name].set_index("run_id")
    for comp in comparators:
        other = sim[sim["method"] == comp].set_index("run_id").loc[mine.index]
        for metric, label in [("acc_drop_pp", "Accuracy drop (pp)"), ("energy_saved_pct", "Energy saved (%)")]:
            diff = mine[metric] - other[metric]
            try:
                p = wilcoxon(mine[metric], other[metric]).pvalue if (diff != 0).any() else 1.0
            except ValueError:
                p = 1.0
            tests.append({"Method": model_name, "Compared with": comp, "Metric": label,
                          "Median paired difference": float(diff.median()),
                          "Mean paired difference": float(diff.mean()), "p (raw)": p})
tests_df = pd.DataFrame(tests)
tests_df["p (Holm)"] = holm(tests_df["p (raw)"].to_numpy())
tests_df["Significant (Holm p<0.05)"] = tests_df["p (Holm)"] < 0.05
tests_df.to_csv(os.path.join(OUT, "significance_tests.csv"), index=False)
print(tests_df[tests_df["Compared with"] == best_patience].round(4).to_string(index=False))

# ---------------------------------------------------------------------------
# SECTION F: Ablation - with / without energy-related features (Random Forest, LODO)
# ---------------------------------------------------------------------------
print("\nAblation: energy-related features")
non_energy = [f for f in FEATURES if f not in ENERGY_FEATURES]
abl_rows = []
for label, feats in [("With energy features", FEATURES), ("Without energy features", non_energy)]:
    d = df.copy()
    pred, _ = cv_predict(d[feats].fillna(0), y, groups_dataset, MODELS["Random Forest"])
    d["abl_pred"] = pred
    s = simulate_all(d, {"RF": "abl_pred"}, patience_values=[])
    s = s[(s["method"] == "RF") & (~s["diverged"])]
    abl_rows.append({"Condition": label, "Features": len(feats),
                     "Classification Accuracy": accuracy_score(y, pred),
                     "Acc Drop (pp, mean)": s["acc_drop_pp"].mean(),
                     "Energy Saved (%, mean)": s["energy_saved_pct"].mean(),
                     "Avg Epochs": s["stop_epoch"].mean()})
abl = pd.DataFrame(abl_rows)
abl.to_csv(os.path.join(OUT, "ablation_results.csv"), index=False)
print(abl.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# SECTION G: Sensitivity to the labelling thresholds (Random Forest, LODO)
# ---------------------------------------------------------------------------
print("\nSensitivity analysis")


def run_labelling(acc_tol, gain):
    d = pd.concat([label_run(g, acc_tol, gain) for _, g in df.groupby("run_id")], ignore_index=True)
    d = d.drop(columns=[c for c in d.columns if c.startswith("pred_") or c.startswith("proba_")], errors="ignore")
    pred, _ = cv_predict(d[FEATURES].fillna(0), d["label"], d["dataset"], MODELS["Random Forest"])
    d["rf_pred"] = pred
    s = simulate_all(d, {"RF": "rf_pred"}, patience_values=[])
    s = s[(s["method"] == "RF") & (~s["diverged"])]
    return {"Stop Share of Labels": d["label"].mean(),
            "Energy-rule Share of Stop Labels": (d["label_rule"] == "energy").sum() / max(d["label"].sum(), 1),
            "Acc Drop (pp, mean)": s["acc_drop_pp"].mean(),
            "Energy Saved (%, mean)": s["energy_saved_pct"].mean(), "Avg Epochs": s["stop_epoch"].mean()}


sens_rows = []
for tol in [0.1, 0.5, 1.0, 2.0]:
    sens_rows.append({"Varied": "accuracy tolerance (pp)", "Value": tol, **run_labelling(tol, args.gain_per_wh)})
for gain in [0.25, 0.5, 1.0, 2.0, 5.0]:
    sens_rows.append({"Varied": "min gain per Wh (pp/Wh)", "Value": gain, **run_labelling(args.acc_tol, gain)})
sens = pd.DataFrame(sens_rows)
for ref in ["Patience-3", "Patience-5", "Patience-10"]:
    r = primary[primary["Method"] == ref].iloc[0]
    sens = pd.concat([sens, pd.DataFrame([{"Varied": "reference (no labels)", "Value": ref,
                                           "Acc Drop (pp, mean)": r["Acc Drop (pp, mean)"],
                                           "Energy Saved (%, mean)": r["Energy Saved (%, mean)"],
                                           "Avg Epochs": r["Avg Epochs"]}])], ignore_index=True)
sens.to_csv(os.path.join(OUT, "sensitivity_results.csv"), index=False)
print(sens.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# SECTION H: Plots
# ---------------------------------------------------------------------------
print("\nGenerating plots...")
C_OURS, C_BASE, C_NONE = "#2E7D32", "#C0504D", "#9E9E9E"
short = {"No Stopping": "No stop", "Patience-3": "Pat-3", "Patience-5": "Pat-5", "Patience-10": "Pat-10"}
cmp_methods = ["No Stopping", "Patience-3", "Patience-5", "Patience-10", best_model]
cmp_methods = list(dict.fromkeys(cmp_methods))


def lab(m):
    return short.get(m, "Ours\n" + m.replace(" (Ours)", "")) if m in short or m == best_model else m


def colour(m):
    return C_NONE if m == "No Stopping" else (C_OURS if "(Ours)" in m else C_BASE)


# Pipeline diagram (counts come from the data)
fig, ax = plt.subplots(figsize=(14, 4.5))
ax.axis("off"); ax.set_xlim(0, 14); ax.set_ylim(0, 5)
boxes = [
    (1.2, f"{num_runs} training runs\n({', '.join(df['dataset'].unique())})", "#E3F2FD", "#1565C0"),
    (3.9, f"Per-epoch log\n({len(df):,} rows: loss, accuracy,\ntime, energy)", "#E3F2FD", "#1565C0"),
    (6.6, "Retrospective labels\n(stop / continue)", "#E8F5E9", "#2E7D32"),
    (9.0, f"Features\n({len(FEATURES)}, observable live)", "#E8F5E9", "#2E7D32"),
    (11.4, "Meta-models\n(leave-one-dataset-out)", "#E8F5E9", "#2E7D32"),
]
for x, t, fc, ec in boxes:
    ax.text(x, 3.0, t, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.5", fc=fc, ec=ec, lw=2))
for a, b in [(2.4, 2.8), (5.1, 5.6), (7.6, 8.1), (9.9, 10.3)]:
    ax.annotate("", xy=(b, 3.0), xytext=(a, 3.0), arrowprops=dict(arrowstyle="-|>", lw=2, color="#555"))
ax.text(7.0, 1.0, f"Evaluation: replay each run, stop where the model says stop, compare with patience 1-10\n"
        f"(paired accuracy drop and energy saved per run, {ACC_BASIS} accuracy, Wilcoxon + Holm)",
        ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.5", fc="#FFF3E0", ec="#E65100", lw=2))
ax.set_title("Pipeline: energy-aware early stopping", fontweight="bold")
save("pipeline_diagram.png")

# Confusion matrices (LODO)
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
for ax, name in zip(axes.flat, MODELS):
    cm = confusion_matrix(y, df[pred_cols["LODO"][OURS[name]]])
    ConfusionMatrixDisplay(cm, display_labels=["Continue", "Stop"]).plot(ax=ax, colorbar=False, cmap="Blues")
    acc = accuracy_score(y, df[pred_cols["LODO"][OURS[name]]])
    ax.set_title(f"{name}  (accuracy {acc:.3f})", fontweight="bold")
fig.suptitle("Stop / continue predictions, leave-one-dataset-out", fontweight="bold")
plt.tight_layout(); save("confusion_matrices.png")

# Feature importance (RF fitted on all rows; for interpretation only)
rf_all = MODELS["Random Forest"]().fit(X, y)
imp = rf_all.named_steps["clf"].feature_importances_
order = np.argsort(imp)
fig, ax = plt.subplots(figsize=(9, 6.5))
ax.barh(range(len(order)), imp[order], color=[C_OURS if FEATURES[i] not in ENERGY_FEATURES else "#1565C0" for i in order])
ax.set_yticks(range(len(order))); ax.set_yticklabels([NICE_NAMES[FEATURES[i]] for i in order])
ax.set_xlabel("Feature importance"); ax.set_title("Which signals matter? (Random Forest on all rows; blue = energy/time features)", fontweight="bold")
plt.tight_layout(); save("feature_importance.png")

# Overall comparison with confidence intervals
fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
t = primary.set_index("Method").loc[cmp_methods]
x = np.arange(len(cmp_methods))
for ax, (col, lo, hi, ttl, yl) in zip(axes, [
        ("Acc Drop (pp, mean)", "Acc Drop CI low", "Acc Drop CI high", "Accuracy drop vs training all epochs", "percentage points (lower is better)"),
        ("Energy Saved (%, mean)", "Energy Saved CI low", "Energy Saved CI high", "Energy saved", "% (higher is better)"),
        ("Avg Epochs", None, None, "Epochs used", "epochs")]):
    vals = t[col].to_numpy()
    err = None if lo is None else np.vstack([vals - t[lo].to_numpy(), t[hi].to_numpy() - vals])
    ax.bar(x, vals, color=[colour(m) for m in cmp_methods], yerr=err, capsize=4)
    ax.set_xticks(x); ax.set_xticklabels([lab(m) for m in cmp_methods], fontsize=8)
    ax.set_title(ttl); ax.set_ylabel(yl)
fig.suptitle(f"Learned stopping vs patience ({ACC_BASIS} accuracy, runs that learned, leave-one-dataset-out, 95% CI)", fontweight="bold")
plt.tight_layout(); save("comparison.png")

# Per-dataset
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
datasets_ = list(df["dataset"].unique())
w = 0.8 / len(cmp_methods)
for ax, (col, ttl) in zip(axes, [("Acc Drop (pp, mean)", "Accuracy drop (pp)"), ("Energy Saved (%, mean)", "Energy saved (%)")]):
    for k, m in enumerate(cmp_methods):
        v = [per_dataset[(per_dataset["Dataset"] == d) & (per_dataset["Method"] == m)][col].iloc[0] for d in datasets_]
        ax.bar(np.arange(len(datasets_)) + k * w, v, w, label=lab(m).replace("\n", " "), color=colour(m), alpha=0.55 + 0.45 * ("(Ours)" in m))
    ax.set_xticks(np.arange(len(datasets_)) + w * (len(cmp_methods) - 1) / 2); ax.set_xticklabels(datasets_)
    ax.set_title(ttl); ax.axhline(0, color="k", lw=0.6)
axes[0].legend(fontsize=7)
fig.suptitle("Per-dataset results (runs that learned, leave-one-dataset-out)", fontweight="bold")
plt.tight_layout(); save("per_dataset_comparison.png")

# Meta-model comparison
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for k, scheme in enumerate(schemes):
    c = clf_df[clf_df["Scheme"] == scheme]
    axes[0].bar(np.arange(len(c)) + k * 0.38, c["Classification Accuracy"], 0.38, label=scheme)
axes[0].set_xticks(np.arange(len(MODELS)) + 0.19 * (len(schemes) - 1)); axes[0].set_xticklabels(list(MODELS), fontsize=8, rotation=15)
axes[0].set_ylabel("Stop/continue classification accuracy"); axes[0].set_title("Classification accuracy"); axes[0].legend()
axes[0].set_ylim(0.5, 1.0)
for name, mname in OURS.items():
    r = primary[primary["Method"] == mname].iloc[0]
    axes[1].scatter(r["Acc Drop (pp, mean)"], r["Energy Saved (%, mean)"], s=90, label=name)
for p in patience_names:
    r = primary[primary["Method"] == p].iloc[0]
    axes[1].scatter(r["Acc Drop (pp, mean)"], r["Energy Saved (%, mean)"], s=25, c=C_BASE, alpha=0.6)
axes[1].set_xlabel("Accuracy drop (pp)"); axes[1].set_ylabel("Energy saved (%)")
axes[1].set_title("Outcome: meta-models (large) vs patience 1-10 (small red)"); axes[1].legend(fontsize=8)
plt.tight_layout(); save("meta_model_comparison.png")

# Pareto
fig, ax = plt.subplots(figsize=(8.5, 6))
pt = primary[primary["Method"].isin(patience_names)]
ax.plot(pt["Acc Drop (pp, mean)"], pt["Energy Saved (%, mean)"], "-o", c=C_BASE, label="Patience 1-10")
for _, r in pt.iterrows():
    ax.annotate(r["Method"].split("-")[1], (r["Acc Drop (pp, mean)"], r["Energy Saved (%, mean)"]), fontsize=7, xytext=(3, 3), textcoords="offset points")
for name, mname in OURS.items():
    r = primary[primary["Method"] == mname].iloc[0]
    ax.errorbar(r["Acc Drop (pp, mean)"], r["Energy Saved (%, mean)"],
                xerr=[[r["Acc Drop (pp, mean)"] - r["Acc Drop CI low"]], [r["Acc Drop CI high"] - r["Acc Drop (pp, mean)"]]],
                fmt="s", ms=8, label=name, capsize=3)
ax.axvline(args.max_drop, ls=":", c="grey"); ax.text(args.max_drop, ax.get_ylim()[0], " accuracy budget", fontsize=7, va="bottom")
ax.set_xlabel("Accuracy drop vs training all epochs (pp)"); ax.set_ylabel("Energy saved (%)")
ax.set_title("Trade-off: learned stopping vs the patience curve", fontweight="bold"); ax.legend(fontsize=8)
plt.tight_layout(); save("pareto_frontier.png")

# Sensitivity
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
for ax, varied in zip(axes, ["accuracy tolerance (pp)", "min gain per Wh (pp/Wh)"]):
    s = sens[sens["Varied"] == varied]
    ax.plot(s["Value"].astype(float), s["Energy Saved (%, mean)"], "-o", c="#1565C0", label="Energy saved (%)")
    ax.set_xlabel(varied); ax.set_ylabel("Energy saved (%)", color="#1565C0")
    ax2 = ax.twinx(); ax2.plot(s["Value"].astype(float), s["Acc Drop (pp, mean)"], "-s", c=C_BASE)
    ax2.set_ylabel("Accuracy drop (pp)", color=C_BASE)
    if varied.startswith("min gain"):
        ax.set_xscale("log")
fig.suptitle("Sensitivity of the Random Forest stopper to the labelling thresholds", fontweight="bold")
plt.tight_layout(); save("sensitivity_analysis.png")

# Example curves (first run that learned, per dataset), using the best model under LODO
fig, axes = plt.subplots(1, len(datasets_), figsize=(5 * len(datasets_), 4.2))
best_col = pred_cols["LODO"][best_model]
for ax, ds in zip(np.atleast_1d(axes), datasets_):
    g = df[(df["dataset"] == ds) & (~df["diverged"])]
    rid = g["run_id"].iloc[0]
    r = df[df["run_id"] == rid]
    ax.plot(r["epoch"], r["val_loss"], "b-", lw=2, label="Val loss")
    ax.plot(r["epoch"], r["train_loss"], "b--", alpha=0.4, label="Train loss")
    p3 = simulate_patience(r, 3)["stop_epoch"]
    ours = simulate_stop(r, r[best_col].to_numpy())["stop_epoch"]
    ax.axvline(p3, c=C_BASE, ls=":", label=f"Patience-3 (ep {p3})")
    ax.axvline(ours, c=C_OURS, lw=2, label=f"{best_model.replace(' (Ours)', '')} (ep {ours})")
    ax.set_title(f"{ds} - run {rid}", fontweight="bold"); ax.set_xlabel("Epoch"); ax.legend(fontsize=7)
fig.suptitle("Example learning curves and stop points", fontweight="bold")
plt.tight_layout(); save("example_curves.png")

# Energy-accuracy scatter, one panel per dataset (different scales)
fig, axes = plt.subplots(1, len(datasets_), figsize=(5 * len(datasets_), 4.5))
for ax, ds in zip(np.atleast_1d(axes), datasets_):
    for m, c in [("No Stopping", C_NONE), ("Patience-3", C_BASE), ("Patience-10", "#E8A09E"), (best_model, C_OURS)]:
        s = sims["LODO"][(sims["LODO"]["dataset"] == ds) & (sims["LODO"]["method"] == m) & (~sims["LODO"]["diverged"])]
        ax.scatter(s["energy_kwh"] * 1000, s["accuracy"], s=22, alpha=0.6, c=c, label=m)
    ax.set_xscale("log"); ax.set_xlabel("Energy per run (Wh, log)"); ax.set_ylabel(f"{ACC_BASIS} accuracy"); ax.set_title(ds, fontweight="bold")
np.atleast_1d(axes)[0].legend(fontsize=7)
fig.suptitle("Energy-accuracy trade-off: each point is one training run", fontweight="bold")
plt.tight_layout(); save("energy_accuracy_tradeoff.png")

print(f"\nDone. Tables and plots are in '{OUT}/'.")
