"""
common.py - helpers shared by the Step 1 and Step 2 scripts.

Contents
  Reproducibility : set_seed, run_seed
  Experiment design: balanced_configs
  Energy           : EnergyMeter (powermetrics when run with sudo on a Mac,
                     otherwise CodeCarbon's run total attributed to epochs by time)
  Analysis         : load_logs, flag_diverged, repair_legacy_energy, label_run,
                     add_features, simulate_patience, simulate_stop
"""

import os
import re
import sys
import time
import random
import shutil
import threading
import subprocess

import numpy as np
import pandas as pd

SEED = 42


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed):
    """Seed python, numpy and torch (torch is imported lazily so Step 2 does not need it)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_seed(base_seed, run_id):
    """Every run gets its own deterministic seed, so a resumed experiment is reproducible."""
    return base_seed + int(run_id)


# ---------------------------------------------------------------------------
# Experiment design
# ---------------------------------------------------------------------------

def balanced_configs(architectures, learning_rates, batch_sizes, seed=SEED):
    """
    Choose half of the full grid so that EVERY learning rate and EVERY batch size is used,
    and learning rate is not tied to batch size.

    For each architecture we pick two different cyclic shifts s of the 4x4
    (learning rate, batch size) grid and keep the cells (lr_i, bs_(i+s) mod 4).
    That gives 8 configs per architecture in which each learning rate appears twice and
    each batch size appears twice. The shifts are drawn from a seeded RNG, so they differ
    between architectures and the design is reproducible.

    (The old code used configs[::2], which only ever selected batch sizes 32 and 128.)
    """
    n = len(learning_rates)
    assert n == len(batch_sizes) == 4, "balanced_configs expects 4 learning rates and 4 batch sizes"
    rng = random.Random(seed)
    # Deal shifts out of repeatedly shuffled decks of [0,1,2,3] so each shift is used
    # about equally often across architectures (2 or 3 times for 5 architectures).
    deck = []
    while len(deck) < 2 * len(architectures):
        d = list(range(n))
        rng.shuffle(d)
        deck.extend(d)
    configs = []
    for k, arch in enumerate(architectures):
        shifts = sorted(deck[2 * k: 2 * k + 2])
        for i, lr in enumerate(learning_rates):
            for s in shifts:
                configs.append({"arch": arch, "learning_rate": lr,
                                "batch_size": batch_sizes[(i + s) % n]})
    return configs


# ---------------------------------------------------------------------------
# Energy measurement
# ---------------------------------------------------------------------------

_PM_LINE = re.compile(r"Combined Power \(CPU \+ GPU \+ ANE\):\s*([\d.]+)\s*mW")
_PM_ELAPSED = re.compile(r"\(([\d.]+)ms elapsed\)")


class EnergyMeter:
    """
    Measures the energy of ONE training run and splits it over the epochs.

    Preferred method ("powermetrics_1s"): macOS `powermetrics` sampled every second while
    training runs. Needs root, so start the script with `sudo`. Per-epoch energy is the
    integral of CPU+GPU+ANE package power over that epoch's time window. It excludes DRAM,
    display and the rest of the machine.

    Fallback ("codecarbon_time_proportional"): CodeCarbon's total for the run, divided over
    the epochs in proportion to epoch time. CodeCarbon only updates about every 15 s, so
    its raw per-epoch readings are mostly zeros followed by jumps; the run total is reliable
    but a per-epoch split is not, so the split is by time and is labelled as such.

    Last resort ("constant_15W_estimate"): 15 W x time.
    """

    def __init__(self, label, country_iso_code="GBR", prefer_powermetrics=True):
        self.label = label
        self.country = country_iso_code
        self.method = None
        self._samples = []          # (timestamp, mW)
        self._proc = None
        self._thread = None
        self._tracker = None
        self._prefer_pm = prefer_powermetrics

    # -- start / stop -----------------------------------------------------
    def start(self):
        if (self._prefer_pm and sys.platform == "darwin" and shutil.which("powermetrics")
                and hasattr(os, "geteuid") and os.geteuid() == 0):
            try:
                # -b 1: line-buffered, so each reading is time-stamped when it is produced
                self._proc = subprocess.Popen(
                    ["powermetrics", "--samplers", "cpu_power", "-i", "1000", "-b", "1"],
                    stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
                time.sleep(0.5)
                if self._proc.poll() is not None:      # refused to start (bad option, no permission)
                    self._proc = None
                else:
                    self._thread = threading.Thread(target=self._read_powermetrics, daemon=True)
                    self._thread.start()
                    self.method = "powermetrics_1s"
                    return self
            except Exception:
                self._proc = None
        try:
            from codecarbon import OfflineEmissionsTracker
            self._tracker = OfflineEmissionsTracker(
                country_iso_code=self.country, project_name=self.label,
                log_level="error", save_to_file=False)
            self._tracker.start()
            self.method = "codecarbon_time_proportional"
        except Exception:
            self._tracker = None
            self.method = "constant_15W_estimate"
        return self

    def _read_powermetrics(self):
        elapsed_s = 1.0
        for line in self._proc.stdout:
            e = _PM_ELAPSED.search(line)
            if e:
                elapsed_s = float(e.group(1)) / 1000.0     # length of the interval this reading covers
            m = _PM_LINE.search(line)
            if m:
                self._samples.append((time.time(), float(m.group(1)), elapsed_s))

    @staticmethod
    def parse_powermetrics_text(text):
        """Return the list of combined-power readings (mW) found in powermetrics output."""
        return [float(x) for x in _PM_LINE.findall(text)]

    # -- finish -------------------------------------------------------------
    def finish(self, windows):
        """
        windows: list of (t_start, t_end) wall-clock times, one per epoch.
        Returns (list of per-epoch energy in kWh, method name).
        """
        durations = np.array([max(b - a, 1e-9) for a, b in windows], dtype=float)

        if self.method == "powermetrics_1s":
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                pass
            if self._thread is not None:
                self._thread.join(timeout=2)
            energies = self._integrate(windows)
            if energies is not None and np.sum(energies) > 0:
                return list(energies), self.method
            self.method = "constant_15W_estimate"   # no usable samples

        if self.method == "codecarbon_time_proportional" and self._tracker is not None:
            total_kwh = None
            try:
                self._tracker.stop()
                data = getattr(self._tracker, "final_emissions_data", None)
                if data is not None:
                    total_kwh = float(data.energy_consumed)
            except Exception:
                pass
            if total_kwh is None:
                try:
                    total_kwh = float(self._tracker._total_energy.kWh)   # private attr, last resort
                except Exception:
                    total_kwh = None
            if total_kwh is not None and total_kwh > 0:
                return list(total_kwh * durations / durations.sum()), self.method
            self.method = "constant_15W_estimate"

        return list(durations * 15.0 / 3.6e6), "constant_15W_estimate"

    def _integrate(self, windows):
        if len(self._samples) < 2:
            return None
        ts = np.array([s[0] for s in self._samples])
        mw = np.array([s[1] for s in self._samples])
        el = np.array([s[2] if len(s) > 2 else 1.0 for s in self._samples])
        starts = ts - el                                     # each reading covers the interval before it
        out = []
        for a, b in windows:
            overlap = np.clip(np.minimum(ts, b) - np.maximum(starts, a), 0, None)
            joules = float(np.sum(mw / 1000.0 * overlap))
            out.append(joules / 3.6e6)
        return out


# ---------------------------------------------------------------------------
# Analysis helpers (Step 2)
# ---------------------------------------------------------------------------

FEATURES = [
    "epoch", "train_loss", "val_loss", "val_acc",
    "val_loss_improvement", "val_acc_improvement",
    "avg_loss_improvement_3", "avg_acc_improvement_3",
    "loss_getting_worse", "no_improve_streak", "train_val_gap",
    "best_acc_gap",
    "epoch_time_secs", "epoch_energy_kwh", "cumulative_energy_kwh",
    "acc_per_energy", "recent_gain_per_wh",
]
ENERGY_FEATURES = ["epoch_time_secs", "epoch_energy_kwh", "cumulative_energy_kwh",
                   "acc_per_energy", "recent_gain_per_wh"]

NICE_NAMES = {
    "epoch": "Epoch number", "train_loss": "Training loss", "val_loss": "Validation loss",
    "val_acc": "Validation accuracy", "val_loss_improvement": "Loss improvement (this epoch)",
    "val_acc_improvement": "Accuracy improvement (this epoch)",
    "avg_loss_improvement_3": "Avg loss improvement (3 epochs)",
    "avg_acc_improvement_3": "Avg accuracy improvement (3 epochs)",
    "loss_getting_worse": "Is loss increasing?", "no_improve_streak": "Loss-worsening streak",
    "train_val_gap": "Train-val loss gap", "best_acc_gap": "Gap to best accuracy so far",
    "epoch_time_secs": "Time per epoch", "epoch_energy_kwh": "Energy per epoch",
    "cumulative_energy_kwh": "Cumulative energy", "acc_per_energy": "Accuracy per unit energy",
    "recent_gain_per_wh": "Recent accuracy gain per Wh",
}


def flag_diverged(df, max_acc=0.30, max_range=0.02):
    """A run 'never learned' if its validation accuracy stays flat near chance for all epochs."""
    g = df.groupby("run_id")["val_acc"].agg(["max", "min"])
    bad = g[((g["max"] - g["min"]) < max_range) & (g["max"] < max_acc)].index
    df = df.copy()
    df["diverged"] = df["run_id"].isin(bad)
    return df


def repair_legacy_energy(df):
    """
    Logs written by the original scripts have per-epoch energy read from CodeCarbon, which
    only updates about every 15 s (most epochs read exactly 0). The run total is reliable, so
    re-split it over the epochs in proportion to epoch time. Logs that already carry an
    `energy_method` column are left alone.
    """
    if "energy_method" in df.columns:
        return df, False
    df = df.sort_values(["run_id", "epoch"]).copy()
    parts = []
    for _, r in df.groupby("run_id"):
        r = r.copy()
        total = r["cumulative_energy_kwh"].iloc[-1]
        t = r["epoch_time_secs"].clip(lower=1e-6)
        r["raw_epoch_energy_kwh"] = r["epoch_energy_kwh"]
        r["epoch_energy_kwh"] = total * t / t.sum()
        r["cumulative_energy_kwh"] = r["epoch_energy_kwh"].cumsum()
        parts.append(r)
    out = pd.concat(parts, ignore_index=True)
    out["energy_method"] = "codecarbon_time_proportional (re-split from run total)"
    return out, True


def load_logs(path):
    df = pd.read_csv(path)
    if "dataset" not in df.columns:
        df["dataset"] = "mnist"          # the original Step 1 only used MNIST
    df, repaired = repair_legacy_energy(df)
    df = flag_diverged(df)
    return df.sort_values(["run_id", "epoch"]).reset_index(drop=True), repaired


def label_run(run_df, acc_tol_pp=0.5, gain_per_wh=1.0):
    """
    Retrospective stop/continue label for every epoch of one run.

    At epoch t, look at what the REST of the run still offers:
      remaining_acc_pp = (best val accuracy from t onwards - val accuracy at t) * 100
      remaining_wh     = energy still to be spent after t, in Wh
    STOP if
      (a) remaining_acc_pp < acc_tol_pp                 -> nothing worth waiting for, or
      (b) remaining_acc_pp / remaining_wh < gain_per_wh -> the gain is too expensive
    Otherwise CONTINUE.

    Returns a DataFrame copy with columns label (1 = stop) and label_rule ('accuracy', 'energy', '').
    """
    r = run_df.copy()
    acc = r["val_acc"].to_numpy()
    future_best = np.maximum.accumulate(acc[::-1])[::-1]
    remaining_acc = (future_best - acc) * 100.0
    remaining_wh = (r["cumulative_energy_kwh"].iloc[-1] - r["cumulative_energy_kwh"].to_numpy()) * 1000.0
    rule_a = remaining_acc < acc_tol_pp
    with np.errstate(divide="ignore", invalid="ignore"):
        rule_b = (~rule_a) & (remaining_wh > 0) & ((remaining_acc / remaining_wh) < gain_per_wh)
    r["label"] = (rule_a | rule_b).astype(int)
    r["label_rule"] = np.where(rule_a, "accuracy", np.where(rule_b, "energy", ""))
    return r


def add_features(run_df):
    """Signals that could be observed live, epoch by epoch (nothing here looks at future epochs)."""
    r = run_df.copy()
    r["avg_loss_improvement_3"] = r["val_loss_improvement"].rolling(3, min_periods=1).mean()
    r["avg_acc_improvement_3"] = r["val_acc_improvement"].rolling(3, min_periods=1).mean()
    r["loss_getting_worse"] = (r["val_loss_improvement"] < 0).astype(int)
    streak, out = 0, []
    for v in r["val_loss_improvement"]:
        streak = streak + 1 if v < 0 else 0
        out.append(streak)
    r["no_improve_streak"] = out
    r["train_val_gap"] = r["train_loss"] - r["val_loss"]
    r["best_acc_gap"] = r["val_acc"].cummax() - r["val_acc"]
    r["acc_per_energy"] = r["val_acc"] / (r["cumulative_energy_kwh"] * 1000.0 + 1e-9)
    gain_pp = r["avg_acc_improvement_3"] * 100.0
    r["recent_gain_per_wh"] = gain_pp / (r["epoch_energy_kwh"] * 1000.0 + 1e-9)
    r["recent_gain_per_wh"] = r["recent_gain_per_wh"].clip(-1e4, 1e4)
    return r


def _final_metrics(run_df, stop_epoch):
    sub = run_df[run_df["epoch"] <= stop_epoch]
    row = {
        "stop_epoch": int(stop_epoch),
        "final_acc": float(sub["val_acc"].iloc[-1]),
        "energy_used": float(sub["cumulative_energy_kwh"].iloc[-1]),
    }
    if "test_acc" in run_df.columns:
        row["final_test_acc"] = float(sub["test_acc"].iloc[-1])
    return row


def simulate_no_stopping(run_df):
    return _final_metrics(run_df, run_df["epoch"].max())


def simulate_patience(run_df, patience):
    """Stop when validation loss has not improved for `patience` epochs in a row."""
    best, wait, stop = float("inf"), 0, run_df["epoch"].max()
    for ep, vl in zip(run_df["epoch"], run_df["val_loss"]):
        if vl < best:
            best, wait = vl, 0
        else:
            wait += 1
        if wait >= patience:
            stop = ep
            break
    return _final_metrics(run_df, stop)


def simulate_stop(run_df, pred):
    """Stop at the first epoch where the meta-model predicts 'stop' (pred: array aligned to run_df)."""
    pred = np.asarray(pred)
    hits = np.flatnonzero(pred == 1)
    stop = run_df["epoch"].iloc[hits[0]] if len(hits) else run_df["epoch"].max()
    return _final_metrics(run_df, stop)


# ---------------------------------------------------------------------------
# Evaluation helpers (Step 2)
# ---------------------------------------------------------------------------

def make_pipeline(estimator):
    """Scaler + model, so scaling is learned on the training folds only."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    return Pipeline([("scale", StandardScaler()), ("clf", estimator)])


def cv_predict(X, y, groups, make_model, n_jobs=1):
    """
    Leave-one-group-out cross-validation. Every row is predicted by a model that never saw
    its group. Returns (class predictions, stop probabilities) aligned with X.
    """
    from joblib import Parallel, delayed
    from sklearn.model_selection import LeaveOneGroupOut

    def one_fold(train_idx, test_idx):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[test_idx])[:, 1]
        return test_idx, model.predict(X.iloc[test_idx]), proba

    folds = list(LeaveOneGroupOut().split(X, y, groups))
    out = Parallel(n_jobs=n_jobs)(delayed(one_fold)(tr, te) for tr, te in folds)
    pred = np.zeros(len(y), dtype=int)
    proba = np.zeros(len(y))
    for te, p, pr in out:
        pred[te] = p
        proba[te] = pr
    return pred, proba


def bootstrap_ci(values, n_boot=2000, seed=SEED):
    """95% percentile bootstrap confidence interval for the mean."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n_boot, len(v)), replace=True).mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * p[idx])
        adj[idx] = min(1.0, running)
    return adj


def simulate_all(df, pred_cols, patience_values=range(1, 11)):
    """
    Replay every run under every stopping method. One output row per (run, method) with
    the accuracy drop (percentage points) and energy saved (%) relative to training for
    all epochs. pred_cols maps a method name to a column of 0/1 stop predictions in df.
    """
    use_test = "test_acc" in df.columns
    rows = []
    for run_id, r in df.groupby("run_id"):
        r = r.sort_values("epoch")
        base = simulate_no_stopping(r)
        base_acc = base["final_test_acc"] if use_test else base["final_acc"]
        methods = {"No Stopping": base}
        for p in patience_values:
            methods[f"Patience-{p}"] = simulate_patience(r, p)
        for name, col in pred_cols.items():
            methods[name] = simulate_stop(r, r[col].to_numpy())
        for name, m in methods.items():
            acc = m["final_test_acc"] if use_test else m["final_acc"]
            rows.append({
                "run_id": run_id, "dataset": r["dataset"].iloc[0], "diverged": bool(r["diverged"].iloc[0]),
                "method": name, "stop_epoch": m["stop_epoch"], "accuracy": acc,
                "val_accuracy": m["final_acc"], "energy_kwh": m["energy_used"],
                "acc_drop_pp": (base_acc - acc) * 100.0,
                "energy_saved_pct": (base["energy_used"] - m["energy_used"]) / base["energy_used"] * 100.0
                if base["energy_used"] > 0 else 0.0,
            })
    return pd.DataFrame(rows)


def summarise(sim, order=None):
    """Mean +/- 95% CI of the paired accuracy drop and the per-run energy saved, per method."""
    rows = []
    for method, g in sim.groupby("method", sort=False):
        d_lo, d_hi = bootstrap_ci(g["acc_drop_pp"])
        s_lo, s_hi = bootstrap_ci(g["energy_saved_pct"])
        rows.append({
            "Method": method, "Runs": len(g),
            "Acc Drop (pp, mean)": g["acc_drop_pp"].mean(), "Acc Drop CI low": d_lo, "Acc Drop CI high": d_hi,
            "Acc Drop (pp, std)": g["acc_drop_pp"].std(),
            "Energy Saved (%, mean)": g["energy_saved_pct"].mean(), "Energy Saved CI low": s_lo,
            "Energy Saved CI high": s_hi, "Energy Saved (%, std)": g["energy_saved_pct"].std(),
            "Avg Epochs": g["stop_epoch"].mean(), "Avg Accuracy": g["accuracy"].mean(),
            "Avg Energy (Wh)": g["energy_kwh"].mean() * 1000.0,
        })
    out = pd.DataFrame(rows)
    if order:
        out["_o"] = out["Method"].map({m: i for i, m in enumerate(order)})
        out = out.sort_values("_o").drop(columns="_o")
    return out.reset_index(drop=True)


def best_by_criterion(summary, candidates, max_drop_pp=0.5):
    """
    Pick the candidate that saves the most energy while its mean accuracy drop stays within
    max_drop_pp. If none qualifies, pick the one with the smallest drop.
    """
    s = summary[summary["Method"].isin(candidates)]
    ok = s[s["Acc Drop (pp, mean)"] <= max_drop_pp]
    if len(ok):
        return ok.sort_values("Energy Saved (%, mean)", ascending=False)["Method"].iloc[0]
    return s.sort_values("Acc Drop (pp, mean)")["Method"].iloc[0]
