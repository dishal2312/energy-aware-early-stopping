"""
STEP 1: Baseline Model Training (original, MNIST only)

What this does
  - Trains 40 small neural networks (MLPs) on MNIST (handwritten digits)
  - Each model has different settings (learning rate, batch size, network shape)
  - At every epoch it logs loss, accuracy, time and energy
  - All logs are saved as CSV files in the 'training_logs/' folder

Changes compared with the first run of this script
  - Reproducible: every run is seeded (SEED + run_id) and the train/validation split is seeded.
  - Balanced design: the 40 configs use ALL four learning rates and ALL four batch sizes.
    (The old `all_configs[::2]` only ever selected batch sizes 32 and 128.)
  - Test accuracy is logged every epoch (for reporting; never used to choose stopping points).
  - Energy: run with `sudo` on a Mac to integrate real package power from `powermetrics`;
    otherwise CodeCarbon's run total is split over epochs by time (see EnergyMeter in
    common.py). The method used is saved in the `energy_method` column.
  - A finished run is skipped if its CSV already exists, so the script can be restarted.

How to run (from the project root, virtual environment active)
  python step1_baseline_training.py
  sudo -E venv/bin/python step1_baseline_training.py     # real power via powermetrics

Output
  training_logs/run_001.csv ... run_040.csv
  training_logs/all_runs_combined.csv
"""

import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from common import SEED, EnergyMeter, balanced_configs, run_seed, set_seed

# ---------------------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------------------
EPOCHS_PER_MODEL = 30
NUM_RUNS = 40
VAL_SIZE = 10000
DATA_DIR = "./data"
LOG_DIR = "./training_logs"

set_seed(SEED)

# ---------------------------------------------------------------------------
# STEP A: Load MNIST (50,000 train / 10,000 validation / 10,000 test)
# ---------------------------------------------------------------------------
print("\nLoading MNIST dataset...")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)
train_data, val_data = random_split(full_train, [len(full_train) - VAL_SIZE, VAL_SIZE],
                                    generator=torch.Generator().manual_seed(SEED))
print(f"  Training: {len(train_data)}  Validation: {len(val_data)}  Test: {len(test_data)}")


# ---------------------------------------------------------------------------
# STEP B: The network
# ---------------------------------------------------------------------------
class SimpleNetwork(nn.Module):
    """Feed-forward network: 784 pixels -> hidden layers -> 10 digit classes."""

    def __init__(self, hidden_sizes):
        super().__init__()
        layers, prev = [], 784
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


# ---------------------------------------------------------------------------
# STEP C: Train one model
# ---------------------------------------------------------------------------
def evaluate(model, loader, loss_function):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            out = model(images)
            total_loss += loss_function(out, labels).item() * images.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train_one_model(run_id, hidden_sizes, learning_rate, batch_size):
    seed = run_seed(SEED, run_id)
    set_seed(seed)

    print(f"\n{'=' * 60}")
    print(f"  Run {run_id}/{NUM_RUNS}  shape {hidden_sizes}  lr {learning_rate}  "
          f"batch {batch_size}  seed {seed}")
    print(f"{'=' * 60}")

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_data, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=1000, shuffle=False)

    model = SimpleNetwork(hidden_sizes)
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    meter = EnergyMeter(f"run_{run_id}").start()
    rows, windows = [], []

    for epoch in range(1, EPOCHS_PER_MODEL + 1):
        t0 = time.time()

        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            optimiser.zero_grad()
            out = model(images)
            loss = loss_function(out, labels)
            loss.backward()
            optimiser.step()
            loss_sum += loss.item() * images.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += images.size(0)
        train_loss, train_acc = loss_sum / total, correct / total

        val_loss, val_acc = evaluate(model, val_loader, loss_function)
        t1 = time.time()
        windows.append((t0, t1))               # energy window = training + validation only
        _, test_acc = evaluate(model, test_loader, loss_function)   # reporting only

        if epoch == 1:
            d_loss = d_acc = 0.0
        else:
            d_loss = rows[-1]["val_loss"] - val_loss      # positive = loss went down
            d_acc = val_acc - rows[-1]["val_acc"]         # positive = accuracy went up

        rows.append({
            "run_id": run_id, "epoch": epoch, "dataset": "mnist",
            "hidden_sizes": str(hidden_sizes), "learning_rate": learning_rate,
            "batch_size": batch_size, "seed": seed,
            "train_loss": round(train_loss, 6), "train_acc": round(train_acc, 4),
            "val_loss": round(val_loss, 6), "val_acc": round(val_acc, 4),
            "test_acc": round(test_acc, 4),
            "val_loss_improvement": round(d_loss, 6), "val_acc_improvement": round(d_acc, 6),
            "epoch_time_secs": round(t1 - t0, 3),
        })
        print(f"  Epoch {epoch:2d}/{EPOCHS_PER_MODEL} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Time: {t1 - t0:.1f}s")

    energies, method = meter.finish(windows)
    cumulative = np.cumsum(energies)
    for row, e, c in zip(rows, energies, cumulative):
        row["epoch_energy_kwh"] = e
        row["cumulative_energy_kwh"] = float(c)
        row["energy_method"] = method

    df = pd.DataFrame(rows)
    path = os.path.join(LOG_DIR, f"run_{run_id:03d}.csv")
    df.to_csv(path, index=False)
    print(f"  Saved {path}  (energy: {method}, total {cumulative[-1] * 1000:.4f} Wh)")
    return df


# ---------------------------------------------------------------------------
# STEP D: The 40 experiments (balanced over architecture, learning rate and batch size)
# ---------------------------------------------------------------------------
architectures = [[64], [128], [128, 64], [256, 128], [256, 128, 64]]
learning_rates = [0.01, 0.005, 0.001, 0.0005]
batch_sizes = [32, 64, 128, 256]

selected_configs = balanced_configs(architectures, learning_rates, batch_sizes, seed=SEED)[:NUM_RUNS]

# ---------------------------------------------------------------------------
# STEP E: Run (resumable)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"\n{'#' * 60}\n  {len(selected_configs)} runs x {EPOCHS_PER_MODEL} epochs\n{'#' * 60}")

    for i, cfg in enumerate(selected_configs):
        run_id = i + 1
        if os.path.exists(os.path.join(LOG_DIR, f"run_{run_id:03d}.csv")):
            print(f"  Run {run_id}/{NUM_RUNS} already done - skipping")
            continue
        train_one_model(run_id, cfg["arch"], cfg["learning_rate"], cfg["batch_size"])

    parts = [pd.read_csv(os.path.join(LOG_DIR, f"run_{i + 1:03d}.csv")) for i in range(len(selected_configs))
             if os.path.exists(os.path.join(LOG_DIR, f"run_{i + 1:03d}.csv"))]
    combined = pd.concat(parts, ignore_index=True)
    combined_path = os.path.join(LOG_DIR, "all_runs_combined.csv")
    combined.to_csv(combined_path, index=False)
    print(f"\n{'#' * 60}\n  Done: {len(parts)}/{NUM_RUNS} runs, {len(combined)} rows\n"
          f"  Combined log: {combined_path}\n{'#' * 60}")
