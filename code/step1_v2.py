"""
STEP 1 (Version 2): Extended Baseline Training

Trains small neural networks on four datasets and logs, for every epoch, the loss,
accuracy, time and energy. Step 2 uses these logs to learn when to stop training.

Datasets and models (40 runs each, 160 runs in total)
  - MNIST         (handwritten digits, 28x28 grey)    -> MLPs   runs   1-40
  - FashionMNIST  (clothing, 28x28 grey)              -> MLPs   runs  41-80
  - CIFAR-10      (object photos, 32x32 colour)       -> CNNs   runs  81-120
  - SVHN          (house numbers, 32x32 colour)       -> CNNs   runs 121-160

What changed compared with the first version of this script
  - Reproducible: every run is seeded (SEED + run_id) and the train/validation split is seeded.
  - Balanced design: the 40 configs per dataset use ALL four learning rates and ALL four
    batch sizes (the old `configs[::2]` only ever used batch sizes 32 and 128).
  - CNN learning rates are [0.002, 0.001, 0.0005, 0.0001]. With 0.005 and 0.01, 20 of the
    160 original runs never left chance accuracy.
  - Test accuracy is logged every epoch, so Step 2 can report accuracy on data that was
    not used to choose the stopping point. SVHN now has a 10,000-image validation set like
    the other datasets.
  - Energy: see EnergyMeter in common.py. Run with `sudo` on a Mac to integrate real
    package power from `powermetrics` (1 s samples). Without sudo, CodeCarbon's run total is
    split over epochs by time. The method is saved in the `energy_method` column.
  - batch_size, hidden_sizes / filters and seed are logged again.

How to run (from the project root, with the virtual environment active)
  python step1_v2.py                # energy via CodeCarbon (split by time)
  sudo -E venv/bin/python step1_v2.py   # energy via powermetrics (recommended on a Mac)
  python step1_v2.py --datasets mnist fashionmnist --log-dir training_logs_v2_rerun
                                    # only the two fast MLP datasets, into a separate folder

  Runs on CUDA if available, otherwise on the CPU. A finished run is skipped if its CSV
  already exists, so an interrupted experiment can simply be started again.

Output
  training_logs_v2/run_001.csv ... run_160.csv
  training_logs_v2/all_runs_combined.csv
"""

import glob
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
RUNS_PER_DATASET = 40
VAL_SIZE = 10000
DATA_DIR = "./data"
LOG_DIR = "./training_logs_v2"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
set_seed(SEED)


def split_generator():
    return torch.Generator().manual_seed(SEED)


# ---------------------------------------------------------------------------
# SECTION A: Load the four datasets (train / validation / test)
# ---------------------------------------------------------------------------
print("\nLoading datasets...")


def make_split(full, name):
    train, val = random_split(full, [len(full) - VAL_SIZE, VAL_SIZE], generator=split_generator())
    print(f"  {name:<13} {len(train)} train, {len(val)} val")
    return train, val


mnist_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
mnist_train, mnist_val = make_split(datasets.MNIST(DATA_DIR, train=True, download=True, transform=mnist_tf), "MNIST")
mnist_test = datasets.MNIST(DATA_DIR, train=False, download=True, transform=mnist_tf)

fm_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))])
fm_train, fm_val = make_split(datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=fm_tf), "FashionMNIST")
fm_test = datasets.FashionMNIST(DATA_DIR, train=False, download=True, transform=fm_tf)

cifar_tf = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
cifar_train, cifar_val = make_split(datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=cifar_tf), "CIFAR-10")
cifar_test = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=cifar_tf)

svhn_tf = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))])
svhn_train, svhn_val = make_split(datasets.SVHN(DATA_DIR, split="train", download=True, transform=svhn_tf), "SVHN")
svhn_test = datasets.SVHN(DATA_DIR, split="test", download=True, transform=svhn_tf)

dataset_map = {
    "mnist": (mnist_train, mnist_val, mnist_test),
    "fashionmnist": (fm_train, fm_val, fm_test),
    "cifar10": (cifar_train, cifar_val, cifar_test),
    "svhn": (svhn_train, svhn_val, svhn_test),
}


# ---------------------------------------------------------------------------
# SECTION B: The networks
# ---------------------------------------------------------------------------
class SimpleNetwork(nn.Module):
    """Feed-forward network (MLP) for the 28x28 grey-scale datasets."""

    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers, prev = [], input_size
        for size in hidden_sizes:
            layers += [nn.Linear(prev, size), nn.ReLU()]
            prev = size
        layers.append(nn.Linear(prev, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x.view(x.size(0), -1))


class SimpleCNN(nn.Module):
    """Two convolution + pooling stages followed by two fully connected layers."""

    def __init__(self, num_filters_1, num_filters_2, input_channels=3, image_size=32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_filters_1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        flat = num_filters_2 * (image_size // 4) * (image_size // 4)
        self.fc1 = nn.Linear(flat, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.relu(self.fc1(x.view(x.size(0), -1)))
        return self.fc2(x)


# ---------------------------------------------------------------------------
# SECTION C: Training one model
# ---------------------------------------------------------------------------
def evaluate(model, loader, loss_function):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            out = model(images)
            total_loss += loss_function(out, labels).item() * images.size(0)
            correct += out.argmax(1).eq(labels).sum().item()
            total += images.size(0)
    return total_loss / total, correct / total


def train_one_model(config, total_runs):
    run_id = config["run_id"]
    train_set, val_set, test_set = dataset_map[config["dataset"]]
    seed = run_seed(SEED, run_id)
    set_seed(seed)

    print(f"\n{'=' * 60}")
    print(f"  Run {run_id}/{total_runs} [{config['dataset'].upper()}]  {config['config_label']}")
    print(f"  Learning rate: {config['learning_rate']}  Batch size: {config['batch_size']}  Seed: {seed}")
    print(f"{'=' * 60}")

    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=1000, shuffle=False)

    if config["model_type"] == "mlp":
        model = SimpleNetwork(784, config["hidden_sizes"])
    else:
        f1, f2 = config["filters"]
        model = SimpleCNN(f1, f2, input_channels=3, image_size=32)
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=config["learning_rate"])

    meter = EnergyMeter(f"run_{run_id}").start()
    rows, windows = [], []

    for epoch in range(1, EPOCHS_PER_MODEL + 1):
        t0 = time.time()

        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
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
        windows.append((t0, t1))          # energy window = training + validation only

        # Test accuracy is for reporting only; it is measured outside the energy window
        # and is never used to label epochs or train the stopping model.
        _, test_acc = evaluate(model, test_loader, loss_function)

        if epoch == 1:
            d_loss = d_acc = 0.0
        else:
            d_loss = rows[-1]["val_loss"] - val_loss
            d_acc = val_acc - rows[-1]["val_acc"]

        rows.append({
            "run_id": run_id, "epoch": epoch,
            "dataset": config["dataset"], "model_type": config["model_type"],
            "config": config["config_label"],
            "hidden_sizes": str(config.get("hidden_sizes", "")),
            "filters": str(config.get("filters", "")),
            "learning_rate": config["learning_rate"], "batch_size": config["batch_size"],
            "seed": seed,
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
# SECTION D: The 160 experiments
# ---------------------------------------------------------------------------
MLP_LEARNING_RATES = [0.01, 0.005, 0.001, 0.0005]
CNN_LEARNING_RATES = [0.002, 0.001, 0.0005, 0.0001]     # lower: 0.005 / 0.01 diverged before
BATCH_SIZES = [32, 64, 128, 256]

MLP_ARCHITECTURES = [[64], [128], [128, 64], [256, 128], [256, 128, 64]]
CNN_ARCHITECTURES = [(16, 32), (32, 64), (64, 128), (32, 32), (16, 64)]


def make_configs(dataset, model_type, start_id):
    if model_type == "mlp":
        picks = balanced_configs(MLP_ARCHITECTURES, MLP_LEARNING_RATES, BATCH_SIZES, seed=SEED)
    else:
        picks = balanced_configs(CNN_ARCHITECTURES, CNN_LEARNING_RATES, BATCH_SIZES, seed=SEED)
    assert len(picks) == RUNS_PER_DATASET
    configs = []
    for i, p in enumerate(picks):
        c = {"run_id": start_id + i, "dataset": dataset, "model_type": model_type,
             "learning_rate": p["learning_rate"], "batch_size": p["batch_size"]}
        if model_type == "mlp":
            c["hidden_sizes"] = p["arch"]
            c["config_label"] = f"MLP{p['arch']}"
        else:
            c["filters"] = tuple(p["arch"])
            c["config_label"] = f"CNN({p['arch'][0]},{p['arch'][1]})"
        configs.append(c)
    return configs


all_configs = (make_configs("mnist", "mlp", 1) + make_configs("fashionmnist", "mlp", 41)
               + make_configs("cifar10", "cnn", 81) + make_configs("svhn", "cnn", 121))
total_runs = len(all_configs)


# ---------------------------------------------------------------------------
# SECTION E: Run everything (resumable)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(dataset_map),
                        choices=list(dataset_map), help="which datasets to run (default: all four)")
    parser.add_argument("--log-dir", default=LOG_DIR, help="where to write the run CSVs")
    cli = parser.parse_args()
    LOG_DIR = cli.log_dir
    all_configs = [c for c in all_configs if c["dataset"] in cli.datasets]
    total_runs = len(all_configs)

    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"\n{'#' * 60}\n  {total_runs} runs x {EPOCHS_PER_MODEL} epochs on {device}\n{'#' * 60}")

    skipped = 0
    for config in all_configs:
        csv_path = os.path.join(LOG_DIR, f"run_{config['run_id']:03d}.csv")
        if os.path.exists(csv_path):
            skipped += 1
            print(f"  Run {config['run_id']}/{total_runs} already done - skipping")
            continue
        train_one_model(config, total_runs)

    # The combined log holds EVERY run CSV in the folder, so running the datasets in separate
    # sessions (e.g. MNIST + FashionMNIST first, CIFAR-10 + SVHN later) builds up one file.
    done_here = sum(os.path.exists(os.path.join(LOG_DIR, f"run_{c['run_id']:03d}.csv")) for c in all_configs)
    parts = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(LOG_DIR, "run_*.csv")))]
    combined = pd.concat(parts, ignore_index=True)
    combined_path = os.path.join(LOG_DIR, "all_runs_combined.csv")
    combined.to_csv(combined_path, index=False)

    print(f"\n{'#' * 60}")
    if done_here == total_runs:
        print(f"  ALL DONE: {total_runs} models x {EPOCHS_PER_MODEL} epochs this session; "
              f"combined log has {len(parts)} runs, {len(combined)} rows")
    else:
        print(f"  {done_here}/{total_runs} runs completed - run the script again to continue")
    print("  Datasets run: " + ", ".join(cli.datasets) + "  (run ids: 1-40 MNIST, 41-80 FashionMNIST, "
          "81-120 CIFAR-10, 121-160 SVHN)")
    print(f"  Combined log: {combined_path}\n{'#' * 60}")
