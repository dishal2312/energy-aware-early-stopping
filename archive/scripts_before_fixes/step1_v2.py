"""
STEP 1 (Version 2): Extended Baseline Training
This is the COMPLETE version that includes everything from V1 plus additions.

What's new compared to V1:
  - FashionMNIST dataset (clothing images) with MLPs
  - CIFAR-10 dataset (real-world photos) with CNNs
  - GPU support (for Google Colab)
  - 120 total runs instead of 40

Datasets:
  - MNIST (handwritten digits)     → 40 runs with MLPs
  - FashionMNIST (clothing items)  → 40 runs with MLPs
  - CIFAR-10 (real-world photos)   → 40 runs with CNNs

How to run:
  1. Upload this file to Google Colab
  2. In Colab, run: !pip install codecarbon torch torchvision pandas numpy
  3. Run: !python step1_v2.py
  4. Download the training_logs_v2/ folder when done

Output:
  - training_logs_v2/run_001.csv to run_120.csv
  - training_logs_v2/all_runs_combined.csv
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import pandas as pd
import numpy as np
import time
import os

# Try to import CodeCarbon for energy tracking
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
    print("✅ CodeCarbon loaded — will track real energy usage")
except ImportError:
    CODECARBON_AVAILABLE = False
    print("⚠️  CodeCarbon not available — will estimate energy from time")



# SETTINGS

EPOCHS_PER_MODEL = 30
DATA_DIR = "./data"
LOG_DIR = "./training_logs_v2"

# Detect GPU (Google Colab gives you a free GPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️  Using device: {device}")
if device.type == "cuda":
    print(f"   GPU: {torch.cuda.get_device_name(0)}")



# SECTION A: Load All Four Datasets

print("\nLoading datasets...")

# MNIST (handwritten digits, 28x28, grayscale)
mnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])
mnist_full = datasets.MNIST(DATA_DIR, train=True, download=True, transform=mnist_transform)
mnist_train, mnist_val = random_split(mnist_full, [50000, 10000])
print(f"  MNIST:        {len(mnist_train)} train, {len(mnist_val)} val")

# FashionMNIST (clothing items, 28x28, grayscale)
# Same format as MNIST but with t-shirts, trousers, dresses, etc.
# This tests whether our approach works on a harder version of the same format
fmnist_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530,))
])
fmnist_full = datasets.FashionMNIST(DATA_DIR, train=True, download=True, transform=fmnist_transform)
fmnist_train, fmnist_val = random_split(fmnist_full, [50000, 10000])
print(f"  FashionMNIST: {len(fmnist_train)} train, {len(fmnist_val)} val")

# CIFAR-10 (real photos of objects, 32x32, colour)
# Cars, planes, cats, dogs, etc. — a genuine step up in difficulty
# Uses CNNs instead of MLPs because CNNs understand images much better
cifar_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
cifar_full = datasets.CIFAR10(DATA_DIR, train=True, download=True, transform=cifar_transform)
cifar_train, cifar_val = random_split(cifar_full, [40000, 10000])
print(f"  CIFAR-10:     {len(cifar_train)} train, {len(cifar_val)} val")

# SVHN (Street View House Numbers, 32x32, colour) NEW
# Real photographs of house numbers taken from Google Street View
# This is a REAL-WORLD dataset — not curated or artificial
# Same format as CIFAR-10 (32x32 colour, 10 classes) so uses CNNs too
svhn_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))
])
svhn_full = datasets.SVHN(DATA_DIR, split='train', download=True, transform=svhn_transform)
svhn_train, svhn_val = random_split(svhn_full, [60000, len(svhn_full) - 60000])
print(f"  SVHN:         {len(svhn_train)} train, {len(svhn_val)} val")



# SECTION B: Define the Neural Networks


# MLP (same as V1) Used for MNIST and FashionMNIST
class SimpleNetwork(nn.Module):
    """
    A basic feedforward neural network (MLP).
    Takes a flattened image and outputs 10 class predictions.
    Used for MNIST and FashionMNIST.
    """
    def __init__(self, input_size, hidden_sizes):
        super().__init__()
        layers = []
        prev_size = input_size

        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))
            layers.append(nn.ReLU())
            prev_size = size

        layers.append(nn.Linear(prev_size, 10))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten image to 1D
        return self.network(x)


# CNN (NEW) Used for CIFAR-10
class SimpleCNN(nn.Module):
    """
    A simple Convolutional Neural Network (CNN).

    Unlike an MLP which just sees raw pixel numbers, a CNN looks for
    PATTERNS (edges, shapes, textures) in small regions of the image.
    This is much better for understanding pictures.

    How it works:
      1. Conv layers scan the image looking for patterns (like a magnifying glass)
      2. Pool layers shrink the image (keeps important info, removes noise)
      3. FC layers make the final classification decision

    num_filters controls how many different patterns each layer looks for.
    More filters = more patterns = smarter but slower.
    """
    def __init__(self, num_filters_1, num_filters_2, input_channels=3, image_size=32):
        super().__init__()

        # Convolutional layers — scan for patterns
        self.conv1 = nn.Conv2d(input_channels, num_filters_1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(num_filters_1, num_filters_2, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2)  # Shrinks image by half each time
        self.relu = nn.ReLU()

        # After 2 pool layers: image shrinks from 32x32 → 16x16 → 8x8
        reduced_size = image_size // 4
        flat_size = num_filters_2 * reduced_size * reduced_size

        # Fully connected layers — make the final decision
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Layer 1: find simple patterns (edges, corners)
        x = self.pool(self.relu(self.conv1(x)))
        # Layer 2: find complex patterns (shapes, textures)
        x = self.pool(self.relu(self.conv2(x)))
        # Flatten and classify
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x



# SECTION C: Training Function (works for any model + dataset)

def train_one_model(run_id, total_runs, model, train_loader, val_loader,
                    learning_rate, dataset_name, model_type, config_label):
    """
    Trains one model for 30 epochs and logs everything.
    Works for both MLPs and CNNs — the model handles the difference.
    """

    print(f"\n{'='*60}")
    print(f"  Run {run_id}/{total_runs} [{dataset_name.upper()}]")
    print(f"  Model: {model_type}  |  Config: {config_label}")
    print(f"  Learning rate: {learning_rate}")
    print(f"{'='*60}")

    # Move model to GPU if available
    model = model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    # Start energy tracking
    energy_tracker = None
    if CODECARBON_AVAILABLE:
        try:
            energy_tracker = OfflineEmissionsTracker(
                country_iso_code="GBR",
                project_name=f"run_{run_id}",
                log_level="error",
                save_to_file=False,
            )
            energy_tracker.start()
        except Exception:
            energy_tracker = None

    epoch_logs = []
    total_energy_so_far = 0.0

    for epoch in range(1, EPOCHS_PER_MODEL + 1):
        epoch_start_time = time.time()

        #- TRAINING-
        model.train()
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimiser.zero_grad()
            predictions = model(images)
            loss = loss_function(predictions, labels)
            loss.backward()
            optimiser.step()

            total_train_loss += loss.item() * images.size(0)
            _, predicted_class = predictions.max(1)
            train_correct += predicted_class.eq(labels).sum().item()
            train_total += images.size(0)

        train_loss = total_train_loss / train_total
        train_accuracy = train_correct / train_total

        #- VALIDATION-
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                predictions = model(images)
                loss = loss_function(predictions, labels)

                total_val_loss += loss.item() * images.size(0)
                _, predicted_class = predictions.max(1)
                val_correct += predicted_class.eq(labels).sum().item()
                val_total += images.size(0)

        val_loss = total_val_loss / val_total
        val_accuracy = val_correct / val_total

        #- TIMING AND ENERGY-
        epoch_time = time.time() - epoch_start_time

        if energy_tracker is not None:
            try:
                cumulative = energy_tracker._total_energy.kWh
                epoch_energy = cumulative - total_energy_so_far
                total_energy_so_far = cumulative
            except Exception:
                epoch_energy = epoch_time * 15.0 / 3600000.0
                total_energy_so_far += epoch_energy
        else:
            epoch_energy = epoch_time * 15.0 / 3600000.0
            total_energy_so_far += epoch_energy

        #- IMPROVEMENT-
        if epoch == 1:
            val_loss_improvement = 0.0
            val_acc_improvement = 0.0
        else:
            prev = epoch_logs[-1]
            val_loss_improvement = prev["val_loss"] - val_loss
            val_acc_improvement = val_accuracy - prev["val_acc"]

        #- LOG THIS EPOCH-
        epoch_logs.append({
            "run_id": run_id,
            "epoch": epoch,
            "dataset": dataset_name,
            "model_type": model_type,
            "config": config_label,
            "learning_rate": learning_rate,
            "train_loss": round(train_loss, 6),
            "train_acc": round(train_accuracy, 4),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_accuracy, 4),
            "val_loss_improvement": round(val_loss_improvement, 6),
            "val_acc_improvement": round(val_acc_improvement, 6),
            "epoch_time_secs": round(epoch_time, 3),
            "epoch_energy_kwh": epoch_energy,
            "cumulative_energy_kwh": total_energy_so_far,
        })

        print(f"  Epoch {epoch:2d}/{EPOCHS_PER_MODEL} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_accuracy:.4f} | "
              f"Time: {epoch_time:.1f}s")

    # Stop energy tracking
    if energy_tracker is not None:
        try:
            energy_tracker.stop()
        except Exception:
            pass

    # Save to CSV
    df = pd.DataFrame(epoch_logs)
    filepath = os.path.join(LOG_DIR, f"run_{run_id:03d}.csv")
    df.to_csv(filepath, index=False)
    print(f"  ✅ Saved to {filepath}")

    return df



# SECTION D: Define All 120 Experiments


# Shared settings
learning_rates = [0.01, 0.005, 0.001, 0.0005]
batch_sizes = [32, 64, 128, 256]

# MLP architectures (for MNIST and FashionMNIST)
mlp_architectures = [
    [64],
    [128],
    [128, 64],
    [256, 128],
    [256, 128, 64],
]

# CNN architectures (for CIFAR-10)
# (filters_layer_1, filters_layer_2)
cnn_architectures = [
    (16, 32),      # Small
    (32, 64),      # Medium
    (64, 128),     # Large
    (32, 32),      # Uniform
    (16, 64),      # Expanding
]


def generate_mlp_configs(dataset_name, start_run_id):
    """Generate 40 MLP experiment configs for a dataset."""
    configs = []
    for arch in mlp_architectures:
        for lr in learning_rates:
            for bs in batch_sizes:
                configs.append({
                    "dataset": dataset_name,
                    "model_type": "mlp",
                    "hidden_sizes": arch,
                    "learning_rate": lr,
                    "batch_size": bs,
                    "config_label": f"MLP{arch}",
                })
    # 5 × 4 × 4 = 80 combos, take every 2nd = 40
    configs = configs[::2][:40]
    # Assign run IDs
    for i, c in enumerate(configs):
        c["run_id"] = start_run_id + i
    return configs


def generate_cnn_configs(dataset_name, start_run_id):
    """Generate 40 CNN experiment configs for a dataset."""
    configs = []
    for (f1, f2) in cnn_architectures:
        for lr in learning_rates:
            for bs in batch_sizes:
                configs.append({
                    "dataset": dataset_name,
                    "model_type": "cnn",
                    "filters": (f1, f2),
                    "learning_rate": lr,
                    "batch_size": bs,
                    "config_label": f"CNN({f1},{f2})",
                })
    configs = configs[::2][:40]
    for i, c in enumerate(configs):
        c["run_id"] = start_run_id + i
    return configs


# Generate all configs
mnist_configs = generate_mlp_configs("mnist", start_run_id=1)
fmnist_configs = generate_mlp_configs("fashionmnist", start_run_id=41)
cifar_configs = generate_cnn_configs("cifar10", start_run_id=81)
svhn_configs = generate_cnn_configs("svhn", start_run_id=121)

all_configs = mnist_configs + fmnist_configs + cifar_configs + svhn_configs
total_runs = len(all_configs)

print(f"\n  Total experiments: {total_runs}")
print(f"    MNIST MLPs:        {len(mnist_configs)} runs")
print(f"    FashionMNIST MLPs: {len(fmnist_configs)} runs")
print(f"    CIFAR-10 CNNs:     {len(cifar_configs)} runs")
print(f"    SVHN CNNs:         {len(svhn_configs)} runs")



# SECTION E: Run All Experiments

os.makedirs(LOG_DIR, exist_ok=True)

# Map dataset names to their data
dataset_map = {
    "mnist": (mnist_train, mnist_val),
    "fashionmnist": (fmnist_train, fmnist_val),
    "cifar10": (cifar_train, cifar_val),
    "svhn": (svhn_train, svhn_val),
}

print(f"\n{'#'*60}")
print(f"  Starting {total_runs} training runs on {device}")
print(f"  {EPOCHS_PER_MODEL} epochs each")
print(f"{'#'*60}")

all_results = []

skipped_count = 0

for config in all_configs:
    run_id = config["run_id"]
    csv_path = os.path.join(LOG_DIR, f"run_{run_id:03d}.csv")

    # RESUME FEATURE: If this run's CSV already exists, skip it
    if os.path.exists(csv_path):
        skipped_count += 1
        print(f"  ⏭️  Run {run_id}/{total_runs} already done — skipping")
        continue

    # Get the right dataset
    train_set, val_set = dataset_map[config["dataset"]]
    train_loader = DataLoader(train_set, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=1000, shuffle=False)

    # Create the right model
    if config["model_type"] == "mlp":
        # MNIST and FashionMNIST images are 28×28 = 784 pixels
        model = SimpleNetwork(input_size=784, hidden_sizes=config["hidden_sizes"])
    else:
        # CIFAR-10 images are 32×32×3 (colour)
        f1, f2 = config["filters"]
        model = SimpleCNN(num_filters_1=f1, num_filters_2=f2,
                          input_channels=3, image_size=32)

    # Train it
    result = train_one_model(
        run_id=run_id,
        total_runs=total_runs,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config["learning_rate"],
        dataset_name=config["dataset"],
        model_type=config["model_type"],
        config_label=config["config_label"],
    )

if skipped_count > 0:
    print(f"\n  ⏭️  Skipped {skipped_count} already-completed runs")

# Combine ALL existing CSVs (including previously completed ones)
all_csvs = []
for config in all_configs:
    csv_path = os.path.join(LOG_DIR, f"run_{config['run_id']:03d}.csv")
    if os.path.exists(csv_path):
        all_csvs.append(pd.read_csv(csv_path))

combined = pd.concat(all_csvs, ignore_index=True)
combined_path = os.path.join(LOG_DIR, "all_runs_combined.csv")
combined.to_csv(combined_path, index=False)

completed = len(all_csvs)
remaining = total_runs - completed

print(f"\n{'#'*60}")
if remaining == 0:
    print(f"  ✅ ALL DONE!")
    print(f"  Trained {total_runs} models × {EPOCHS_PER_MODEL} epochs")
    print(f"  Total data points: {len(combined)}")
else:
    print(f"  ⚠️  {completed}/{total_runs} runs completed. {remaining} still missing.")
    print(f"  Run this script again to continue.")
print(f"")
print(f"  MNIST runs:        1-40   ({len(mnist_configs)} runs)")
print(f"  FashionMNIST runs: 41-80  ({len(fmnist_configs)} runs)")
print(f"  CIFAR-10 runs:     81-120 ({len(cifar_configs)} runs)")
print(f"  SVHN runs:         121-160 ({len(svhn_configs)} runs)")
print(f"")
print(f"  Combined log: {combined_path}")
print(f"{'#'*60}")
