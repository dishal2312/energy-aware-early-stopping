"""
STEP 1: Baseline Model Training
What this does:
  - Trains 40 small neural networks on MNIST (handwritten digits)
  - Each model has different settings (learning rate, batch size, network shape)
  - At every epoch, we log: loss, accuracy, time, and energy used
  - All logs are saved to CSV files in the 'training_logs/' folder

How to run:
  1. Install libraries:  install the dependencies using requirements.txt
  2. Run this script:    python step1_baseline_training.py

Output:
  - training_logs/run_001.csv, run_002.csv, etc.  (one per model)
  - training_logs/all_runs_combined.csv            (everything in one file)
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
# If it doesn't work on your machine, we fall back to time-based estimation
try:
    from codecarbon import OfflineEmissionsTracker
    CODECARBON_AVAILABLE = True
    print("CodeCarbon loaded — will track real energy usage")
except ImportError:
    CODECARBON_AVAILABLE = False
    print("CodeCarbon not installed — will estimate energy from time")
    print("   (Install with: pip install codecarbon)")



# SETTINGS — you can change these if you want

EPOCHS_PER_MODEL = 30     # How many epochs to train each model
NUM_RUNS = 40             # How many different models to train
DATA_DIR = "./data"       # Where to download MNIST
LOG_DIR = "./training_logs"  # Where to save CSV logs



# STEP A: Load the MNIST dataset

print("\nLoading MNIST dataset...")

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Standard MNIST normalisation
])

full_train = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transform)
test_data = datasets.MNIST(DATA_DIR, train=False, download=True, transform=transform)

# Split training data: 50,000 for training, 10,000 for validation
train_data, val_data = random_split(full_train, [50000, 10000])

print(f"  Training samples:   {len(train_data)}")
print(f"  Validation samples: {len(val_data)}")
print(f"  Test samples:       {len(test_data)}")



# STEP B: Define a simple neural network

class SimpleNetwork(nn.Module):
    """
    A basic feedforward neural network.
    Takes a 28x28 image (flattened to 784 numbers) and outputs 10 predictions
    (one for each digit 0-9).

    hidden_sizes controls the shape of the network.
    Example: [128, 64] means two hidden layers with 128 and 64 neurons.
    """

    def __init__(self, hidden_sizes):
        super().__init__()

        # Build layers one by one
        layers = []
        prev_size = 784  # MNIST images are 28x28 = 784 pixels

        for size in hidden_sizes:
            layers.append(nn.Linear(prev_size, size))  # Connections between layers
            layers.append(nn.ReLU())                    # Activation function
            prev_size = size

        layers.append(nn.Linear(prev_size, 10))  # Final layer: 10 digit classes
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the 28x28 image into a 1D vector
        return self.network(x)



# STEP C: Function to train one model

def train_one_model(run_id, hidden_sizes, learning_rate, batch_size):
    """
    Trains one model for 30 epochs and returns a list of logged data.
    Each epoch records: loss, accuracy, time, and energy.
    """

    print(f"\n{'='*60}")
    print(f"  Run {run_id}/{NUM_RUNS}")
    print(f"  Network shape: {hidden_sizes}")
    print(f"  Learning rate:  {learning_rate}")
    print(f"  Batch size:     {batch_size}")
    print(f"{'='*60}")

    # Set up data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=1000, shuffle=False)

    # Create the model, loss function, and optimiser
    model = SimpleNetwork(hidden_sizes)
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=learning_rate)

    # Start energy tracking for this entire run
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

    # This list will hold one row of data per epoch
    epoch_logs = []
    total_energy_so_far = 0.0

    for epoch in range(1, EPOCHS_PER_MODEL + 1):
        epoch_start_time = time.time()

        # ---- TRAINING PHASE ----
        model.train() 
        total_train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            optimiser.zero_grad()          # Reset gradients
            predictions = model(images)    # Forward pass
            loss = loss_function(predictions, labels)  # Calculate loss
            loss.backward()                # Backward pass (compute gradients)
            optimiser.step()               # Update weights

            # Track running totals
            total_train_loss += loss.item() * images.size(0)
            _, predicted_class = predictions.max(1)
            train_correct += predicted_class.eq(labels).sum().item()
            train_total += images.size(0)

        train_loss = total_train_loss / train_total
        train_accuracy = train_correct / train_total

        # ---- VALIDATION PHASE ----
        model.eval()
        total_val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():  # Don't track gradients during validation
            for images, labels in val_loader:
                predictions = model(images)
                loss = loss_function(predictions, labels)

                total_val_loss += loss.item() * images.size(0)
                _, predicted_class = predictions.max(1)
                val_correct += predicted_class.eq(labels).sum().item()
                val_total += images.size(0)

        val_loss = total_val_loss / val_total
        val_accuracy = val_correct / val_total

        # ---- TIMING AND ENERGY ----
        epoch_time = time.time() - epoch_start_time

        # Estimate energy for this epoch
        # If CodeCarbon is running, we estimate proportionally from cumulative energy
        # Otherwise, we use: time × estimated CPU power (15W is typical for a laptop)
        if energy_tracker is not None:
            try:
                cumulative = energy_tracker._total_energy.kWh
                epoch_energy = cumulative - total_energy_so_far
                total_energy_so_far = cumulative
            except Exception:
                epoch_energy = epoch_time * 15.0 / 3600000.0  # 15W fallback
                total_energy_so_far += epoch_energy
        else:
            # Fallback: estimate using 15 watts (typical laptop CPU)
            epoch_energy = epoch_time * 15.0 / 3600000.0  # Convert W·s to kWh
            total_energy_so_far += epoch_energy

        # ---- CALCULATE IMPROVEMENT ----
        if epoch == 1:
            val_loss_improvement = 0.0
            val_acc_improvement = 0.0
        else:
            prev = epoch_logs[-1]
            val_loss_improvement = prev["val_loss"] - val_loss   # Positive = loss went down (good)
            val_acc_improvement = val_accuracy - prev["val_acc"]  # Positive = accuracy went up (good)

        # ---- SAVE THIS EPOCH'S DATA ----
        epoch_logs.append({
            "run_id": run_id,
            "epoch": epoch,
            "hidden_sizes": str(hidden_sizes),
            "learning_rate": learning_rate,
            "batch_size": batch_size,
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

        # Print progress
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

    # Save this run's logs to its own CSV file
    df = pd.DataFrame(epoch_logs)
    filepath = os.path.join(LOG_DIR, f"run_{run_id:03d}.csv")
    df.to_csv(filepath, index=False)
    print(f"  ✅Saved to {filepath}")

    return df



# STEP D: Define the 40 experiments

# We vary three things to get different training curves:
#   1. Network shape (how many neurons in each layer)
#   2. Learning rate (how fast the model learns)
#   3. Batch size (how many images per training step)

architectures = [
    [64],            # Tiny:   1 layer, 64 neurons
    [128],           # Small:  1 layer, 128 neurons
    [128, 64],       # Medium: 2 layers
    [256, 128],      # Large:  2 layers
    [256, 128, 64],  # Deep:   3 layers
]

learning_rates = [0.01, 0.005, 0.001, 0.0005]
batch_sizes = [32, 64, 128, 256]

# Generate all possible combinations
all_configs = []
for arch in architectures:
    for lr in learning_rates:
        for bs in batch_sizes:
            all_configs.append({
                "hidden_sizes": arch,
                "learning_rate": lr,
                "batch_size": bs,
            })

# We have 5 × 4 × 4 = 80 combinations — pick 40 evenly spread ones
selected_configs = all_configs[::2]  # Take every 2nd one
selected_configs = selected_configs[:NUM_RUNS]  # Make sure we have exactly 40



# STEP E: Run all experiments

os.makedirs(LOG_DIR, exist_ok=True)

print(f"\n{'#'*60}")
print(f"  Starting {len(selected_configs)} training runs")
print(f"  {EPOCHS_PER_MODEL} epochs each")
print(f"  Estimated time: {len(selected_configs) * 1} - {len(selected_configs) * 2} minutes")
print(f"{'#'*60}")

all_run_results = []

for i, config in enumerate(selected_configs):
    result = train_one_model(
        run_id=i + 1,
        hidden_sizes=config["hidden_sizes"],
        learning_rate=config["learning_rate"],
        batch_size=config["batch_size"],
    )
    all_run_results.append(result)

# Combine all runs into one big CSV
combined = pd.concat(all_run_results, ignore_index=True)
combined_path = os.path.join(LOG_DIR, "all_runs_combined.csv")
combined.to_csv(combined_path, index=False)

print(f"\n{'#'*60}")
print(f"  ALL DONE!")
print(f"  Trained {len(selected_configs)} models × {EPOCHS_PER_MODEL} epochs")
print(f"  Total data points: {len(combined)}")
print(f"  Individual logs:   {LOG_DIR}/run_XXX.csv")
print(f"  Combined log:      {combined_path}")
print(f"{'#'*60}")
