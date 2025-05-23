#!/usr/bin/env python3
import argparse
import re
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# ----- Directories -----
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "Results_figures"  # This folder contains subfolders like rep{n}
TRAINS_DIR = RESULTS_DIR / "Results_trains"
TRAINS_DIR.mkdir(parents=True, exist_ok=True)
SUMMARY_DIR = RESULTS_DIR / "Results_summary"   # Used for output
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# ----- Thresholds for filtering -----
MIN_LOG_RANK = -1
MAX_LOG_RANK = 1
MIN_C_INDEX  = 0

# ----- Color mapping for losses -----
loss_color_map = {'coxph': 'green', 'rank': 'blue', 'MSE': 'orange', 'SurvPLE': 'purple'}

###############################################################################
# REGEX PATTERN FOR SINGLE-LOSS FILES 
###############################################################################
# Examples:
# With external dataset:
# ABMIL_rank_lr1e-4_w1.00__Epc10_[0.123-0.456-0.789]_[0.800-0.750-0.770]_ExtABC_11.png
# Without external dataset:
# ABMIL_rank_lr1e-4_w1.00__Epc10_[0.123-0.456-0.789]_[0.800-0.750-0.770]_Exp_11.png
pattern_single = re.compile(r"""
    ^
    (?P<model>ACMIL|CLAM_SB|CLAM_MB|TransMIL|DSMIL|MeanMIL|MaxMIL|ABMIL|GABMIL)  # model
    _
    (?P<loss>coxph|rank|MSE|SurvPLE)                                            # loss
    _lr(?P<lr>[0-9.+\-eE]+)                                                      # learning-rate
    _w(?P<weight>[-\d.]+)                                                        # weight
    _Epc(?P<epoch>\d+)                                                           # epoch
    _LogRanks\[(?P<lr_train>[-\d.]+)-(?P<lr_valid>[-\d.]+)-(?P<lr_test>[-\d.]+)\] # log-rank tests
    _CIdx\[(?P<c_train>[-\d.]+)-(?P<c_valid>[-\d.]+)-(?P<c_test>[-\d.]+)\]        # C-indices
    _(?P<prefix>Ext(?:[A-Za-z0-9_]+)?|Exp)                                       # Ext… or Exp
    _(?P<repeat>\d+)                                                             # repeat
    \.png$                                                                       # png suffix
""", re.VERBOSE)

###############################################################################
# PARSING FUNCTION FOR SINGLE-LOSS FILES 
###############################################################################
def extract_file_info(filename: str) -> dict:
    """
    Parse single-loss filenames (with mandatory repeat).
    Returns a dict with keys: model, loss, learning_rate, weight, epoch,
    lr_train, lr_valid, lr_test, c_train, c_valid, c_test, external, repeat.
    
    Examples:
      ABMIL_rank_lr1.00e-04_w1.0__Epc10_[0.138-0.183-0.899]_[0.465-0.421-0.508]_Ext_11.png  -> external=""
      ABMIL_rank_lr1.00e-04_w1.0__Epc10_[0.918-0.159-0.026]_[0.508-0.687-0.476]_ExtCHOL_12.png   -> external="CHOL"
      ABMIL_rank_lr1.00e-04_w1.0__Epc10_[... ]_[... ]_Exp_11.png -> external="" (Exp means no external)
    """
    match = pattern_single.match(filename)
    if match:
        prefix = match.group("prefix")
        # If prefix starts with "Ext" and is longer than "Ext", remove the "Ext" part.
        if prefix.startswith("Ext") and prefix != "Ext":
            external = prefix[3:]
        else:
            external = ""
        return {
            "model": match.group("model"),
            "loss": match.group("loss"),
            "learning_rate": float(match.group("lr")),
            "weight": match.group("weight"),
            "epoch": int(match.group("epoch")),
            "lr_train": float(match.group("lr_train")),
            "lr_valid": float(match.group("lr_valid")),
            "lr_test": float(match.group("lr_test")),
            "c_train": float(match.group("c_train")),
            "c_valid": float(match.group("c_valid")),
            "c_test": float(match.group("c_test")),
            "external": external,
            "repeat": match.group("repeat")
        }
    return None

def process_files(input_folder: Path) -> dict:
    """
    Collect file info from the given input_folder.
    Returns a dict keyed by (model, loss, learning_rate, weight, external)
    with a list of tuples:
      (epoch, lr_train, lr_valid, lr_test, c_train, c_valid, c_test, filename, external).
    """
    data = defaultdict(list)
    print(f"Processing folder: {input_folder}")
    for filename in os.listdir(input_folder):
        info = extract_file_info(filename)
        if info is None:
            continue
        key = (
            info["model"],
            info["loss"],
            info["learning_rate"],
            info["weight"],
            info["external"]
        )
        data[key].append((
            info["epoch"],
            info["lr_train"],
            info["lr_valid"],
            info["lr_test"],
            info["c_train"],
            info["c_valid"],
            info["c_test"],
            filename,
            info["external"]
        ))
    for key in data:
        data[key].sort(key=lambda x: x[0])
    return data

def plot_group(key, values, repeat_override: str = "") -> None:
    """
    Given a key (model, loss, learning_rate, weight, external) and the aggregated values,
    create a plot showing Log-Rank and C-Index progress over epochs.
    The plot title and filename will include the external dataset name if present.
    The figure is saved under TRAINS_DIR/rep{repeat_override} (or rep_{external} if repeat_override is empty).
    """
    model, loss, learning_rate, _, external = key  # omit weight from label
    values.sort(key=lambda x: x[0])
    offset = 0.001
    epochs = [v[0] for v in values]
    lr_train_vals = [v[1] + offset for v in values]
    lr_valid_vals = [v[2] + offset for v in values]
    lr_test_vals  = [v[3] + offset for v in values]
    c_train_vals  = [v[4] for v in values]
    c_valid_vals  = [v[5] for v in values]
    c_test_vals   = [v[6] for v in values]

    best_index, best_c_test = max(enumerate(c_test_vals), key=lambda x: x[1])
    best_epoch = epochs[best_index]
    best_lr_test = lr_test_vals[best_index]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    rep_str = repeat_override if repeat_override else ""
    external_str = f"Ext{external}" if external else ""
    title_str = f"{model} - {loss}, lr={learning_rate:.1e} {external_str} {('rep=' + rep_str) if rep_str else ''}"
    title_str += f"\nBest: Test C-Index = {best_c_test:.3f} @ epoch {best_epoch} (Test LR = {best_lr_test:.3f})"
    fig.suptitle(title_str, fontsize=16)

    axs[0].plot(epochs, lr_train_vals, marker="o", label="Train", color="green")
    axs[0].plot(epochs, lr_valid_vals, marker="o", label="Valid", color="blue")
    axs[0].plot(epochs, lr_test_vals, marker="o", label="Test", color="red")
    axs[0].axhline(y=0.05, color="gray", linestyle=":", label="LR=0.05")
    axs[0].axhline(y=0.01, color="gray", linestyle=":", label="LR=0.01")
    axs[0].set_yscale("log")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Log-Rank (log scale)")
    axs[0].set_title("Log-Rank Over Epochs")
    axs[0].legend(fontsize=8, loc="upper right")

    axs[1].plot(epochs, c_train_vals, marker="o", label="Train", color="green")
    axs[1].plot(epochs, c_valid_vals, marker="o", label="Valid", color="blue")
    axs[1].plot(epochs, c_test_vals, marker="o", label="Test", color="red")
    axs[1].axhline(y=0.5, color="gray", linestyle=":", label="C=0.5")
    axs[1].axhline(y=0.6, color="gray", linestyle=":", label="C=0.6")
    axs[1].axhline(y=0.7, color="gray", linestyle=":", label="C=0.7")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("C-Index")
    axs[1].set_title("C-Index Over Epochs")
    axs[1].legend(fontsize=8, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    if external_str:
        out_filename = f"{model}_{loss}_lr{learning_rate:.1e}_{external_str}_rep{rep_str}.png"
    else:
        out_filename = f"{model}_{loss}_lr{learning_rate:.1e}_rep{rep_str}.png"
    if rep_str:
        rep_folder = TRAINS_DIR / f"rep{rep_str}"
    else:
        rep_folder = TRAINS_DIR / (f"rep_{external_str}" if external_str else "rep_all")
    rep_folder.mkdir(parents=True, exist_ok=True)
    output_path = rep_folder / out_filename
    plt.savefig(output_path)
    plt.close(fig)
    print(f"Saved aggregated plot to {output_path}")

def main(args):
    # Process each rep folder provided in --repeat (multiple values allowed)
    repeats = args.repeat  # list of repeat values (as strings)
    for rep in repeats:
        input_folder = FIGURES_DIR / f"rep{rep}"
        if not input_folder.exists():
            print(f"Warning: Folder {input_folder} does not exist. Skipping.")
            continue
        print(f"Processing folder: {input_folder}")
        data = process_files(input_folder)
        for key, values in data.items():
            plot_group(key, values, repeat_override=str(rep))
    
    # Create a new combined folder with a name like rep3_4_5 from the provided repeats.
    combined_folder_name = "rep" + "_".join(sorted(repeats))
    combined_folder = TRAINS_DIR / combined_folder_name
    combined_folder.mkdir(parents=True, exist_ok=True)
    print(f"Combined folder created: {combined_folder}")
    # (Additional combined aggregation can be implemented later here.)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate single‑loss figures from specified rep folders in FIGURES_DIR and produce aggregated plots."
    )
    parser.add_argument("--repeat", type=str, nargs='+', required=True,
                        help="List of repeat suffixes to aggregate (e.g., --repeat 3 4 5). Only files in FIGURES_DIR/rep{n} will be processed.")
    args = parser.parse_args()
    main(args)
