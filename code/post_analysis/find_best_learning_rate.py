#!/usr/bin/env python3
import argparse
import re
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os

# ----- Directories -----
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "Results_figures"  # Contains subfolders like rep3, rep4, etc.
SUMMARY_DIR = RESULTS_DIR / "Results_summary"    # Used for output
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# ----- Thresholds for filtering -----
MIN_LOG_RANK = -1
MAX_LOG_RANK = 1
MIN_C_INDEX  = 0

loss_color_map = {'coxph': 'green', 'rank': 'blue', 'MSE': 'orange', 'SurvPLE': 'purple'}

# ----- Regex pattern for single-loss files -----
# Example valid filename:
# ACMIL_coxph_lr0.0001_w1.00__Epc10_[0.123-0.456-0.789]_[0.800-0.750-0.770]_Ext_3.png
pattern_single = re.compile(
    r"""
    ^
    (?P<model>ACMIL|CLAM_SB|CLAM_MB|TransMIL|DSMIL|MeanMIL|MaxMIL|ABMIL|GABMIL)
    _
    (?P<loss>coxph|rank|MSE|SurvPLE)
    _lr(?P<lr>[0-9.\-+eE]+)
    _w(?P<weight>[-\d.]+)
    __Epc(?P<epoch>\d+)
    _\[(?P<lr_train>[-\d.]+)-(?P<lr_valid>[-\d.]+)-(?P<lr_test>[-\d.]+)\]
    _\[(?P<c_train>[-\d.]+)-(?P<c_valid>[-\d.]+)-(?P<c_test>[-\d.]+)\]
    _Ext_(?P<repeat>\d+)
    \.png$
    """,
    re.VERBOSE
)

# ----- Parsing function -----
def extract_file_info(filename: str) -> dict:
    """
    Parse single-loss filenames (with mandatory repeat).
    Returns a dict with keys: model, loss, learning_rate, weight, epoch,
    lr_train, lr_valid, lr_test, c_train, c_valid, c_test, repeat.
    Returns None if no match.
    """
    match = pattern_single.match(filename)
    # Debug print (optional): print(match)
    if match:
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
            "repeat": match.group("repeat")
        }
    return None

def parse_single_loss_files_from_folder(input_folder: Path) -> dict:
    """
    Parse all single-loss files in the given input_folder.
    Returns a dict keyed by (model, loss, learning_rate, weight, repeat) with values containing run info.
    """
    results = {}
    for file in input_folder.iterdir():
        if not file.is_file():
            continue
        info = extract_file_info(file.name)
        if info is None:
            continue
        key = (info["model"], info["loss"], info["learning_rate"], info["weight"], info["repeat"])
        # Keep best run (highest c_test) for this key.
        if key not in results or info["c_test"] > results[key]["c_test"]:
            results[key] = {
                'model': info["model"],
                'loss': info["loss"],
                'lr': info["learning_rate"],
                'epoch': info["epoch"],
                'filename': file.name,
                'lr_test': info["lr_test"],
                'c_test': info["c_test"],
                'repeat': info["repeat"]
            }
    return results

def aggregate_best_across_reps(input_folders: list) -> pd.DataFrame:
    """
    Given a list of input folders (each corresponding to a rep folder),
    aggregate the best runs across folders by grouping by (model, loss, lr).
    Returns a DataFrame that includes a 'rep' column (the best run's rep) for each group.
    """
    records = []
    for folder in input_folders:
        folder_results = parse_single_loss_files_from_folder(folder)
        for key, info in folder_results.items():
            record = {
                'model': info['model'],
                'loss': info['loss'],
                'lr': info['lr'],
                'epoch': info['epoch'],
                'c_test': info['c_test'],
                'rep': info['repeat'],
                'filename': info['filename']
            }
            records.append(record)
    df = pd.DataFrame(records)
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate single‑loss figures from specified rep folders in FIGURES_DIR and produce learning rate vs. best Test C-Index analysis."
    )
    parser.add_argument('--repeat', type=int, nargs='+', required=True,
                        help="List of repeat numbers to aggregate (e.g., --repeat 3 4 5).")
    args = parser.parse_args()

    # Build list of input folders based on --repeat.
    input_folders = []
    for r in args.repeat:
        folder = FIGURES_DIR / f"rep{r}"
        if folder.exists():
            input_folders.append(folder)
        else:
            print(f"Warning: Folder {folder} does not exist.")
    if not input_folders:
        print("No valid rep folders found. Exiting.")
        return

    # Create a combined output folder under SUMMARY_DIR with a name based on repeats.
    repeat_suffix = "rep" + "_".join(sorted({str(r) for r in args.repeat}))
    output_folder = SUMMARY_DIR / repeat_suffix
    output_folder.mkdir(parents=True, exist_ok=True)

    # Aggregate best results from all specified rep folders.
    df = aggregate_best_across_reps(input_folders)
    if df.empty:
        print("No valid single-loss files found in the specified rep folders. Exiting.")
        return

    # Save individual summary text.
    df_sorted = df.sort_values(by='c_test', ascending=False)
    summary_file = output_folder / "best_learning_rate_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=== Best Learning Rate Summary (Individual) ===\n")
        for _, row in df_sorted.iterrows():
            f.write(f"lr={row['lr']:.2e}, c_test={row['c_test']:.3f}, epoch={row['epoch']}, rep={row['rep']}, file={row['filename']}\n")
    print(f"Saved individual summary to {summary_file}")

    # Compute aggregated statistics across reps for each learning rate.
    # Group by (model, loss, lr) since we assume same model-loss combo, then compute mean and std.
    group_cols = ['model', 'loss', 'lr']
    agg_df = df.groupby(group_cols).agg(
        mean_c_test=('c_test', 'mean'),
        std_c_test=('c_test', 'std'),
        count=('c_test', 'count')
    ).reset_index()
    # Sort by mean_c_test descending.
    agg_df.sort_values(by='mean_c_test', ascending=False, inplace=True)

    # Save aggregated summary text.
    agg_summary_file = output_folder / "best_learning_rate_summary_combined.txt"
    with open(agg_summary_file, "w") as f:
        f.write("=== Aggregated Best Learning Rate Summary ===\n")
        for _, row in agg_df.iterrows():
            f.write(f"lr={row['lr']:.2e}, mean_c_test={row['mean_c_test']:.3f}, std_c_test={row['std_c_test']:.3f}, count={int(row['count'])}\n")
    print(f"Saved aggregated summary to {agg_summary_file}")

    # Overlay plot: Plot each rep's curve and the aggregated mean with error bars.
    plt.figure(figsize=(10, 5))
    # Plot individual rep curves.
    unique_reps = df['rep'].unique()
    colors = sns.color_palette("tab10", len(unique_reps))
    for i, rep in enumerate(sorted(unique_reps)):
        df_rep = df[df['rep'] == rep].sort_values(by='lr')
        plt.plot(df_rep['lr'], df_rep['c_test'], marker='o', linestyle='None', color=colors[i], label=f"rep {rep}")
    plt.axhline(y=0.5, color="gray", linestyle=":", label="C=0.5")
    plt.axhline(y=0.6, color="gray", linestyle=":", label="C=0.6")
    plt.axhline(y=0.7, color="gray", linestyle=":", label="C=0.7")        
    # Plot aggregated mean with error bars.
    # Here we assume that the learning rate values are the same across reps.
    plt.errorbar(agg_df['lr'], agg_df['mean_c_test'], yerr=agg_df['std_c_test'], fmt='o', color='black', label="Mean ± STD", capsize=5)
    plt.xscale('log')
    plt.xlabel("Learning Rate (log scale)")
    plt.ylabel("Best Test C-Index")
    unique_models = df['model'].unique()
    unique_losses = df['loss'].unique()
    title_model = unique_models[0] if len(unique_models) == 1 else "Multiple"
    title_loss  = unique_losses[0] if len(unique_losses) == 1 else "Multiple"
    plt.title(f"{title_model} - {title_loss} (Combined Reps: {repeat_suffix})\nLearning Rate vs. Best Test C-Index")
    plt.legend()
    lr_plot_file = output_folder / "best_learning_rate_plot.png"
    plt.savefig(lr_plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved learning rate plot to {lr_plot_file}")
    plt.show()

if __name__ == "__main__":
    main()
