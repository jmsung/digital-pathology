#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

# ----- Directories -----
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "Results_figures"  # Contains subfolders like rep21, rep22, etc.
SUMMARY_DIR = RESULTS_DIR / "Results_summary"    # Used for output
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# ----- Regex pattern for single-loss files -----
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
    _Ext
    (?P<ext_datasets>[A-Za-z0-9_]*)
    _
    (?P<repeat>\d+)
    \.png$
    """,
    re.VERBOSE
)

# 1) Expanded map from repeat # to label
REPEAT_DATASET_MAP = {
    '21': "PDAC Only",
    '22': "PDAC+CHOL",
    '23': "PDAC+COAD",
    '24': "PDAC+LUAD",
    '25': "PDAC+LIHC",
    '26': "PDAC+ESCA",  # Adjust label as needed.
}

# 2) Preferred order for all possible datasets:
PREFERRED_ORDER = [
    "PDAC Only",
    "PDAC+CHOL",
    "PDAC+COAD",
    "PDAC+LUAD",
    "PDAC+ESCA",
    "PDAC+LIHC",
]

def extract_file_info(filename):
    """
    Parse single-loss filenames.
    Returns a dict with keys: model, loss, learning_rate, weight, epoch,
    lr_train, lr_valid, lr_test, c_train, c_valid, c_test, repeat.
    """
    match = pattern_single.match(filename)
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

def parse_single_loss_files_from_folder(input_folder):
    """
    Parse all single-loss .png files in the given input_folder.
    Returns a dict keyed by (model, loss, lr, weight, repeat)
    with values = the best run for that combo (based on highest c_test).
    """
    results = {}
    for file in input_folder.iterdir():
        if not file.is_file():
            continue
        info = extract_file_info(file.name)
        if info is None:
            continue
        key = (info["model"], info["loss"], info["learning_rate"], info["weight"], info["repeat"])
        if key not in results or info["c_test"] > results[key]["c_test"]:
            results[key] = info
    return results

def aggregate_best_across_folders(folders):
    """
    Parse all single-loss runs from these folders and return a DataFrame of the best run per (model, loss, lr, repeat).
    Includes c_train and c_valid.
    """
    records = []
    for folder in folders:
        folder_results = parse_single_loss_files_from_folder(folder)
        for key, info in folder_results.items():
            record = {
                'model': info['model'],
                'loss': info['loss'],
                'lr': info['learning_rate'],
                'weight': info['weight'],
                'epoch': info['epoch'],
                'c_train': info['c_train'],
                'c_valid': info['c_valid'],
                'c_test': info['c_test'],
                'repeat': info['repeat']
            }
            records.append(record)
    df = pd.DataFrame(records)
    return df

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate single‑loss runs from specified rep folders and compare c_test across multiple datasets."
    )
    parser.add_argument('--repeat', type=int, nargs='+', required=True,
                        help="List of repeat numbers to aggregate (e.g. --repeat 21 22 23 24).")
    args = parser.parse_args()

    # 1) Gather rep folders
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

    # 2) Build output folder
    repeats_str = "_".join(sorted(str(r) for r in args.repeat))
    output_folder = SUMMARY_DIR / f"rep{repeats_str}_combined"
    output_folder.mkdir(parents=True, exist_ok=True)

    # 3) Parse & aggregate
    df = aggregate_best_across_folders(input_folders)
    if df.empty:
        print("No valid single-loss files found. Exiting.")
        return

    # 4) Map repeat -> dataset
    df['repeat'] = df['repeat'].astype(str)
    df['dataset'] = df['repeat'].map(REPEAT_DATASET_MAP).fillna(df['repeat'])

    # 5) Determine datasets present
    unique_datasets = df['dataset'].unique().tolist()

    # 6) Filter preferred order
    dataset_order = [d for d in PREFERRED_ORDER if d in unique_datasets]

    # 7) Convert to categorical
    df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)

    # 8) Save raw CSV
    raw_csv_file = output_folder / "raw_aggregated_data.csv"
    df.to_csv(raw_csv_file, index=False)
    print(f"Saved raw data to {raw_csv_file}")

    # 9) Write sorted text file
    sorted_txt_file = output_folder / "sorted_by_c_test.txt"
    with open(sorted_txt_file, "w") as f:
        for repeat, group_df in df.groupby("repeat"):
            group_sorted = group_df.sort_values(by="c_test", ascending=False)
            dataset_label = group_sorted["dataset"].iloc[0]
            f.write(f"Group (repeat {repeat} - {dataset_label}):\n")
            f.write("-" * 80 + "\n")
            for idx, row in group_sorted.iterrows():
                f.write(
                    f"Model: {row['model']}, Loss: {row['loss']}, LR: {row['lr']:.2e}, "
                    f"Weight: {row['weight']}, Epoch: {row['epoch']}, "
                    f"c_train: {row['c_train']:.3f}, c_valid: {row['c_valid']:.3f}, c_test: {row['c_test']:.3f}\n"
                )
            f.write("\n")
    print(f"Saved sorted results to {sorted_txt_file}")

    # 10) Create aggregated plot using Stata s1 colors.
    # Use a palette for each dataset group.
    s1_palette = [
        "#000000",  # black (darker for extra group)
        "#3B78CB",  # blue
        "#E41A1C",  # red
        "#4DAF4A",  # green
        "#984EA3",  # purple
        "#FF7F00",  # orange
        "#999999",  # gray
        "#F781BF"   # pink
    ]
    group_palette = dict(zip(dataset_order, s1_palette[:len(dataset_order)]))

    fig, ax = plt.subplots(figsize=(12, 8))
    sns.set_style("whitegrid")
    # Plot individual points with jitter
    sns.stripplot(
        data=df,
        x='dataset',
        y='c_test',
        jitter=0.1,
        palette=group_palette,
        alpha=0.9,
        size=8,
        ax=ax,
        dodge=False
    )
    # Overlay pointplot for mean ± 95% CI using same palette.
    sns.pointplot(
        data=df,
        x='dataset',
        y='c_test',
        errorbar=('ci', 95),
        join=False,
        markers='D',
        color='black',
        ax=ax,
        dodge=False,
        ci=95
    )
    ax.axhline(y=0.5, color="gray", linestyle=":", label="C=0.5")
    ax.axhline(y=0.6, color="gray", linestyle=":", label="C=0.6")
    ax.axhline(y=0.7, color="gray", linestyle=":", label="C=0.7")  
    ax.set_title(f"Comparison of runs across repeats {repeats_str}\nBest Test C-Index per LR")
    ax.set_xlabel("Dataset")
    ax.set_ylabel("Best Test C-Index")
    
    # Build custom legend handles.
    handles = []
    for d in dataset_order:
        # Count how many data points belong to this dataset
        d_count = (df['dataset'] == d).sum()
        label_with_count = f"{d} ({d_count})"  # e.g. "PDAC+CHOL (39)"
        patch = mpatches.Patch(color=group_palette[d], label=label_with_count)
        handles.append(patch)

    # Add a dummy marker for Mean ± 95% CI
    dummy = mlines.Line2D([], [], color='black', marker='D', linestyle='None',
                          markersize=9, label="Mean ± 95% CI")
    handles.append(dummy)
    ax.legend(handles=handles, loc='best')
    
    out_fig = output_folder / "compare_datasets.png"
    fig.tight_layout()
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    print(f"Saved comparison figure to {out_fig}")
    plt.close(fig)

if __name__ == "__main__":
    main()
