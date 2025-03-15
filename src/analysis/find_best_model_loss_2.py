#!/usr/bin/env python3
from collections import defaultdict
import matplotlib.pyplot as plt
import os
import pandas as pd
from pathlib import Path
import re
import seaborn as sns
import argparse

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / 'Results_figures'
TRAINS_DIR = RESULTS_DIR / 'Results_trains'
SUMMARY_DIR = RESULTS_DIR / "Results_summary"
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

###############################################################################
# REGEX PATTERN FOR COMBINED-LOSS (TWO-LOSS) FILES
###############################################################################
pattern_combined = re.compile(r"""
^
(?P<model>ACMIL|CLAM_SB|CLAM_MB|TransMIL|DSMIL|MeanMIL|MaxMIL|ABMIL|GABMIL)
_
(?P<loss1>coxph)_(?P<loss2>SurvPLE)
_lr(?P<lr>[-\d.]+)
_w(?P<weight>[-\d.]+)
__Epc(?P<epoch>\d+)
_\[
(?P<lr_train>[-\d.]+)-(?P<lr_valid>[-\d.]+)-(?P<lr_test>[-\d.]+)
\]_
\[
(?P<c_train>[-\d.]+)-(?P<c_valid>[-\d.]+)-(?P<c_test>[-\d.]+)
\]_Ext
(?:_(?P<repeat>\d+))?
\.png$
""", re.VERBOSE)

def extract_file_info(filename: str) -> dict:
    """
    Extract info from a combined-loss filename.
    Returns a dictionary with keys:
      model, loss1, loss2, lr, weight, epoch,
      lr_train, lr_valid, lr_test, c_train, c_valid, c_test, repeat.
    Returns None if the filename doesn't match.
    """
    match = pattern_combined.match(filename)
    if not match:
        return None
    return {
        "model":  match.group("model"),
        "loss1":  match.group("loss1"),
        "loss2":  match.group("loss2"),
        "lr":     float(match.group("lr")),
        "weight": float(match.group("weight")),
        "epoch":  int(match.group("epoch")),
        "lr_train": float(match.group("lr_train")),
        "lr_valid": float(match.group("lr_valid")),
        "lr_test":  float(match.group("lr_test")),
        "c_train":  float(match.group("c_train")),
        "c_valid":  float(match.group("c_valid")),
        "c_test":   float(match.group("c_test")),
        "repeat":   match.group("repeat") if match.group("repeat") else "0"
    }

def process_files(keywords: list = None) -> dict:
    """
    Process all PNG files in RESULTS_FIGURE_DIR that match the combined-loss pattern.
    If keywords is provided, only process filenames that contain all keywords (case-insensitive).
    Returns a dictionary keyed by (model, loss1, loss2) with a list of records.
    Each record is a dict with: weight, lr, c_test, filename, epoch, and repeat.
    """
    data = defaultdict(list)
    for filename in os.listdir(RESULTS_FIGURE_DIR):
        if not filename.endswith(".png"):
            continue
        if keywords and not all(kw.lower() in filename.lower() for kw in keywords):
            continue
        info = extract_file_info(filename)
        if info is None:
            continue
        key = (info["model"], info["loss1"], info["loss2"])
        data[key].append({
            "weight": info["weight"],
            "lr": info["lr"],
            "c_test": info["c_test"],
            "filename": filename,
            "epoch": info["epoch"],
            "repeat": info["repeat"]
        })
    return data

def plot_scatter_weight_cindex(data: dict, kw_tag: str) -> None:
    """
    For each (model, loss1, loss2) group, create a scatter plot with:
      - x-axis: weight
      - y-axis: test C-index (c_test)
      - color: learning rate (lr)
    The plot title shows the best case (highest test C-index) and its details.
    The figure is saved into RESULTS_SUMMARY with the kw_tag appended.
    """

    for (model, loss1, loss2), records in data.items():
        df = pd.DataFrame(records)
        if df.empty:
            continue
        
        # Determine the best record (highest test C-index)
        best_idx = df["c_test"].idxmax()
        best_row = df.loc[best_idx]
        
        plt.figure(figsize=(8, 5))
        sns.scatterplot(data=df, x="weight", y="c_test", hue="lr",
                        palette="viridis", s=80)
        
        title_str = f"{model} - {loss1}_{loss2}\nBest: c_test={best_row['c_test']:.3f} at w={best_row['weight']:.2f}, lr={best_row['lr']}"
        plt.title(title_str, fontsize=12)
        plt.xlabel("Weight")
        plt.ylabel("Test C-Index")
        plt.tight_layout()
        
        out_filename = f"{model}_{loss1}_{loss2}_scatter{kw_tag}.png"
        out_path = SUMMARY_DIR / out_filename
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved scatter plot to {out_path}")

def output_sorted_results(data: dict, kw_tag: str) -> None:
    """
    Flatten all records from all groups, sort them by test C-index (descending),
    and output the sorted list (filenames only) to a text file in RESULTS_DIR.
    The kw_tag is appended to the output filename.
    """
    records = []
    for rec_list in data.values():
        records.extend(rec_list)
    if not records:
        print("No records found.")
        return
    sorted_records = sorted(records, key=lambda r: r["c_test"], reverse=True)
    
    lines = ["=== Best Combined-Loss Results Sorted by Test C-Index (Descending) ===\n"]
    for rec in sorted_records:
        lines.append(rec["filename"] + "\n")
    
    out_file = SUMMARY_DIR / f"best_combined_results{kw_tag}.txt"
    out_file.write_text("".join(lines))
    print(f"Saved sorted results to {out_file}")

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate combined-loss figures by keywords and produce scatter plots and sorted results."
    )
    parser.add_argument("--keywords", type=str, nargs="+", default=None,
                        help="Only process files whose filenames contain all these keywords (e.g., _0).")
    args = parser.parse_args()
    
    # Build a keyword tag from the provided keywords (e.g., if keywords are ['_0'], tag becomes '_0')
    kw_tag = ""
    if args.keywords:
        kw_tag = "_" + "_".join(args.keywords)
    
    combined_data = process_files(keywords=args.keywords)
    plot_scatter_weight_cindex(combined_data, kw_tag)
    output_sorted_results(combined_data, kw_tag)

if __name__ == "__main__":
    main()
