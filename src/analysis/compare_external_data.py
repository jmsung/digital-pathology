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

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "Results_figures"  # Contains subfolders like rep61, rep62, etc.
SUMMARY_DIR = RESULTS_DIR / "Results_summary"    # Used for output
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Regex pattern updated for new filename format AND corrected LogRank group names
pattern_single = re.compile(
    r"""
    ^
    (?P<model>ACMIL|CLAM_SB|CLAM_MB|TransMIL|DSMIL|MeanMIL|MaxMIL|ABMIL|GABMIL)
    _
    (?P<loss>coxph|rank|MSE|SurvPLE) 
    _lr(?P<lr>[0-9.\-+eE]+)  # This is the actual Learning Rate
    _w(?P<weight>[-\d.]+)
    _Epc(?P<epoch>\d+)       # Corrected: Single underscore based on filenames from training script
    _LogRanks\[(?P<logrank_train>[-\d.]+)-(?P<logrank_valid>[-\d.]+)-(?P<logrank_test>[-\d.]+)\]  # Using 'logrank_train' etc.
    _CIdx\[(?P<c_train>[-\d.]+)-(?P<c_valid>[-\d.]+)-(?P<c_test>[-\d.]+)\]
    _Ext(?P<ext_datasets>[A-Za-z0-9_]*) # Captures '', 'None', 'BRCA', 'BRCA_CHOL'
    _
    (?P<repeat>\d+)
    \.png$
    """,
    re.VERBOSE
)

def extract_file_info(filename):
    match = pattern_single.match(filename)
    if match:
        return {
            "model": match.group("model"),
            "loss": match.group("loss"),
            "learning_rate": float(match.group("lr")), # Actual learning rate
            "weight": match.group("weight"), 
            "epoch": int(match.group("epoch")),
            "logrank_train": float(match.group("logrank_train")), # CORRECTED key
            "logrank_valid": float(match.group("logrank_valid")), # CORRECTED key
            "logrank_test": float(match.group("logrank_test")),   # CORRECTED key
            "c_train": float(match.group("c_train")),
            "c_valid": float(match.group("c_valid")),
            "c_test": float(match.group("c_test")),
            "ext": match.group("ext_datasets"),
            "repeat": match.group("repeat"),
        }
    return None

def parse_single_loss_files_from_folder(input_folder):
    results = {}
    for file in input_folder.iterdir():
        if not file.is_file() or not file.name.endswith(".png"): # Ensure it's a PNG file
            continue
        info = extract_file_info(file.name) 
        if info is None:
            # print(f"Debug: Skipping non-matching file: {file.name} in folder {input_folder}") # Optional debug
            continue
        key = (info["model"], info["loss"], info["learning_rate"], info["weight"], info["repeat"])
        if key not in results or info["c_test"] > results[key]["c_test"]:
            results[key] = info
    return results

def aggregate_best_across_folders(folders):
    records = []
    for folder in folders:
        print(f"Processing folder: {folder}")
        folder_results = parse_single_loss_files_from_folder(folder)
        for key, info in folder_results.items():
            records.append({
                'model': info['model'],
                'loss': info['loss'],
                'lr': info['learning_rate'],
                'weight': info['weight'],
                'epoch': info['epoch'],
                'c_train': info['c_train'],
                'c_valid': info['c_valid'],
                'c_test': info['c_test'],
                'repeat': info['repeat'],
                'ext': info['ext'],
                # Optionally include logrank stats if needed in the final DataFrame
                # 'logrank_train': info['logrank_train'],
                # 'logrank_valid': info['logrank_valid'],
                # 'logrank_test': info['logrank_test'],
            })
    return pd.DataFrame(records)

def map_ext_to_dataset(ext_label):
    # Handles old empty _Ext_ and new _ExtNone_
    if ext_label == "" or (isinstance(ext_label, str) and ext_label.upper() == "NONE"):
        return "PDAC Only"
    else:
        return f"PDAC+{ext_label}"

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate single‑loss runs and compare c_test across multiple datasets."
    )
    parser.add_argument('--repeat', type=int, nargs='+', required=True,
                        help="List of repeat numbers to aggregate (e.g. --repeat 21 22 23 24).")
    args = parser.parse_args()

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

    repeats_str = "_".join(sorted(str(r) for r in args.repeat))
    output_folder = SUMMARY_DIR / f"rep{repeats_str}_summary_agg" 
    output_folder.mkdir(parents=True, exist_ok=True)

    df = aggregate_best_across_folders(input_folders)
    if df.empty:
        print("No valid single-loss files found matching the pattern. Exiting.")
        return

    df['dataset'] = df['ext'].apply(map_ext_to_dataset)

    unique_datasets_found = df['dataset'].unique().tolist()

    preferred_order = [
        "PDAC Only", "PDAC+CHOL", "PDAC+ESCA", "PDAC+LUAD",
        "PDAC+BRCA", "PDAC+LIHC", "PDAC+COAD",
    ]
    
    dataset_order = [d for d in preferred_order if d in unique_datasets_found]
    for d_unique in unique_datasets_found:
        if d_unique not in dataset_order:
            dataset_order.append(d_unique)
    
    if not dataset_order and unique_datasets_found: 
        dataset_order = sorted(unique_datasets_found)
    elif not unique_datasets_found:
        print("No datasets to plot after mapping 'ext' labels. Exiting.")
        return

    df['dataset'] = pd.Categorical(df['dataset'], categories=dataset_order, ordered=True)
    # Sorting by 'dataset' category is implicitly handled by seaborn's 'order' parameter

    raw_csv_file = output_folder / "aggregated_best_c_test_data.csv"
    df.to_csv(raw_csv_file, index=False)
    print(f"Saved aggregated data to {raw_csv_file}")

    sorted_txt_file = output_folder / "summary_sorted_by_c_test.txt"
    with open(sorted_txt_file, "w") as f:
        for dataset_name, dataset_group_df in df.groupby("dataset", observed=True):
            f.write(f"Dataset Group: {dataset_name}\n")
            f.write("=" * 80 + "\n")
            group_sorted = dataset_group_df.sort_values(by="c_test", ascending=False)
            for idx, row in group_sorted.iterrows():
                f.write(
                    f"  Repeat: {row['repeat']}, Model: {row['model']}, Loss: {row['loss']}, LR: {row['lr']:.2e}, "
                    f"Weight: {row['weight']}, Epoch: {row['epoch']}, "
                    f"c_train: {row['c_train']:.3f}, c_valid: {row['c_valid']:.3f}, c_test: {row['c_test']:.3f}, ExtRaw: '{row['ext']}'\n"
                )
            f.write("\n")
    print(f"Saved sorted summary to {sorted_txt_file}")

    color_map = {
        "PDAC Only": "#000000",   # black
        "PDAC+CHOL": "#3B78CB",   # blue
        "PDAC+COAD": "#E41A1C",   # red
        "PDAC+LUAD": "#4DAF4A",   # green
        "PDAC+ESCA": "#984EA3",   # purple
        "PDAC+LIHC": "#FF7F00",   # orange
        "PDAC+BRCA": "#A65628",   # brown 
    }
    # Create the palette dictionary for sns.stripplot
    # Keys are dataset names, values are colors
    plot_palette_dict = {ds: color_map.get(ds, "#999999") for ds in dataset_order}


    fig, ax = plt.subplots(figsize=(max(12, int(1.8 * len(dataset_order))), 8)) 
    sns.set_style("whitegrid")

    sns.stripplot(
        data=df, x='dataset', y='c_test', order=dataset_order, 
        jitter=0.1, palette=plot_palette_dict, alpha=0.8, size=7, ax=ax, dodge=False # Use dict palette
    )
    sns.pointplot(
        data=df, x='dataset', y='c_test', order=dataset_order, 
        errorbar=('ci', 95), 
        join=False, markers='D', color='black', ax=ax, dodge=False,
    )

    # Set y-axis limits
    ref_lines_y_values = [0.5, 0.6, 0.7]
    padding_around_ref_lines = 0.05 
    min_y_view_for_refs = min(ref_lines_y_values) - padding_around_ref_lines
    max_y_view_for_refs = max(ref_lines_y_values) + padding_around_ref_lines

    if df.empty or df['c_test'].isnull().all():
        plot_bottom_limit = min_y_view_for_refs
        plot_top_limit = max_y_view_for_refs
    else:
        data_min_val = df['c_test'].min()
        data_max_val = df['c_test'].max()
        padding_around_data = 0.05
        plot_bottom_limit = min(data_min_val - padding_around_data, min_y_view_for_refs)
        plot_top_limit = max(data_max_val + padding_around_data, max_y_view_for_refs)

    final_bottom = max(0.0, plot_bottom_limit) 
    final_top = min(1.0, plot_top_limit)   
    min_yaxis_height = 0.2 
    if (final_top - final_bottom) < min_yaxis_height:
        mid_point = (final_top + final_bottom) / 2.0
        final_bottom = mid_point - (min_yaxis_height / 2.0)
        final_top = mid_point + (min_yaxis_height / 2.0)
        final_bottom = min(final_bottom, min_y_view_for_refs)
        final_top = max(final_top, max_y_view_for_refs)
        final_bottom = max(0.0, final_bottom) 
        final_top = min(1.0, final_top)     
    ax.set_ylim(bottom=final_bottom, top=final_top)

    # Add horizontal lines (these will be automatically picked up by legend if labeled)
    ax.axhline(y=0.5, color="dimgray", linestyle="--", linewidth=0.8, label="C-Index = 0.5 (Random)")
    ax.axhline(y=0.6, color="dimgray", linestyle="--", linewidth=0.8, label="C-Index = 0.6")
    ax.axhline(y=0.7, color="dimgray", linestyle="--", linewidth=0.8, label="C-Index = 0.7")
    
    ax.set_title(f"Comparison of Best Runs (Test C-Index) across Repeats: {repeats_str}", fontsize=16)
    ax.set_xlabel("Dataset Combination", fontsize=14)
    ax.set_ylabel("Best Test C-Index (per LR & Repeat)", fontsize=14)
    ax.tick_params(axis='x', labelrotation=45, labelsize=12, bottom=True)
    ax.tick_params(axis='y', labelsize=12)

    # --- UPDATED LEGEND CREATION ---
    legend_handles = []

    # 1. Create handles for dataset series (stripplot colors)
    for ds_name in dataset_order:
        # Check if there's actually data for this dataset category to avoid creating empty legend entries
        if (df['dataset'] == ds_name).any(): 
            count = (df['dataset'] == ds_name).sum()
            label_text = f"{ds_name} (N = {count})" # Correct desired format
            legend_handles.append(mpatches.Patch(color=plot_palette_dict[ds_name], label=label_text))
    
    # # 2. Create handle for Mean ± 95% CI (pointplot)
    # #    Only add if pointplot actually plotted something (i.e., df was not empty for c_test)
    # if not df.empty and 'c_test' in df.columns and not df['c_test'].isnull().all():
    #      legend_handles.append(mlines.Line2D([], [], color='black', marker='D', linestyle='None', 
    #                                        markersize=8, label="Mean ± 95% CI"))

    # # 3. Create handles for axhlines (reference C-Index lines)
    # legend_handles.append(mlines.Line2D([], [], color='dimgray', linestyle='--', 
    #                                    linewidth=0.8, label='C-Index = 0.5 (Random)'))
    # legend_handles.append(mlines.Line2D([], [], color='dimgray', linestyle='--', 
    #                                    linewidth=0.8, label='C-Index = 0.6'))
    # legend_handles.append(mlines.Line2D([], [], color='dimgray', linestyle='--', 
    #                                    linewidth=0.8, label='C-Index = 0.7'))
    
    # Display the legend
    if legend_handles: # Only create legend if there are handles to show
        if len(legend_handles) > 6: # Heuristic for when to move legend below
            ax.legend(handles=legend_handles, loc='upper center', 
                      bbox_to_anchor=(0.5, -0.20 - (0.03 * (len(dataset_order)//2.5))), # Adjust bbox based on number of items
                      ncol=min(3, (len(legend_handles) + 2) // 3), fontsize=10)
        else:
            ax.legend(handles=legend_handles, loc='best', fontsize=10)
    # --- END OF UPDATED LEGEND CREATION ---

    out_fig = output_folder / "comparison_c_test_stripplot.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    print(f"Saved comparison figure to {out_fig}")
    plt.close(fig)

if __name__ == "__main__":
    main()

