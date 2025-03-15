#!/usr/bin/env python3
import argparse
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import re
import argparse

# Directories
BASE_DIR = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / 'Results_figures'  # This folder contains rep folders (e.g., rep3)
SUMMARY_DIR = RESULTS_DIR / "Results_summary"  # Used for output
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds
MIN_LOG_RANK = -1
MAX_LOG_RANK = 1
MIN_C_INDEX  = 0

loss_color_map = {'coxph': 'green', 'rank': 'blue', 'MSE': 'orange', 'SurvPLE': 'purple'}

###############################################################################
# REGEX PATTERN FOR SINGLE-LOSS FILES 
###############################################################################
# Example valid filename:
# ACMIL_coxph_lr0.0001_w1.00__Epc10_[0.123-0.456-0.789]_[0.800-0.750-0.770]_Ext_4.png
pattern_single = re.compile(
    r"""
    ^
    (?P<model>ACMIL|CLAM_SB|CLAM_MB|TransMIL|DSMIL|MeanMIL|MaxMIL|ABMIL|GABMIL)
    _
    (?P<loss>coxph|rank|MSE|SurvPLE)
    _lr(?P<lr>[-\d.]+)
    _w(?P<weight>[-\d.]+)
    __Epc(?P<epoch>\d+)
    _\[(?P<lr_train>[-\d.]+)-(?P<lr_valid>[-\d.]+)-(?P<lr_test>[-\d.]+)\]
    _\[(?P<c_train>[-\d.]+)-(?P<c_valid>[-\d.]+)-(?P<c_test>[-\d.]+)\]
    _Ext_(?P<repeat>\d+)
    \.png$
    """,
    re.VERBOSE
)

###############################################################################
# PARSING FUNCTION FOR SINGLE-LOSS FILES 
###############################################################################
def parse_single_loss_files(input_folder: Path) -> dict:
    """
    Parse all single‑loss files in input_folder and, for each (model, loss, lr, repeat) combination,
    store the run with the best Test C‑Index (c_test).
    
    Returns:
        A dictionary with keys (model, loss, lr, repeat) and values containing the best run info.
    """
    best_results = {}
    for file in input_folder.iterdir():
        if not file.is_file():
            continue
        match = pattern_single.match(file.name)
        if not match:
            continue

        model   = match.group('model')
        loss    = match.group('loss')
        epoch   = int(match.group('epoch'))
        lr      = float(match.group('lr'))
        lr_test = float(match.group('lr_test'))
        c_test  = float(match.group('c_test'))
        repeat  = match.group('repeat')  # stored as string

        # Skip files that don't meet thresholds.
        if lr_test <= MIN_LOG_RANK or lr_test > MAX_LOG_RANK:
            continue
        if c_test < MIN_C_INDEX:
            continue

        key = (model, loss, lr, repeat)
        if key not in best_results or c_test > best_results[key]['c_test']:
            best_results[key] = {
                'model': model,
                'loss': loss,
                'lr': lr,
                'epoch': epoch,
                'filename': file.name,
                'lr_test': lr_test,
                'c_test': c_test,
                'repeat': repeat
            }
    return best_results

###############################################################################
# OUTPUT & PLOTTING FUNCTIONS FOR TOTAL RUNS (ALL Runs)
###############################################################################
def output_and_plot_total_combined(best_results: dict, output_folder: Path) -> None:
    """
    Create a text summary and a figure with two subplots:
      - Left: grouped by model (ordered by median Test C-Index)
      - Right: grouped by loss (ordered by median Test C-Index)
    Both the text file and PNG are saved in output_folder.
    """
    rows = []
    for key, info in best_results.items():
        model, loss, lr, repeat = key
        rows.append({
            'model': model,
            'loss': loss,
            'lr': lr,
            'c_test': info['c_test'],
            'lr_test': info['lr_test'],
            'epoch': info['epoch'],
            'repeat': repeat,
            'filename': info['filename']
        })
    df_total = pd.DataFrame(rows)
    
    # Text output: sort by best-case (c_test descending)
    df_text = df_total.sort_values(by='c_test', ascending=False)
    txt_lines = ["=== Combined Total Single-Loss Runs Summary (sorted by best-case) ===\n"]
    for idx, row in df_text.iterrows():
        line = (f"[{row['model']} | {row['loss']} | lr={row['lr']:.2e} | rep={row['repeat']}] "
                f"C-Index = {row['c_test']:.3f}, Log-Rank = {row['lr_test']:.3f}, "
                f"Epoch = {row['epoch']} (File: {row['filename']})\n")
        txt_lines.append(line)
        print(line, end="")
    output_txt_total = output_folder / "best_model_loss_single_total.txt"
    output_txt_total.write_text("".join(txt_lines))
    
    # Left subplot: Order models by median Test C-Index (descending)
    median_df = df_total.groupby('model')['c_test'].median().reset_index().rename(columns={'c_test': 'median_c'})
    ordered_models = median_df.sort_values(by='median_c', ascending=False)['model'].tolist()
    
    # Right subplot: Order losses by median Test C-Index (descending)
    loss_medians = df_total.groupby('loss')['c_test'].median().reset_index().rename(columns={'c_test': 'median_c'})
    ordered_losses = loss_medians.sort_values(by='median_c', ascending=False)['loss'].tolist()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sns.stripplot(x='model', y='c_test', data=df_total, order=ordered_models,
                  jitter=False, size=8, alpha=0.7, hue='loss', palette=loss_color_map,
                  dodge=False, ax=axes[0])
    sns.pointplot(x='model', y='c_test', data=df_total, order=ordered_models,
                  estimator='median', errorbar=None, color='red', markers='D',
                  linewidth=0, markersize=10, dodge=False, ax=axes[0])
    axes[0].axhline(y=0.5, color='gray', linestyle='dotted', linewidth=1.5, label="C=0.5")
    axes[0].axhline(y=0.6, color='gray', linestyle='dotted', linewidth=1.5, label="C=0.6")
    axes[0].axhline(y=0.7, color='gray', linestyle='dotted', linewidth=1.5, label="C=0.7")
    axes[0].set_xlabel('Model (sorted by median Test C-Index)')
    axes[0].set_ylabel('Test C-Index')
    axes[0].set_title('Grouped by Model')
    axes[0].legend(title='Loss', fontsize=8, loc="upper right")
    
    sns.stripplot(x='loss', y='c_test', data=df_total, order=ordered_losses,
                  jitter=False, dodge=False, hue='model', palette="tab10", ax=axes[1])
    sns.pointplot(x='loss', y='c_test', data=df_total, order=ordered_losses,
                  estimator='median', errorbar=None, color='red', markers='D',
                  linewidth=0, markersize=10, dodge=False, ax=axes[1])
    axes[1].axhline(y=0.5, color='gray', linestyle='dotted', linewidth=1.5, label="C=0.5")
    axes[1].axhline(y=0.6, color='gray', linestyle='dotted', linewidth=1.5, label="C=0.6")
    axes[1].axhline(y=0.7, color='gray', linestyle='dotted', linewidth=1.5, label="C=0.7")
    axes[1].set_xlabel('Loss Function')
    axes[1].set_ylabel('Test C-Index')
    axes[1].set_title('Grouped by Loss Function (sorted by median)')
    axes[1].legend(title='Model', fontsize=8, loc="upper right")
    
    plt.suptitle('Combined Total Single-Loss Results', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    output_plot_total = output_folder / "best_model_loss_single_total.png"
    plt.savefig(output_plot_total, dpi=300)
    plt.show()

###############################################################################
# HEATMAP FUNCTION
###############################################################################
def plot_heatmap_best_results(best_results: dict, output_path: Path = None):
    """
    Given best_results (a dict with keys (model, loss, lr, repeat) and values containing c_test, epoch, etc.),
    group by model and loss (ignoring lr and repeat) by selecting the row with the highest c_test for each combination.
    Then, create a 2D heatmap with rows as loss functions and columns as models.
    Each cell is annotated with:
         "<c_test>\nlr: <lr>\nep: <epoch>"
    The rows and columns are sorted in descending order based on their mean c-index.
    If output_path is provided, the figure is saved there.
    """
    records = []
    for key, info in best_results.items():
        model, loss, lr, repeat = key
        records.append({
            'model': model,
            'loss': loss,
            'lr': lr,
            'epoch': info['epoch'],
            'c_test': info['c_test']
        })
    
    df = pd.DataFrame(records)
    grouped = df.groupby(['model', 'loss']).apply(lambda x: x.loc[x['c_test'].idxmax()]).reset_index(drop=True)
    
    heatmap_data = grouped.pivot(index='loss', columns='model', values='c_test')
    # Sort rows (loss) in descending order (best at the top) and columns (models) in descending order
    ordered_losses = heatmap_data.mean(axis=1).sort_values(ascending=False).index.tolist()
    ordered_models = heatmap_data.mean(axis=0).sort_values(ascending=False).index.tolist()
    heatmap_data = heatmap_data.reindex(index=ordered_losses, columns=ordered_models)
    
    grouped['annot'] = grouped.apply(lambda row: f"{row['c_test']:.3f}\nlr: {row['lr']:.2e}\nep: {row['epoch']}", axis=1)
    annot_data = grouped.pivot(index='loss', columns='model', values='annot')
    annot_data = annot_data.reindex(index=ordered_losses, columns=ordered_models)
    
    plt.figure(figsize=(12, 6))
    ax = sns.heatmap(heatmap_data, annot=annot_data, fmt='', cmap='viridis', cbar_kws={'label': 'Test C-Index'})
    ax.set_title("Best Test C-Index (with LR and Epoch) by Loss and Model")
    ax.set_ylabel("Loss Function")
    ax.set_xlabel("Model")
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to {output_path}")
    plt.show()

###############################################################################
# MAIN FUNCTION
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Aggregate single‑loss figures from FIGURES_DIR rep folders and produce aggregated plots. "
                    "If --repeat is provided, only aggregate data from the corresponding rep folder (e.g. --repeat 3)."
    )
    parser.add_argument('--repeat', type=int, nargs='+', default=[],
                        help="List of repeat numbers to aggregate (e.g., --repeat 3 6). If multiple are provided, files from all those rep folders are combined.")
    args = parser.parse_args()

    # Determine input folders based on --repeat
    input_folders = []
    if args.repeat:
        for r in args.repeat:
            rep_folder = FIGURES_DIR / f"rep{r}"
            if rep_folder.exists():
                input_folders.append(rep_folder)
            else:
                print(f"Warning: {rep_folder} does not exist.")
        if not input_folders:
            print("No valid rep folders found. Exiting.")
            return
        repeat_suffix = "rep" + "_".join(sorted({str(r) for r in args.repeat}))
    else:
        input_folders = [FIGURES_DIR]
        repeat_suffix = "all"

    # Aggregate best results from all selected input folders
    best_single = {}
    for folder in input_folders:
        folder_results = parse_single_loss_files(folder)
        best_single.update(folder_results)

    if not best_single:
        print("No valid single-loss files found.")
        return

    # Create output folder under SUMMARY_DIR
    output_folder = SUMMARY_DIR / repeat_suffix
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Produce combined outputs, saving both TXT and PNG in output_folder
    output_and_plot_total_combined(best_single, output_folder=output_folder)
    
    # Produce heatmap of best results and save under output_folder
    heatmap_output = output_folder / "heatmap_best_results.png"
    plot_heatmap_best_results(best_single, output_path=heatmap_output)

if __name__ == "__main__":
    main()
