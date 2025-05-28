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

# base directories
BASE_DIR    = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = BASE_DIR / "results"
FIGURES_DIR = RESULTS_DIR / "Results_figures"   # subfolders: repXX
SUMMARY_DIR = RESULTS_DIR / "Results_summary"   # output here
SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# filename‐regex now supports either one or two losses
pattern = re.compile(r"""
^
(?P<model>ACMIL|CLAM_SB|CLAM_MB|TransMIL|DSMIL|MeanMIL|MaxMIL|ABMIL|GABMIL)
_
(?P<loss1>coxph|rank|MSE|SurvPLE)           # first loss
(?:_(?P<loss2>coxph|rank|MSE|SurvPLE))?     # optional second loss
_lr(?P<lr>[0-9.+\-eE]+)                     # learning rate
_w(?P<weight>[-\d.]+)
_Epc(?P<epoch>\d+)
_LogRanks\[(?P<logrank_train>[-\d.]+)
           \-(?P<logrank_valid>[-\d.]+)
           \-(?P<logrank_test>[-\d.]+)\]
_CIdx\[(?P<c_train>[-\d.]+)
       \-(?P<c_valid>[-\d.]+)
       \-(?P<c_test>[-\d.]+)\]
_Ext(?P<ext_datasets>[A-Za-z0-9_]*)
_(?P<repeat>\d+)
\.png$
""", re.VERBOSE)

def extract_file_info(filename: str):
    m = pattern.match(filename)
    if not m:
        return None

    # merge first and optional second loss into single string
    l1 = m.group("loss1")
    l2 = m.group("loss2")
    loss = f"{l1}_{l2}" if l2 else l1

    return {
        "model":         m.group("model"),
        "loss":          loss,
        "learning_rate": float(m.group("lr")),
        "weight":        m.group("weight"),
        "epoch":         int(m.group("epoch")),
        "logrank_train": float(m.group("logrank_train")),
        "logrank_valid": float(m.group("logrank_valid")),
        "logrank_test":  float(m.group("logrank_test")),
        "c_train":       float(m.group("c_train")),
        "c_valid":       float(m.group("c_valid")),
        "c_test":        float(m.group("c_test")),
        "ext":           m.group("ext_datasets"),
        "repeat":        m.group("repeat"),
    }

def parse_single_loss_files_from_folder(input_folder: Path):
    results = {}
    for file in input_folder.iterdir():
        if not file.is_file() or not file.name.endswith(".png"):
            continue
        info = extract_file_info(file.name)
        if info is None:
            continue
        key = (info["model"], info["loss"], info["learning_rate"], info["weight"], info["repeat"])
        # keep only the best‐c_test for each (model,loss,lr,weight,repeat)
        if key not in results or info["c_test"] > results[key]["c_test"]:
            results[key] = info
    return results

def aggregate_best_across_folders(folders):
    records = []
    for folder in folders:
        print(f"Processing folder: {folder}")
        folder_results = parse_single_loss_files_from_folder(folder)
        for info in folder_results.values():
            records.append({
                'model': info['model'],
                'loss':  info['loss'],
                'lr':    info['learning_rate'],
                'weight':info['weight'],
                'epoch': info['epoch'],
                'c_train': info['c_train'],
                'c_valid': info['c_valid'],
                'c_test':  info['c_test'],
                'repeat':  info['repeat'],
                'ext':     info['ext'],
                # you can uncomment to include logrank stats:
                # 'logrank_train': info['logrank_train'],
                # 'logrank_valid': info['logrank_valid'],
                # 'logrank_test':  info['logrank_test'],
            })
    return pd.DataFrame(records)

def map_ext_to_dataset(ext_label):
    if ext_label == "" or ext_label.upper() == "NONE":
        return "PDAC Only"
    else:
        return f"PDAC+{ext_label}"

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate single- or dual-loss runs and compare best test C-Index."
    )
    parser.add_argument('--repeat', type=int, nargs='+', required=True,
                        help="List of repeat numbers (e.g. --repeat 21 22 23).")
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
        print("No matching files found. Exiting.")
        return

    df['dataset'] = df['ext'].apply(map_ext_to_dataset)

    # define plot order
    unique_ds = df['dataset'].unique().tolist()
    pref = ["PDAC Only","PDAC+CHOL","PDAC+ESCA","PDAC+LUAD",
            "PDAC+BRCA","PDAC+LIHC","PDAC+COAD"]
    order = [d for d in pref if d in unique_ds] + [d for d in unique_ds if d not in pref]
    df['dataset'] = pd.Categorical(df['dataset'], categories=order, ordered=True)

    # save raw CSV and sorted text summary
    raw_csv = output_folder / "aggregated_best_c_test_data.csv"
    df.to_csv(raw_csv, index=False)
    print(f"Saved CSV → {raw_csv}")

    sorted_txt = output_folder / "summary_sorted_by_c_test.txt"
    with open(sorted_txt, "w") as f:
        for ds, grp in df.groupby("dataset", observed=True):
            f.write(f"Dataset Group: {ds}\n" + "="*60 + "\n")
            for _, row in grp.sort_values("c_test", ascending=False).iterrows():
                f.write(
                    f"  Repeat: {row['repeat']}, Model: {row['model']}, Loss: {row['loss']}, "
                    f"LR: {row['lr']:.2e}, W: {row['weight']}, Epc: {row['epoch']}, "
                    f"c_train: {row['c_train']:.3f}, c_valid: {row['c_valid']:.3f}, "
                    f"c_test: {row['c_test']:.3f}, ExtRaw: '{row['ext']}'\n"
                )
            f.write("\n")
    print(f"Saved text summary → {sorted_txt}")

    # color palette
    color_map = {
        "PDAC Only": "#000000",
        "PDAC+CHOL": "#3B78CB",
        "PDAC+COAD": "#E41A1C",
        "PDAC+LUAD": "#4DAF4A",
        "PDAC+ESCA": "#984EA3",
        "PDAC+LIHC": "#FF7F00",
        "PDAC+BRCA": "#A65628",
    }
    palette = {ds: color_map.get(ds, "#999999") for ds in order}

    # plotting (exactly as original)
    fig, ax = plt.subplots(figsize=(max(12, 1.8*len(order)), 8))
    sns.set_style("whitegrid")

    sns.stripplot(
        data=df, x='dataset', y='c_test', order=order,
        jitter=0.1, palette=palette, alpha=0.8, size=7, ax=ax
    )
    sns.pointplot(
        data=df, x='dataset', y='c_test', order=order,
        errorbar=('ci',95), join=False, markers='D',
        color='black', ax=ax
    )

    # dynamic y‐limits and reference lines
    ref_vals = [0.5,0.6,0.7]
    pad = 0.05
    data_min, data_max = df['c_test'].min(), df['c_test'].max()
    bottom = min(data_min - pad, min(ref_vals)-pad)
    top    = max(data_max + pad, max(ref_vals)+pad)
    bottom, top = max(0.0, bottom), min(1.0, top)
    if (top-bottom) < 0.2:
        mid = (top+bottom)/2
        bottom, top = max(0.0, mid-0.1), min(1.0, mid+0.1)
    ax.set_ylim(bottom=bottom, top=top)

    for y in ref_vals:
        ax.axhline(y=y, color="dimgray", linestyle="--", linewidth=0.8)

    ax.set_title(f"Best Test C-Index across Repeats: {repeats_str}", fontsize=16)
    ax.set_xlabel("Dataset Combination", fontsize=14)
    ax.set_ylabel("Best Test C-Index", fontsize=14)
    ax.tick_params(axis='x', labelrotation=45, labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    # legend for dataset (count)
    handles = []
    for ds in order:
        cnt = (df['dataset']==ds).sum()
        if cnt>0:
            handles.append(mpatches.Patch(color=palette[ds], label=f"{ds} (N={cnt})"))
    if handles:
        ax.legend(handles=handles, loc='best', fontsize=10)

    out_fig = output_folder / "comparison_c_test_stripplot.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    print(f"Saved comparison figure → {out_fig}")
    plt.close(fig)

if __name__ == "__main__":
    main()
