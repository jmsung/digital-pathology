#!/usr/bin/env python3
import argparse
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
import random
from sklearn.model_selection import train_test_split
import sys
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms
from tqdm import tqdm
import yaml

# local library imports
current_dir = Path(__file__).resolve().parent
modules_dir = current_dir.parent / 'modules'
sys.path.append(str(modules_dir))
from architecture.abmil import ABMIL, GatedABMIL
from architecture.acmil import AttnMIL6 as AttnMIL
from architecture.clam import CLAM_MB, CLAM_SB
from architecture.dsmil import BClassifier, FCLayer, MILNet
from architecture.transMIL import TransMIL
from loss.loss import cox_ph_loss, MSE_loss, recon_loss, rank_loss, SurvMLE, SurvPLE
from modules import mean_max
from util_functions import (DrawUmapHeatmap, featureRandomSelection, logRankTest,
                            minMax, plotSurvival, plotSurvival_three, plotSurvival_two, run_one_sample)
from utils.utils import Struct

# GPU memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CODE_DIR = BASE_DIR / "code"
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = BASE_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"

FIGURES_DIR = RESULTS_DIR / 'Results_figures'
WEIGHTS_DIR = RESULTS_DIR / 'Results_weights'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

#############################################
# Data Loading Functions
#############################################
def getData(args):
    clinical_file = DATA_DIR / "clinical.txt"
    Clinical = pd.read_csv(clinical_file, sep="\t")
    Clinical.set_index('Study_no', inplace=True)

    PDAC_path = FEATURES_DIR / "Features_PDAC"
    PDAC_Path_Coord   = PDAC_path / f"Coord_{args.Backbone}"
    PDAC_Path_Feature = PDAC_path / f"Feature_{args.Backbone}"
    PDAC_Path_Heatmap = PDAC_path / f"Heatmap_{args.Backbone}"

    Filenames = sorted([p.stem for p in PDAC_Path_Coord.iterdir() if p.suffix == ".pickle"])
    Features, Coords, Barcodes = [], [], []
    for curBarcode in tqdm(Filenames):
        feature_file = PDAC_Path_Feature / f"{curBarcode}.pickle"
        coord_file = PDAC_Path_Coord / f"{curBarcode}.pickle"
        with open(feature_file, 'rb') as f:
            Feature = pickle.load(f)
        with open(coord_file, 'rb') as f:
            Coord = pickle.load(f)
        Features.append(Feature)
        Coords.append(np.array(Coord))
        Barcodes.append(curBarcode)
    Clinical = Clinical.loc[[fn.split('.svs')[0] for fn in Filenames]]
    Time, Event = Clinical['PFS_mo'], Clinical['PFS_event_Up220713']
    return Features, Coords, Barcodes, Time, Event

def getDataTCGA_All(args, cancer_types=['LIHC', 'CHOL', 'LUAD', 'COAD', 'ESCA']):
    Clinical = pd.read_csv(DATA_DIR / "TCGA_clinical_data.tsv", sep="\t")
    Clinical.set_index('case_submitter_id', inplace=True)
    Clinical = Clinical.loc[Clinical['cancer_type'].isin(cancer_types)]
    TCGAALL_path = FEATURES_DIR / "Features_ALLTypes"
    TCGAALL_Path_Coord   = TCGAALL_path / f"Coord_{args.Backbone}"
    TCGAALL_Path_Feature = TCGAALL_path / f"Feature_{args.Backbone}"
    TCGAALL_Path_Heatmap = TCGAALL_path / f"Heatmap_{args.Backbone}"  
    Filenames = [i.split('.pickle')[0] for i in sorted(os.listdir(TCGAALL_Path_Coord))]
    Barcodes = [filename[:12] for filename in Filenames if filename[:12] in Clinical.index.tolist()]
    Filenames = [filename for filename in Filenames if filename[:12] in Clinical.index.tolist()]
    Clinical = Clinical.loc[Barcodes]
    Features, Coords, Barcodes_out = [], [], []
    for slide_idx in tqdm(range(len(Filenames))):
        curBarcode = Filenames[slide_idx]
        try:
            with open(f'{TCGAALL_Path_Feature}/{curBarcode}.pickle', 'rb') as f:
                Feature = pickle.load(f)
            with open(f'{TCGAALL_Path_Coord}/{curBarcode}.pickle', 'rb') as f:
                Coord = pickle.load(f)
        except Exception as e:
            print(f'{curBarcode} is not found: {e}')
            continue
        Features.append(Feature)
        Coords.append(np.array(Coord))
        Barcodes_out.append(curBarcode)
    return Features, Coords, Barcodes_out, Clinical['time'], (Clinical['status'] == "Dead").astype(int)

def getDataTCGA_PDAC(args):
    TCGA_path = FEATURES_DIR / "Features_TCGA"
    TCGA_Path_Coord   = TCGA_path / f"Coord_{args.Backbone}"
    TCGA_Path_Feature = TCGA_path / f"Feature_{args.Backbone}"
    TCGA_Path_Heatmap = TCGA_path / f"Heatmap_{args.Backbone}"  
    Filenames_TCGA = [i.split('.pickle')[0] for i in os.listdir(TCGA_Path_Coord) if ".svs" in i]
    Clinical_TCGA = pd.read_csv(DATA_DIR / 'clinical.tsv', sep="\t")
    Clinical_TCGA.set_index('case_submitter_id', inplace=True)
    Clinical_TCGA = Clinical_TCGA[['vital_status', 'days_to_death', 'days_to_last_follow_up']]
    Clinical_TCGA = Clinical_TCGA.applymap(lambda x: 0 if x == '\'--' else x)
    Clinical_TCGA_final = []
    for idx, row in Clinical_TCGA.iterrows():
        status, day1, day2 = row
        Clinical_TCGA_final.append({
            'Barcode': idx,
            'Status': status,
            'Time': max([float(day1), float(day2)])
        })
    Clinical_TCGA_final = pd.DataFrame(Clinical_TCGA_final).set_index('Barcode')
    Clinical_TCGA_final = Clinical_TCGA_final.loc[~Clinical_TCGA_final.duplicated()]
    Barcodes_TCGA = [i[:12] for i in Filenames_TCGA]
    Barcodes_TCGA_idx = [idx for idx, i in enumerate(Barcodes_TCGA) if i in Clinical_TCGA_final.index]
    Features_TCGA, Coords_TCGA, Barcodes_TCGA_out = [], [], []
    for slide_idx in tqdm(Barcodes_TCGA_idx):
        curBarcode = Filenames_TCGA[slide_idx]
        with open(f'{TCGA_Path_Feature}/{curBarcode}.pickle', 'rb') as f:
            Feature_TCGA = pickle.load(f)
        with open(f'{TCGA_Path_Coord}/{curBarcode}.pickle', 'rb') as f:
            Coord_TCGA = pickle.load(f)
        Features_TCGA.append(Feature_TCGA)
        Coords_TCGA.append(np.array(Coord_TCGA))
        Barcodes_TCGA_out.append(curBarcode)
    Clinical_TCGA_final = Clinical_TCGA_final.loc[[i[:12] for i in Barcodes_TCGA_out]]
    return Features_TCGA, Coords_TCGA, Barcodes_TCGA_out, Clinical_TCGA_final

#############################################
# Model Loading Function
#############################################
def getModel(config_file, feature_dim, model_name='ACMIL', lr=0.0001):
    with open(config_file, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        conf = Struct(**c)
    conf.n_token = 5
    conf.n_masked_patch = 10
    conf.mask_drop = 0.3
    conf.D_feat = feature_dim
    conf.n_class = 1
    if model_name == 'ACMIL':
        MODEL = AttnMIL(conf)
    elif model_name == 'CLAM_SB':
        MODEL = CLAM_SB(conf)
    elif model_name == 'CLAM_MB':
        MODEL = CLAM_MB(conf)
    elif model_name == 'TransMIL':
        MODEL = TransMIL(conf)
    elif model_name == 'DSMIL':
        i_classifier = FCLayer(conf.D_feat, conf.n_class)
        b_classifier = BClassifier(conf, nonlinear=False)
        MODEL = MILNet(i_classifier, b_classifier)
    elif model_name == 'MeanMIL':
        MODEL = mean_max.MeanMIL(conf)
    elif model_name == 'MaxMIL':
        MODEL = mean_max.MaxMIL(conf)
    elif model_name == 'ABMIL':
        MODEL = ABMIL(conf)
    elif model_name == 'GABMIL':
        MODEL = GatedABMIL(conf)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, MODEL.parameters()), lr=lr, weight_decay=conf.wd)
    return MODEL, criterion, optimizer, conf

#############################################
# Survival Loss Helper
#############################################
def compute_surv_loss(slide_preds, times, numerical_events, device, losses, weight):
    """
    Compute a weighted combination of two survival losses.
    """
    preds = torch.vstack(slide_preds)[:, 0]
    times_tensor = torch.tensor(times).to(device)
    events_tensor = torch.tensor(numerical_events, dtype=torch.float32).to(device)
    surv_loss1 = losses[0](preds, times_tensor, events_tensor)
    surv_loss2 = losses[1](preds, times_tensor, events_tensor)
    surv_loss = surv_loss1 * weight + surv_loss2 * (1 - weight)
    return surv_loss


#############################################
# Main Training Function
#############################################
def main(args, configs):
    # Load PDAC dataset (TCGA)
    TCGA_PDAC_Features, Coords_TCGA_PDAC, Barcodes_TCGA_PDAC, Clinical_TCGA_PDAC = getDataTCGA_PDAC(args)
    TCGA_PDAC_TIME = Clinical_TCGA_PDAC['Time']
    TCGA_PDAC_EVENT = Clinical_TCGA_PDAC['Status']

    # Split PDAC dataset into train/valid indices
    data = np.arange(len(TCGA_PDAC_Features))
    train_index, valid_index = train_test_split(data, test_size=0.2, random_state=42)

    # Optionally load external datasets
    if args.ExternalDatasets:
        TCGA_ALL_Features, Coords_TCGA_ALL, Barcodes_TCGAALL, Time_TCGA_ALL, Event_TCGA_ALL = getDataTCGA_All(args, cancer_types=args.ExternalDatasets)
        TRAIN_FEATURES = [TCGA_PDAC_Features[i] for i in train_index] + TCGA_ALL_Features
        TRAIN_TIME  = pd.concat([TCGA_PDAC_TIME.iloc[train_index], Time_TCGA_ALL], axis=0)
        TRAIN_EVENT = pd.concat([TCGA_PDAC_EVENT.iloc[train_index], Event_TCGA_ALL], axis=0)
    else:
        TRAIN_FEATURES = [TCGA_PDAC_Features[i] for i in train_index]
        TRAIN_TIME  = TCGA_PDAC_TIME.iloc[train_index]
        TRAIN_EVENT = TCGA_PDAC_EVENT.iloc[train_index]

    VALID_FEATURES = [TCGA_PDAC_Features[i] for i in valid_index]
    VALID_TIME = TCGA_PDAC_TIME.iloc[valid_index]
    VALID_EVENT = TCGA_PDAC_EVENT.iloc[valid_index]
    
    # Load NCC test dataset
    NCC_Features, Coords_TEST, Barcodes_TEST, NCC_TIME, NCC_EVENT = getData(args)

    # Load Model & Loss (using args.learning_rate)
    MODEL, criterion, optimizer, conf = getModel(configs['config_file'], configs['feature_dim'], model_name=args.model_name, lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, verbose=True)
    MODEL = MODEL.to(args.device)
    
    # Define available survival losses and names
    loss_names = ['coxph', 'rank', 'MSE', 'SurvPLE']
    loss_funcs = [cox_ph_loss, rank_loss, MSE_loss, SurvPLE()]

    # Parse the --loss argument. Allow one or two comma-separated loss names.
    loss_input = [l.strip() for l in args.loss.split(',')]
    indices = []
    for lname in loss_input:
        if lname in loss_names:
            indices.append(loss_names.index(lname))
        else:
            raise ValueError(f"Loss name '{lname}' not recognized. Available: {loss_names}")

    if len(indices) == 1:
        selected_loss_name = loss_names[indices[0]]
    elif len(indices) == 2:
        selected_loss_name = f"{loss_names[indices[0]]}_{loss_names[indices[1]]}"
    else:
        raise ValueError("Please provide one or two loss names for --loss.")

    event_mapping = {'Alive': 0, 'Dead': 1}
    
    for Epoch in range(args.Epoch):
        MODEL.train()
        if args.model_name == 'TransMIL' or args.loss == 'rank':
            batch_size = random.randint(8, 16)
        else:
            batch_size = random.randint(8, 32)

        train_idx_list = list(range(len(TRAIN_FEATURES)))
        random.shuffle(train_idx_list)
        
        trn_loss = 0
        slide_preds_save, times_save, events_save = [], [], []
        
        for batch_idx in tqdm(range(0, len(train_idx_list), batch_size)):
            torch.cuda.empty_cache()  # Clear unused GPU memory before each batch

            batch_self_loss = 0
            slide_preds = []
            times = []
            events = []
            
            optimizer.zero_grad()
            for idx in range(batch_idx, batch_idx + batch_size):
                if idx >= len(train_idx_list):
                    break
                cur_idx = train_idx_list[idx]
                if args.model_name == 'ACMIL':
                    self_loss, slide_pred = run_one_sample(featureRandomSelection(TRAIN_FEATURES[cur_idx]), MODEL, conf, args.device)
                    batch_self_loss += self_loss
                else:
                    Feature = torch.from_numpy(featureRandomSelection(TRAIN_FEATURES[cur_idx])).unsqueeze(0).float().to(args.device)
                    slide_pred = MODEL(Feature)
                slide_preds.append(slide_pred)
                times.append(TRAIN_TIME.iloc[cur_idx])
                events.append(TRAIN_EVENT.iloc[cur_idx])
            
            numerical_events = [event_mapping[e] if isinstance(e, str) else e for e in events]
            preds = torch.vstack(slide_preds)[:, 0]
            times_tensor = torch.tensor(times).to(args.device)
            events_tensor = torch.tensor(numerical_events, dtype=torch.float32).to(args.device)
            
            if len(indices) == 1:
                selected_loss = loss_funcs[indices[0]]
                Surv_loss = selected_loss(preds, times_tensor, events_tensor)

                # If using SurvPLE, negate the loss
                if loss_names[indices[0]] == "SurvPLE":
                    Surv_loss = -Surv_loss  # Invert loss for correct optimization

            elif len(indices) == 2:
                loss1_val = loss_funcs[indices[0]](preds, times_tensor, events_tensor)
                loss2_val = loss_funcs[indices[1]](preds, times_tensor, events_tensor)

                # If using SurvPLE, negate its contribution
                if loss_names[indices[0]] == "SurvPLE":
                    loss1_val = -loss1_val
                if loss_names[indices[1]] == "SurvPLE":
                    loss2_val = -loss2_val

                Surv_loss = loss1_val * args.weight + loss2_val * (1 - args.weight)
            
            print(f'batch_idx: {batch_idx}, Surv_loss: {(Surv_loss / batch_size * 100).item():.3f}')
            
            loss_weights = [0.05, 0.95]
            final_loss = batch_self_loss * loss_weights[0] + Surv_loss * loss_weights[1]
            trn_loss += final_loss.item()
            final_loss.backward()
            optimizer.step()
            
            slide_preds_save += torch.vstack(slide_preds)[:, 0].detach().cpu().tolist()
            times_save += times
            events_save += events
        
        data_train = pd.DataFrame({
            'PredClass': np.array(slide_preds_save) > np.median(slide_preds_save),
            'times': times_save,
            'events': pd.Series(events_save).map(event_mapping)
        })
        
        print(f"Epoch {Epoch}, Training Loss: {trn_loss}")
        
        MODEL.eval()
        val_slide_preds = []
        val_times = []
        val_events = []
        for idx in range(len(VALID_FEATURES)):
            with torch.no_grad():
                if args.model_name == 'ACMIL':
                    _, slide_pred = run_one_sample(VALID_FEATURES[idx], MODEL, conf, args.device)
                else:
                    Feature = torch.from_numpy(VALID_FEATURES[idx]).unsqueeze(0).float().to(args.device)
                    slide_pred = MODEL(Feature)
                val_slide_preds.append(slide_pred.item())
                val_times.append(VALID_TIME.iloc[idx])
                val_events.append(VALID_EVENT.iloc[idx])
        
        data_valid = pd.DataFrame({
            'PredClass': np.array(val_slide_preds) > np.median(val_slide_preds),
            'times': val_times,
            'events': pd.Series(val_events).map(event_mapping)
        })
        
        ncc_slide_preds = []
        ncc_times = []
        ncc_events = []
        for idx in range(len(NCC_Features)):
            with torch.no_grad():
                if args.model_name == 'ACMIL':
                    _, slide_pred = run_one_sample(NCC_Features[idx], MODEL, conf, args.device)
                else:
                    Feature = torch.from_numpy(NCC_Features[idx]).unsqueeze(0).float().to(args.device)
                    slide_pred = MODEL(Feature)
                ncc_slide_preds.append(slide_pred.item())
                ncc_times.append(NCC_TIME.iloc[idx])
                ncc_events.append(NCC_EVENT.iloc[idx])
        
        data_NCC = pd.DataFrame({
            'PredClass': np.array(ncc_slide_preds) > np.median(ncc_slide_preds),
            'times': ncc_times,
            'events': pd.to_numeric(ncc_events, errors='coerce')
        }).dropna(subset=['events'])
        
        log_ranks = [logRankTest(data_train), logRankTest(data_valid), logRankTest(data_NCC)]
        c_index = [
            concordance_index(data_train['times'], -data_train['PredClass'].astype(int), data_train['events']),
            concordance_index(data_valid['times'], -data_valid['PredClass'].astype(int), data_valid['events']),
            concordance_index(data_NCC['times'], -data_NCC['PredClass'].astype(int), data_NCC['events'])
        ]
        scheduler.step(1 - c_index[1])

        log_ranks_str = '-'.join([f'{i:.3f}' for i in log_ranks])
        c_index_str = '-'.join([f'{i:.3f}' for i in c_index])
        
        Result_str = (f"{args.model_name}_{selected_loss_name}_lr{args.learning_rate:.4f}_w{args.weight:.2f}__Epc{Epoch}_[{log_ranks_str}]"
                    f"_[{c_index_str}]_Ext{'_'.join(args.ExternalDatasets)}_{args.repeat}")

        figure_file = FIGURES_DIR / f"{Result_str}.png" 
        weights_file = WEIGHTS_DIR / f"{Result_str}.pth"
        
        plotSurvival_three(data_train, data_valid, data_NCC, filename=figure_file)
        torch.save(MODEL.state_dict(), weights_file)
        print(f"Saved model weights to {weights_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--Backbone', type=str, default='UNI')
    parser.add_argument('--cuda_divce', type=int, default=0)
    parser.add_argument('--loss', type=str, default='coxph')
    parser.add_argument('--model_name', type=str, default='ACMIL')
    parser.add_argument('--Epoch', type=int, default=30)
    parser.add_argument('--weight', type=float, default=1.0,
                        help="Weight for the first survival loss when combining two losses (value between 0 and 1)")
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                        help="Learning rate for the optimizer")
    parser.add_argument('--ExternalDatasets', nargs='+', type=str, default=[])
    parser.add_argument('--repeat', type=int, default=0,
                        help="Manually specify a repeat number to append to output filenames.")
    
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.cuda_divce}" if torch.cuda.is_available() else "cpu")
    
    configs = {
        'STAD': {'stride': 224, 'patch_size': 224, 'batch_size': 32, 'downsample': 1},
        'TCGA': {'stride': 448, 'patch_size': 448, 'batch_size': 32, 'downsample': 2},
        'config_file': CODE_DIR / 'config/huaxi_medical_ssl_config.yml',
        'feature_dim': 1024,
    }
    
    main(args, configs)
