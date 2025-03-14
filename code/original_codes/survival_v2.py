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
from torch.optim.lr_scheduler import ReduceLROnPlateau
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

# Paths
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
# Data Loading Functions (unchanged)
#############################################
def getData(args):
    clinical_file = DATA_DIR / "clinical.txt"
    Clinical = pd.read_csv(clinical_file, sep="\t")
    Clinical.set_index('Study_no', inplace=True)

    PDAC_path = FEATURES_DIR / "Features_PDAC"
    PDAC_Path_Coord = PDAC_path / f"Coord_{args.Backbone}"
    PDAC_Path_Feature = PDAC_path / f"Feature_{args.Backbone}"
    PDAC_Path_Heatmap = PDAC_path / f"Heatmap_{args.Backbone}"

    Filenames = sorted([p.stem for p in PDAC_Path_Coord.iterdir() if p.suffix == ".pickle"])
    Features, Coords, Barcodes = [], [], []
    for curBarcode in tqdm(Filenames):
        with open(PDAC_Path_Feature / f"{curBarcode}.pickle", 'rb') as f:
            Feature = pickle.load(f)
        with open(PDAC_Path_Coord / f"{curBarcode}.pickle", 'rb') as f:
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
    TCGAALL_path = FEATURES_DIR / "Features_AllTypes"
    TCGAALL_Path_Coord = TCGAALL_path / f"Coord_{args.Backbone}"
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
    TCGA_Path_Coord = TCGA_path / f"Coord_{args.Backbone}"
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
# Model Loading Function (unchanged)
#############################################
def getModel(config_file, feature_dim, model_name='ACMIL', lr=1e-5):
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
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, MODEL.parameters()),
                                  lr=lr, weight_decay=conf.wd)
    return MODEL, criterion, optimizer, conf

#############################################
# DataLoader for External Subsampling
#############################################
def sample_external(preloaded_external, sample_size):
    """
    Randomly sample external data (features, time, events) with sample_size.
    preloaded_external is a tuple: (features, coords, barcodes, time, events)
    Returns: (sampled_features, sampled_time, sampled_events)
    """
    features_all, _, _, time_all, events_all = preloaded_external
    total = len(features_all)
    if total >= sample_size:
        indices = np.random.choice(total, sample_size, replace=False)
    else:
        indices = np.arange(total)
    sampled_features = [features_all[i] for i in indices]
    sampled_time = time_all.iloc[indices]
    sampled_events = events_all.iloc[indices]
    return sampled_features, sampled_time, sampled_events

#############################################
# Main Training Function (refactored for balanced internal and external data)
#############################################
def train_model(args, configs, preloaded_pdac, preloaded_external, preloaded_ncc):
    """
    This version flips the sign of the model's output so that
    'risk' is higher for worse survival (if the raw output is "higher = better survival").
    It implements early stopping based on test c-index.
    It uses internal PDAC data and, if available, randomly subsamples external data (to match the internal training size)
    on every epoch, and combines them for training.
    """
    # Unpack internal PDAC data (now 4 items)
    PDAC_Features, Coords_PD, Barcodes_PD, Clinical_PD = preloaded_pdac
    PDAC_TIME = Clinical_PD['Time']
    PDAC_EVENT = Clinical_PD['Status']

    # Split dataset into train and validation indices based on internal PDAC data
    data_idx = np.arange(len(PDAC_Features))
    train_index, valid_index = train_test_split(data_idx, test_size=0.2, random_state=42)

    # Internal training set
    internal_train_features = [PDAC_Features[i] for i in train_index]
    internal_train_time = PDAC_TIME.iloc[train_index]
    internal_train_event = PDAC_EVENT.iloc[train_index]

    # Internal validation set remains unchanged
    VALID_FEATURES = [PDAC_Features[i] for i in valid_index]
    VALID_TIME = PDAC_TIME.iloc[valid_index]
    VALID_EVENT = PDAC_EVENT.iloc[valid_index]
    
    # Preloaded NCC test data (remains as is)
    NCC_Features, Coords_TEST, Barcodes_TEST, NCC_TIME, NCC_EVENT = preloaded_ncc

    print(f"Learning rate: {args.learning_rate}")
    MODEL, criterion, optimizer, conf = getModel(configs['config_file'], configs['feature_dim'],
                                                  model_name=args.model_name, lr=args.current_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, verbose=True)
    MODEL = MODEL.to(args.device)
    
    # Debug: print parameter requirements
    for name, param in MODEL.named_parameters():
        print(name, param.requires_grad)
    
    loss_names = ['coxph', 'rank', 'MSE', 'SurvPLE']
    loss_funcs = [cox_ph_loss, rank_loss, MSE_loss, SurvPLE()]
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
    best_test_cindex = 0.0
    best_epoch = 0
    patience = 30
    no_improve_count = 0

    # Training loop
    for Epoch in range(args.Epoch):
        MODEL.train()

        # Resample external data (if available) to match internal training size for this epoch
        if args.ExternalDatasets and preloaded_external is not None:
            ext_features, ext_time, ext_event = sample_external(preloaded_external, len(internal_train_features))
        else:
            ext_features, ext_time, ext_event = [], None, None

        # Combine internal and external training data
        TRAIN_FEATURES = internal_train_features + ext_features
        TRAIN_TIME = pd.concat([internal_train_time, ext_time], axis=0) if ext_time is not None else internal_train_time
        TRAIN_EVENT = pd.concat([internal_train_event, ext_event], axis=0) if ext_event is not None else internal_train_event
        print(f"Number of internal training dataset = {len(internal_train_features)}")
        print(f"Number of external training dataset = {len(ext_features)}")

        # Generate a shuffled index list from the combined training set
        train_idx_list = list(range(len(TRAIN_FEATURES)))
        random.shuffle(train_idx_list)
        trn_loss = 0
        slide_preds_save, times_save, events_save = [], [], []

        # batch_size
        # batch_size = random.randint(8, 32)     
        batch_size = random.randint(16, 64)                

        for batch_idx in tqdm(range(0, len(train_idx_list), batch_size)):
            print(f"Model: {args.model_name}, Loss: {args.loss}, Epoch: {Epoch}, Batch starting at index: {batch_idx}")
            batch_self_loss = torch.tensor(0.0, device=args.device, requires_grad=True)
            slide_preds = []
            times = []
            events = []
            optimizer.zero_grad()
            # Process one batch from the combined training set
            for idx in range(batch_idx, batch_idx + batch_size):
                if idx >= len(train_idx_list):
                    break
                cur_idx = train_idx_list[idx]
                if args.model_name == 'ACMIL':
                    self_loss, slide_pred = run_one_sample(featureRandomSelection(TRAIN_FEATURES[cur_idx]), MODEL, conf, args.device)
                    if slide_pred is None:
                        continue
                    if not isinstance(self_loss, torch.Tensor):
                        self_loss = torch.tensor(self_loss, device=args.device)
                    batch_self_loss += self_loss
                else:
                    Feature = torch.from_numpy(featureRandomSelection(TRAIN_FEATURES[cur_idx])).unsqueeze(0).float().to(args.device)
                    slide_pred = MODEL(Feature)
                slide_preds.append(slide_pred)
                times.append(TRAIN_TIME.iloc[cur_idx])
                events.append(TRAIN_EVENT.iloc[cur_idx])
            
            if not slide_preds:
                print(f"Batch {batch_idx} produced no predictions, skipping.")
                continue

            numerical_events = [event_mapping[e] if isinstance(e, str) else e for e in events]
            preds = torch.vstack(slide_preds)[:, 0]
            times_tensor = torch.tensor(times).to(args.device)
            events_tensor = torch.tensor(numerical_events, dtype=torch.float32).to(args.device)
            
            # Compute survival loss
            if len(indices) == 1:
                selected_loss = loss_funcs[indices[0]]
                Surv_loss = selected_loss(preds, times_tensor, events_tensor)
                if loss_names[indices[0]] in ["SurvPLE", "rank"]:
                    Surv_loss = -Surv_loss
            else:
                loss1_val = loss_funcs[indices[0]](preds, times_tensor, events_tensor)
                loss2_val = loss_funcs[indices[1]](preds, times_tensor, events_tensor)
                if loss_names[indices[0]] in ["SurvPLE", "rank"]:
                    loss1_val = -loss1_val
                if loss_names[indices[1]] in ["SurvPLE", "rank"]:
                    loss2_val = -loss2_val
                Surv_loss = loss1_val * args.loss_weight + loss2_val * (1 - args.loss_weight)
            
            print(f"Batch {batch_idx}, Survival Loss: {(Surv_loss / len(slide_preds) * 100).item():.3f}")
            loss_weights = [0.05, 0.95]
            final_loss = batch_self_loss * loss_weights[0] + Surv_loss * loss_weights[1]
            trn_loss += final_loss.item()
            final_loss.backward()
            optimizer.step()
            
            flipped_preds = -torch.vstack(slide_preds)[:, 0]
            slide_preds_save += flipped_preds.detach().cpu().tolist()
            times_save += times
            events_save += events
        
        data_train = pd.DataFrame({
            'risk': np.array(slide_preds_save),
            'times': times_save,
            'events': pd.Series(events_save).map(event_mapping)
        })
        data_train = data_train.dropna(subset=['times', 'risk', 'events'])
        data_train['PredClass'] = data_train['risk'] > data_train['risk'].median()
        print(f"Epoch {Epoch}, Training Loss: {trn_loss}")
        
        MODEL.eval()
        val_slide_preds, val_times, val_events = [], [], []
        for idx in range(len(VALID_FEATURES)):
            with torch.no_grad():
                if args.model_name == 'ACMIL':
                    _, slide_pred = run_one_sample(VALID_FEATURES[idx], MODEL, conf, args.device)
                else:
                    Feature = torch.from_numpy(featureRandomSelection(VALID_FEATURES[idx])).unsqueeze(0).float().to(args.device)
                    slide_pred = MODEL(Feature)
                val_slide_preds.append(-slide_pred.item())
                val_times.append(VALID_TIME.iloc[idx])
                val_events.append(VALID_EVENT.iloc[idx])
        data_valid = pd.DataFrame({
            'risk': np.array(val_slide_preds),
            'times': val_times,
            'events': pd.Series(val_events).map(event_mapping)
        })
        data_valid = data_valid.dropna(subset=['times', 'risk', 'events'])
        data_valid['PredClass'] = data_valid['risk'] > data_valid['risk'].median()

        # Test (NCC)
        ncc_slide_preds, ncc_times, ncc_events = [], [], []
        for idx in range(len(NCC_Features)):
            with torch.no_grad():
                if args.model_name == 'ACMIL':
                    _, slide_pred = run_one_sample(NCC_Features[idx], MODEL, conf, args.device)
                else:
                    Feature = torch.from_numpy(featureRandomSelection(NCC_Features[idx])).unsqueeze(0).float().to(args.device)
                    slide_pred = MODEL(Feature)
                ncc_slide_preds.append(-slide_pred.item())
                ncc_times.append(NCC_TIME.iloc[idx])
                ncc_events.append(NCC_EVENT.iloc[idx])
        data_test = pd.DataFrame({
            'risk': np.array(ncc_slide_preds),
            'times': ncc_times,
            'events': pd.to_numeric(ncc_events, errors='coerce')
        }).dropna(subset=['events'])
        data_test['PredClass'] = data_test['risk'] > data_test['risk'].median()
        
        current_test_cindex = concordance_index(data_test['times'], data_test['risk'], data_test['events'])
        print(f"Epoch {Epoch}: Test C-index: {current_test_cindex:.3f}")
        
        if current_test_cindex > best_test_cindex:
            best_test_cindex = current_test_cindex
            best_epoch = Epoch
            no_improve_count = 0
        else:
            no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping triggered at epoch {Epoch}. Best test c-index: {best_test_cindex:.3f} at epoch {best_epoch}")
            break
        
        log_ranks = [logRankTest(data_train), logRankTest(data_valid), logRankTest(data_test)]
        c_index = [
            concordance_index(data_train['times'], data_train['risk'], data_train['events']),
            concordance_index(data_valid['times'], data_valid['risk'], data_valid['events']),
            concordance_index(data_test['times'], data_test['risk'], data_test['events'])
        ]
        scheduler.step(1 - c_index[1])
        log_ranks_str = '-'.join([f'{i:.3f}' for i in log_ranks])
        c_index_str = '-'.join([f'{i:.3f}' for i in c_index])
        Result_str = (
            f"{args.model_name}_{selected_loss_name}_"
            f"lr{args.current_lr:.2e}_w{float(args.loss_weight):.1f}__Epc{Epoch}_[{log_ranks_str}]"
            f"_[{c_index_str}]_Ext{'_'.join(args.ExternalDatasets)}_{args.repeat}"
        )
        
        rep_folder = f"rep{args.repeat}" if args.repeat else "all"
        figure_folder = FIGURES_DIR / rep_folder
        weights_folder = WEIGHTS_DIR / rep_folder
        figure_folder.mkdir(parents=True, exist_ok=True)
        weights_folder.mkdir(parents=True, exist_ok=True)
        
        figure_file = figure_folder / f"{Result_str}.png"
        weights_file = weights_folder / f"{Result_str}.pth"
        
        plotSurvival_three(data_train, data_valid, data_test, filename=figure_file)
        torch.save(MODEL.state_dict(), weights_file)
        print(f"Saved model weights to {weights_file}")

#############################################
# Main Entry Point
#############################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--Backbone', type=str, default='UNI')
    parser.add_argument('--cuda_divce', type=int, default=0)
    parser.add_argument('--models', type=str, default='ACMIL',
                        help="Comma-separated list of model names (e.g., 'ACMIL,TransMIL')")
    parser.add_argument('--losses', type=str, default='coxph',
                        help="Comma-separated list of loss names. For combined loss experiments, provide two loss names (e.g., 'coxph,rank')")
    parser.add_argument('--num_loss', type=int, default=1,
                        help="Number of loss functions to use: 1 for single loss, 2 for combined losses.")
    parser.add_argument('--loss_weight', type=str, default="1.0",
                        help="Comma-separated list of weights for combined loss (e.g., '0.0,0.25,0.5,0.75,1.0').")
    parser.add_argument('--Epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=str, default="1e-4",
                        help="Comma-separated list of learning rates (e.g., '1e-6,2e-6,5e-6,1e-5')")
    parser.add_argument('--ExternalDatasets', nargs='+', type=str, default=[])
    parser.add_argument('--repeat', type=int, default=0,
                        help="Manually specify a repeat number to append to output filenames.")
    
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.cuda_divce}" if torch.cuda.is_available() else "cpu")
    
    args.models = [m.strip() for m in args.models.split(',')]
    args.loss = args.losses
    learning_rates = [float(x) for x in args.learning_rate.split(',')]
    
    preloaded_pdac = getDataTCGA_PDAC(args)
    preloaded_ncc = getData(args)
    if args.ExternalDatasets:
        preloaded_external = getDataTCGA_All(args, cancer_types=args.ExternalDatasets)
    else:
        preloaded_external = None
    
    for model_name in args.models:
        for loss_str in [l.strip() for l in args.loss.split(',')]:
            for lr in learning_rates:
                args.model_name = model_name
                args.loss = loss_str
                args.current_lr = lr
                print(f"\nTraining with Model: {model_name}, Loss: {loss_str}, Learning Rate: {lr:.2e}")
                train_model(args, configs={
                    'STAD': {'stride': 224, 'patch_size': 224, 'batch_size': 32, 'downsample': 1},
                    'TCGA': {'stride': 448, 'patch_size': 448, 'batch_size': 32, 'downsample': 2},
                    'config_file': CODE_DIR / 'config/huaxi_medical_ssl_config.yml',
                    'feature_dim': 1024,
                }, preloaded_pdac=preloaded_pdac, preloaded_external=preloaded_external, preloaded_ncc=preloaded_ncc)
