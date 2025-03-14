#!/usr/bin/env python3
"""
Updated survival.py for predicting pancreatic cancer patient survival from WSIs.
Improvements include modular training/evaluation functions, logging, and better error handling.
"""

import argparse
import logging
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
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

# Local library imports
current_dir = Path(__file__).resolve().parent
modules_dir = current_dir.parent / 'modules'
sys.path.append(str(modules_dir))
from architecture.abmil import ABMIL, GatedABMIL
from architecture.acmil import AttnMIL6 as AttnMIL
from architecture.clam import CLAM_MB, CLAM_SB
from architecture.dsmil import BClassifier, FCLayer, MILNet
from architecture.transMIL import TransMIL
from loss.loss import cox_ph_loss, MSE_loss, rank_loss, SurvPLE
from modules import mean_max
from util_functions import (DrawUmapHeatmap, featureRandomSelection, logRankTest,
                            plotSurvival_three, run_one_sample)
from utils.utils import Struct

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# GPU memory optimization setting
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

def set_seed(seed=42):
    """Set seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#############################################
# Data Loading Functions (unchanged)
#############################################
def getData(args):
    clinical_file = DATA_DIR / "clinical.txt"
    Clinical = pd.read_csv(clinical_file, sep="\t")
    Clinical.set_index('Study_no', inplace=True)

    PDAC_path = FEATURES_DIR / "Features_PDAC"
    PDAC_Path_Coord   = PDAC_path / f"Coord_{args.Backbone}"
    PDAC_Path_Feature = PDAC_path / f"Feature_{args.Backbone}"

    Filenames = sorted([p.stem for p in PDAC_Path_Coord.iterdir() if p.suffix == ".pickle"])
    Features, Coords, Barcodes = [], [], []
    for curBarcode in tqdm(Filenames, desc="Loading PDAC Data"):
        try:
            with open(PDAC_Path_Feature / f"{curBarcode}.pickle", 'rb') as f:
                Feature = pickle.load(f)
            with open(PDAC_Path_Coord / f"{curBarcode}.pickle", 'rb') as f:
                Coord = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading data for {curBarcode}: {e}")
            continue
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
    TCGAALL_Path_Coord = TCGAALL_path / f"Coord_{args.Backbone}"
    TCGAALL_Path_Feature = TCGAALL_path / f"Feature_{args.Backbone}"
    Filenames = [i.split('.pickle')[0] for i in sorted(os.listdir(TCGAALL_Path_Coord))]
    Barcodes = [filename[:12] for filename in Filenames if filename[:12] in Clinical.index.tolist()]
    Filenames = [filename for filename in Filenames if filename[:12] in Clinical.index.tolist()]
    Clinical = Clinical.loc[Barcodes]
    Features, Coords, Barcodes_out = [], [], []
    for slide_idx in tqdm(range(len(Filenames)), desc="Loading TCGA All Data"):
        curBarcode = Filenames[slide_idx]
        try:
            with open(f'{TCGAALL_Path_Feature}/{curBarcode}.pickle', 'rb') as f:
                Feature = pickle.load(f)
            with open(f'{TCGAALL_Path_Coord}/{curBarcode}.pickle', 'rb') as f:
                Coord = pickle.load(f)
        except Exception as e:
            logging.warning(f"{curBarcode} is missing: {e}")
            continue
        Features.append(Feature)
        Coords.append(np.array(Coord))
        Barcodes_out.append(curBarcode)
    return Features, Coords, Barcodes_out, Clinical['time'], (Clinical['status'] == "Dead").astype(int)

def getDataTCGA_PDAC(args):
    TCGA_path = FEATURES_DIR / "Features_TCGA"
    TCGA_Path_Coord = TCGA_path / f"Coord_{args.Backbone}"
    TCGA_Path_Feature = TCGA_path / f"Feature_{args.Backbone}"
    Filenames_TCGA = [i.split('.pickle')[0] for i in os.listdir(TCGA_Path_Coord) if ".svs" in i]
    Clinical_TCGA = pd.read_csv(DATA_DIR / 'clinical.tsv', sep="\t")
    Clinical_TCGA.set_index('case_submitter_id', inplace=True)
    Clinical_TCGA = Clinical_TCGA[['vital_status', 'days_to_death', 'days_to_last_follow_up']]
    Clinical_TCGA = Clinical_TCGA.applymap(lambda x: 0 if x == '\'--' else x)
    Clinical_TCGA_final = []
    for idx, row in Clinical_TCGA.iterrows():
        status, day1, day2 = row
        Clinical_TCGA_final.append({'Barcode': idx, 'Status': status, 'Time': max([float(day1), float(day2)])})
    Clinical_TCGA_final = pd.DataFrame(Clinical_TCGA_final).set_index('Barcode')
    Clinical_TCGA_final = Clinical_TCGA_final.loc[~Clinical_TCGA_final.duplicated()]
    Barcodes_TCGA = [i[:12] for i in Filenames_TCGA]
    Barcodes_TCGA_idx = [idx for idx, i in enumerate(Barcodes_TCGA) if i in Clinical_TCGA_final.index]
    Features_TCGA, Coords_TCGA, Barcodes_TCGA_out = [], [], []
    for slide_idx in tqdm(Barcodes_TCGA_idx, desc="Loading TCGA PDAC Data"):
        curBarcode = Filenames_TCGA[slide_idx]
        try:
            with open(f'{TCGA_Path_Feature}/{curBarcode}.pickle', 'rb') as f:
                Feature_TCGA = pickle.load(f)
            with open(f'{TCGA_Path_Coord}/{curBarcode}.pickle', 'rb') as f:
                Coord_TCGA = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading TCGA PDAC for {curBarcode}: {e}")
            continue
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
    # Update configuration parameters
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
# Training and Evaluation Functions
#############################################
def train_one_epoch(model, optimizer, features, times, events, conf, device, loss_func, event_mapping, batch_size):
    """Train the model for one epoch."""
    model.train()
    epoch_loss = 0.0
    slide_preds = []
    epoch_times = []
    epoch_events = []
    
    indices = list(range(len(features)))
    random.shuffle(indices)
    
    for i in tqdm(range(0, len(indices), batch_size), desc="Training Epoch"):
        batch_indices = indices[i:i+batch_size]
        optimizer.zero_grad()
        batch_loss = 0.0
        batch_preds = []
        batch_times = []
        batch_events = []
        
        for idx in batch_indices:
            # For ACMIL (AttnMIL), use run_one_sample; adjust for other models as needed.
            if model.__class__.__name__ == 'AttnMIL':
                loss_val, pred = run_one_sample(featureRandomSelection(features[idx]), model, conf, device)
                if pred is None:
                    continue
                if not isinstance(loss_val, torch.Tensor):
                    loss_val = torch.tensor(loss_val, device=device)
                batch_loss += loss_val
            else:
                feat = torch.from_numpy(featureRandomSelection(features[idx])).unsqueeze(0).float().to(device)
                pred = model(feat)
            batch_preds.append(pred)
            batch_times.append(times.iloc[idx])
            batch_events.append(events.iloc[idx])
        
        if len(batch_preds) == 0:
            continue
        
        preds_tensor = torch.vstack(batch_preds)[:, 0]
        times_tensor = torch.tensor(batch_times).to(device)
        events_tensor = torch.tensor([event_mapping.get(e, e) for e in batch_events], dtype=torch.float32).to(device)
        
        loss_value = loss_func(preds_tensor, times_tensor, events_tensor)
        if loss_func.__name__ in ["SurvPLE", "rank_loss"]:
            loss_value = -loss_value
        batch_loss += loss_value
        batch_loss.backward()
        optimizer.step()
        
        epoch_loss += batch_loss.item()
        slide_preds.extend(preds_tensor.detach().cpu().tolist())
        epoch_times.extend(batch_times)
        epoch_events.extend(batch_events)
    
    return epoch_loss, slide_preds, epoch_times, epoch_events

def evaluate_model(model, features, times, events, conf, device, event_mapping):
    """Evaluate the model on a dataset."""
    model.eval()
    preds = []
    eval_times = []
    eval_events = []
    with torch.no_grad():
        for idx in range(len(features)):
            if model.__class__.__name__ == 'AttnMIL':
                _, pred = run_one_sample(features[idx], model, conf, device)
            else:
                feat = torch.from_numpy(features[idx]).unsqueeze(0).float().to(device)
                pred = model(feat)
            preds.append(pred.item())
            eval_times.append(times.iloc[idx])
            eval_events.append(events.iloc[idx])
    return preds, eval_times, eval_events

#############################################
# Main Training Loop
#############################################
def train_model(args, configs, preloaded_pdac, preloaded_external, preloaded_ncc):
    # Unpack TCGA PDAC data
    TCGA_PDAC_Features, _, _, Clinical_TCGA_PDAC = preloaded_pdac
    TCGA_PDAC_TIME = Clinical_TCGA_PDAC['Time']
    TCGA_PDAC_EVENT = Clinical_TCGA_PDAC['Status']

    # Split into training and validation sets
    data_idx = np.arange(len(TCGA_PDAC_Features))
    train_index, valid_index = train_test_split(data_idx, test_size=0.2, random_state=42)

    if args.ExternalDatasets and preloaded_external is not None:
        TCGA_ALL_Features, _, _, Time_TCGA_ALL, Event_TCGA_ALL = preloaded_external
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
    
    # Load NCC test data
    NCC_Features, _, _, NCC_TIME, NCC_EVENT = preloaded_ncc

    logging.info(f"Learning rate: {args.learning_rate}")
    MODEL, criterion, optimizer, conf = getModel(configs['config_file'], configs['feature_dim'],
                                                  model_name=args.model_name, lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, verbose=True)
    MODEL = MODEL.to(args.device)

    for name, param in MODEL.named_parameters():
        logging.debug(f"{name}: requires_grad={param.requires_grad}")

    loss_names = ['coxph', 'rank', 'MSE', 'SurvPLE']
    loss_funcs = [cox_ph_loss, rank_loss, MSE_loss, SurvPLE()]
    loss_input = [l.strip() for l in args.loss.split(',')]
    indices = [loss_names.index(l) for l in loss_input if l in loss_names]
    
    if len(indices) == 1:
        selected_loss_name = loss_names[indices[0]]
        loss_func = loss_funcs[indices[0]]
    elif len(indices) == 2:
        selected_loss_name = f"{loss_names[indices[0]]}_{loss_names[indices[1]]}"
        def combined_loss(preds, times, events):
            loss1 = loss_funcs[indices[0]](preds, times, events)
            loss2 = loss_funcs[indices[1]](preds, times, events)
            if loss_names[indices[0]] in ["SurvPLE", "rank"]:
                loss1 = -loss1
            if loss_names[indices[1]] in ["SurvPLE", "rank"]:
                loss2 = -loss2
            return loss1 * args.loss_weight + loss2 * (1 - args.loss_weight)
        loss_func = combined_loss
    else:
        raise ValueError("Provide one or two loss names for --loss.")
    
    event_mapping = {'Alive': 0, 'Dead': 1}

    # Main epoch loop
    for epoch in range(args.Epoch):
        logging.info(f"Epoch {epoch} starting...")
        train_loss, train_preds, train_times, train_events = train_one_epoch(
            MODEL, optimizer, TRAIN_FEATURES, TRAIN_TIME, TRAIN_EVENT, conf, args.device, loss_func, event_mapping, batch_size=random.randint(8,32)
        )
        logging.info(f"Epoch {epoch}: Training Loss = {train_loss:.4f}")
        
        val_preds, val_times, val_events = evaluate_model(MODEL, VALID_FEATURES, VALID_TIME, VALID_EVENT, conf, args.device, event_mapping)
        ncc_preds, ncc_times, ncc_events = evaluate_model(MODEL, NCC_Features, NCC_TIME, NCC_EVENT, conf, args.device, event_mapping)
        
        data_train = pd.DataFrame({
            'PredClass': np.array(train_preds) > np.median(train_preds),
            'times': train_times,
            'events': pd.Series(train_events).map(event_mapping)
        })
        data_valid = pd.DataFrame({
            'PredClass': np.array(val_preds) > np.median(val_preds),
            'times': val_times,
            'events': pd.Series(val_events).map(event_mapping)
        })
        data_NCC = pd.DataFrame({
            'PredClass': np.array(ncc_preds) > np.median(ncc_preds),
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
        logging.info(f"Epoch {epoch}: Log-Ranks: {log_ranks}, C-Indices: {c_index}")
        
        log_ranks_str = '-'.join([f'{lr:.3f}' for lr in log_ranks])
        c_index_str = '-'.join([f'{ci:.3f}' for ci in c_index])
        result_str = (f"{args.model_name}_{selected_loss_name}_lr{args.learning_rate:.5f}_w{float(args.loss_weight):.2f}__Epc{epoch}_[{log_ranks_str}]"
                      f"_[{c_index_str}]_Ext{'_'.join(args.ExternalDatasets)}_{args.repeat}")
        
        figure_file = FIGURES_DIR / f"{result_str}.png"
        weights_file = WEIGHTS_DIR / f"{result_str}.pth"
        
        plotSurvival_three(data_train, data_valid, data_NCC, filename=figure_file)
        torch.save(MODEL.state_dict(), weights_file)
        logging.info(f"Saved model weights to {weights_file}")

if __name__ == '__main__':
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--Backbone', type=str, default='UNI')
    parser.add_argument('--cuda_divce', type=int, default=0)
    parser.add_argument('--models', type=str, default='ACMIL', help="Comma-separated list of model names")
    parser.add_argument('--losses', type=str, default='coxph', help="Comma-separated list of loss names")
    parser.add_argument('--num_loss', type=int, default=1, help="Number of loss functions to use")
    parser.add_argument('--loss_weight', type=str, default="1.0", help="Weight(s) for combined loss")
    parser.add_argument('--Epoch', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--ExternalDatasets', nargs='+', type=str, default=[])
    parser.add_argument('--repeat', type=int, default=0, help="Repeat number for output filenames")
    
    args = parser.parse_args()
    args.device = torch.device(f"cuda:{args.cuda_divce}" if torch.cuda.is_available() else "cpu")
    args.models = [m.strip() for m in args.models.split(',')]
    
    preloaded_pdac = getDataTCGA_PDAC(args)
    preloaded_ncc = getData(args)
    if args.ExternalDatasets:
        preloaded_external = getDataTCGA_All(args, cancer_types=args.ExternalDatasets)
    else:
        preloaded_external = None
    
    if args.num_loss == 1:
        for model_name in args.models:
            for loss_str in args.losses.split(','):
                args.model_name = model_name
                args.loss = loss_str.strip()
                if args.loss == 'rank':
                    logging.info("Detected 'rank' loss. Overriding learning rate to 1e-5.")
                    args.learning_rate = 1e-5
                logging.info(f"Training with Model: {model_name} and Single Loss: {args.loss}")
                train_model(args, configs={
                    'STAD': {'stride': 224, 'patch_size': 224, 'batch_size': 32, 'downsample': 1},
                    'TCGA': {'stride': 448, 'patch_size': 448, 'batch_size': 32, 'downsample': 2},
                    'config_file': CODE_DIR / 'config/huaxi_medical_ssl_config.yml',
                    'feature_dim': 1024,
                }, preloaded_pdac=preloaded_pdac, preloaded_external=preloaded_external, preloaded_ncc=preloaded_ncc)
    elif args.num_loss == 2:
        loss_list = [l.strip() for l in args.losses.split(',')]
        if len(loss_list) != 2:
            raise ValueError("For combined loss experiments, please provide exactly two loss names in --losses")
        combined_loss_str = ','.join(loss_list)
        args.loss = combined_loss_str
        weight_list = [float(w) for w in args.loss_weight.split(',')]
        for model_name in args.models:
            args.model_name = model_name
            for w in weight_list:
                args.loss_weight = w
                logging.info(f"Training with Model: {model_name} and Combined Loss: {combined_loss_str} with weight: {w}")
                train_model(args, configs={
                    'STAD': {'stride': 224, 'patch_size': 224, 'batch_size': 32, 'downsample': 1},
                    'TCGA': {'stride': 448, 'patch_size': 448, 'batch_size': 32, 'downsample': 2},
                    'config_file': CODE_DIR / 'config/huaxi_medical_ssl_config.yml',
                    'feature_dim': 1024,
                }, preloaded_pdac=preloaded_pdac, preloaded_external=preloaded_external, preloaded_ncc=preloaded_ncc)
    else:
        raise ValueError("Unsupported num_loss value. Use 1 for single loss or 2 for combined losses.")
