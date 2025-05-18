#!/usr/bin/env python3
import argparse
from lifelines import KaplanMeierFitter
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
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import yaml

# ---------------- Standard Setup ----------------
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
from util_functions import (
    DrawUmapHeatmap, featureRandomSelection, logRankTest, 
    minMax, plotSurvival, plotSurvival_three, plotSurvival_two, run_one_sample
)
from utils.utils import Struct

# GPU memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CODE_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = BASE_DIR / "features"
RESULTS_DIR = BASE_DIR / "results"

FIGURES_DIR = RESULTS_DIR / 'Results_figures'
WEIGHTS_DIR = RESULTS_DIR / 'Results_weights'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

BATCH_SIZE_MIN, BATCH_SIZE_MAX = 16, 64
CINDEX_MIN = 0.5         # Minimum acceptable test c-index after epoch=0
MAX_REINIT_ATTEMPTS = 100       # Max times we re-initialize if threshold not met
BEST_EPOCH_PACIENCE = 50
NUM_EXTERNAL = 160
DROPOUT_RATE = 0.0

# ---------------- Global Hyperparameters ----------------
# Define n_class hyperparameter (set to 1 for survival only, or 2 for survival + classification)
N_CLASS = 1

BATCH_SELF_LOSS_WEIGHT = 0.05
if N_CLASS == 1:
    print('Number of class = 1')
    CLS_LOSS_WEIGHT = 0
elif N_CLASS == 2:
    print('Number of class = 2')
    CLS_LOSS_WEIGHT = 0.1
else:
    raise ValueError("N_CLASS must be either 1 or 2.")
SURV_LOSS_WEIGHT = 1 - BATCH_SELF_LOSS_WEIGHT - CLS_LOSS_WEIGHT

#############################################
# Data Loading Functions (unchanged)
#############################################

# Load NCC/PDAC data
def getData(args):
    clinical_file = DATA_DIR / 'clinical' / "clinical.txt"
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

# Load TCGA/External data
def getDataTCGA_All(args, cancer_types=['LIHC', 'CHOL', 'LUAD', 'COAD', 'ESCA', 'BRCA']):
    Clinical = pd.read_csv(DATA_DIR / 'clinical' / "TCGA_clinical_data.tsv", sep="\t")
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

# Load TCGA/PDAC data
def getDataTCGA_PDAC(args):
    TCGA_path = FEATURES_DIR / "Features_TCGA"
    TCGA_Path_Coord = TCGA_path / f"Coord_{args.Backbone}"
    TCGA_Path_Feature = TCGA_path / f"Feature_{args.Backbone}"
    TCGA_Path_Heatmap = TCGA_path / f"Heatmap_{args.Backbone}"
    Filenames_TCGA = [i.split('.pickle')[0] for i in os.listdir(TCGA_Path_Coord) if ".svs" in i]
    Clinical_TCGA = pd.read_csv(DATA_DIR / 'clinical' / 'clinical.tsv', sep="\t")
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
# Model Loading Function (updated to use global N_CLASS)
#############################################
def getModel(config_file, feature_dim, model_name='ACMIL', lr=1e-5):
    with open(config_file, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader)
        conf = Struct(**c)
    conf.n_token = 5
    conf.n_masked_patch = 10
    conf.mask_drop = 0.3
    conf.D_feat = feature_dim
    conf.n_class = N_CLASS  # Set n_class using the global variable
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
    # criterion is only used for classification loss (if N_CLASS == 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, MODEL.parameters()),
                                  lr=lr, weight_decay=conf.wd)
    return MODEL, criterion, optimizer, conf

#############################################
# DataLoader for External Subsampling
#############################################
def sample_external(preloaded_external, sample_size):
    """
    Randomly sample external data (features, time, events).
    If sample_size > total number of external samples,
    return all external samples (no sampling with replacement).
    
    preloaded_external is a tuple: (features, coords, barcodes, time, events)
    Returns: (sampled_features, sampled_time, sampled_events)
    """
    features_all, _, _, time_all, events_all = preloaded_external
    total = len(features_all)
    # If sample_size exceeds total, we just use all external data
    actual_size = min(sample_size, total)

    indices = np.random.choice(total, size=actual_size, replace=False)
    sampled_features = [features_all[i] for i in indices]
    sampled_time = time_all.iloc[indices]
    sampled_events = events_all.iloc[indices]

    return sampled_features, sampled_time, sampled_events

def feature_dropout(features, dropout_rate=0.0):
    """
    Randomly drop (set to zero) a fraction of elements along the feature dimension.
    Assumes that the feature dimension is the last dimension of the input tensor.
    """
    # Create a mask for the feature dimension (features.size(2))
    mask = (torch.rand(features.size(2)) > dropout_rate).float().to(features.device)
    # Reshape mask to (1, 1, features.size(2)) so it can broadcast over the first two dimensions
    mask = mask.unsqueeze(0).unsqueeze(0)
    # Apply the mask
    features = features * mask
    return features

#############################################
# Main Training Function with Loop Approach (UPDATED)
#############################################
def train_model(args, configs, preloaded_pdac, preloaded_external, preloaded_ncc):
    """
    This version:
      1) Requires the epoch=0 test C-index to be >= CINDEX_MIN,
         else we re-initialize data + model (up to MAX_REINIT_ATTEMPTS).
      2) Once threshold is met, normal early-stopping is applied for subsequent epochs.
      3) Flips the sign of model output so 'risk' is higher => worse survival.
      4) Combines internal PDAC data with subsampled external data each epoch.
    """

    reinit_attempts = 0
    while True:
        if reinit_attempts >= MAX_REINIT_ATTEMPTS:
            raise RuntimeError(
                f"Reached {MAX_REINIT_ATTEMPTS} attempts without surpassing CINDEX_MIN={CINDEX_MIN:.3f} at epoch=0. Aborting."
            )

        print(f"\n*** Attempt #{reinit_attempts+1} / {MAX_REINIT_ATTEMPTS}: must achieve C-index >= {CINDEX_MIN} at epoch=0 ***")

        # -------------------------------
        # Unpack internal PDAC & do train/val split
        # -------------------------------
        PDAC_Features, Coords_PD, Barcodes_PD, Clinical_PD = preloaded_pdac
        PDAC_TIME = Clinical_PD['Time']
        PDAC_EVENT = Clinical_PD['Status']

        data_idx = np.arange(len(PDAC_Features))
        train_index, valid_index = train_test_split(data_idx, test_size=0.2, random_state=random.randint(0,9999))

        # Internal training set
        internal_train_features = [PDAC_Features[i] for i in train_index]
        internal_train_time = PDAC_TIME.iloc[train_index]
        internal_train_event = PDAC_EVENT.iloc[train_index]

        # Internal validation set
        VALID_FEATURES = [PDAC_Features[i] for i in valid_index]
        VALID_TIME = PDAC_TIME.iloc[valid_index]
        VALID_EVENT = PDAC_EVENT.iloc[valid_index]

        # NCC test set
        NCC_Features, Coords_TEST, Barcodes_TEST, NCC_TIME, NCC_EVENT = preloaded_ncc

        # -------------------------------
        # Initialize model, optimizer, scheduler
        # -------------------------------
        print(f"Learning rate: {args.current_lr}")
        MODEL, criterion, optimizer, conf = getModel(
            configs['config_file'],
            configs['feature_dim'],
            model_name=args.model_name,
            lr=args.current_lr
        )
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2, verbose=True)
        MODEL = MODEL.to(args.device)

        # Decide which loss functions we are using
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

        # ---- EPOCH 0 (initial check) ----
        Epoch = 0
        print(f"=== EPOCH {Epoch} (initial check) ===")
        MODEL.train()

        # By default, use all internal data
        TRAIN_FEATURES = internal_train_features[:]
        TRAIN_TIME = internal_train_time.copy()
        TRAIN_EVENT = internal_train_event.copy()

        # Only add external data if available
        if args.ExternalDatasets and preloaded_external is not None:
            ext_features, ext_time, ext_event = sample_external(preloaded_external, NUM_EXTERNAL)
            TRAIN_FEATURES += ext_features
            TRAIN_TIME = pd.concat([TRAIN_TIME, ext_time], axis=0)
            TRAIN_EVENT = pd.concat([TRAIN_EVENT, ext_event], axis=0)

        print(f"Number of TRAIN_FEATURES = {len(TRAIN_FEATURES)}")
        train_idx_list = list(range(len(TRAIN_FEATURES)))
        random.shuffle(train_idx_list)

        epoch_loss = 0
        slide_preds_save, times_save, events_save = [], [], []
        slide_cls_list = []  # For collecting classification outputs if applicable
        batch_size = random.randint(BATCH_SIZE_MIN, BATCH_SIZE_MAX)

        for batch_idx in tqdm(range(0, len(train_idx_list), batch_size)):
            batch_self_loss = torch.tensor(0.0, device=args.device, requires_grad=True)
            slide_preds = []
            slide_cls = []  # Temporary list for classification outputs in current batch
            times_ = []
            events_ = []
            optimizer.zero_grad()

            for idx_ in range(batch_idx, batch_idx + batch_size):
                if idx_ >= len(train_idx_list):
                    break
                cur_idx = train_idx_list[idx_]

                # For ACMIL branch using run_one_sample
                if args.model_name == 'ACMIL':
                    if conf.n_class == 2:
                        self_loss, outputs = run_one_sample(featureRandomSelection(TRAIN_FEATURES[cur_idx]),
                                                            MODEL, conf, args.device)
                        if outputs is None:
                            continue
                        slide_pred, slide_class = outputs
                        if not isinstance(self_loss, torch.Tensor):
                            self_loss = torch.tensor(self_loss, device=args.device)
                        batch_self_loss += self_loss
                        slide_cls.append(slide_class)
                    else:
                        self_loss, slide_pred = run_one_sample(featureRandomSelection(TRAIN_FEATURES[cur_idx]),
                                                               MODEL, conf, args.device)
                        if slide_pred is None:
                            continue
                        if not isinstance(self_loss, torch.Tensor):
                            self_loss = torch.tensor(self_loss, device=args.device)
                        batch_self_loss += self_loss
                else:
                    Feature_ = torch.from_numpy(featureRandomSelection(TRAIN_FEATURES[cur_idx])).unsqueeze(0).float().to(args.device)
                    if conf.n_class == 2:
                        slide_pred, slide_class = MODEL(Feature_)
                        slide_cls.append(slide_class)
                    else:
                        slide_pred = MODEL(Feature_)
                # Ensure slide_pred is a tensor (if it is a tuple, extract the first element)
                if isinstance(slide_pred, (tuple, list)):
                    slide_pred = slide_pred[0]
                slide_preds.append(slide_pred)
                times_.append(TRAIN_TIME.iloc[cur_idx])
                events_.append(TRAIN_EVENT.iloc[cur_idx])

            if not slide_preds:
                continue

            preds = torch.vstack(slide_preds)[:, 0]
            if conf.n_class == 2:
                cls_logits = torch.vstack(slide_cls)
            numerical_events = [event_mapping[e] if isinstance(e, str) else e for e in events_]
            times_tensor = torch.tensor(times_).to(args.device)
            events_tensor = torch.tensor(numerical_events, dtype=torch.float32).to(args.device)

            # Compute survival loss
            if len(indices) == 1:
                sel_loss_fn = loss_funcs[indices[0]]
                Surv_loss = sel_loss_fn(preds, times_tensor, events_tensor)
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

            if conf.n_class == 2:
                target_cls = torch.tensor(numerical_events, dtype=torch.long).to(args.device)
                cls_loss = criterion(cls_logits, target_cls)
                final_loss = batch_self_loss * BATCH_SELF_LOSS_WEIGHT + Surv_loss * SURV_LOSS_WEIGHT + cls_loss * CLS_LOSS_WEIGHT
            else:
                final_loss = batch_self_loss * BATCH_SELF_LOSS_WEIGHT + Surv_loss * (SURV_LOSS_WEIGHT + CLS_LOSS_WEIGHT)

            epoch_loss += final_loss.item()
            final_loss.backward()
            optimizer.step()

            # Flip sign => higher = worse
            flipped_preds = -preds.detach().cpu()
            slide_preds_save.extend(flipped_preds.tolist())
            times_save.extend(times_)
            events_save.extend(events_)

        print(f"Epoch {Epoch}, Training Loss: {epoch_loss:.4f}")

        data_train_0 = pd.DataFrame({
            'risk': np.array(slide_preds_save),
            'times': times_save,
            'events': pd.Series(events_save).map(event_mapping)
        }).dropna(subset=['times', 'risk', 'events'])
        train_cindex_0 = concordance_index(data_train_0['times'], data_train_0['risk'], data_train_0['events'])

        # ---- Validation Evaluation ----
        val_slide_preds, val_times, val_events = [], [], []
        for idx_ in range(len(VALID_FEATURES)):
            with torch.no_grad():
                if args.model_name == 'ACMIL':
                    if conf.n_class == 2:
                        _, outputs = run_one_sample(VALID_FEATURES[idx_], MODEL, conf, args.device)
                        slide_pred, _ = outputs
                    else:
                        _, slide_pred = run_one_sample(VALID_FEATURES[idx_], MODEL, conf, args.device)
                else:
                    feat_ = torch.from_numpy(featureRandomSelection(VALID_FEATURES[idx_])).unsqueeze(0).float().to(args.device)
                    if conf.n_class == 2:
                        slide_pred, _ = MODEL(feat_)
                    else:
                        slide_pred = MODEL(feat_)
                if isinstance(slide_pred, (tuple,list)):
                    slide_pred = slide_pred[0]
                val_slide_preds.append(-slide_pred.item())
                val_times.append(VALID_TIME.iloc[idx_])
                val_events.append(VALID_EVENT.iloc[idx_])
        data_valid_0 = pd.DataFrame({
            'risk': np.array(val_slide_preds),
            'times': val_times,
            'events': pd.Series(val_events).map(event_mapping)
        }).dropna(subset=['times','risk','events'])
        valid_cindex_0 = concordance_index(data_valid_0['times'], data_valid_0['risk'], data_valid_0['events'])

        # ---- Test Evaluation (NCC) ----
        MODEL.eval()
        ncc_preds, ncc_times, ncc_events_ = [], [], []
        for idx_ in range(len(NCC_Features)):
            with torch.no_grad():
                if args.model_name == 'ACMIL':
                    if conf.n_class == 2:
                        _, outputs = run_one_sample(NCC_Features[idx_], MODEL, conf, args.device)
                        slide_pred, _ = outputs
                    else:
                        _, slide_pred = run_one_sample(NCC_Features[idx_], MODEL, conf, args.device)
                else:
                    feat_ = torch.from_numpy(featureRandomSelection(NCC_Features[idx_])).unsqueeze(0).float().to(args.device)
                    if conf.n_class == 2:
                        slide_pred, _ = MODEL(feat_)
                    else:
                        slide_pred = MODEL(feat_)
                if isinstance(slide_pred, (tuple,list)):
                    slide_pred = slide_pred[0]
                ncc_preds.append(-slide_pred.item())
                ncc_times.append(NCC_TIME.iloc[idx_])
                ncc_events_.append(NCC_EVENT.iloc[idx_])
        data_test_0 = pd.DataFrame({
            'risk': np.array(ncc_preds),
            'times': ncc_times,
            'events': pd.to_numeric(NCC_EVENT, errors='coerce')
        }).dropna(subset=['events'])
        test_cindex_0 = concordance_index(data_test_0['times'], data_test_0['risk'], data_test_0['events'])
        current_test_cindex = test_cindex_0

        if min([train_cindex_0, valid_cindex_0, test_cindex_0]) < CINDEX_MIN:
            print(f"*** C-index={train_cindex_0:.2f}, {valid_cindex_0:.2f}, {test_cindex_0:.2f} < {CINDEX_MIN}. Re-initializing data/model... ***")
            reinit_attempts += 1
            continue
        else:
            break

    # ------------------------------------------------------
    # Remaining Epochs
    # ------------------------------------------------------
    best_test_cindex = current_test_cindex
    best_epoch = 0
    no_improve_count = 0

    for Epoch in range(1, args.Epoch):
        MODEL.train()
        TRAIN_FEATURES = internal_train_features[:]
        TRAIN_TIME = internal_train_time.copy()
        TRAIN_EVENT = internal_train_event.copy()

        if args.ExternalDatasets and preloaded_external is not None:
            ext_features, ext_time, ext_event = sample_external(preloaded_external, NUM_EXTERNAL)
            TRAIN_FEATURES += ext_features
            TRAIN_TIME = pd.concat([TRAIN_TIME, ext_time], axis=0)
            TRAIN_EVENT = pd.concat([TRAIN_EVENT, ext_event], axis=0)

        print(f"Number of TRAIN_FEATURES = {len(TRAIN_FEATURES)}")
        train_idx_list = list(range(len(TRAIN_FEATURES)))
        random.shuffle(train_idx_list)

        trn_loss = 0
        slide_preds_save, times_save, events_save = [], [], []
        slide_cls_list = []  # for classification outputs if applicable
        batch_size = random.randint(BATCH_SIZE_MIN, BATCH_SIZE_MAX)

        for batch_idx in tqdm(range(0, len(train_idx_list), batch_size)):
            batch_self_loss = torch.tensor(0.0, device=args.device, requires_grad=True)
            slide_preds = []
            slide_cls = []  # for current batch classification outputs
            times_ = []
            events_ = []
            optimizer.zero_grad()

            for idx_ in range(batch_idx, batch_idx + batch_size):
                if idx_ >= len(train_idx_list):
                    break
                cur_idx = train_idx_list[idx_]
                if args.model_name == 'ACMIL':
                    if conf.n_class == 2:
                        self_loss, outputs = run_one_sample(featureRandomSelection(TRAIN_FEATURES[cur_idx]),
                                                            MODEL, conf, args.device)
                        if outputs is None:
                            continue
                        slide_pred, slide_class = outputs
                        if not isinstance(self_loss, torch.Tensor):
                            self_loss = torch.tensor(self_loss, device=args.device)
                        batch_self_loss += self_loss
                        slide_cls.append(slide_class)
                    else:
                        self_loss, slide_pred = run_one_sample(featureRandomSelection(TRAIN_FEATURES[cur_idx]),
                                                               MODEL, conf, args.device)
                        if slide_pred is None:
                            continue
                        if not isinstance(self_loss, torch.Tensor):
                            self_loss = torch.tensor(self_loss, device=args.device)
                        batch_self_loss += self_loss
                else:
                    feat_ = torch.from_numpy(featureRandomSelection(TRAIN_FEATURES[cur_idx])).unsqueeze(0).float().to(args.device)
                    feat_ = feature_dropout(feat_, dropout_rate=DROPOUT_RATE)
                    if conf.n_class == 2:
                        slide_pred, slide_class = MODEL(feat_)
                        slide_cls.append(slide_class)
                    else:
                        slide_pred = MODEL(feat_)
                if isinstance(slide_pred, (tuple,list)):
                    slide_pred = slide_pred[0]
                slide_preds.append(slide_pred)
                times_.append(TRAIN_TIME.iloc[cur_idx])
                events_.append(TRAIN_EVENT.iloc[cur_idx])
                
            if not slide_preds:
                continue

            preds = torch.vstack(slide_preds)[:, 0]
            if conf.n_class == 2:
                cls_logits = torch.vstack(slide_cls)
            numerical_events = [event_mapping[e] if isinstance(e, str) else e for e in events_]
            times_tensor = torch.tensor(times_).to(args.device)
            events_tensor = torch.tensor(numerical_events, dtype=torch.float32).to(args.device)

            if len(indices) == 1:
                sel_loss_fn = loss_funcs[indices[0]]
                Surv_loss = sel_loss_fn(preds, times_tensor, events_tensor)
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

            if conf.n_class == 2:
                target_cls = torch.tensor(numerical_events, dtype=torch.long).to(args.device)
                cls_loss = criterion(cls_logits, target_cls)
                final_loss = batch_self_loss * BATCH_SELF_LOSS_WEIGHT + Surv_loss * SURV_LOSS_WEIGHT + cls_loss * CLS_LOSS_WEIGHT
            else:
                final_loss = batch_self_loss * BATCH_SELF_LOSS_WEIGHT + Surv_loss * (SURV_LOSS_WEIGHT + CLS_LOSS_WEIGHT)

            trn_loss += final_loss.item()
            final_loss.backward()
            optimizer.step()

            flipped_preds = -preds.detach().cpu()
            slide_preds_save.extend(flipped_preds.tolist())
            times_save.extend(times_)
            events_save.extend(events_)

        print(f"Epoch {Epoch}, Training Loss: {trn_loss:.4f}")
        torch.cuda.empty_cache()

        MODEL.eval()
        ncc_preds, ncc_times, ncc_events_ = [], [], []
        for idx_ in range(len(NCC_Features)):
            with torch.no_grad():
                if args.model_name == 'ACMIL':
                    if conf.n_class == 2:
                        _, outputs = run_one_sample(NCC_Features[idx_], MODEL, conf, args.device)
                        slide_pred, _ = outputs
                    else:
                        _, slide_pred = run_one_sample(NCC_Features[idx_], MODEL, conf, args.device)
                else:
                    feat_ = torch.from_numpy(featureRandomSelection(NCC_Features[idx_])).unsqueeze(0).float().to(args.device)
                    if conf.n_class == 2:
                        slide_pred, _ = MODEL(feat_)
                    else:
                        slide_pred = MODEL(feat_)
                if isinstance(slide_pred, (tuple,list)):
                    slide_pred = slide_pred[0]
                ncc_preds.append(-slide_pred.item())
                ncc_times.append(NCC_TIME.iloc[idx_])
                ncc_events_.append(NCC_EVENT.iloc[idx_])
        data_test = pd.DataFrame({
            'risk': np.array(ncc_preds),
            'times': ncc_times,
            'events': pd.to_numeric(NCC_EVENT, errors='coerce')
        }).dropna(subset=['events'])
        data_test['PredClass'] = data_test['risk'] < data_test['risk'].median()

        current_test_cindex = concordance_index(data_test['times'], data_test['risk'], data_test['events'])
        print(f"Epoch {Epoch}: Test C-index: {current_test_cindex:.3f}")

        if current_test_cindex > best_test_cindex:
            best_test_cindex = current_test_cindex
            best_epoch = Epoch
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > BEST_EPOCH_PACIENCE//2:
                print(f"no_improve_count = {no_improve_count} (BEST_EPOCH_PACIENCE = {BEST_EPOCH_PACIENCE})")

        if no_improve_count >= BEST_EPOCH_PACIENCE:
            print(f"Early stopping triggered at epoch {Epoch}. Best test c-index: {best_test_cindex:.3f} at epoch {best_epoch}")
            break

        data_train = pd.DataFrame({
            'risk': np.array(slide_preds_save),
            'times': times_save,
            'events': pd.Series(events_save).map(event_mapping)
        }).dropna(subset=['times', 'risk', 'events'])
        data_train['PredClass'] = data_train['risk'] < data_train['risk'].median()

        val_slide_preds, val_times, val_events = [], [], []
        for idx_ in range(len(VALID_FEATURES)):
            with torch.no_grad():
                if args.model_name == 'ACMIL':
                    if conf.n_class == 2:
                        _, outputs = run_one_sample(VALID_FEATURES[idx_], MODEL, conf, args.device)
                        slide_pred, _ = outputs
                    else:
                        _, slide_pred = run_one_sample(VALID_FEATURES[idx_], MODEL, conf, args.device)
                else:
                    feat_ = torch.from_numpy(featureRandomSelection(VALID_FEATURES[idx_])).unsqueeze(0).float().to(args.device)
                    if conf.n_class == 2:
                        slide_pred, _ = MODEL(feat_)
                    else:
                        slide_pred = MODEL(feat_)
                if isinstance(slide_pred, (tuple,list)):
                    slide_pred = slide_pred[0]
                val_slide_preds.append(-slide_pred.item())
                val_times.append(VALID_TIME.iloc[idx_])
                val_events.append(VALID_EVENT.iloc[idx_])
        data_valid = pd.DataFrame({
            'risk': np.array(val_slide_preds),
            'times': val_times,
            'events': pd.Series(val_events).map(event_mapping)
        }).dropna(subset=['times','risk','events'])
        data_valid['PredClass'] = data_valid['risk'] < data_valid['risk'].median()

        print("data_train shape:", data_train.shape)
        print(data_train.head())

        log_ranks = [
            logRankTest(data_train),
            logRankTest(data_valid),
            logRankTest(data_test)
        ]
        c_index_all = [
            concordance_index(data_train['times'], data_train['risk'], data_train['events']),
            concordance_index(data_valid['times'], data_valid['risk'], data_valid['events']),
            concordance_index(data_test['times'], data_test['risk'], data_test['events'])
        ]
        scheduler.step(1 - c_index_all[1])

        log_ranks_str = '-'.join([f'{i:.3f}' for i in log_ranks])
        c_index_str = '-'.join([f'{i:.3f}' for i in c_index_all])
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
        print(f"Saved model weights to {weights_file}\n")

    print(f"\nTraining complete. Best test c-index={best_test_cindex:.3f} at epoch {best_epoch} (initial attempt(s) needed={reinit_attempts}).")

def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--Backbone',     type=str, default='UNI')
    parser.add_argument('--cuda_divce',   type=int, default=0)
    parser.add_argument('--models',       type=str, default='ACMIL',
                        help="Comma-separated list of model names (e.g., 'ACMIL,TransMIL')")
    parser.add_argument('--losses',       type=str, default='coxph',
                        help="Comma-separated list of loss names. For combined‐loss experiments, provide two (e.g., 'coxph,rank')")
    parser.add_argument('--num_loss',     type=int, default=1,
                        help="Number of loss functions to use: 1 or 2.")
    parser.add_argument('--loss_weight',  type=str, default="1.0",
                        help="Comma-separated list of weights for combined loss (e.g., '0.0,0.25,0.5,0.75,1.0').")
    parser.add_argument('--Epoch',        type=int, default=100)
    parser.add_argument('--learning_rate',type=str, default="1e-4",
                        help="Comma-separated list of learning rates (e.g., '1e-6,2e-6,5e-6,1e-5')")
    parser.add_argument('--ExternalDatasets', nargs='+', type=str, default=[])
    parser.add_argument('--repeat',       type=int, default=0,
                        help="Manually specify a repeat number to append to output filenames.")
    return parser


def run_experiment(args):
    # mirror your existing __main__ body here:
    args.device = torch.device(f"cuda:{args.cuda_divce}"
                               if torch.cuda.is_available() else "cpu")
    args.models = [m.strip() for m in args.models.split(',')]
    learning_rates = [float(x) for x in args.learning_rate.split(',')]

    preloaded_pdac = getDataTCGA_PDAC(args)
    preloaded_ncc  = getData(args)
    preloaded_ext  = ( getDataTCGA_All(args, cancer_types=args.ExternalDatasets)
                      if args.ExternalDatasets else None )

    for model_name in args.models:
        for loss_str in [l.strip() for l in args.losses.split(',')]:
            for lr in learning_rates:
                args.model_name = model_name
                args.loss       = loss_str
                args.current_lr = lr

                print(f"\nTraining with Model: {model_name}, "
                      f"Loss: {loss_str}, LR: {lr:.2e}")
                train_model(
                    args,
                    configs={
                        'STAD': {'stride': 224, 'patch_size': 224, 'batch_size': 32, 'downsample': 1},
                        'TCGA': {'stride': 448, 'patch_size': 448, 'batch_size': 32, 'downsample': 2},
                        'config_file': CODE_DIR / 'config/huaxi_medical_ssl_config.yml',
                        'feature_dim': 1024,
                    },
                    preloaded_pdac=preloaded_pdac,
                    preloaded_external=preloaded_ext,
                    preloaded_ncc=preloaded_ncc
                )


def main(argv=None):
    parser = build_parser()
    # use parse_known_args if you want to ignore Jupyter flags, or parse_args(argv)
    if argv is None:
        args, _ = parser.parse_known_args()
    else:
        args = parser.parse_args(argv)
    run_experiment(args)


if __name__ == '__main__':
    main()
