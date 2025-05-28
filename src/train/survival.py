#!/usr/bin/env python3
import argparse
import io
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
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import yaml
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Tuple, Optional, Callable
from scipy.stats import rankdata # For percentile rank calculation
from sklearn.model_selection import train_test_split
from dataclasses import dataclass, field

# ---------------- Standard Setup ----------------
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# local library imports
current_script_file = Path(__file__).resolve()
project_root_dir = current_script_file.parent.parent.parent
module_dir = project_root_dir / 'src' / 'modules'
if str(module_dir) not in sys.path:
    sys.path.insert(0, str(module_dir))

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

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ---------------- Configuration ----------------
@dataclass
class PathsConfig:
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    CODE_DIR: Path = field(init=False)
    DATA_DIR: Path = field(init=False)
    FEATURES_DIR: Path = field(init=False)
    RESULTS_DIR: Path = field(init=False)
    FIGURES_DIR: Path = field(init=False)
    WEIGHTS_DIR: Path = field(init=False)
    MODEL_CONFIG_FILE: Path = field(init=False)

    def __post_init__(self):
        self.CODE_DIR = self.BASE_DIR / "src"
        self.DATA_DIR = self.BASE_DIR / "data"
        self.FEATURES_DIR = self.BASE_DIR / "features"
        self.RESULTS_DIR = self.BASE_DIR / "results"
        self.FIGURES_DIR = self.RESULTS_DIR / 'Results_figures'
        self.WEIGHTS_DIR = self.RESULTS_DIR / 'Results_weights'
        self.MODEL_CONFIG_FILE = self.CODE_DIR / 'config/huaxi_medical_ssl_config.yml'
        self.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        self.WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


logger = logging.getLogger(__name__)

@dataclass
class TrainingParams:
    # core settings
    BATCH_SIZE_INT: int = 60
    NUM_SAMPLES_INT: int = 180
    NUM_SAMPLES_EXT: int = 180
    CINDEX_MIN_EPOCH0: float = 0.5
    MAX_REINIT_ATTEMPTS: int = 50
    EARLY_STOPPING_PATIENCE: int = 30
    FEATURE_DIM: int = 1024

    # mappings
    EVENT_MAPPING: Dict[str, int] = field(
        default_factory=lambda: {'Alive': 0, 'Dead': 1}
    )
    CANCER_TYPES: List[str] = field(
        default_factory=lambda: ['PDAC','CHOL','ESCA','LIHC','COAD','LUAD','BRCA']
    )
    CANCER_MAPPING: Dict[str,int] = field(init=False)

    # loss‐weight slots
    BATCH_SELF_LOSS_WEIGHT: float = 0.00
    EVENT_LOSS_WEIGHT: float = 0.10  # for Alive/Dead head
    CANCER_LOSS_WEIGHT: float = 0.10 # for cancer‐type head
    SURV_LOSS_WEIGHT: float = field(init=False)  # the remainder

    def __post_init__(self):
        # build the cancer→index map
        self.CANCER_MAPPING = {c: i for i, c in enumerate(self.CANCER_TYPES)}
        logger.info(f"CANCER_MAPPING = {self.CANCER_MAPPING}")

        # now compute survival‐head weight so everything sums to 1.0
        total_fixed = (
            self.BATCH_SELF_LOSS_WEIGHT
          + self.EVENT_LOSS_WEIGHT
          + self.CANCER_LOSS_WEIGHT
        )
        self.SURV_LOSS_WEIGHT = 1.0 - total_fixed

        if self.SURV_LOSS_WEIGHT < 0:
            raise ValueError(
                f"Loss weights sum to >1.0 (fixed={total_fixed:.2f})"
            )

        logger.info(
            f"Loss weights → self:{self.BATCH_SELF_LOSS_WEIGHT:.2f}, "
            f"surv:{self.SURV_LOSS_WEIGHT:.2f}, "
            f"event:{self.EVENT_LOSS_WEIGHT:.2f}, "
            f"cancer:{self.CANCER_LOSS_WEIGHT:.2f}"
        )


paths_cfg = PathsConfig()
train_cfg = TrainingParams()

# ---------------- Helper for Percentile Rank Normalization ----------------
def calculate_percentile_ranks(times: pd.Series) -> pd.Series:
    """
    Calculates percentile ranks for a series of times.
    Smallest time gets smallest rank. Handles ties by averaging.
    Scales ranks to be approximately in [0, 1].
    """
    if not isinstance(times, pd.Series):
        times = pd.Series(times) # Ensure it's a Series for index access
    if times.empty or times.isnull().all(): # Handle empty or all-NaN series
        return pd.Series(dtype='float64', index=times.index)
    
    # Use rankdata for robust ranking, method='average' handles ties.
    # rankdata output is 1-based.
    ranks = rankdata(times.fillna(np.inf), method='average') # Fill NaNs to rank them last if any

    # Filter out original NaNs before scaling if necessary, or ensure they map to NaN rank
    # For now, rankdata with fillna(np.inf) pushes NaNs to the end.
    # We want ranks relative to observed data.

    valid_indices = times.notna()
    if not valid_indices.any(): # No valid data points
         return pd.Series(dtype='float64', index=times.index)

    num_valid = valid_indices.sum()

    if num_valid <= 1: # If only one valid data point or none
        # Assign a mid-rank (0.5) to the single point, NaN to others
        percentile_ranks_values = np.full(len(times), np.nan)
        if num_valid == 1:
            percentile_ranks_values[valid_indices] = 0.5
    else:
        # Scale ranks of valid data to be approximately [0, 1]
        # (rank - 1) makes it 0-based, then divide by (num_valid - 1)
        scaled_ranks = (ranks[valid_indices] - 1) / (num_valid - 1)
        percentile_ranks_values = np.full(len(times), np.nan)
        percentile_ranks_values[valid_indices] = scaled_ranks
        
    return pd.Series(percentile_ranks_values, index=times.index, dtype='float64')

# ---------------- Data Loading Functions (Modified for Percentile Rank) ----------------

# For NCC Test Data (Returns RAW Times for Evaluation)
def get_data_pdac_test(backbone_name: str, data_dir: Path, features_dir: Path) -> Tuple[List, List, List, pd.DataFrame]:
    logger.info(f"Loading NCC PDAC (Test) data for backbone: {backbone_name} - using RAW survival times for evaluation.")
    clinical_file = data_dir / 'clinical' / "clinical.txt" # Path to NCC clinical data
    clinical_df = pd.read_csv(clinical_file, sep="\t")
    clinical_df.set_index('Study_no', inplace=True)

    # Assuming "Features_PDAC" here refers to NCC PDAC features based on original script context
    # If NCC features are in a different folder, adjust this path.
    pdac_path = features_dir / "Features_PDAC" # Path to NCC feature directory
    pdac_coord_path = pdac_path / f"Coord_{backbone_name}"
    pdac_feature_path = pdac_path / f"Feature_{backbone_name}"

    filenames = sorted([p.stem for p in pdac_coord_path.iterdir() if p.suffix == ".pickle"])
    features_list, coords_list, barcodes_list = [], [], []

    for barcode_stem in tqdm(filenames, desc="Loading NCC PDAC (Test) features"):
        try:
            with open(pdac_feature_path / f"{barcode_stem}.pickle", 'rb') as f: feature = pickle.load(f)
            with open(pdac_coord_path / f"{barcode_stem}.pickle", 'rb') as f: coord = pickle.load(f)
            features_list.append(feature)
            coords_list.append(np.array(coord))
            barcodes_list.append(barcode_stem)
        except FileNotFoundError:
            logger.warning(f"NCC PDAC Test: Feature/Coord file not found for {barcode_stem}")
            continue
        
    slide_ids = [fn.split('.svs')[0] for fn in barcodes_list]
    clinical_df_filtered = clinical_df.loc[clinical_df.index.isin(slide_ids)].reindex(slide_ids)
    
    # Rename for consistency, keep times RAW
    clinical_df_filtered = clinical_df_filtered.rename(columns={'PFS_mo': 'Time_Raw', 'PFS_event_Up220713': 'Status'})
    # For test set evaluation, 'Time' should be the raw time.
    clinical_df_filtered['Time'] = clinical_df_filtered['Time_Raw'] * (365/12) # convert months to days
    
    # Ensure 'Status' is mapped if it's string
    if clinical_df_filtered['Status'].dtype == 'object':
        clinical_df_filtered['Status'] = clinical_df_filtered['Status'].map(train_cfg.EVENT_MAPPING).fillna(clinical_df_filtered['Status'])
    clinical_df_filtered['Status'] = pd.to_numeric(clinical_df_filtered['Status'], errors='coerce')

    return features_list, coords_list, barcodes_list, clinical_df_filtered[['Time', 'Status', 'Time_Raw']] # Return Time (raw) and Status

# For TCGA PDAC Training/Validation Data (Returns RAW and Percentile Ranked Times)
def get_data_tcga_pdac_train_val(backbone_name: str, data_dir: Path, features_dir: Path) -> Tuple[List, List, List, pd.DataFrame]:
    logger.info(f"Loading TCGA PDAC (Train/Val) data for backbone: {backbone_name} - applying percentile rank normalization.")
    tcga_path = features_dir / "Features_TCGA" # Path to TCGA PDAC features
    tcga_coord_path = tcga_path / f"Coord_{backbone_name}"
    tcga_feature_path = tcga_path / f"Feature_{backbone_name}"

    filenames_tcga_stems = [p.stem for p in tcga_coord_path.iterdir() if p.suffix == ".pickle" and ".svs" in p.name]

    clinical_tcga_df = pd.read_csv(data_dir / 'clinical' / 'clinical.tsv', sep="\t") # TCGA PDAC clinical
    clinical_tcga_df.set_index('case_submitter_id', inplace=True)
    clinical_tcga_df = clinical_tcga_df[['vital_status', 'days_to_death', 'days_to_last_follow_up']]
    clinical_tcga_df = clinical_tcga_df.applymap(lambda x: 0 if x == '\'--' else x)

    clinical_data_processed = []
    for idx, row in clinical_tcga_df.iterrows():
        status_val = train_cfg.EVENT_MAPPING.get(row['vital_status'], row['vital_status'])
        if not isinstance(status_val, (int,float)): status_val = pd.to_numeric(status_val, errors='coerce')
        if pd.isna(status_val): continue
        time_val = max(float(row['days_to_death']), float(row['days_to_last_follow_up']))
        clinical_data_processed.append({'Barcode': idx, 'Status': status_val, 'Time_Raw': time_val})
    
    clinical_final_df = pd.DataFrame(clinical_data_processed).set_index('Barcode')
    clinical_final_df = clinical_final_df.loc[~clinical_final_df.index.duplicated(keep='first')]

    features_list, coords_list, barcodes_out_list = [], [], []
    valid_tcga_stems_info = [{'stem': stem, 'case_id': stem[:12]} for stem in filenames_tcga_stems if stem[:12] in clinical_final_df.index]
            
    for item_info in tqdm(valid_tcga_stems_info, desc="Loading TCGA PDAC (Train/Val) features"):
        stem = item_info['stem']
        try:
            with open(tcga_feature_path / f"{stem}.pickle", 'rb') as f: feature = pickle.load(f)
            with open(tcga_coord_path / f"{stem}.pickle", 'rb') as f: coord = pickle.load(f)
            features_list.append(feature)
            coords_list.append(np.array(coord))
            barcodes_out_list.append(stem)
        except FileNotFoundError: logger.warning(f"TCGA PDAC Train/Val: File not found {stem}"); continue
        except Exception as e: logger.error(f"Error loading TCGA PDAC {stem}: {e}"); continue

    loaded_case_ids = [b[:12] for b in barcodes_out_list]
    clinical_final_df_filtered = clinical_final_df.loc[clinical_final_df.index.isin(loaded_case_ids)].reindex(loaded_case_ids)
    
    # Add cancer type for consistent processing, though it's all PDAC here
    clinical_final_df_filtered['cancer_type'] = 'PDAC_TCGA' 
    # Calculate Percentile Rank Time within this PDAC cohort
    clinical_final_df_filtered['Time_PercentileRank'] = calculate_percentile_ranks(clinical_final_df_filtered['Time_Raw'])
    # 'Time' used for training will be the percentile rank
    clinical_final_df_filtered['Time'] = clinical_final_df_filtered['Time_PercentileRank']

    return features_list, coords_list, barcodes_out_list, clinical_final_df_filtered[['Time', 'Status', 'Time_Raw', 'Time_PercentileRank', 'cancer_type']]

# For External TCGA Data (Returns RAW and Percentile Ranked Times, grouped by cancer type)
def get_data_tcga_all_external(
    backbone_name: str,
    cancer_types: List[str],
    data_dir: Path,
    features_dir: Path
) -> Tuple[List[np.ndarray], List[np.ndarray], List[str], pd.DataFrame]:
    """
    Load Features_AllTypes for multiple cancer types, apply
    percentile-rank normalization within each type.

    Returns:
      features:     list of N feature arrays
      coords:       list of N coord arrays
      barcodes:     list of N filename stems
      clinical_df:  DataFrame with index=barcode, cols=[
                     Time, Status, Time_Raw, Time_PercentileRank, cancer_type
                   ]
    """
    logger.info(f"External: loading types={cancer_types}, backbone={backbone_name}")

    # 1) read & filter clinical
    clin = pd.read_csv(data_dir/'clinical'/"TCGA_clinical_data.tsv", sep="\t")
    clin = clin[clin.cancer_type.isin(cancer_types)].copy()
    clin.index = clin.case_submitter_id
    clin = clin.loc[:, ['time','status','cancer_type']].rename(
        columns={'time':'Time_Raw','status':'Status'}
    )

    # map status strings to 0/1
    clin['Status'] = clin['Status'].map({'Alive':0,'Dead':1}).astype(float)

    # 2) scan pickle stems
    base = features_dir/"Features_AllTypes"
    coord_dir = base/f"Coord_{backbone_name}"
    feat_dir  = base/f"Feature_{backbone_name}"
    stems = [p.stem for p in coord_dir.glob("*.pickle")]

    # only keep those whose first 12 chars in clinical
    valid = [s for s in stems if s[:12] in clin.index]
    if not valid:
        logger.warning("No matching external .pickle files found.")
        return [], [], [], pd.DataFrame()

    # 3) load features & coords, build aligned clinical records
    feats, coords, barcodes, recs = [], [], [], []
    for stem in tqdm(valid, desc="Loading external features"):
        try:
            with open(feat_dir/f"{stem}.pickle",'rb') as f:
                feats.append(pickle.load(io.BytesIO(f.read())))
            with open(coord_dir/f"{stem}.pickle",'rb') as f:
                coords.append(np.array(pickle.load(io.BytesIO(f.read()))))
        except Exception as e:
            logger.error(f"Skipping {stem}: {e}")
            continue

        barcodes.append(stem)
        row = clin.loc[stem[:12]]
        recs.append({
            'barcode':         stem,
            'Time_Raw':        float(row.Time_Raw),
            'Status':          row.Status,
            'cancer_type':     row.cancer_type
        })

    if not recs:
        logger.warning("After loading, no valid external samples remain.")
        return [], [], [], pd.DataFrame()

    # 4) assemble DataFrame
    df = pd.DataFrame(recs).set_index('barcode')
    # per‐type percentile rank
    df['Time_PercentileRank'] = (
        df.groupby('cancer_type')['Time_Raw']
          .transform(calculate_percentile_ranks)
    )
    df['Time'] = df['Time_PercentileRank']

    return feats, coords, barcodes, df[['Time','Status','Time_Raw','Time_PercentileRank','cancer_type']]


# ---------------- Model Loading Function ----------------
def get_model(
    model_config_path: Path, # Parameter name changed from 'config_file'
    feature_dim: int,
    n_classes: int,          # This comes from TrainingParams.N_CLASS
    model_name: str = 'ACMIL',
    lr: float = 1e-5
) -> Tuple[nn.Module, nn.Module, torch.optim.Optimizer, Struct]:
    
    with open(model_config_path, "r") as ymlfile:
        c = yaml.load(ymlfile, Loader=yaml.FullLoader) # c is the dictionary from YAML
    conf = Struct(**c) # Attributes are created from keys in c

    # Replicate original behavior:
    # These values are either newly set or override any same-named values from the YAML.
    conf.n_token = 5
    conf.n_masked_patch = 10
    conf.mask_drop = 0.3
    
    # These are set from function arguments / TrainingParams
    conf.D_feat = feature_dim
    conf.n_class = n_classes # Use the n_classes argument

    logger.info(f"Initializing model: {model_name} with D_feat={conf.D_feat}, n_class={conf.n_class}, "
                f"n_token={conf.n_token}, n_masked_patch={conf.n_masked_patch}, mask_drop={conf.mask_drop}")

    model: nn.Module
    if model_name == 'ACMIL':
        model = AttnMIL(conf)
    elif model_name == 'CLAM_SB':
        model = CLAM_SB(conf)
    elif model_name == 'CLAM_MB':
        model = CLAM_MB(conf)
    elif model_name == 'TransMIL':
        model = TransMIL(conf)
    elif model_name == 'DSMIL':
        i_classifier = FCLayer(conf.D_feat, conf.n_class)
        b_classifier = BClassifier(conf, nonlinear=False)
        model = MILNet(i_classifier, b_classifier)
    elif model_name == 'MeanMIL':
        model = mean_max.MeanMIL(conf)
    elif model_name == 'MaxMIL':
        model = mean_max.MaxMIL(conf)
    elif model_name == 'ABMIL':
        model = ABMIL(conf)
    elif model_name == 'GABMIL':
        model = GatedABMIL(conf)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    # Improved criterion handling from refactoring
    criterion_event  = nn.CrossEntropyLoss()
    criterion_cancer = nn.CrossEntropyLoss()
    
    # For weight_decay: Assume 'wd' is present in the YAML and thus an attribute of 'conf'
    # This matches the original working code. If 'wd' can be missing, this will error.
    # If 'wd' can be missing and you need a default, then `getattr(conf, 'wd', 1e-4)` would be better.
    # But to strictly match the original which worked:
    try:
        weight_decay_val = conf.wd
    except AttributeError:
        logger.warning(f"'wd' (weight_decay) not found in config file {model_config_path}. Using default 1e-4.")
        weight_decay_val = 1e-4 # Provide a default if it's missing, to prevent crash

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay_val
    )
    return model, criterion_event, criterion_cancer, optimizer, conf


def sample_external_data(
    preloaded_external: Tuple[List, List, List, pd.DataFrame],
    sample_size: int
) -> Tuple[List, pd.DataFrame]:
    features_all, _, barcodes_all, clinical_df_all = preloaded_external

    if not features_all or clinical_df_all.empty or len(features_all) == 0:
        logger.info("[sample_external_data] No external features or clinical data available (loader returned empty).")
        return [], pd.DataFrame()

    total_valid_samples = len(features_all) # This is the number of synchronized, valid samples
    if total_valid_samples < sample_size:
        # Sample with replacement if fewer valid samples than requested sample_size
        sampled_indices = np.random.choice(total_valid_samples, size=sample_size, replace=True)
    else:
        # Sample without replacement
        sampled_indices = np.random.choice(total_valid_samples, size=sample_size, replace=False)

    sampled_features = [features_all[i] for i in sampled_indices]
    sampled_original_barcodes = [barcodes_all[i] for i in sampled_indices]
    sampled_clinical_df = clinical_df_all.iloc[sampled_indices].copy()
    sampled_clinical_df.index = sampled_original_barcodes

    return sampled_features, sampled_clinical_df


def select_loss_functions(loss_arg: str, device: torch.device) -> Tuple[List[Callable], str, List[str]]:
    """Parses loss argument and returns loss functions and their names."""
    loss_name_map = {
        'coxph': cox_ph_loss,
        'rank': rank_loss,
        'MSE': MSE_loss,
        'SurvPLE': SurvPLE() # This is a class instance
    }
    loss_input_names = [l.strip() for l in loss_arg.split(',')]
    loss_funcs = []
    parsed_loss_names = []

    for lname in loss_input_names:
        if lname in loss_name_map:
            loss_funcs.append(loss_name_map[lname])
            parsed_loss_names.append(lname)
        else:
            raise ValueError(f"Loss name '{lname}' not recognized. Available: {list(loss_name_map.keys())}")

    if not 1 <= len(loss_funcs) <= 2:
        raise ValueError("Please provide one or two loss names for --loss.")

    combined_name = "_".join(parsed_loss_names)
    return loss_funcs, combined_name, parsed_loss_names


def calculate_survival_loss(
    preds: torch.Tensor,
    times: torch.Tensor,
    events: torch.Tensor,
    loss_funcs: List[Callable],
    loss_names: List[str], # Original names like 'coxph', 'rank'
    loss_weight_alpha: float # Weight for the first loss if two are used
) -> torch.Tensor:
    """Calculates the survival loss, handling single or combined losses."""
    
    # Ensure preds is 1D
    if preds.dim() > 1 and preds.size(1) == 1:
        preds = preds.squeeze(1)
    elif preds.dim() > 1:
        raise ValueError(f"Predictions tensor should be 1D or Nx1, but got shape {preds.shape}")

    if len(loss_funcs) == 1:
        loss_val = loss_funcs[0](preds, times, events)
        if loss_names[0] in ["SurvPLE", "rank"]: # Names requiring sign flip
            loss_val = -loss_val
    elif len(loss_funcs) == 2:
        loss1_val = loss_funcs[0](preds, times, events)
        loss2_val = loss_funcs[1](preds, times, events)
        if loss_names[0] in ["SurvPLE", "rank"]:
            loss1_val = -loss1_val
        if loss_names[1] in ["SurvPLE", "rank"]:
            loss2_val = -loss2_val
        loss_val = loss1_val * loss_weight_alpha + loss2_val * (1.0 - loss_weight_alpha)
    else: # Should not happen due to check in select_loss_functions
        raise ValueError("Invalid number of loss functions.")
    return loss_val


# ---------------- Core Training and Evaluation Logic (Adjusted for new Time columns) ----------------
def process_batch(
    batch_features_raw: List[np.ndarray],
    batch_times_processed: List[float],    # percentile‐ranked times
    batch_events_raw: List[Any],           # “Alive” / “Dead”
    batch_cancers_raw: List[str],          # e.g. “PDAC”, “BRCA”, …
    model: nn.Module,
    model_conf: Struct,
    criterion_event: nn.Module,            # nn.CrossEntropyLoss for event head
    criterion_cancer: nn.Module,           # nn.CrossEntropyLoss for cancer‐type head
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    train_cfg: TrainingParams,
    loss_funcs: List[Callable],
    loss_names: List[str]
) -> Tuple[
    float,            # total loss
    List[float],      # flipped risk predictions for C-index
    List[float],      # times used in loss
    List[int],        # event labels
    List[int],        # cancer‐type labels
]:
    model.train()
    optimizer.zero_grad()

    # 1) Self-supervision loss (only for the AttnMIL variant):
    self_loss_acc = torch.tensor(0., device=args.device, requires_grad=True)

    # 2) Collectors
    risks:       List[torch.Tensor] = []
    event_logits:  List[torch.Tensor] = []
    cancer_logits: List[torch.Tensor] = []
    times_for_loss:  List[float] = []
    events_idx:      List[int]   = []
    cancers_idx:     List[int]   = []

    for feat_raw, t, e_str, c_str in zip(
        batch_features_raw,
        batch_times_processed,
        batch_events_raw,
        batch_cancers_raw
    ):
        sel = featureRandomSelection(feat_raw)
        if sel is None or sel.size == 0:
            continue
        x = torch.from_numpy(sel).unsqueeze(0).float().to(args.device)

        # --- forward + collect self‐loss for ACMIL ---
        if args.model_name == 'ACMIL':
            sl, outputs = run_one_sample(feat_raw, model, model_conf, args.device)
            if outputs is None:
                continue
            risk_pred, evt_pred, can_pred = outputs   # now expect 3 heads
            # accumulate
            sl = sl if isinstance(sl, torch.Tensor) else torch.tensor(sl, device=args.device)
            self_loss_acc = self_loss_acc + sl
        else:
            # for other MILs you must have updated them to return (risk, evt_logits, cancer_logits)
            risk_pred, evt_pred, can_pred = model(x)

        # flatten if needed
        if isinstance(risk_pred, (tuple,list)):
            risk_pred = risk_pred[0]
        if isinstance(evt_pred,  (tuple,list)):
            evt_pred  = evt_pred[0]
        if isinstance(can_pred,  (tuple,list)):
            can_pred  = can_pred[0]

        # collect
        risks.append(risk_pred)
        event_logits.append(evt_pred)
        cancer_logits.append(can_pred)
        times_for_loss.append(t)

        # turn string labels into ints
        evt_idx = train_cfg.EVENT_MAPPING.get(e_str, e_str)
        events_idx.append(int(evt_idx))
        can_idx = train_cfg.CANCER_MAPPING.get(c_str)
        cancers_idx.append(int(can_idx))

    if not risks:
        return 0.0, [], [], [], []

    # stack
    P = torch.vstack(risks)                              # [B,1]
    E = torch.tensor(events_idx, device=args.device)     # [B]
    C = torch.tensor(cancers_idx, device=args.device)    # [B]
    T = torch.tensor(times_for_loss, device=args.device)

    # 3) survival loss (coxph/rank/etc)
    surv_loss = calculate_survival_loss(
        P, T, E.float(),
        loss_funcs, loss_names,
        args.loss_weight
    )

    # 4) event classification loss
    evt_logits_tensor = torch.vstack(event_logits)      # [B,2]
    cls_loss_event = criterion_event(evt_logits_tensor, E.long())

    # 5) cancer‐type classification loss
    can_logits_tensor = torch.vstack(cancer_logits)     # [B, num_cancers]
    cls_loss_cancer = criterion_cancer(can_logits_tensor, C.long())

    # 6) combine
    total = (
        self_loss_acc   * train_cfg.BATCH_SELF_LOSS_WEIGHT +
        surv_loss       * train_cfg.SURV_LOSS_WEIGHT +
        cls_loss_event  * train_cfg.CLS_LOSS_WEIGHT +
        cls_loss_cancer * train_cfg.CANCER_LOSS_WEIGHT
    )

    if torch.isnan(total) or torch.isinf(total):
        logger.error(f"bad loss {total.item()}, skipping")
        return 0.0, [], [], [], []

    total.backward()
    optimizer.step()

    # prepare risk for C-index (flip so higher=risk→worse)
    flipped = (-P.squeeze(1)).detach().cpu().tolist()
    return total.item(), flipped, times_for_loss, events_idx, cancers_idx


def evaluate_model_on_set(
    features_set: List[np.ndarray],
    # For C-index: raw times (test) or percentile ranks (train/val)
    time_set_for_eval: pd.Series,
    event_set: pd.Series,          # e.g. “Alive”/“Dead”
    cancer_set: pd.Series,         # e.g. “PDAC”/“BRCA”/… 
    model: nn.Module,
    model_conf: Struct,
    device: torch.device,
    args: argparse.Namespace,
    train_cfg: TrainingParams,
    set_name: str = "Test",
    raw_times_for_plotting_optional: Optional[pd.Series] = None
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Runs the model on a held-out set and returns:
      - df:      DataFrame with columns [risk, times, true_event, pred_event, true_cancer, pred_cancer]
      - times_c: Series of times used for C-index
      - ev_c:    Series of true events (0/1) for C-index
    """
    model.eval()
    risks, times_cidx, events_true = [], [], []
    pred_events, true_events   = [], []
    pred_cancers, true_cancers = [], []
    times_plot = []

    for i in tqdm(range(len(features_set)), desc=f"Evaluating on {set_name} set"):
        feat_raw = features_set[i]
        sel = featureRandomSelection(feat_raw)
        if sel is None or sel.size == 0:
            continue

        x = torch.from_numpy(sel).unsqueeze(0).float().to(device)
        with torch.no_grad():
            if args.model_name == 'ACMIL':
                # run_one_sample must now return (risk, event_logits, cancer_logits)
                _, outputs = run_one_sample(feat_raw, model, model_conf, device)
                if outputs is None:
                    continue
                risk_logits, event_logits, cancer_logits = outputs
            else:
                risk_logits, event_logits, cancer_logits = model(x)

        # flatten if necessary
        if isinstance(risk_logits, (tuple, list)):       risk_logits  = risk_logits[0]
        if isinstance(event_logits, (tuple, list)):      event_logits = event_logits[0]
        if isinstance(cancer_logits, (tuple, list)):     cancer_logits= cancer_logits[0]

        # extract scalar risk
        risk = -risk_logits.item()  # flip so higher = worse
        risks.append(risk)

        # record the time for C-index
        t = time_set_for_eval.iloc[i]
        times_cidx.append(t)

        # record times for Kaplan–Meier plotting
        times_plot.append(
            raw_times_for_plotting_optional.iloc[i]
            if raw_times_for_plotting_optional is not None else t
        )

        # true / predicted event
        ev_str = event_set.iloc[i]
        ev_true = train_cfg.EVENT_MAPPING.get(ev_str, ev_str)
        ev_pred = int(torch.argmax(event_logits, dim=1).item())
        events_true.append(ev_true)
        true_events.append(int(ev_true))
        pred_events.append(ev_pred)

        # true / predicted cancer
        c_str = cancer_set.iloc[i]
        c_true = train_cfg.CANCER_MAPPING[c_str]
        c_pred = int(torch.argmax(cancer_logits, dim=1).item())
        true_cancers.append(c_true)
        pred_cancers.append(c_pred)

    # assemble DataFrame
    df = pd.DataFrame({
        'risk':       risks,
        'times':      times_plot,
        'true_event': true_events,
        'pred_event': pred_events,
        'true_cancer':true_cancers,
        'pred_cancer':pred_cancers,
    })

    # for C-index we need the raw list of events & times
    times_ser = pd.Series(times_cidx, name='times_for_c_index')
    events_ser= pd.Series(events_true, name='events')

    return df, times_ser, events_ser


# Main training function
def train_and_evaluate_model(
    args: argparse.Namespace, paths: PathsConfig, train_params: TrainingParams,
    # Clinical DFs now contain Time, Status, Time_Raw, Time_PercentileRank, cancer_type
    preloaded_pdac_train_val_data: Tuple[List, List, List, pd.DataFrame], 
    preloaded_external_data: Optional[Tuple[List, List, List, pd.DataFrame]],
    # NCC Test clinical DF contains Time (raw), Status, Time_Raw
    preloaded_ncc_test_data: Tuple[List, List, List, pd.DataFrame] 
):

    # Load PDAC training/validation data
    pdac_features_all, _, _, pdac_clinical_df_all = preloaded_pdac_train_val_data
    
    # For NCC test, Time is raw
    ncc_features, _, _, ncc_clinical_df_test = preloaded_ncc_test_data
    ncc_time_raw_test = ncc_clinical_df_test['Time'] # Raw times for test C-index
    ncc_event_test = ncc_clinical_df_test['Status'] # Raw events

    loss_funcs, loss_name_str, loss_orig_names = select_loss_functions(args.loss, args.device)

    num_sample_total = train_params.NUM_SAMPLES_INT + train_params.NUM_SAMPLES_EXT if args.ExternalDatasets else train_params.NUM_SAMPLES_INT
    batch_size_int = train_params.BATCH_SIZE_INT 
    batch_size_ext = int(batch_size_int * train_params.NUM_SAMPLES_EXT / train_params.NUM_SAMPLES_INT) if args.ExternalDatasets else 0
    batch_size_total = batch_size_int + batch_size_ext
    num_batches = num_sample_total // batch_size_total
    
    # pick exactly NUM_INTERNAL_SAMPLES unique training cases
    all_idx = np.arange(len(pdac_features_all))        
    n_int = min(len(all_idx), train_params.NUM_SAMPLES_INT)

    current_random_state = random.randint(0, 99999)
    rs = np.random.RandomState(current_random_state)
    train_idx = rs.choice(all_idx, size=n_int, replace=False)
    internal_train_features = [pdac_features_all[i] for i in train_idx]
    internal_train_clinical_df = pdac_clinical_df_all.iloc[train_idx]

    # everyone else is validation
    valid_idx = np.setdiff1d(all_idx, train_idx)
    valid_features = [pdac_features_all[i] for i in valid_idx]
    internal_valid_clinical_df = pdac_clinical_df_all.iloc[valid_idx]

    model, crit_event, crit_cancer, optimizer, model_conf = get_model(
        paths.MODEL_CONFIG_FILE, train_params.FEATURE_DIM,
        model_name=args.model_name, lr=args.current_lr
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=10)
    model.to(args.device)

    for epoch in range(args.Epoch):
        logger.info(f"=== EPOCH {epoch}/{args.Epoch} ===")
        
        current_train_features_epoch = list(internal_train_features)
        current_train_clinical_dfs_list_epoch = [internal_train_clinical_df.copy()]
        
        if args.ExternalDatasets and preloaded_external_data:
            ext_feat, ext_clinical_df_sampled = sample_external_data(preloaded_external_data, train_params.NUM_SAMPLES_EXT)
            if not ext_clinical_df_sampled.empty:
                current_train_features_epoch.extend(ext_feat)
                current_train_clinical_dfs_list_epoch.append(ext_clinical_df_sampled)
        
        combined_train_clinical_df_epoch = pd.concat(current_train_clinical_dfs_list_epoch, ignore_index=True)
        current_train_time_processed_epoch = combined_train_clinical_df_epoch['Time_PercentileRank']
        current_train_event_epoch = combined_train_clinical_df_epoch['Status']
        current_train_time_raw_epoch = combined_train_clinical_df_epoch['Time_Raw'] # For plotting

        logger.info(f"Epoch {epoch}: Total training samples = {len(current_train_features_epoch)}")

        int_data_indices_shuffled = list(range(train_params.NUM_SAMPLES_INT))
        random.shuffle(int_data_indices_shuffled)
        ext_data_indices_shuffled = list(range(train_params.NUM_SAMPLES_INT, num_sample_total))
        random.shuffle(ext_data_indices_shuffled)

        epoch_loss_sum = 0.0; num_batches_current_epoch = 0
        epoch_train_preds_all, epoch_train_times_processed_all, epoch_train_events_all = [], [], []
        epoch_train_times_raw_all_for_plot = []

        for batch_idx in tqdm(range(num_batches), desc=f"Epoch {epoch} Training"):
            batch_indices = int_data_indices_shuffled[batch_idx*batch_size_int:(batch_idx+1)*batch_size_int] 
            batch_indices += ext_data_indices_shuffled[batch_idx*batch_size_ext:(batch_idx+1)*batch_size_ext] 
            if not batch_indices: continue
            # print(f"Batch {batch_idx + 1}/{num_batches} - Indices: {batch_indices}")

            batch_feat_raw = [current_train_features_epoch[i] for i in batch_indices]
            batch_time_processed = [current_train_time_processed_epoch.iloc[i] for i in batch_indices]
            batch_event_raw = [current_train_event_epoch.iloc[i] for i in batch_indices]
            batch_time_raw_for_df = [current_train_time_raw_epoch.iloc[i] for i in batch_indices]


            batch_loss, batch_preds, batch_times_ret, batch_events_ret, _ = process_batch(
                batch_feat_raw, batch_time_processed, batch_event_raw, model, model_conf, crit_event, crit_cancer,
                optimizer, args, train_params, loss_funcs, loss_orig_names
            )
            epoch_loss_sum += batch_loss; num_batches_current_epoch +=1
            epoch_train_preds_all.extend(batch_preds)
            epoch_train_times_processed_all.extend(batch_times_ret) # Percentile ranks
            epoch_train_events_all.extend(batch_events_ret)
            epoch_train_times_raw_all_for_plot.extend(batch_time_raw_for_df) # Raw times


        if num_batches_current_epoch > 0: logger.info(f"Epoch {epoch} Training Loss: {epoch_loss_sum / num_batches_current_epoch:.4f}")
        torch.cuda.empty_cache()

        # Evaluate model
        # For train_df_current, C-index against percentile ranks, plot with raw times
        # This requires a bit of careful handling in evaluate_model_on_set or how data is passed
        # Let's use the collected preds and times directly for train_df_current
        train_df_current_plot = pd.DataFrame({
            'risk': epoch_train_preds_all, 
            'times': epoch_train_times_raw_all_for_plot, # Use RAW times for plotting
            'events': epoch_train_events_all
        }).dropna()
        # C-index for training set uses percentile-ranked times
        c_train = concordance_index(pd.Series(epoch_train_times_processed_all), pd.Series(epoch_train_preds_all), pd.Series(epoch_train_events_all)) if epoch_train_preds_all else 0.0

        valid_df_current_plot, valid_times_cidx, valid_events_cidx = evaluate_model_on_set(
            valid_features, internal_valid_clinical_df['Time_PercentileRank'], internal_valid_clinical_df['Status'],
            model, model_conf, args.device, args, train_params, f"Valid (E{epoch})",
            raw_times_for_plotting_optional=internal_valid_clinical_df['Time_Raw']
        )
        c_valid = concordance_index(valid_times_cidx, valid_df_current_plot['risk'], valid_events_cidx) if not valid_df_current_plot.empty else 0.0
        
        test_df_current_plot, test_times_cidx, test_events_cidx = evaluate_model_on_set(
            ncc_features, ncc_time_raw_test, ncc_event_test, model, model_conf, args.device, args,
            train_params, f"NCC Test (E{epoch})"
        )
        c_test = concordance_index(test_times_cidx, test_df_current_plot['risk'], test_events_cidx) if not test_df_current_plot.empty else 0.0
        logger.info(f"E{epoch} C-Idx: Tr={c_train:.3f}, Vl={c_valid:.3f}, Ts={c_test:.3f}")

        for df in [train_df_current_plot, valid_df_current_plot, test_df_current_plot]:
            if not df.empty and 'risk' in df.columns and not df['risk'].isnull().all() and len(df['risk']) > 0:
                median_risk = df['risk'].median(); df['PredClass'] = df['risk'] < median_risk
            else: df['PredClass'] = False
        
        # 1. Still track the best test C-index and manage early stopping counters
        if c_test > best_test_cindex:
            best_test_cindex = c_test
            best_epoch = epoch
            epochs_no_improve = 0
            logger.info(f"New best test C-index: {best_test_cindex:.3f} at epoch {epoch}")
        else:
            epochs_no_improve += 1

        # 2. Prepare information for filenames (using current epoch's metrics)
        # Note: The variable names log_ranks_best and c_indices_best are slightly misleading now,
        # as they represent the *current* epoch's values for the filename.
        # You might rename them to log_ranks_current_epoch, c_indices_current_epoch if preferred for clarity.
        log_ranks_current = [logRankTest(df) if not df.empty and 'PredClass' in df.columns and df['PredClass'].any() else 0.0 
                             for df in [train_df_current_plot, valid_df_current_plot, test_df_current_plot]]
        c_indices_current = [c_train, c_valid, c_test] # Use the calculated C-indices for the current epoch

        log_ranks_str = '-'.join([f'{val:.3f}' for val in log_ranks_current])
        c_indices_str = '-'.join([f'{val:.3f}' for val in c_indices_current])
        
        # Modify filename to indicate it's for the current epoch, not necessarily the "BEST"
        # Removed "_BEST_" from the filename stem for per-epoch saves.
        # You could add a specific marker like "_EpochState_" or just rely on the epoch number.
        result_filename_stem = (
            f"{args.model_name}_{loss_name_str}_lr{args.current_lr:.2e}_"
            f"w{float(args.loss_weight):.1f}_Epc{epoch}_" # Just Epc{epoch} now
            f"LogRanks[{log_ranks_str}]_CIdx[{c_indices_str}]_"
            f"Ext{'_'.join(args.ExternalDatasets if args.ExternalDatasets else ['None'])}_{args.repeat}" # Handle empty ExternalDatasets
        )

        rep_folder_name = f"rep{args.repeat}" if args.repeat is not None else "all_reps"
        figure_save_dir = paths.FIGURES_DIR / rep_folder_name
        weights_save_dir = paths.WEIGHTS_DIR / rep_folder_name
        figure_save_dir.mkdir(parents=True, exist_ok=True); weights_save_dir.mkdir(parents=True, exist_ok=True)

        weights_file = weights_save_dir / f"{result_filename_stem}.pth"
        figure_file = figure_save_dir / f"{result_filename_stem}.png"

        # 3. Save model state and plot for the current epoch unconditionally
        try:
            torch.save(model.state_dict(), weights_file)
            logger.info(f"Saved model for epoch {epoch}: {weights_file}")
        except Exception as e_save:
            logger.error(f"Error saving model weights for epoch {epoch} to {weights_file}: {e_save}")

        try:
            # Ensure DataFrames are not empty before plotting
            if not train_df_current_plot.empty and not valid_df_current_plot.empty and not test_df_current_plot.empty:
                plotSurvival_three(train_df_current_plot, valid_df_current_plot, test_df_current_plot, filename=str(figure_file))
                logger.info(f"Saved plot for epoch {epoch}: {figure_file}")
            else:
                logger.warning(f"Skipping plot for epoch {epoch} due to empty dataframes for plotting.")
        except Exception as e_plot:
            logger.error(f"Error saving plot for epoch {epoch} to {figure_file}: {e_plot}")
        
        # 4. Scheduler step and Early stopping logic (remains the same)
        scheduler.step(1.0 - c_valid) # Step scheduler based on validation C-index
        
        if epochs_no_improve >= train_params.EARLY_STOPPING_PATIENCE:
            logger.info(f"Early stopping triggered at epoch {epoch}. "
                        f"Best test C-index was: {best_test_cindex:.3f} at epoch {best_epoch}.")
            break # Exit the training loop
        
        if epochs_no_improve > 0 and epochs_no_improve % (train_params.EARLY_STOPPING_PATIENCE // 4) == 0 : # Log periodically
             logger.info(f"No improvement in test C-index for {epochs_no_improve} epochs. "
                         f"Best test C-index remains: {best_test_cindex:.3f} from epoch {best_epoch}.")
        # --- Refactored Logic Ends Here ---


    logger.info(f"Training complete. Best Test C-idx={best_test_cindex:.3f} @E{best_epoch}. Re-inits={reinit_attempts}.")

# ---------------- Argument Parsing and Main Execution ----------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MIL models for survival analysis with percentile rank time normalization.")
    parser.add_argument('--Backbone', type=str, default='UNI', help="Feature extraction backbone")
    parser.add_argument('--cuda_device', type=int, default=0, help="CUDA device ID")
    parser.add_argument('--models', type=str, default='ACMIL', help="Comma-separated model names")
    parser.add_argument('--losses', type=str, default='coxph', help="Comma-separated loss names (1 or 2)")
    parser.add_argument('--loss_weight', type=float, default=1.0, help="Weight for FIRST loss if two given (0.0-1.0)")
    parser.add_argument('--Epoch', type=int, default=50, help="Max training epochs")
    parser.add_argument('--learning_rate', type=str, default="1e-4", help="Comma-separated learning rates")
    parser.add_argument('--ExternalDatasets', nargs='*', type=str, default=[], help="External TCGA cancer types for training augmentation")
    parser.add_argument('--repeat', type=int, default=0, help="Repeat number for output filenames")
    return parser

def run_experiment_iterations(args: argparse.Namespace, paths: PathsConfig, base_train_params: TrainingParams):
    args.device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {args.device}")

    model_names_list = [m.strip() for m in args.models.split(',')]
    learning_rates_list = [float(lr.strip()) for lr in args.learning_rate.split(',')]
    
    current_train_params = TrainingParams(
        BATCH_SIZE_INT=base_train_params.BATCH_SIZE_INT,
        NUM_SAMPLES_INT=base_train_params.NUM_SAMPLES_INT,
        NUM_SAMPLES_EXT=base_train_params.NUM_SAMPLES_EXT,
        CINDEX_MIN_EPOCH0=base_train_params.CINDEX_MIN_EPOCH0,
        MAX_REINIT_ATTEMPTS=base_train_params.MAX_REINIT_ATTEMPTS,
        EARLY_STOPPING_PATIENCE=base_train_params.EARLY_STOPPING_PATIENCE,
        FEATURE_DIM=base_train_params.FEATURE_DIM)

    logger.info("Pre-loading TCGA PDAC (Train/Val) data with percentile rank normalization...")
    # This now returns DF with Time_Raw, Time_PercentileRank, Time (set to percentile rank)
    preloaded_pdac_tv_data = get_data_tcga_pdac_train_val(args.Backbone, paths.DATA_DIR, paths.FEATURES_DIR)
    
    logger.info("Pre-loading NCC PDAC (Test) data with RAW times...")
    # This returns DF with Time (raw), Status, Time_Raw
    preloaded_ncc_test_data_raw = get_data_pdac_test(args.Backbone, paths.DATA_DIR, paths.FEATURES_DIR)

    preloaded_external_data_norm: Optional[Tuple[List, List, List, pd.DataFrame]] = None
    if args.ExternalDatasets:
        logger.info(f"Pre-loading external TCGA data ({args.ExternalDatasets}) with percentile rank normalization...")
        # This now returns DF with Time_Raw, Time_PercentileRank, Time (set to percentile rank), cancer_type
        preloaded_external_data_norm = get_data_tcga_all_external(args.Backbone, args.ExternalDatasets, paths.DATA_DIR, paths.FEATURES_DIR)

    for model_name_iter in model_names_list:
        for lr_iter in learning_rates_list:
            current_iter_args = argparse.Namespace(**vars(args))
            current_iter_args.model_name = model_name_iter
            current_iter_args.loss = args.losses
            current_iter_args.current_lr = lr_iter
            
            logger.info(f"\n--- Experiment: Model={model_name_iter}, Loss={args.losses}, LR={lr_iter:.2e} ---")
            try:
                train_and_evaluate_model(
                    current_iter_args, paths, current_train_params,
                    preloaded_pdac_tv_data,
                    preloaded_external_data_norm,
                    preloaded_ncc_test_data_raw
                )
            except Exception as e:
                logger.error(f"Error in experiment (Model: {model_name_iter}, LR: {lr_iter}): {e}", exc_info=True)
                continue

def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    global paths_cfg, train_cfg 
    run_experiment_iterations(args, paths_cfg, train_cfg)

if __name__ == '__main__':
    main()