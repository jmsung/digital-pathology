#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import yaml
from utils.utils import Struct
from architecture.abmil import ABMIL, GatedABMIL
from architecture.acmil import AttnMIL6 as AttnMIL
from architecture.clam import CLAM_SB, CLAM_MB
from architecture.dsmil import BClassifier, FCLayer, MILNet
from architecture.transMIL import TransMIL
from modules import mean_max

# GPU memory optimization
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getModel(config_file, feature_dim, model_name='ABMIL', lr=1e-5):
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

# -----------------------------
# Fine-tuning configuration
# -----------------------------
# Change these paths and hyperparameters as needed
config_file = "path/to/huaxi_medical_ssl_config.yml"
feature_dim = 1024
pretrained_lr = 1e-4       # Learning rate used during pretraining
fine_tune_lr = 1e-5        # Lower learning rate for fine-tuning
fine_tune_epochs = 10      # Number of epochs for fine-tuning
pretrained_model_path = "path/to/pancreatic_pretrained_model.pth"

# -----------------------------
# Load pretrained model
# -----------------------------
model, criterion, optimizer, conf = getModel(config_file, feature_dim, model_name='ABMIL', lr=pretrained_lr)
model.load_state_dict(torch.load(pretrained_model_path))
model = model.to(device)

# -----------------------------
# Freeze early layers (adjust condition as needed)
# -----------------------------
for name, param in model.named_parameters():
    # Example: freeze any parameter whose name contains "early_layer" (change as needed)
    if "early_layer" in name:
        param.requires_grad = False

# Re-create the optimizer for fine-tuning with a lower learning rate
optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=fine_tune_lr, weight_decay=conf.wd)

# -----------------------------
# External dataloader setup (fill in with your own dataloader)
# -----------------------------
# For example, assume you have a PyTorch DataLoader for external data:
# external_dataloader = torch.utils.data.DataLoader(external_dataset, batch_size=32, shuffle=True)
# Replace the following line with your actual external dataloader.
external_dataloader = ...  # <--- Define your external data loader here

# -----------------------------
# Fine-tuning loop
# -----------------------------
for epoch in range(fine_tune_epochs):
    model.train()
    running_loss = 0.0
    for batch in external_dataloader:
        inputs, labels = batch  # External data inputs and labels
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(external_dataloader)
    print(f"Epoch {epoch+1}/{fine_tune_epochs}, Loss: {avg_loss:.4f}")
    
    # (Optionally, add a validation loop here and compute metrics such as the concordance index)

# -----------------------------
# Save fine-tuned model (optional)
# -----------------------------
fine_tuned_model_path = "path/to/fine_tuned_model.pth"
torch.save(model.state_dict(), fine_tuned_model_path)
print(f"Fine-tuned model saved to {fine_tuned_model_path}")
