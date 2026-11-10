import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
import time
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from model_drp_cnn import ContrastGNN
from utils import GDSCDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
from splitters_drp import split
from copy import deepcopy


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_params(model, grad_flag=False):
    for param in model.parameters():
        param.requires_grad = grad_flag


seed = 46
set_seed(seed)

gdsc_version = 'gdsc1'

gdscdataset = GDSCDataset(gdsc_version, 'rma', 'data/hgnc_csv/depmap_gdsc_tcga_gene.csv')
print(f"Dataset version: {gdsc_version}")
train_idx, val_idx, test_idx = split(gdsc_version, 'rma', 'smiles', 0.2)
print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

train_dataset, val_dataset, test_dataset = Subset(gdscdataset, train_idx), Subset(gdscdataset, val_idx), Subset(gdscdataset, test_idx)
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
expt_folder = os.path.join(f'/home/dataset-assist-0/datafile/quantianhong/MDR/expt_drp/{gdsc_version}', 'warm', f'{timestamp}', f'{seed}')
os.makedirs(expt_folder, exist_ok=True)


def train_one_epoch(model, train_loader, optimizer, device):
    model.train()
    epoch_loss = 0.0
    total_samples = 0
    
    loss_fn = nn.MSELoss()
    
    for idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        graph = batch[0]
        confs = batch[1]
        sfeats = batch[2]
        rna = batch[3]
        label = batch[4]
        graph = graph.to(device)
        confs = confs.to(device)
        sfeats = sfeats.to(device)
        rna = rna.to(device)
        label = label.to(device)
        
        num_samples = rna.shape[0]
        total_samples += num_samples
        
        pred = model(None, graph, confs, sfeats, rna)
        loss = loss_fn(pred, label)
        # print(f'batch_loss:{loss}')
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item() * num_samples
    
    avg_loss = epoch_loss / total_samples
    print(f'epoch_loss:{avg_loss}')

    return avg_loss


def validate(model, val_loader, device):
    model.eval()
    
    epoch_loss = 0.0
    total_samples = 0
    label_list = []
    output_list = []
    score_list = []
    val_metrics = {}
    
    loss_fn = nn.MSELoss()
    
    with torch.no_grad():
        for idx, batch in enumerate(val_loader):

            graph = batch[0]
            confs = batch[1]
            sfeats = batch[2]
            rna = batch[3]
            label = batch[4]
            graph = graph.to(device)
            confs = confs.to(device)
            sfeats = sfeats.to(device)
            rna = rna.to(device)
            label = label.to(device)

            num_samples = rna.shape[0]
            total_samples += num_samples

            pred = model(None, graph, confs, sfeats, rna)
            loss = loss_fn(pred, label)

            epoch_loss += loss.item() * num_samples
            
            label_list.append(label.cpu().detach().numpy().ravel())
            output_list.append(pred.cpu().detach().numpy().ravel())
    
    y_true = np.concatenate(label_list)
    y_output = np.concatenate(output_list)
    
    rmse = np.sqrt(mean_squared_error(y_true, y_output))
    pcc = pearsonr(y_true, y_output)[0]
    spcc = spearmanr(y_true, y_output)[0]
    
    avg_loss = epoch_loss / total_samples
    
    print(f'Val epoch_loss: {avg_loss:.4f} | RMSE: {rmse:.4f} | PCC: {pcc:.4f} | SPCC: {spcc:.4f}')

    return {
        'rmse': rmse,
        'pcc': pcc,
        'spcc': spcc,
    }


model = ContrastGNN(
    num_layer=5,
    emb_dim=300,
    out_dim=256,
    atom_feat_dim=170,
    bond_feat_dim=14,
    drop_ratio=0,
    attn_drop_ratio=0,
    model_head=6,
    backbone="gps"
)


# model.load_state_dict(torch.load("./checkpoint/zinc-gps_best.pt")['wrapper'], strict=False)
model.load_state_dict(torch.load("expt_dti/20260826_062712/DTI_fold_early_stop_0")["model_state_dict"], strict=False)
for module in model.fusion_out_linear.modules():
    if hasattr(module, 'reset_parameters'):
        module.reset_parameters()
# set_params(model.gnn)

device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)


best_score = -10000
val_metric = "pcc"
early_stop = 0

for epoch in range(30):
    print(f"--- Epoch {epoch+1} ---")
    train_one_epoch(model, train_loader, optimizer, device)
    score = validate(model, val_loader, device)
    
    if val_metric in ["pcc", "spcc"]:
        if score[val_metric] > best_score:
            best_score = score[val_metric]
            torch.save(model.state_dict(), f"{expt_folder}/drp_checkpoint")
            best_model = deepcopy(model)
            early_stop = 0
        else:
            early_stop = early_stop + 1
            if early_stop == 5:
                break

validate(best_model, test_loader, device)
