import os
import random
import pickle
import numpy as np
import pandas as pd
import scipy.sparse as sps
from rdkit import Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol, MakeScaffoldGeneric, MurckoScaffoldSmiles
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.data import Batch
from data import *

from transformers import AutoModel, AutoTokenizer
import lmdb


class MoleculeContrastDataset(Dataset):
    def __init__(self, data_dir, data_name):
        data = pd.read_csv(os.path.join(data_dir, data_name + '.csv'))
        self.smiles = data['smiles'].to_list()
        self.smiles_dict = dict(zip(data['smiles'], data['chunk_id']))
        
        self.coord_dicts = []
        
        for coord_id in range(5):
            chunk_dict = {}
            for chunk_id in range(4):
                chunk_dict[chunk_id] = np.load(
                    f"data/chembl_mol_unimol1_{chunk_id}_seed_42_coord_{coord_id}.npy",
                    allow_pickle=True
                ).item()
            self.coord_dicts.append(chunk_dict)

        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        chunk_id = self.smiles_dict[smi]
        coord_id = np.random.choice(range(5))
        confs = self.coord_dicts[coord_id][chunk_id][smi]
        confs = torch.tensor(confs, dtype=torch.float)
        mol = Chem.MolFromSmiles(smi)
        graph = mol_to_graph_data_obj_super_rich(mol)
        graph = self.transform(graph)
        return graph, confs


def log_nested_dict(raw_dict, logger, indent=0):
    for key, value in raw_dict.items():
        if isinstance(value, dict):
            logger.info(f'{"    " * indent}{key}:')
            log_nested_dict(value, logger, indent + 1)
        else:
            logger.info(f'{"    " * indent}{key}: {value}')


def get_scaffold(smi, generic=False, return_mol=False):
    if generic:
        assert not return_mol

    if isinstance(smi, str):
        mol = Chem.MolFromSmiles(smi)
    else:
        mol = smi

    if generic:
        if return_mol:
            return MakeScaffoldGeneric(mol)
        else:
            return Chem.MolToSmiles(MakeScaffoldGeneric(mol))
    else:
        if return_mol:
            return GetScaffoldForMol(mol)
        else:
            return Chem.MolToSmiles(GetScaffoldForMol(mol))


class TrimodalDataset(Dataset):
    def __init__(self, data_dir, data_name):
        data = pd.read_csv(os.path.join(data_dir, data_name + '.csv'))
        self.smiles = data['smiles'].to_list()
        self.smiles_dict = dict(zip(data['smiles'], data['chunk_id']))
        
        self.coord_dicts = []
        
        for coord_id in range(5):
            chunk_dict = {}
            for chunk_id in range(4):
                chunk_dict[chunk_id] = np.load(
                    f"data/chembl_mol_unimol1_{chunk_id}_seed_42_coord_{coord_id}.npy",
                    allow_pickle=True
                ).item()
            self.coord_dicts.append(chunk_dict)

        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        
        model = AutoModel.from_pretrained("ibm/MoLFormer-XL-both-10pct", deterministic_eval=True, trust_remote_code=True, revision="compat-v4")
        tokenizer = AutoTokenizer.from_pretrained("ibm/MoLFormer-XL-both-10pct", trust_remote_code=True, revision="compat-v4")
        model.to('cuda')
        self.seq_model = model
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        chunk_id = self.smiles_dict[smi]
        coord_id = np.random.choice(range(5))
        confs = self.coord_dicts[coord_id][chunk_id][smi]
        confs = torch.tensor(confs, dtype=torch.float)
        mol = Chem.MolFromSmiles(smi)
        graph = mol_to_graph_data_obj_super_rich(mol)
        graph = self.transform(graph)
        with torch.no_grad():
            inputs = self.tokenizer([smi], padding=True, return_tensors="pt")
            inputs = {k: v.to('cuda') for k, v in inputs.items()}
            outputs = self.seq_model(**inputs)
            sfeats = outputs.pooler_output.squeeze()
        return graph, confs, sfeats


# class DTITrimodalDataset(Dataset):
#     def __init__(self, data_dir, data_name, dti_data_file):
#         data = pd.read_csv(os.path.join(data_dir, data_name + '.csv'))
#         self.smiles = data['smiles'].to_list()
#         self.smiles_dict = dict(zip(data['smiles'], data['chunk_id']))
        
#         self.coord_dicts = []
        
#         for coord_id in range(5):
#             chunk_dict = {}
#             for chunk_id in range(4):
#                 chunk_dict[chunk_id] = np.load(
#                     f"data/chembl_mol_unimol1_{chunk_id}_seed_42_coord_{coord_id}.npy",
#                     allow_pickle=True
#                 ).item()
#             self.coord_dicts.append(chunk_dict)

#         self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        
#         dti_data = pd.read_csv(dti_data_file)
#         prot_mapping = pd.read_csv("data/chembl_prot.csv")
#         prot_uniprot_dict = dict(zip(prot_mapping["seq"], prot_mapping["uniprot_id"]))
#         dti_data["uniprot_id"] = dti_data["Protein"].map(prot_uniprot_dict)
        
#         dti_uniprot_ids = dti_data["uniprot_id"].to_list()
#         dti_smiles = dti_data["SMILES"].to_list()
#         dti_labels = dti_data["Y"].to_numpy()
        
#         self.dti_uniprot_ids = dti_uniprot_ids
#         self.dti_smiles = dti_smiles
#         self.dti_labels = dti_labels
        
#         chembl_molf_embs_dict = np.load("data/chembl_molf_embs.npy", allow_pickle=True).item()
#         self.chembl_molf_embs_dict = chembl_molf_embs_dict
    
#     def get_prot_emb(self, key):
#         lmdb_path = "/home/dataset-local/data_local/quantianhong/GDR/data/chembl_target_ESMC"
#         read_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
#         with read_env.begin(write=False) as txn:
#             emb = txn.get(key.encode())
#             emb = pickle.loads(emb)
#         emb = emb.mean(axis=0)
#         return emb

#     def __len__(self):
#         return len(self.dti_smiles)

#     def __getitem__(self, idx):
#         smi = self.dti_smiles[idx]
        
#         chunk_id = self.smiles_dict[smi]
#         coord_id = np.random.choice(range(5))
#         confs = self.coord_dicts[coord_id][chunk_id][smi]
#         confs = torch.tensor(confs, dtype=torch.float)
        
#         mol = Chem.MolFromSmiles(smi)
#         graph = mol_to_graph_data_obj_super_rich(mol)
#         graph = self.transform(graph)
        
#         sfeats = self.chembl_molf_embs_dict[smi]
#         sfeats = torch.tensor(sfeats, dtype=torch.float)
        
#         uid = self.dti_uniprot_ids[idx]
#         prot = self.get_prot_emb(uid)
#         prot = torch.tensor(prot, dtype=torch.float)
        
#         label = torch.Tensor([self.dti_labels[idx]])
        
#         return prot, graph, confs, sfeats, label


class DTITrimodalDataset(Dataset):
    def __init__(self, data_dir, data_name, dti_data_file):
        data = pd.read_csv(os.path.join(data_dir, data_name + '.csv'))
        self.smiles = data['smiles'].to_list()
        self.smiles_dict = dict(zip(data['smiles'], data['chunk_id']))
        
        self.coord_dicts = []
        
        for coord_id in range(5):
            chunk_dict = {}
            for chunk_id in range(4):
                chunk_dict[chunk_id] = np.load(
                    f"data/chembl_mol_unimol1_{chunk_id}_seed_42_coord_{coord_id}.npy",
                    allow_pickle=True
                ).item()
            self.coord_dicts.append(chunk_dict)
        
        dti_data = pd.read_csv(dti_data_file)
        prot_mapping = pd.read_csv("data/chembl_prot.csv")
        prot_uniprot_dict = dict(zip(prot_mapping["seq"], prot_mapping["uniprot_id"]))
        dti_data["uniprot_id"] = dti_data["Protein"].map(prot_uniprot_dict)
        
        dti_uniprot_ids = dti_data["uniprot_id"].to_list()
        dti_smiles = dti_data["SMILES"].to_list()
        dti_labels = dti_data["Y"].to_numpy()
        
        self.dti_uniprot_ids = dti_uniprot_ids
        self.dti_smiles = dti_smiles
        self.dti_labels = dti_labels
        
        chembl_molf_embs_dict = np.load("data/chembl_molf_embs.npy", allow_pickle=True).item()
        self.chembl_molf_embs_dict = chembl_molf_embs_dict
        
        chembl_mol_graphs_dict = torch.load("data/chembl_mol_graphs.pt")
        self.chembl_mol_graphs_dict = chembl_mol_graphs_dict
    
    def get_prot_emb(self, key):
        lmdb_path = "/home/dataset-local/data_local/quantianhong/GDR/data/chembl_target_ESMC"
        read_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with read_env.begin(write=False) as txn:
            emb = txn.get(key.encode())
            emb = pickle.loads(emb)
        emb = emb.mean(axis=0)
        return emb

    def __len__(self):
        return len(self.dti_smiles)

    def __getitem__(self, idx):
        smi = self.dti_smiles[idx]
        
        chunk_id = self.smiles_dict[smi]
        coord_id = np.random.choice(range(5))
        confs = self.coord_dicts[coord_id][chunk_id][smi]
        confs = torch.tensor(confs, dtype=torch.float)
        
        graph = self.chembl_mol_graphs_dict[smi]
        
        sfeats = self.chembl_molf_embs_dict[smi]
        sfeats = torch.tensor(sfeats, dtype=torch.float)
        
        uid = self.dti_uniprot_ids[idx]
        prot = self.get_prot_emb(uid)
        prot = torch.tensor(prot, dtype=torch.float)
        
        label = torch.Tensor([self.dti_labels[idx]])
        
        return prot, graph, confs, sfeats, label


class MultiViewDTITrimodalDataset(Dataset):
    def __init__(self, data_dir, data_name, dti_data_file):
        data = pd.read_csv(os.path.join(data_dir, data_name + '.csv'))
        self.smiles = data['smiles'].to_list()
        self.smiles_dict = dict(zip(data['smiles'], data['chunk_id']))
        
        self.coord_dicts = []
        
        for coord_id in range(5):
            chunk_dict = {}
            for chunk_id in range(4):
                chunk_dict[chunk_id] = np.load(
                    f"data/chembl_mol_unimol1_{chunk_id}_seed_42_coord_{coord_id}.npy",
                    allow_pickle=True
                ).item()
            self.coord_dicts.append(chunk_dict)
        
        dti_data = pd.read_csv(dti_data_file)
        prot_mapping = pd.read_csv("data/chembl_prot.csv")
        prot_uniprot_dict = dict(zip(prot_mapping["seq"], prot_mapping["uniprot_id"]))
        dti_data["uniprot_id"] = dti_data["Protein"].map(prot_uniprot_dict)
        
        dti_uniprot_ids = dti_data["uniprot_id"].to_list()
        dti_smiles = dti_data["SMILES"].to_list()
        dti_labels = dti_data["Y"].to_numpy()
        
        self.dti_uniprot_ids = dti_uniprot_ids
        self.dti_smiles = dti_smiles
        self.dti_labels = dti_labels
        
        chembl_molf_embs_dict = np.load("data/chembl_molf_embs.npy", allow_pickle=True).item()
        self.chembl_molf_embs_dict = chembl_molf_embs_dict
        
        chembl_mol_graphs_dict = torch.load("data/chembl_mol_graphs.pt")
        self.chembl_mol_graphs_dict = chembl_mol_graphs_dict
        
        chembl_molf_scaffold_embs_dict = np.load("/home/dataset-assist-0/datafile/quantianhong/MDR/data/chembl/chembl_molf_scaffold_embs.npy", allow_pickle=True).item()
        self.chembl_molf_scaffold_embs_dict = chembl_molf_scaffold_embs_dict
        
        self.chembl_molf_isostere_embs_dict = {}
        for i in range(4):
            with open(f"/home/dataset-assist-0/datafile/quantianhong/MDR/data/chembl/chembl_molf_isostere_embs_{i}.pkl", 'rb') as f:
                chembl_molf_isostere_embs = pickle.load(f)
            self.chembl_molf_isostere_embs_dict.update(chembl_molf_isostere_embs)
        assert len(self.chembl_molf_isostere_embs_dict) == len(self.chembl_molf_scaffold_embs_dict)
    
    def get_prot_emb(self, key):
        lmdb_path = "/home/dataset-local/data_local/quantianhong/GDR/data/chembl_target_ESMC"
        read_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with read_env.begin(write=False) as txn:
            emb = txn.get(key.encode())
            emb = pickle.loads(emb)
        emb = emb.mean(axis=0)
        return emb

    def __len__(self):
        return len(self.dti_smiles)

    def __getitem__(self, idx):
        smi = self.dti_smiles[idx]
        
        chunk_id = self.smiles_dict[smi]
        coord_id = np.random.choice(range(5))
        confs = self.coord_dicts[coord_id][chunk_id][smi]
        confs = torch.tensor(confs, dtype=torch.float)
        
        graph = self.chembl_mol_graphs_dict[smi]
        
        feat_type = random.choice(['normal', 'scaffold', 'isostere'])
        if feat_type == 'normal':
            sfeats = self.chembl_molf_embs_dict[smi]
        elif feat_type == 'scaffold':
            sfeats = self.chembl_molf_scaffold_embs_dict[smi]
        elif feat_type == 'isostere':
            isostere_dict = self.chembl_molf_isostere_embs_dict[smi]
            sfeats = random.choice(list(isostere_dict.values()))
        sfeats = torch.tensor(sfeats, dtype=torch.float)
        
        uid = self.dti_uniprot_ids[idx]
        prot = self.get_prot_emb(uid)
        prot = torch.tensor(prot, dtype=torch.float)
        
        label = torch.Tensor([self.dti_labels[idx]])
        
        return prot, graph, confs, sfeats, label


class MultiViewDTITrimodalDrugProtDataset(Dataset):
    def __init__(self, data_dir, data_name, dti_data_file, multiview=True):
        data = pd.read_csv(os.path.join(data_dir, data_name + '.csv'))
        self.smiles = data['smiles'].to_list()
        self.smiles_dict = dict(zip(data['smiles'], data['chunk_id']))
        
        self.coord_dicts = []
        
        for coord_id in range(5):
            chunk_dict = {}
            for chunk_id in range(4):
                chunk_dict[chunk_id] = np.load(
                    f"data/chembl_mol_unimol1_{chunk_id}_seed_42_coord_{coord_id}.npy",
                    allow_pickle=True
                ).item()
            self.coord_dicts.append(chunk_dict)
        
        dti_data = pd.read_csv(dti_data_file)
        prot_mapping = pd.read_csv("data/chembl_prot.csv")
        prot_uniprot_dict = dict(zip(prot_mapping["seq"], prot_mapping["uniprot_id"]))
        dti_data["uniprot_id"] = dti_data["Protein"].map(prot_uniprot_dict)
        
        dti_uniprot_ids = dti_data["uniprot_id"].to_list()
        dti_smiles = dti_data["SMILES"].to_list()
        dti_labels = dti_data["Y"].to_numpy()
        
        self.dti_uniprot_ids = dti_uniprot_ids
        self.dti_smiles = dti_smiles
        self.dti_labels = dti_labels
        
        chembl_molf_embs_dict = np.load("data/chembl_molf_embs.npy", allow_pickle=True).item()
        self.chembl_molf_embs_dict = chembl_molf_embs_dict
        
        chembl_mol_graphs_dict = torch.load("data/chembl_mol_graphs.pt")
        self.chembl_mol_graphs_dict = chembl_mol_graphs_dict
        
        chembl_molf_scaffold_embs_dict = np.load("/home/dataset-assist-0/datafile/quantianhong/MDR/data/chembl/chembl_molf_scaffold_embs.npy", allow_pickle=True).item()
        self.chembl_molf_scaffold_embs_dict = chembl_molf_scaffold_embs_dict
        
        self.chembl_molf_isostere_embs_dict = {}
        for i in range(4):
            with open(f"/home/dataset-assist-0/datafile/quantianhong/MDR/data/chembl/chembl_molf_isostere_embs_{i}.pkl", 'rb') as f:
                chembl_molf_isostere_embs = pickle.load(f)
            self.chembl_molf_isostere_embs_dict.update(chembl_molf_isostere_embs)
        assert len(self.chembl_molf_isostere_embs_dict) == len(self.chembl_molf_scaffold_embs_dict)
        
        self.multiview = multiview
    
    def get_prot_emb(self, key):
        lmdb_path = "/home/dataset-local/data_local/quantianhong/GDR/data/chembl_target_ESMC"
        read_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with read_env.begin(write=False) as txn:
            emb = txn.get(key.encode())
            emb = pickle.loads(emb)
        # emb = emb.mean(axis=0)
        return emb

    def __len__(self):
        return len(self.dti_smiles)

    def __getitem__(self, idx):
        smi = self.dti_smiles[idx]
        
        chunk_id = self.smiles_dict[smi]
        coord_id = np.random.choice(range(5))
        confs = self.coord_dicts[coord_id][chunk_id][smi]
        confs = torch.tensor(confs, dtype=torch.float)
        
        graph = self.chembl_mol_graphs_dict[smi]
        
        if self.multiview:
            feat_type = random.choice(['normal', 'scaffold', 'isostere'])
            if feat_type == 'normal':
                sfeats = self.chembl_molf_embs_dict[smi]
            elif feat_type == 'scaffold':
                sfeats = self.chembl_molf_scaffold_embs_dict[smi]
            elif feat_type == 'isostere':
                isostere_dict = self.chembl_molf_isostere_embs_dict[smi]
                sfeats = random.choice(list(isostere_dict.values()))
        else:
            sfeats = self.chembl_molf_embs_dict[smi]
        sfeats = torch.tensor(sfeats, dtype=torch.float)
        
        uid = self.dti_uniprot_ids[idx]
        prot = self.get_prot_emb(uid)
        prot = torch.tensor(prot, dtype=torch.float)
        
        label = torch.Tensor([self.dti_labels[idx]])
        
        return prot, graph, confs, sfeats, label


def custom_collate_fn(batch):
    # 将 batch 中的不同元素解压
    prots, graphs, confs, sfeats, labels = zip(*batch)
    
    # 1. 蛋白质序列补零 (Padding) 
    # 输入的每个 prot 的尺寸为 [seq_len, prot_dim]
    # pad_sequence 后得到 [batch_size, max_seq_len, prot_dim]
    prots_padded = pad_sequence(prots, batch_first=True, padding_value=0.0)
    
    # 2. PyG 的图组成 Batch
    graphs_batched = Batch.from_data_list(graphs)
    
    # 3. 其他向量的堆叠
    confs = torch.stack(confs, dim=0)
    sfeats = torch.stack(sfeats, dim=0)
    labels = torch.stack(labels, dim=0)
    
    return prots_padded, graphs_batched, confs, sfeats, labels


class RectifiedCosineBCELoss(nn.Module):
    def __init__(self):
        super(RectifiedCosineBCELoss, self).__init__()
        self.loss_fn = nn.BCELoss()
        self.cosine = nn.CosineSimilarity()
    
    def forward(self, x, y, t):
        x = F.relu(x)
        y = F.relu(y)
        distance = self.cosine(x, y)
        loss = self.loss_fn(distance, t.squeeze())
        return loss


def set_random_seed(seed=10):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


class EarlyStopping():
    """
    Parameters
    ----------
    mode : str
        * 'higher': Higher metric suggests a better model
        * 'lower': Lower metric suggests a better model
        If ``metric`` is not None, then mode will be determined
        automatically from that.
    patience : int
        The early stopping will happen if we do not observe performance
        improvement for ``patience`` consecutive epochs.
    filename : str or None
        Filename for storing the model checkpoint. If not specified,
        we will automatically generate a file starting with ``early_stop``
        based on the current time.
    metric : str or None
        A metric name that can be used to identify if a higher value is
        better, or vice versa. Default to None. Valid options include:
        ``'r2'``, ``'mae'``, ``'rmse'``, ``'roc_auc_score'``.
    """

    def __init__(self, mode='higher', patience=None, metric=None, n_fold=None, folder=None, logger=None):

        filepath = os.path.join(folder, '{}_fold_early_stop'.format(n_fold))

        if metric is not None:
            assert metric in ['r2', 'mae', 'rmse', 'roc_auc_score', 'pr_auc_score', 'mse'], \
                "Expect metric to be 'r2' or 'mae' or " \
                "'rmse' or 'roc_auc_score' or 'mse', got {}".format(metric)
            if metric in ['r2', 'roc_auc_score', 'pr_auc_score']:
                print('For metric {}, the higher the better'.format(metric))
                mode = 'higher'
            if metric in ['mae', 'rmse', 'mse']:
                print('For metric {}, the lower the better'.format(metric))
                mode = 'lower'

        assert mode in ['higher', 'lower']
        self.mode = mode
        if self.mode == 'higher':
            self._check = self._check_higher
        else:
            self._check = self._check_lower

        self.patience = patience
        self.counter = 0
        self.filepath = filepath
        self.best_score = None
        self.early_stop = False
        self.logger = logger

    def _check_higher(self, score, prev_best_score):
        """Check if the new score is higher than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is higher than the previous best score.
        """
        return score > prev_best_score

    def _check_lower(self, score, prev_best_score):
        """Check if the new score is lower than the previous best score.
        Parameters
        ----------
        score : float
            New score.
        prev_best_score : float
            Previous best score.
        Returns
        -------
        bool
            Whether the new score is lower than the previous best score.
        """
        return score < prev_best_score

    def step(self, score, model, current_epoch, optimizer):
        """Update based on a new score.
        The new score is typically model performance on the validation set
        for a new epoch.
        Parameters
        ----------
        score : float
            New score.
        model : nn.Module
            Model instance.
        Returns
        -------
        bool
            Whether an early stop should be performed.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, current_epoch, optimizer)
        elif self._check(score, self.best_score):
            self.best_score = score
            self.save_checkpoint(model, current_epoch, optimizer)
            self.counter = 0
        else:
            self.counter += 1
            self.logger.info(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

    def save_checkpoint(self, model, current_epoch, optimizer):
        '''Saves model when the metric on the validation set gets improved.
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        torch.save({
            'epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            
        }, f"{self.filepath}_{current_epoch}")
        
        self.saved_epoch = current_epoch

    def load_checkpoint(self, model, optimizer):
        '''Load the latest checkpoint
        Parameters
        ----------
        model : nn.Module
            Model instance.
        '''
        checkpoint = torch.load(f"{self.filepath}_{self.saved_epoch}")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model


class GDSCDataset(Dataset):
    def __init__(self, version='gdsc2', platform='rma', geneset_file='data/hgnc_csv/depmap_gdsc_tcga_gene.csv'):
        drug_df = pd.read_csv("data/gdsc_csv/gdsc_smiles.csv")
        drug_df = drug_df.dropna(subset=["smiles"])
        
        if version == "gdsc1":
            gdsc_df = pd.read_excel("data/gdsc_csv/GDSC1_fitted_dose_response_27Oct23.xlsx")
            drug_df = drug_df[drug_df[" Datasets"].isin(["GDSC1"])]
        elif version == "gdsc2":
            gdsc_df = pd.read_excel("data/gdsc_csv/GDSC2_fitted_dose_response_27Oct23.xlsx")
            drug_df = drug_df[drug_df[" Datasets"].isin(["GDSC2"])]
        
        drugid_smi_dict = dict(zip(drug_df["Drug Id"], drug_df["smiles"]))
        gdsc_df["smiles"] = gdsc_df["DRUG_ID"].map(drugid_smi_dict)
        gdsc_df = gdsc_df.dropna(subset=["smiles"])
        
        if platform == "rma":
            gex_df = pd.read_csv("data/gdsc_csv/cell_line_gex_rma.csv", index_col=0)
            cosmic_ids = [row.split(".")[1] for row in gex_df.index.to_list()]
            # gdsc_df["COSMIC_ID"] = gdsc_df["COSMIC_ID"].astype('str')
            gdsc_df = gdsc_df[gdsc_df["COSMIC_ID"].astype('str').isin(cosmic_ids)]
            self.smiles = gdsc_df["smiles"].to_list()
            self.cells = ["DATA." + str(idx) for idx in gdsc_df["COSMIC_ID"].to_list()]
            self.labels = gdsc_df["LN_IC50"].to_list()
        elif platform == "tpm":
            gex_df = pd.read_csv("data/depmap_csv/depmap_gex_tpm.csv", index_col=0)
            depmap_info = pd.read_csv("data/depmap_csv/Model.csv", index_col=0)
            depmap_sanger_dict = depmap_info.dropna(subset=["SangerModelID"])["SangerModelID"].to_dict()
            gex_df.index = gex_df.index.map(depmap_sanger_dict)
            gex_df = gex_df[~gex_df.index.isna()]
            gdsc_df = gdsc_df[gdsc_df["SANGER_MODEL_ID"].isin(gex_df.index)]
            self.smiles = gdsc_df["smiles"].to_list()
            self.cells = gdsc_df["SANGER_MODEL_ID"].to_list()
            self.labels = gdsc_df["LN_IC50"].to_list()
        
        geneset_list = pd.read_csv(geneset_file)["gene"].to_list()
        gex_df = gex_df[geneset_list]
        gex_df = gex_df.rank(axis=1, pct=True)
        
        # self.tokenizer = MWE("data/vocab_csv/subword.csv")
        self.gex = gex_df
        
        # gdsc_df.to_csv(f"data/gdsc_csv/{version}_{platform}.csv", index=False)
        
        gdsc_molf_embs_dict = np.load("data/gdsc_csv/gdsc_molf_embs.npy", allow_pickle=True).item()
        gdsc_unimol_embs_dict = np.load("data/gdsc_csv/gdsc_unimol_embs.npy", allow_pickle=True).item()
        self.gdsc_molf_embs_dict = gdsc_molf_embs_dict
        self.gdsc_unimol_embs_dict = gdsc_unimol_embs_dict
        
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        
        self.mol_data_dict = {}
        for i, smi in enumerate(list(set(gdsc_df["smiles"]))):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                graph = mol_to_graph_data_obj_super_rich(mol)
                graph = self.transform(graph)
                self.mol_data_dict[smi] = graph
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, index):
        smi = self.smiles[index]
        
        graph = self.mol_data_dict[smi]
        
        confs = self.gdsc_unimol_embs_dict[smi]
        confs = torch.tensor(confs, dtype=torch.float)
        
        sfeats = self.gdsc_molf_embs_dict[smi]
        sfeats = torch.tensor(sfeats, dtype=torch.float)
        
        cell = self.cells[index]
        rna = self.gex.loc[cell].to_numpy()
        rna = torch.tensor(rna, dtype=torch.float)
        
        label = np.array([self.labels[index]])
        label = torch.tensor(label, dtype=torch.float)
        
        return graph, confs, sfeats, rna, label


class GDSCMissingDataset(Dataset):
    def __init__(self, version='gdsc2', platform='rma', geneset_file='data/hgnc_csv/depmap_gdsc_tcga_gene.csv'):
        drug_df = pd.read_csv("data/gdsc_csv/gdsc_smiles.csv")
        drug_df = drug_df.dropna(subset=["smiles"])
        
        if version == "gdsc1":
            gdsc_df = pd.read_excel("data/gdsc_csv/GDSC1_fitted_dose_response_27Oct23.xlsx")
            drug_df = drug_df[drug_df[" Datasets"].isin(["GDSC1"])]
        elif version == "gdsc2":
            gdsc_df = pd.read_excel("data/gdsc_csv/GDSC2_fitted_dose_response_27Oct23.xlsx")
            drug_df = drug_df[drug_df[" Datasets"].isin(["GDSC2"])]
        
        drugid_smi_dict = dict(zip(drug_df["Drug Id"], drug_df["smiles"]))
        gdsc_df["smiles"] = gdsc_df["DRUG_ID"].map(drugid_smi_dict)
        gdsc_df = gdsc_df.dropna(subset=["smiles"])
        
        if platform == "rma":
            gex_df = pd.read_csv("data/gdsc_csv/cell_line_gex_rma.csv", index_col=0)
            cosmic_ids = [row.split(".")[1] for row in gex_df.index.to_list()]
            # gdsc_df["COSMIC_ID"] = gdsc_df["COSMIC_ID"].astype('str')
            gdsc_df = gdsc_df[gdsc_df["COSMIC_ID"].astype('str').isin(cosmic_ids)]
            
            drug_list = gdsc_df["smiles"].value_counts().index
            cell_list = gdsc_df["COSMIC_ID"].value_counts().index
            
            all_df = pd.DataFrame()
            dup_drug = []
            for i in drug_list:
                dup_drug.extend([i] * len(cell_list))
            all_df["smiles"] = dup_drug
            
            dup_cell = []
            for i in range(len(drug_list)):
                dup_cell.extend(cell_list)
            all_df["COSMIC_ID"] = dup_cell
            
            cell_type_dict = dict(zip(gdsc_df["COSMIC_ID"], gdsc_df["TCGA_DESC"]))
            
            all_df["ID"] = all_df["smiles"].astype(str).str.cat(all_df["COSMIC_ID"].astype(str), sep='_')
            print(all_df)
            gdsc_df["ID"] = gdsc_df["smiles"].astype(str).str.cat(gdsc_df["COSMIC_ID"].astype(str), sep='_')
            print(gdsc_df)
            print(gdsc_df.drop_duplicates(subset=["ID"]))
            missing_df = all_df[~all_df["ID"].isin(gdsc_df["ID"])]
            missing_df = missing_df[missing_df.groupby("smiles")["smiles"].transform("count") >= int(len(cell_list)/2)]
            print(missing_df)
            missing_df["TCGA_DESC"] = missing_df["COSMIC_ID"].map(cell_type_dict)
            self.cell_types = missing_df["TCGA_DESC"].to_list()
            
            self.smiles = missing_df["smiles"].to_list()
            self.cells = ["DATA." + str(idx) for idx in missing_df["COSMIC_ID"].to_list()]
            # self.labels = gdsc_df["LN_IC50"].to_list()
        elif platform == "tpm":
            gex_df = pd.read_csv("data/depmap_csv/depmap_gex_tpm.csv", index_col=0)
            depmap_info = pd.read_csv("data/depmap_csv/Model.csv", index_col=0)
            depmap_sanger_dict = depmap_info.dropna(subset=["SangerModelID"])["SangerModelID"].to_dict()
            gex_df.index = gex_df.index.map(depmap_sanger_dict)
            gex_df = gex_df[~gex_df.index.isna()]
            gdsc_df = gdsc_df[gdsc_df["SANGER_MODEL_ID"].isin(gex_df.index)]
            self.smiles = gdsc_df["smiles"].to_list()
            self.cells = gdsc_df["SANGER_MODEL_ID"].to_list()
            self.labels = gdsc_df["LN_IC50"].to_list()
        
        geneset_list = pd.read_csv(geneset_file)["gene"].to_list()
        gex_df = gex_df[geneset_list]
        gex_df = gex_df.rank(axis=1, pct=True)
        
        # self.tokenizer = MWE("data/vocab_csv/subword.csv")
        self.gex = gex_df
        
        # gdsc_df.to_csv(f"data/gdsc_csv/{version}_{platform}.csv", index=False)
        
        gdsc_molf_embs_dict = np.load("data/gdsc_csv/gdsc_molf_embs.npy", allow_pickle=True).item()
        gdsc_unimol_embs_dict = np.load("data/gdsc_csv/gdsc_unimol_embs.npy", allow_pickle=True).item()
        self.gdsc_molf_embs_dict = gdsc_molf_embs_dict
        self.gdsc_unimol_embs_dict = gdsc_unimol_embs_dict
        
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        
        self.mol_data_dict = {}
        for i, smi in enumerate(list(set(gdsc_df["smiles"]))):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                graph = mol_to_graph_data_obj_super_rich(mol)
                graph = self.transform(graph)
                self.mol_data_dict[smi] = graph
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, index):
        smi = self.smiles[index]
        
        graph = self.mol_data_dict[smi]
        
        confs = self.gdsc_unimol_embs_dict[smi]
        confs = torch.tensor(confs, dtype=torch.float)
        
        sfeats = self.gdsc_molf_embs_dict[smi]
        sfeats = torch.tensor(sfeats, dtype=torch.float)
        
        cell = self.cells[index]
        rna = self.gex.loc[cell].to_numpy()
        rna = torch.tensor(rna, dtype=torch.float)
        
        # label = np.array([self.labels[index]])
        # label = torch.tensor(label, dtype=torch.float)
        
        return graph, confs, sfeats, rna


class GDSCDatasetDesc(Dataset):
    def __init__(self, version='gdsc2', platform='rma', geneset_file='data/hgnc_csv/depmap_gdsc_tcga_gene.csv'):
        drug_df = pd.read_csv("data/gdsc_csv/gdsc_smiles.csv")
        drug_df = drug_df.dropna(subset=["smiles"])
        
        if version == "gdsc1":
            gdsc_df = pd.read_excel("data/gdsc_csv/GDSC1_fitted_dose_response_27Oct23.xlsx")
            drug_df = drug_df[drug_df[" Datasets"].isin(["GDSC1"])]
        elif version == "gdsc2":
            gdsc_df = pd.read_excel("data/gdsc_csv/GDSC2_fitted_dose_response_27Oct23.xlsx")
            drug_df = drug_df[drug_df[" Datasets"].isin(["GDSC2"])]
        
        drugid_smi_dict = dict(zip(drug_df["Drug Id"], drug_df["smiles"]))
        gdsc_df["smiles"] = gdsc_df["DRUG_ID"].map(drugid_smi_dict)
        gdsc_df = gdsc_df.dropna(subset=["smiles"])
        
        if platform == "rma":
            gex_df = pd.read_csv("data/gdsc_csv/cell_line_gex_rma.csv", index_col=0)
            cosmic_ids = [row.split(".")[1] for row in gex_df.index.to_list()]
            # gdsc_df["COSMIC_ID"] = gdsc_df["COSMIC_ID"].astype('str')
            gdsc_df = gdsc_df[gdsc_df["COSMIC_ID"].astype('str').isin(cosmic_ids)]
            self.smiles = gdsc_df["smiles"].to_list()
            self.cells = ["DATA." + str(idx) for idx in gdsc_df["COSMIC_ID"].to_list()]
            self.labels = gdsc_df["LN_IC50"].to_list()
            self.cell_types = gdsc_df["TCGA_DESC"].to_list()
        elif platform == "tpm":
            gex_df = pd.read_csv("data/depmap_csv/depmap_gex_tpm.csv", index_col=0)
            depmap_info = pd.read_csv("data/depmap_csv/Model.csv", index_col=0)
            depmap_sanger_dict = depmap_info.dropna(subset=["SangerModelID"])["SangerModelID"].to_dict()
            gex_df.index = gex_df.index.map(depmap_sanger_dict)
            gex_df = gex_df[~gex_df.index.isna()]
            gdsc_df = gdsc_df[gdsc_df["SANGER_MODEL_ID"].isin(gex_df.index)]
            self.smiles = gdsc_df["smiles"].to_list()
            self.cells = gdsc_df["SANGER_MODEL_ID"].to_list()
            self.labels = gdsc_df["LN_IC50"].to_list()
        
        geneset_list = pd.read_csv(geneset_file)["gene"].to_list()
        gex_df = gex_df[geneset_list]
        gex_df = gex_df.rank(axis=1, pct=True)
        
        # self.tokenizer = MWE("data/vocab_csv/subword.csv")
        self.gex = gex_df
        
        # gdsc_df.to_csv(f"data/gdsc_csv/{version}_{platform}.csv", index=False)
        
        gdsc_molf_embs_dict = np.load("data/gdsc_csv/gdsc_molf_embs.npy", allow_pickle=True).item()
        gdsc_unimol_embs_dict = np.load("data/gdsc_csv/gdsc_unimol_embs.npy", allow_pickle=True).item()
        self.gdsc_molf_embs_dict = gdsc_molf_embs_dict
        self.gdsc_unimol_embs_dict = gdsc_unimol_embs_dict
        
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        
        self.mol_data_dict = {}
        for i, smi in enumerate(list(set(gdsc_df["smiles"]))):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                graph = mol_to_graph_data_obj_super_rich(mol)
                graph = self.transform(graph)
                self.mol_data_dict[smi] = graph
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, index):
        smi = self.smiles[index]
        
        graph = self.mol_data_dict[smi]
        
        confs = self.gdsc_unimol_embs_dict[smi]
        confs = torch.tensor(confs, dtype=torch.float)
        
        sfeats = self.gdsc_molf_embs_dict[smi]
        sfeats = torch.tensor(sfeats, dtype=torch.float)
        
        cell = self.cells[index]
        rna = self.gex.loc[cell].to_numpy()
        rna = torch.tensor(rna, dtype=torch.float)
        
        label = np.array([self.labels[index]])
        label = torch.tensor(label, dtype=torch.float)
        
        return graph, confs, sfeats, rna, label


class GDSCAugmentedDataset(Dataset):
    def __init__(self, version='gdsc2', platform='rma', geneset_file='data/hgnc_csv/depmap_gdsc_tcga_gene.csv'):
        drug_df = pd.read_csv("data/gdsc_csv/gdsc_smiles.csv")
        drug_df = drug_df.dropna(subset=["smiles"])
        
        if version == "gdsc1":
            gdsc_df = pd.read_excel("data/gdsc_csv/GDSC1_fitted_dose_response_27Oct23.xlsx")
            drug_df = drug_df[drug_df[" Datasets"].isin(["GDSC1"])]
        elif version == "gdsc2":
            gdsc_df = pd.read_excel("data/gdsc_csv/GDSC2_fitted_dose_response_27Oct23.xlsx")
            drug_df = drug_df[drug_df[" Datasets"].isin(["GDSC2"])]
        
        drugid_smi_dict = dict(zip(drug_df["Drug Id"], drug_df["smiles"]))
        gdsc_df["smiles"] = gdsc_df["DRUG_ID"].map(drugid_smi_dict)
        gdsc_df = gdsc_df.dropna(subset=["smiles"])
        
        if platform == "rma":
            gex_df = pd.read_csv("data/gdsc_csv/cell_line_gex_rma.csv", index_col=0)
            cosmic_ids = [row.split(".")[1] for row in gex_df.index.to_list()]
            # gdsc_df["COSMIC_ID"] = gdsc_df["COSMIC_ID"].astype('str')
            gdsc_df = gdsc_df[gdsc_df["COSMIC_ID"].astype('str').isin(cosmic_ids)]
            self.smiles = gdsc_df["smiles"].to_list()
            self.cells = ["DATA." + str(idx) for idx in gdsc_df["COSMIC_ID"].to_list()]
            self.labels = gdsc_df["LN_IC50"].to_list()
        elif platform == "tpm":
            gex_df = pd.read_csv("data/depmap_csv/depmap_gex_tpm.csv", index_col=0)
            depmap_info = pd.read_csv("data/depmap_csv/Model.csv", index_col=0)
            depmap_sanger_dict = depmap_info.dropna(subset=["SangerModelID"])["SangerModelID"].to_dict()
            gex_df.index = gex_df.index.map(depmap_sanger_dict)
            gex_df = gex_df[~gex_df.index.isna()]
            gdsc_df = gdsc_df[gdsc_df["SANGER_MODEL_ID"].isin(gex_df.index)]
            self.smiles = gdsc_df["smiles"].to_list()
            self.cells = gdsc_df["SANGER_MODEL_ID"].to_list()
            self.labels = gdsc_df["LN_IC50"].to_list()
        
        geneset_list = pd.read_csv(geneset_file)["gene"].to_list()
        gex_df = gex_df[geneset_list]
        gex_df = gex_df.rank(axis=1, pct=True)
        
        # self.tokenizer = MWE("data/vocab_csv/subword.csv")
        self.gex = gex_df
        
        # gdsc_df.to_csv(f"data/gdsc_csv/{version}_{platform}.csv", index=False)
        
        gdsc_molf_embs_dict = np.load("data/gdsc_csv/gdsc_molf_embs.npy", allow_pickle=True).item()
        gdsc_unimol_embs_dict = np.load("data/gdsc_csv/gdsc_unimol_embs.npy", allow_pickle=True).item()
        self.gdsc_molf_embs_dict = gdsc_molf_embs_dict
        self.gdsc_unimol_embs_dict = gdsc_unimol_embs_dict
        
        self.transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        
        self.mol_data_dict = {}
        for i, smi in enumerate(list(set(gdsc_df["smiles"]))):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                graph = mol_to_graph_data_obj_super_rich(mol)
                graph = self.transform(graph)
                self.mol_data_dict[smi] = graph
        
        self.coord_dicts = []
        
        for coord_id in range(5):
            coord_dict = np.load(f"data/gdsc_csv/gdsc_mol_unimol1_seed_42_coord_{coord_id}.npy", allow_pickle=True).item()
            self.coord_dicts.append(coord_dict)
        
        with open(f"data/gdsc_csv/gdsc_molf_isostere_embs.pkl", 'rb') as f:
            gdsc_molf_isostere_embs = pickle.load(f)
        self.gdsc_molf_isostere_embs_dict = gdsc_molf_isostere_embs
        
        gdsc_molf_scaffold_embs = np.load(f"data/gdsc_csv/gdsc_molf_scaffold_embs.npy", allow_pickle=True).item()
        self.gdsc_molf_scaffold_embs_dict = gdsc_molf_scaffold_embs
        
        self.multiview = True
    
    def __len__(self):
        return len(self.smiles)
    
    def __getitem__(self, index):
        smi = self.smiles[index]
        
        graph = self.mol_data_dict[smi]
        
        if self.multiview:
            coord_id = np.random.choice(range(5))
            confs = self.coord_dicts[coord_id][smi]
            confs = torch.tensor(confs, dtype=torch.float)
            
            feat_type = random.choice(['normal', 'scaffold', 'isostere'])
            if feat_type == 'normal':
                sfeats = self.gdsc_molf_embs_dict[smi]
            elif feat_type == 'scaffold':
                sfeats = self.gdsc_molf_scaffold_embs_dict[smi]
            elif feat_type == 'isostere':
                isostere_dict = self.gdsc_molf_isostere_embs_dict[smi]
                sfeats = random.choice(list(isostere_dict.values()))
            sfeats = torch.tensor(sfeats, dtype=torch.float)
        else:
            confs = self.gdsc_unimol_embs_dict[smi]
            confs = torch.tensor(confs, dtype=torch.float)

            sfeats = self.gdsc_molf_embs_dict[smi]
            sfeats = torch.tensor(sfeats, dtype=torch.float)
        
        cell = self.cells[index]
        rna = self.gex.loc[cell].to_numpy()
        rna = torch.tensor(rna, dtype=torch.float)
        
        label = np.array([self.labels[index]])
        label = torch.tensor(label, dtype=torch.float)
        
        return graph, confs, sfeats, rna, label


class MultiViewDTITrimodalDatasetOriginal(Dataset):
    def __init__(self, data_dir, data_name, dti_data_file):
        data = pd.read_csv(os.path.join(data_dir, data_name + '.csv'))
        self.smiles = data['smiles'].to_list()
        self.smiles_dict = dict(zip(data['smiles'], data['chunk_id']))
        
        self.coord_dicts = []
        
        for coord_id in range(5):
            chunk_dict = {}
            for chunk_id in range(4):
                chunk_dict[chunk_id] = np.load(
                    f"data/chembl_mol_unimol1_{chunk_id}_seed_42_coord_{coord_id}.npy",
                    allow_pickle=True
                ).item()
            self.coord_dicts.append(chunk_dict)
        
        dti_data = pd.read_csv(dti_data_file)
        prot_mapping = pd.read_csv("data/chembl_prot.csv")
        prot_uniprot_dict = dict(zip(prot_mapping["seq"], prot_mapping["uniprot_id"]))
        dti_data["uniprot_id"] = dti_data["Protein"].map(prot_uniprot_dict)
        
        dti_uniprot_ids = dti_data["uniprot_id"].to_list()
        dti_smiles = dti_data["SMILES"].to_list()
        dti_labels = dti_data["Y"].to_numpy()
        
        self.dti_uniprot_ids = dti_uniprot_ids
        self.dti_smiles = dti_smiles
        self.dti_labels = dti_labels
        
        chembl_molf_embs_dict = np.load("data/chembl_molf_embs.npy", allow_pickle=True).item()
        self.chembl_molf_embs_dict = chembl_molf_embs_dict
        
        chembl_mol_graphs_dict = torch.load("data/chembl_mol_graphs.pt")
        self.chembl_mol_graphs_dict = chembl_mol_graphs_dict
        
        chembl_molf_scaffold_embs_dict = np.load("/home/dataset-assist-0/datafile/quantianhong/MDR/data/chembl/chembl_molf_scaffold_embs.npy", allow_pickle=True).item()
        self.chembl_molf_scaffold_embs_dict = chembl_molf_scaffold_embs_dict
        
        self.chembl_molf_isostere_embs_dict = {}
        for i in range(4):
            with open(f"/home/dataset-assist-0/datafile/quantianhong/MDR/data/chembl/chembl_molf_isostere_embs_{i}.pkl", 'rb') as f:
                chembl_molf_isostere_embs = pickle.load(f)
            self.chembl_molf_isostere_embs_dict.update(chembl_molf_isostere_embs)
        assert len(self.chembl_molf_isostere_embs_dict) == len(self.chembl_molf_scaffold_embs_dict)
    
    def get_prot_emb(self, key):
        lmdb_path = "/home/dataset-local/data_local/quantianhong/GDR/data/chembl_target_ESMC"
        read_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with read_env.begin(write=False) as txn:
            emb = txn.get(key.encode())
            emb = pickle.loads(emb)
        emb = emb.mean(axis=0)
        return emb

    def __len__(self):
        return len(self.dti_smiles)

    def __getitem__(self, idx):
        smi = self.dti_smiles[idx]
        
        chunk_id = self.smiles_dict[smi]
        coord_id = 0
        confs = self.coord_dicts[coord_id][chunk_id][smi]
        confs = torch.tensor(confs, dtype=torch.float)
        
        graph = self.chembl_mol_graphs_dict[smi]
        
        feat_type = 'normal'
        if feat_type == 'normal':
            sfeats = self.chembl_molf_embs_dict[smi]
        elif feat_type == 'scaffold':
            sfeats = self.chembl_molf_scaffold_embs_dict[smi]
        elif feat_type == 'isostere':
            isostere_dict = self.chembl_molf_isostere_embs_dict[smi]
            sfeats = random.choice(list(isostere_dict.values()))
        sfeats = torch.tensor(sfeats, dtype=torch.float)
        
        uid = self.dti_uniprot_ids[idx]
        prot = self.get_prot_emb(uid)
        prot = torch.tensor(prot, dtype=torch.float)
        
        label = torch.Tensor([self.dti_labels[idx]])
        
        return prot, graph, confs, sfeats, label


class MultiViewDTITrimodalDataset1DV(Dataset):
    def __init__(self, data_dir, data_name, dti_data_file):
        data = pd.read_csv(os.path.join(data_dir, data_name + '.csv'))
        self.smiles = data['smiles'].to_list()
        self.smiles_dict = dict(zip(data['smiles'], data['chunk_id']))
        
        self.coord_dicts = []
        
        for coord_id in range(5):
            chunk_dict = {}
            for chunk_id in range(4):
                chunk_dict[chunk_id] = np.load(
                    f"data/chembl_mol_unimol1_{chunk_id}_seed_42_coord_{coord_id}.npy",
                    allow_pickle=True
                ).item()
            self.coord_dicts.append(chunk_dict)
        
        dti_data = pd.read_csv(dti_data_file)
        prot_mapping = pd.read_csv("data/chembl_prot.csv")
        prot_uniprot_dict = dict(zip(prot_mapping["seq"], prot_mapping["uniprot_id"]))
        dti_data["uniprot_id"] = dti_data["Protein"].map(prot_uniprot_dict)
        
        dti_uniprot_ids = dti_data["uniprot_id"].to_list()
        dti_smiles = dti_data["SMILES"].to_list()
        dti_labels = dti_data["Y"].to_numpy()
        
        self.dti_uniprot_ids = dti_uniprot_ids
        self.dti_smiles = dti_smiles
        self.dti_labels = dti_labels
        
        chembl_molf_embs_dict = np.load("data/chembl_molf_embs.npy", allow_pickle=True).item()
        self.chembl_molf_embs_dict = chembl_molf_embs_dict
        
        chembl_mol_graphs_dict = torch.load("data/chembl_mol_graphs.pt")
        self.chembl_mol_graphs_dict = chembl_mol_graphs_dict
        
        chembl_molf_scaffold_embs_dict = np.load("/home/dataset-assist-0/datafile/quantianhong/MDR/data/chembl/chembl_molf_scaffold_embs.npy", allow_pickle=True).item()
        self.chembl_molf_scaffold_embs_dict = chembl_molf_scaffold_embs_dict
        
        self.chembl_molf_isostere_embs_dict = {}
        for i in range(4):
            with open(f"/home/dataset-assist-0/datafile/quantianhong/MDR/data/chembl/chembl_molf_isostere_embs_{i}.pkl", 'rb') as f:
                chembl_molf_isostere_embs = pickle.load(f)
            self.chembl_molf_isostere_embs_dict.update(chembl_molf_isostere_embs)
        assert len(self.chembl_molf_isostere_embs_dict) == len(self.chembl_molf_scaffold_embs_dict)
    
    def get_prot_emb(self, key):
        lmdb_path = "/home/dataset-local/data_local/quantianhong/GDR/data/chembl_target_ESMC"
        read_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with read_env.begin(write=False) as txn:
            emb = txn.get(key.encode())
            emb = pickle.loads(emb)
        emb = emb.mean(axis=0)
        return emb

    def __len__(self):
        return len(self.dti_smiles)

    def __getitem__(self, idx):
        smi = self.dti_smiles[idx]
        
        chunk_id = self.smiles_dict[smi]
        coord_id = 0
        confs = self.coord_dicts[coord_id][chunk_id][smi]
        confs = torch.tensor(confs, dtype=torch.float)
        
        graph = self.chembl_mol_graphs_dict[smi]
        
        feat_type = random.choice(['normal', 'scaffold', 'isostere'])
        if feat_type == 'normal':
            sfeats = self.chembl_molf_embs_dict[smi]
        elif feat_type == 'scaffold':
            sfeats = self.chembl_molf_scaffold_embs_dict[smi]
        elif feat_type == 'isostere':
            isostere_dict = self.chembl_molf_isostere_embs_dict[smi]
            sfeats = random.choice(list(isostere_dict.values()))
        sfeats = torch.tensor(sfeats, dtype=torch.float)
        
        uid = self.dti_uniprot_ids[idx]
        prot = self.get_prot_emb(uid)
        prot = torch.tensor(prot, dtype=torch.float)
        
        label = torch.Tensor([self.dti_labels[idx]])
        
        return prot, graph, confs, sfeats, label


class MultiViewDTITrimodalDataset3DV(Dataset):
    def __init__(self, data_dir, data_name, dti_data_file):
        data = pd.read_csv(os.path.join(data_dir, data_name + '.csv'))
        self.smiles = data['smiles'].to_list()
        self.smiles_dict = dict(zip(data['smiles'], data['chunk_id']))
        
        self.coord_dicts = []
        
        for coord_id in range(5):
            chunk_dict = {}
            for chunk_id in range(4):
                chunk_dict[chunk_id] = np.load(
                    f"data/chembl_mol_unimol1_{chunk_id}_seed_42_coord_{coord_id}.npy",
                    allow_pickle=True
                ).item()
            self.coord_dicts.append(chunk_dict)
        
        dti_data = pd.read_csv(dti_data_file)
        prot_mapping = pd.read_csv("data/chembl_prot.csv")
        prot_uniprot_dict = dict(zip(prot_mapping["seq"], prot_mapping["uniprot_id"]))
        dti_data["uniprot_id"] = dti_data["Protein"].map(prot_uniprot_dict)
        
        dti_uniprot_ids = dti_data["uniprot_id"].to_list()
        dti_smiles = dti_data["SMILES"].to_list()
        dti_labels = dti_data["Y"].to_numpy()
        
        self.dti_uniprot_ids = dti_uniprot_ids
        self.dti_smiles = dti_smiles
        self.dti_labels = dti_labels
        
        chembl_molf_embs_dict = np.load("data/chembl_molf_embs.npy", allow_pickle=True).item()
        self.chembl_molf_embs_dict = chembl_molf_embs_dict
        
        chembl_mol_graphs_dict = torch.load("data/chembl_mol_graphs.pt")
        self.chembl_mol_graphs_dict = chembl_mol_graphs_dict
        
        chembl_molf_scaffold_embs_dict = np.load("/home/dataset-assist-0/datafile/quantianhong/MDR/data/chembl/chembl_molf_scaffold_embs.npy", allow_pickle=True).item()
        self.chembl_molf_scaffold_embs_dict = chembl_molf_scaffold_embs_dict
        
        self.chembl_molf_isostere_embs_dict = {}
        for i in range(4):
            with open(f"/home/dataset-assist-0/datafile/quantianhong/MDR/data/chembl/chembl_molf_isostere_embs_{i}.pkl", 'rb') as f:
                chembl_molf_isostere_embs = pickle.load(f)
            self.chembl_molf_isostere_embs_dict.update(chembl_molf_isostere_embs)
        assert len(self.chembl_molf_isostere_embs_dict) == len(self.chembl_molf_scaffold_embs_dict)
    
    def get_prot_emb(self, key):
        lmdb_path = "/home/dataset-local/data_local/quantianhong/GDR/data/chembl_target_ESMC"
        read_env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        with read_env.begin(write=False) as txn:
            emb = txn.get(key.encode())
            emb = pickle.loads(emb)
        emb = emb.mean(axis=0)
        return emb

    def __len__(self):
        return len(self.dti_smiles)

    def __getitem__(self, idx):
        smi = self.dti_smiles[idx]
        
        chunk_id = self.smiles_dict[smi]
        coord_id = np.random.choice(range(5))
        confs = self.coord_dicts[coord_id][chunk_id][smi]
        confs = torch.tensor(confs, dtype=torch.float)
        
        graph = self.chembl_mol_graphs_dict[smi]
        
        feat_type = 'normal'
        if feat_type == 'normal':
            sfeats = self.chembl_molf_embs_dict[smi]
        elif feat_type == 'scaffold':
            sfeats = self.chembl_molf_scaffold_embs_dict[smi]
        elif feat_type == 'isostere':
            isostere_dict = self.chembl_molf_isostere_embs_dict[smi]
            sfeats = random.choice(list(isostere_dict.values()))
        sfeats = torch.tensor(sfeats, dtype=torch.float)
        
        uid = self.dti_uniprot_ids[idx]
        prot = self.get_prot_emb(uid)
        prot = torch.tensor(prot, dtype=torch.float)
        
        label = torch.Tensor([self.dti_labels[idx]])
        
        return prot, graph, confs, sfeats, label
