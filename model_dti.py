import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from gnn import GNN, GPS


class ContrastGNN(nn.Module):
    def __init__(self, num_layer, emb_dim, out_dim, confs_dim=512, smiles_dim=768, prot_dim=1152, JK="last", drop_ratio=0, attn_drop_ratio=0, model_head=4, atom_feat_dim=None, bond_feat_dim=None, backbone='gnn'):
        super(ContrastGNN, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.attn_drop_ratio = attn_drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim
        self.out_dim = out_dim
        self.confs_dim = confs_dim
        self.smiles_dim = smiles_dim
        self.prot_dim = prot_dim
        self.backbone = backbone

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")

        if self.backbone == 'gnn':
            self.gnn = GNN(self.num_layer, self.emb_dim, JK=self.JK, drop_ratio=self.drop_ratio,
                           atom_feat_dim=atom_feat_dim, bond_feat_dim=bond_feat_dim)
        elif self.backbone == 'gps':
            self.gnn = GPS(channels=self.emb_dim, pe_dim=20,
                           node_dim=atom_feat_dim, edge_dim=bond_feat_dim,
                           num_layers=self.num_layer, heads=model_head,
                           attn_type='multihead', dropout=self.drop_ratio,
                           attn_dropout=self.attn_drop_ratio)
        
        self.graph_out_linear = nn.Sequential(
            nn.Linear(self.emb_dim, 2 * self.emb_dim),
            nn.ReLU(),
            nn.Linear(2 * self.emb_dim, self.out_dim)
        )
        
        self.confs_out_linear = nn.Sequential(
            nn.Linear(self.confs_dim, 2 * self.confs_dim),
            nn.ReLU(),
            nn.Linear(2 * self.confs_dim, self.out_dim)
        )
        
        self.smiles_out_linear = nn.Sequential(
            nn.Linear(self.smiles_dim, 2 * self.smiles_dim),
            nn.ReLU(),
            nn.Linear(2 * self.smiles_dim, self.out_dim)
        )
        
        self.fusion_out_linear = nn.Sequential(
            nn.Linear(3 * self.out_dim, 2 * self.out_dim),
            nn.ReLU(),
            nn.Linear(2 * self.out_dim, self.out_dim)
        )
        
        self.prot_out_linear = nn.Sequential(
            nn.Linear(self.prot_dim, 2 * self.prot_dim),
            nn.ReLU(),
            nn.Linear(2 * self.prot_dim, self.out_dim)
        )
    
    def get_representations(self, data):
        if self.backbone == 'gps':
            h_g, node_repres = self.gnn(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        else:
            h_g, node_repres = self.gnn(data.x, data.edge_index, data.edge_attr, data.batch)

        return h_g
    
    def get_graph_out(self, data):
        h_g = self.get_representations(data)
        g_out = self.graph_out_linear(h_g)
        return g_out
    
    def get_confs_out(self, data):
        c_out = self.confs_out_linear(data)
        return c_out
    
    def get_smiles_out(self, data):
        s_out = self.smiles_out_linear(data)
        return s_out
    
    def get_fusion_out(self, g_out, c_out, s_out):
        f_out = torch.cat([g_out, c_out, s_out], dim=-1)
        f_out = self.fusion_out_linear(f_out)
        return f_out
    
    def get_prot_out(self, data):
        p_out = self.prot_out_linear(data)
        return p_out
    
    def get_drug_out(self, graph, confs, smiles):
        g_out = self.get_graph_out(graph)
        c_out = self.get_confs_out(confs)
        s_out = self.get_smiles_out(smiles)
        
        f_out = self.get_fusion_out(g_out, c_out, s_out)
        
        return f_out
    
    def forward(self, prot, graph, confs, smiles):
        g_out = self.get_graph_out(graph)
        c_out = self.get_confs_out(confs)
        s_out = self.get_smiles_out(smiles)
        
        f_out = self.get_fusion_out(g_out, c_out, s_out)
        
        p_out = self.get_prot_out(prot)
        
        return p_out, f_out
