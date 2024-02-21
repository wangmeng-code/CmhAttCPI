import math
import time
import pickle
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from utils import *

#some predefined parameters
elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6

class CmhAttCPI(nn.Module):
    def __init__(self, init_atom_features, init_bond_features, num_features_xd, embed_dim, dropout, \
                  hidden_size, kernels, dropout_cnn, cnn_hid_dim, num_heads, num_layers, N_word, params):
        super(CmhAttCPI, self).__init__()


        self.init_atom_features = init_atom_features
        self.init_bond_features = init_bond_features

        self.num_features_xd = num_features_xd
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.kernels = kernels
        self.dropout_cnn = dropout_cnn
        self.cnn_hid_dim = cnn_hid_dim
        self.N_word = N_word
        
        """hyper part"""
        GNN_depth, CNN_depth, k_head, kernel_size, hidden_size1, hidden_size2 = params
        self.GNN_depth = GNN_depth
        self.k_head = k_head
        self.CNN_depth = CNN_depth
        self.kernel_size = kernel_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        
        """GraphConv Module"""
        self.vertex_embedding = nn.Linear(atom_fdim, self.hidden_size1) #first transform vertex features into hidden representations
        
        # GWM parameters
        self.W_a_main = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.k_head)]) for i in range(self.GNN_depth)])
        self.W_a_super = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.k_head)]) for i in range(self.GNN_depth)])
        self.W_main = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.k_head)]) for i in range(self.GNN_depth)])
        self.W_bmm = nn.ModuleList([nn.ModuleList([nn.Linear(self.hidden_size1, 1) for i in range(self.k_head)]) for i in range(self.GNN_depth)])
        
        self.W_super = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])
        self.W_main_to_super = nn.ModuleList([nn.Linear(self.hidden_size1*self.k_head, self.hidden_size1) for i in range(self.GNN_depth)])
        self.W_super_to_main = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])
        
        self.W_zm1 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])
        self.W_zm2 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])
        self.W_zs1 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])
        self.W_zs2 = nn.ModuleList([nn.Linear(self.hidden_size1, self.hidden_size1) for i in range(self.GNN_depth)])
        self.GRU_main = nn.GRUCell(self.hidden_size1, self.hidden_size1)
        self.GRU_super = nn.GRUCell(self.hidden_size1, self.hidden_size1)
        
        """WLN parameters"""
        self.label_U2 = nn.ModuleList([nn.Linear(self.hidden_size1+bond_fdim, self.hidden_size1) for i in range(self.GNN_depth)]) #assume no edge feature transformation
        self.label_U1 = nn.ModuleList([nn.Linear(self.hidden_size1*2, self.hidden_size1) for i in range(self.GNN_depth)])

        
        """CNN module"""
        self.pro_embed = nn.Embedding(self.N_word, self.embed_dim)
        self.paddings = [(x-1)//2 for x in self.kernels]
        self.conv = nn.ModuleList([nn.Conv1d(self.embed_dim, self.cnn_hid_dim, kernel_size=self.kernels[i], padding=self.paddings[i]) for i in range(self.CNN_depth)])
        
        self.p_dropout = SpatialDropout(0.2)
        self.conv_bn = nn.BatchNorm1d(self.cnn_hid_dim)
        self.p_fc = nn.Linear(self.cnn_hid_dim*self.CNN_depth, 128)
        self.p_fc_bn = nn.BatchNorm1d(128)

        self.elu = nn.ELU()
        self.Dropout_cnn = nn.Dropout(self.dropout_cnn)
        
        """ multiheadattention """
        self.multihead_attn_p_to_d = nn.MultiheadAttention(embed_dim=self.hidden_size*2, num_heads=self.num_heads[1], kdim=self.cnn_hid_dim*2, vdim=self.cnn_hid_dim*2)
        self.multihead_attn_d_to_p = nn.MultiheadAttention(embed_dim=self.cnn_hid_dim*2, num_heads=self.num_heads[0], kdim=self.hidden_size*2, vdim=self.hidden_size*2)

        # hyperattention
        self.drug_att_layer = nn.Linear(self.hidden_size1, self.hidden_size1)
        self.prot_att_layer = nn.Linear(self.hidden_size1, self.hidden_size1)
        self.attention_layer = nn.Linear(self.hidden_size1, self.hidden_size1)


        # combined layers
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.out = nn.Linear(64, 2)
        
      
    def mask_softmax(self,a, mask, dim=-1):
        a_max = torch.max(a,dim,keepdim=True)[0]
        a_exp = torch.exp(a-a_max)
        a_exp = a_exp*mask
        a_softmax = a_exp/(torch.sum(a_exp,dim,keepdim=True)+1e-6)
        return a_softmax
    
    def GraphConv_module(self, batch_size, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask):
        # vertex_mask:[30,最大原子数], vertex:[30,最大原子数], edge:[30,最大原子数], atom_adj:[30*最大原子数*6], bond_adj:[30*最大原子数*6], nbs_mask:[30,最大原子数,6]
        n_vertex = vertex_mask.size(1)
        
        # initial features
        vertex_initial = torch.index_select(self.init_atom_features, 0, vertex.view(-1))   # [bsz*最大atom_num, 82]
        vertex_initial = vertex_initial.view(batch_size, -1, atom_fdim)       # [bsz, 最大atom_num, 82]
        edge_initial = torch.index_select(self.init_bond_features, 0, edge.view(-1))  # [bsz*最大atom_num, 6]
        edge_initial = edge_initial.view(batch_size, -1, bond_fdim)   # [bsz, 最大atom_num, 6]
        
        vertex_feature = F.leaky_relu(self.vertex_embedding(vertex_initial), 0.1)    # [bsz, 最大atom_num, 128]
        super_feature = torch.sum(vertex_feature*vertex_mask.view(batch_size,-1,1), dim=1, keepdim=True)   # [bsz, 1, 128]

        for GWM_iter in range(self.GNN_depth):
                # prepare main node features
            for k in range(self.k_head):
                a_main = torch.tanh(self.W_a_main[GWM_iter][k](vertex_feature))    # [bsz, 最大atom_num, 128]
                a_super = torch.tanh(self.W_a_super[GWM_iter][k](super_feature))   # [bsz, 1, 128]
                a = self.W_bmm[GWM_iter][k](a_main*super_feature)  # [bsz, 最大atom_num, 1]

                attn = self.mask_softmax(a.view(batch_size,-1), vertex_mask).view(batch_size,-1,1) 
                k_main_to_super = torch.bmm(attn.transpose(1,2), self.W_main[GWM_iter][k](vertex_feature)) 
                if k == 0:
                        m_main_to_super = k_main_to_super
                else:
                        m_main_to_super = torch.cat([m_main_to_super, k_main_to_super], dim=-1)  # concat k-head
        
            main_to_super = torch.tanh(self.W_main_to_super[GWM_iter](m_main_to_super))  # [bsz, 1, 128]
            main_self = self.wln_unit(batch_size, vertex_mask, vertex_feature, edge_initial, atom_adj, bond_adj, nbs_mask, GWM_iter)  
            # main_self:[bsz, 最大原子数, 128]

            super_to_main = torch.tanh(self.W_super_to_main[GWM_iter](super_feature))   # [bsz, 1, 128]

            # warp gate and GRU for update main node features, use main_self and super_to_main
            z_main = torch.sigmoid(self.W_zm1[GWM_iter](main_self) + self.W_zm2[GWM_iter](super_to_main)) 
            # z_main:[bsz, 最大原子数, 128]
            hidden_main = (1-z_main)*main_self + z_main*super_to_main
            # hidden_main:[bsz, 最大原子数, 128]
            vertex_feature = self.GRU_main(hidden_main.view(-1, self.hidden_size1), vertex_feature.view(-1, self.hidden_size1))  
            # vertex_feature:[bsz*最大原子数, 128]
            vertex_feature = vertex_feature.view(batch_size, n_vertex, self.hidden_size1)  # [bsz, 最大原子数, 128]
    
        return vertex_feature

    
    def wln_unit(self, batch_size, vertex_mask, vertex_features, edge_initial, atom_adj, bond_adj, nbs_mask, GNN_iter):
        n_vertex = vertex_mask.size(1)  # 最大原子数
        n_nbs = nbs_mask.size(2)   # 6

        vertex_mask = vertex_mask.view(batch_size,n_vertex,1)  # [bsz, 最大原子数, 1]
        nbs_mask = nbs_mask.view(batch_size,n_vertex,n_nbs,1)   # [bsz, 最大原子数, 6, 1]
        
        vertex_nei = torch.index_select(vertex_features.view(-1, self.hidden_size1), 0, atom_adj).view(batch_size, n_vertex, n_nbs,self.hidden_size1)
        # vertex_nei:[bsz, 最大原子数, 6, 128]
        edge_nei = torch.index_select(edge_initial.view(-1, bond_fdim), 0, bond_adj).view(batch_size,n_vertex,n_nbs,bond_fdim)
        # vertex_nei:[bsz, 最大原子数, 6, 6]

        # Weisfeiler Lehman relabelling
        l_nei = torch.cat((vertex_nei, edge_nei), -1)   # [bsz, 最大原子数, 6, 134]
        nei_label = F.leaky_relu(self.label_U2[GNN_iter](l_nei), 0.1)   # [bsz, 最大原子数, 6, 128]
        nei_label = torch.sum(nei_label*nbs_mask, dim=-2)   # [bsz, 最大原子数, 128]
        new_label = torch.cat((vertex_features, nei_label), 2)   # [bsz, 最大原子数, 256]
        new_label = self.label_U1[GNN_iter](new_label)    # [bsz, 最大原子数, 256]
        vertex_features = F.leaky_relu(new_label, 0.1)    # [bsz, 最大原子数, 256]
        
        return vertex_features 
    

    
    def CNN_model(self, sequence):
        embedded_xt = self.pro_embed(sequence)
        p_feats = self.p_dropout(embedded_xt, noise_shape=(embedded_xt.size(0), 1, embedded_xt.size(2)))

        feats = [self.elu(self.conv_bn(self.conv[i](p_feats.transpose(2,1)))) for i in range(len(self.conv))]
        
        prot_feats = torch.cat(feats, 1)  # [bsz, 192, pro_len]
        
        prot_feats = self.Dropout_cnn(self.elu(self.p_fc_bn(self.p_fc(prot_feats.transpose(2,1)).transpose(2,1))))
        return prot_feats.transpose(2,1)


    def attention_module(self, drug_feature, prot_feature, vertex_mask, seq_mask):
        
        vertex_mask = vertex_mask.bool() == False
        seq_mask = seq_mask.bool() == False

        drug_att_output, drug_scores = self.multihead_attn_p_to_d(query=drug_feature.transpose(1,0), key=prot_feature.transpose(1,0),\
                                                      value=prot_feature.transpose(1,0), key_padding_mask=seq_mask.bool())
        
        prot_att_output, prot_scores = self.multihead_attn_d_to_p(query=prot_feature.transpose(1,0), key=drug_feature.transpose(1,0),\
                                                      value=drug_feature.transpose(1,0), key_padding_mask=vertex_mask.bool())
        
        return drug_att_output.transpose(1,0), drug_scores, prot_att_output.transpose(1,0), prot_scores


    def DTI_pred_module(self, atom_feature, drug_att_output, vertex_mask, prot_feature, prot_att_output, seq_mask):
        
        drug_att_output = torch.sum(drug_att_output, dim=1) / torch.sum(vertex_mask.unsqueeze(dim=-1), dim=1)
        prot_att_output = torch.sum(prot_att_output, dim=1) / torch.sum(seq_mask.unsqueeze(dim=-1), dim=1)
        
        drug_feats = torch.sum(atom_feature*(vertex_mask.unsqueeze(dim=-1)), dim=1)
        prot_feats = torch.sum(prot_feature*(seq_mask.unsqueeze(dim=-1)), dim=1)

        drug_feats = drug_feats + drug_att_output  
        prot_feats = prot_feats + prot_att_output
        
        xc = torch.cat([drug_feats, prot_feats], dim=1)
       
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu(xc)
        xc = self.dropout(xc)
        xc = self.fc3(xc)
        xc = self.relu(xc)
        output = self.out(xc)

        return output  

    def forward(self, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence):
        
        batch_size = vertex.size(0) 
        
        prot_feature = self.CNN_model(sequence)
        atom_feature = self.GraphConv_module(batch_size, vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask)
        
        drug_att, drug_scores, prot_att, prot_scores = self.attention_module(atom_feature, prot_feature, vertex_mask, seq_mask)
        
        pred_output = self.DTI_pred_module(atom_feature, drug_att, vertex_mask, prot_feature, prot_att, seq_mask)

        return pred_output, drug_att, prot_att

class SpatialDropout(nn.Module):
    
    def __init__(self, drop=0.5):
        super(SpatialDropout, self).__init__()
        self.drop = drop
        
    def forward(self, inputs, noise_shape=None):
        """
        @param: inputs, tensor
        @param: noise_shape, tuple, 应当与inputs的shape一致，其中值为1的即沿着drop的轴
        """
        outputs = inputs.clone()
        if noise_shape is None:
            noise_shape = (inputs.shape[0], *repeat(1, inputs.dim()-2), inputs.shape[-1])   # 默认沿着中间所有的shape
        
        self.noise_shape = noise_shape
        if not self.training or self.drop == 0:
            return inputs
        else:
            noises = self._make_noises(inputs)
            if self.drop == 1:
                noises.fill_(0.0)
            else:
                noises.bernoulli_(1 - self.drop).div_(1 - self.drop)
            noises = noises.expand_as(inputs)    
            outputs.mul_(noises)
            return outputs
            
    def _make_noises(self, inputs):
        return inputs.new().resize_(self.noise_shape)