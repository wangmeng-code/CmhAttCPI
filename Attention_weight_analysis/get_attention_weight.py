import os
import pickle
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score

from utils import *
from dataset import ProcessDataSet, Collate_fn
from model_att import *


def predicting(model, device, eval_loader):
    model.eval()
    Correct_Label = []
    Predicted_Score = []
    Predicted_Label = []
    drug_features = []
    prot_features = []
    
    with torch.no_grad():
        for eval_data in eval_loader:
            eval_data = [x.to(device) for x in eval_data]
            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, eval_label = eval_data
            
            
            output, drug_att, prot_att = model(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
            
            drug_feats = torch.mean(drug_att,dim=-1).cpu().numpy()
            prot_feats = torch.mean(prot_att,dim=-1).cpu().numpy()

            loss = F.cross_entropy(output, eval_label)
            
            correct_label = eval_label
            y_score_softmax = F.softmax(output,1).cpu().data.numpy()
            _, predicted_label = torch.max(output,1)
            predicted_scores = list(map(lambda x: x[1], y_score_softmax))

            Correct_Label.extend(correct_label.cpu().numpy().tolist())
            Predicted_Score.extend(predicted_scores)
            Predicted_Label.extend(predicted_label.cpu().numpy().tolist())
            drug_features.append(drug_feats)
            prot_features.append(prot_feats)

    return Correct_Label, Predicted_Score, drug_features, prot_features

if __name__ == "__main__":

    root_path = './analysis_result/' 
    
    LR = 0.0001
    LOG_INTERVAL = 20
    NUM_EPOCHS = 50
    SEED = 1234

    cuda_name = "cuda:1"
    num_features_xd=82
    num_features_xt=25
    
    num_heads=[1,4] 
   
    embed_dim=128
    
    dropout=0.2
    hidden_size=64
    kernels = [11, 15, 21]
    dropout_cnn = 0
    cnn_hid_dim = 64
    
    num_layers=2
   
    GNN_depth, CNN_depth, k_head, kernel_size, hidden_size1, hidden_size2 = 2, 1, 1, 5, 128, 128

    params = [GNN_depth, CNN_depth, k_head, kernel_size, hidden_size1, hidden_size2]

    
    test_data_pack = load_test_data()
    test_data = ProcessDataSet(test_data_pack)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, collate_fn=Collate_fn)
    
    # create dataloader
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    init_atom_features, init_bond_features, word_dict = load_test_dict()
    
    init_atom_features = init_atom_features.to(device)
    init_bond_features = init_bond_features.to(device)
    N_word = len(word_dict)
    
    model = CmhAttCPI(init_atom_features, init_bond_features, num_features_xd, \
            embed_dim, dropout, hidden_size, kernels, dropout_cnn, cnn_hid_dim, num_heads, num_layers, N_word, \
            params).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    model_path = 'pretrain_model.model'
    model_state = torch.load(model_path)
    model.load_state_dict(model_state)
    

    test = pd.read_csv('test.txt', sep=' ', header=None)
    
    test_label, test_predict_scores, drug_features, prot_features = predicting(model, device, test_loader)
    
    atom_symbol = pd.read_csv(root_path+'atom_symbol.csv',header=None)
    drug_feats = pd.DataFrame(drug_features[0]).T
    drug_feature_df = pd.concat([atom_symbol,drug_feats],axis=1)
    drug_feature_df.columns = ['atom', 'weights']

    residue_symbol = pd.read_csv(root_path+'residue_symbol.csv',header=None)
    prot_feats = pd.DataFrame(prot_features[0]).T
    prot_feature_df = pd.concat([residue_symbol,prot_feats],axis=1)
    prot_feature_df.columns = ['residue', 'weights']

    drug_feature_df.to_csv(root_path + 'drug_attention_weights.csv',index=False)
    prot_feature_df.to_csv(root_path + 'protein_attention_weights.csv',index=False)
    

        
