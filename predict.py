import os
import pickle
import numpy as np
import pandas as pd
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score
from utils import *
from dataset import *
from model import *


def predicting(model, device, eval_loader):
    model.eval()
    Correct_Label = []
    Predicted_Score = []
    Predicted_Label = []
    
    eval_data_smiles = []
    eval_data_seqs = []
    
    with torch.no_grad():
        for eval_data in eval_loader:
            eval_feats = [x.to(device) for x in eval_data[:-2]]
            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, seq_input = eval_feats
            
            output = model(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, seq_input)
            
            y_score_softmax = F.softmax(output,1).cpu().data.numpy()
            _, predicted_label = torch.max(output,1)
            predicted_scores = list(map(lambda x: x[1], y_score_softmax))

            Predicted_Score.extend(predicted_scores)
            Predicted_Label.extend(predicted_label.cpu().numpy().tolist())

            eval_data_smiles.extend(eval_data[-2])
            eval_data_seqs.extend(eval_data[-1])
            
    return Predicted_Score,Predicted_Label, eval_data_smiles, eval_data_seqs

if __name__ == "__main__":
    print('*'*20 + ' Predicting on test dataset... ' + '*'*20)

    (cuda_name, preprocessed_testdata_path, model_path, save_results_path, \
    TEST_BATCH_SIZE, num_features_xd, num_head1, num_head2, \
    embed_dim, dropout, hidden_size, kernel1, kernel2, kernel3, dropout_cnn, \
    cnn_hid_dim, num_layers) = sys.argv[1:]


    (TEST_BATCH_SIZE, num_features_xd, num_head1, num_head2, \
    embed_dim, hidden_size, kernel1, kernel2, kernel3, \
    cnn_hid_dim, num_layers) = map(int, [TEST_BATCH_SIZE, num_features_xd, num_head1, num_head2, \
                                         embed_dim, hidden_size, kernel1, kernel2, kernel3, \
                                         cnn_hid_dim, num_layers])
    
    num_heads = [num_head1, num_head2]
    kernels = [kernel1, kernel2, kernel3]

    (dropout, dropout_cnn) = map(float, [dropout, dropout_cnn])
    (cuda_name, preprocessed_testdata_path, model_path, save_results_path) = map(str, [cuda_name, preprocessed_testdata_path, model_path, save_results_path])
    
    os.makedirs(save_results_path, exist_ok=True)

    TEST_BATCH_SIZE = 128
    SEED = 1234
   
    GNN_depth, CNN_depth, k_head, kernel_size, hidden_size1, hidden_size2 = 2, 1, 1, 5, 128, 128

    params = [GNN_depth, CNN_depth, k_head, kernel_size, hidden_size1, hidden_size2]
    
    # create dataloader
    test_data_pack = load_test_data(preprocessed_testdata_path)
    test_data = ProcessTestDataSet(test_data_pack)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=Collate_Test_fn)
    
    # model initialization
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    init_atom_features, init_bond_features, word_dict = load_test_dict(preprocessed_testdata_path)

    init_atom_features = init_atom_features.to(device)
    init_bond_features = init_bond_features.to(device)

    N_word = len(word_dict)

    model = CmhAttCPI(init_atom_features, init_bond_features, num_features_xd, \
            embed_dim, dropout, hidden_size, kernels, dropout_cnn, cnn_hid_dim, num_heads, num_layers, N_word, \
            params).to(device)
    
    result_file_name = save_results_path +  'predict_result.csv'
    
    # model loading
    model.load_state_dict(torch.load(model_path))
    
    # predicting
    test_predict_scores, test_predicted_label, test_smiles, test_seqs = predicting(model, device, test_loader)
    
    predict_result = pd.DataFrame(columns=['Smiles', 'Sequences', 'Predict_scores', 'Predicted_labels'])
    predict_result['Smiles'] = test_smiles
    predict_result['Sequences'] = test_seqs
    predict_result['Predict_scores'] = test_predict_scores
    predict_result['Predicted_labels'] = test_predicted_label

    predict_result.to_csv(result_file_name,index=False, header=True)
    
