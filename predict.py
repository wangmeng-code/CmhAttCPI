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
from model import *


def predicting(model, device, eval_loader):
    model.eval()
    Correct_Label = []
    Predicted_Score = []
    Predicted_Label = []
    eval_loss_sum = 0
    with torch.no_grad():
        for eval_data in eval_loader:
            eval_data = [x.to(device) for x in eval_data]
            vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, eval_label = eval_data
            
            output = model(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
            loss = F.cross_entropy(output, eval_label)
            
            correct_label = eval_label
            y_score_softmax = F.softmax(output,1).cpu().data.numpy()
            _, predicted_label = torch.max(output,1)
            predicted_scores = list(map(lambda x: x[1], y_score_softmax))

            Correct_Label.extend(correct_label.cpu().numpy().tolist())
            Predicted_Score.extend(predicted_scores)
            Predicted_Label.extend(predicted_label.cpu().numpy().tolist())
            
            eval_loss_sum += loss.item()*len(eval_data[0])
    eval_loss = round(eval_loss_sum/float(len(eval_loader.dataset)), 6)
    
    # metrics
    confusion = confusion_matrix(Correct_Label, Predicted_Label)
    TP = confusion[1, 1]
    TN = confusion[0, 0]
    FP = confusion[0, 1]
    FN = confusion[1, 0]
    
    AUC = roc_auc_score(Correct_Label, Predicted_Score)
    acc = (TP+TN)/(TP+TN+FP+FN)
    
    return eval_loss, AUC, acc, Correct_Label, Predicted_Score

if __name__ == "__main__":
    
    DATASET = 'user-dataset' 
    
    TEST_BATCH_SIZE = 128
    
    LR = 0.0001
    LOG_INTERVAL = 20
    NUM_EPOCHS = 50
    SEED = 1234
    
    cuda_name = "cuda:1"
    num_features_xd=82
    
    num_heads=[1,4] 
    embed_dim=128
    dropout=0.2
    hidden_size=64
    kernels = [11, 15, 21]
    dropout_cnn = 0
    cnn_hid_dim = 64
    num_layers=2
    

    root_path = './Predicting_user-dataset/' 
    save_path = root_path + 'results/'
    os.makedirs(save_path, exist_ok=True)
    
    GNN_depth, CNN_depth, k_head, kernel_size, hidden_size1, hidden_size2 = 2, 1, 1, 5, 128, 128

    
    params = [GNN_depth, CNN_depth, k_head, kernel_size, hidden_size1, hidden_size2]

    # create dataloader
    test_data_pack = load_test_data(root_path)
    test_data = ProcessDataSet(test_data_pack)
    test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=Collate_fn)
    
    
    device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
    init_atom_features, init_bond_features, word_dict = load_test_dict(root_path)

    init_atom_features = init_atom_features.to(device)
    init_bond_features = init_bond_features.to(device)

    N_word = len(word_dict)
    
    model = CmhAttCPI(init_atom_features, init_bond_features, num_features_xd, \
            embed_dim, dropout, hidden_size, kernels, dropout_cnn, cnn_hid_dim, num_heads, num_layers, N_word, \
            params).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    model_file_name = 'pretrain.model'   # pretrain model path
    result_file_name = save_path +  'user_datset_test_result.txt'

    model.load_state_dict(torch.load(model_file_name))

    with open(result_file_name, 'w') as f:
        f.write('test_AUC\ttest_Acc' + '\n')
        print('test_AUC   test_Acc')
        test_loss, test_AUC, test_acc, test_label, test_predict_scores = predicting(model, device, test_loader)
        
        print(str(format(test_AUC,'.4f')) + '   ' + str(format(test_acc,'.4f')))
        
        train_results = [float(format(test_AUC,'.4f')), float(format(test_acc,'.4f'))]

        f.write('\t'.join(map(str, train_results))+'\n')
        f.close()
    
    result = [test_AUC, test_acc]
    
    print('==========================================')
    print('The performace on user-datset is:')
    print('AUC:' + str(result[0]))  
    print('ACC:' + str(result[1]))
    print('==========================================')
    