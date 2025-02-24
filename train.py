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
from dataset import ProcessDataSet, Collate_fn
from model import *


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    loss_sum = 0
    
    for train_data in train_loader:
        train_data = [x.to(device) for x in train_data]
        vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, train_label = train_data
        
        optimizer.zero_grad()
        output = model(vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence)
        loss = F.cross_entropy(output, train_label)
        loss.backward()
        
        optimizer.step()
        loss_sum += loss.item()*len(train_data[0])
        
    train_loss = round(loss_sum/float(len(train_loader.dataset)), 6)
    print('Train epoch: {}  Loss:{:.6f}'.format(epoch, train_loss))
    return train_loss

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

    print('*'*20 + ' Training model... ' + '*'*20)
    scenario = sys.argv[1]

    if scenario == 'cross_validation':
        (cuda_name, \
         preprocessed_data_path, save_results_path, \
         TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE, \
         LR, LOG_INTERVAL, NUM_EPOCHS, num_features_xd, num_head1, num_head2, \
         embed_dim, dropout, hidden_size, kernel1, kernel2, kernel3, dropout_cnn, \
         cnn_hid_dim, num_layers) = sys.argv[2:]

    elif scenario == 'unknown_setting':
        (cluster_cross_validation_setting, cluster_threshold, cuda_name, \
         preprocessed_data_path, save_results_path, \
         TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE, \
         LR, LOG_INTERVAL, NUM_EPOCHS, num_features_xd, num_head1, num_head2, \
         embed_dim, dropout, hidden_size, kernel1, kernel2, kernel3, dropout_cnn, \
         cnn_hid_dim, num_layers) = sys.argv[2:]

    (TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE, \
    LOG_INTERVAL, NUM_EPOCHS, num_features_xd, num_head1, num_head2, \
    embed_dim, hidden_size, kernel1, kernel2, kernel3, \
    cnn_hid_dim, num_layers) = map(int, [TRAIN_BATCH_SIZE, VAL_BATCH_SIZE, TEST_BATCH_SIZE, \
                                         LOG_INTERVAL, NUM_EPOCHS, num_features_xd, num_head1, num_head2, \
                                         embed_dim, hidden_size, kernel1, kernel2, kernel3, \
                                         cnn_hid_dim, num_layers])
    
    num_heads = [num_head1, num_head2]
    kernels = [kernel1, kernel2, kernel3]

    (LR, dropout, dropout_cnn) = map(float, [LR, dropout, dropout_cnn])
    (scenario, cuda_name, preprocessed_data_path, save_results_path) = map(str, [scenario, cuda_name, preprocessed_data_path, save_results_path])
    
    Kfolds = 5
    SEED = 1234
    
    if scenario == 'unknown_setting':
        os.makedirs(save_results_path, exist_ok=True)
        
        setting = cluster_cross_validation_setting
        thred = cluster_threshold

        if setting == 'protein_clustering_unknown':
            GNN_depth, CNN_depth, k_head, kernel_size, hidden_size1, hidden_size2 = 4, 3, 1, 5, 128, 128
        else:
            GNN_depth, CNN_depth, k_head, kernel_size, hidden_size1, hidden_size2 = 4, 1, 2, 5, 128, 128

    else:
        os.makedirs(save_results_path, exist_ok=True)

        GNN_depth, CNN_depth, k_head, kernel_size, hidden_size1, hidden_size2 = 2, 1, 1, 5, 128, 128

    
    params = [GNN_depth, CNN_depth, k_head, kernel_size, hidden_size1, hidden_size2]

    results_df = pd.DataFrame(columns=('AUC', 'Acc'))
    
    for fold in range(Kfolds):
        print('-'*25 + ' training on fold ' + str(fold) + ' data ' + '-'*25)
        train_data_pack = load_data(fold, preprocessed_data_path, 'train')
        val_data_pack = load_data(fold, preprocessed_data_path, 'val')
        test_data_pack = load_data(fold, preprocessed_data_path, 'test')

        train_data = ProcessDataSet(train_data_pack)
        val_data = ProcessDataSet(val_data_pack)
        test_data = ProcessDataSet(test_data_pack)
        
        train_loader = DataLoader(train_data, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=Collate_fn)
        val_loader = DataLoader(val_data, batch_size=VAL_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=Collate_fn)
        test_loader = DataLoader(test_data, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=Collate_fn)
        
        # create dataloader
        device = torch.device(cuda_name if torch.cuda.is_available() else 'cpu')
        init_atom_features, init_bond_features, word_dict = load_dict(fold, preprocessed_data_path)

        init_atom_features = init_atom_features.to(device)
        init_bond_features = init_bond_features.to(device)
        N_word = len(word_dict)
        
        model = CmhAttCPI(init_atom_features, init_bond_features, num_features_xd, \
                embed_dim, dropout, hidden_size, kernels, dropout_cnn, cnn_hid_dim, num_heads, num_layers, N_word, \
                params).to(device)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)
        
        model_file_name = save_results_path + str(fold) + 'fold_model.model'
        result_file_name = save_results_path + str(fold) + 'fold_result.txt'
        predict_file_name = save_results_path + str(fold) + 'fold_predict.npy'

        MIN_LOSS = 30
        
        with open(result_file_name, 'w') as f:
            f.write('Epoch\ttrain_loss\tval_loss\tval_AUC\ttest_AUC\ttest_Acc' + '\n')
            print('Epoch   val_loss   val_AUC   test_AUC   test_Acc')
            for epoch in range(NUM_EPOCHS):
                train_loss = train(model, device, train_loader, optimizer, epoch)
                
                val_loss, val_AUC, val_acc,  _, _ = predicting(model, device, val_loader)
                test_loss, test_AUC, test_acc, test_label, test_predict_scores = predicting(model, device, test_loader)
                
                print(str(epoch) + '   ' + str(format(val_loss,'.4f')) + '   ' + str(format(val_AUC,'.4f')) + '   ' + \
                   str(format(test_AUC,'.4f')) + '   ' + str(format(test_acc,'.4f')))
                
                if val_loss < MIN_LOSS:
                    torch.save(model.state_dict(), model_file_name)
                    predict_prob = np.array([test_label, test_predict_scores])
                    np.save(predict_file_name, predict_prob)

                    MIN_LOSS = val_loss
                    n_epoch = epoch
                    best_test_AUC = float(format(test_AUC,'.4f'))
                    best_test_acc = float(format(test_acc,'.4f'))
                
                train_results = [epoch+1, float(format(train_loss,'.4f')), float(format(val_loss,'.4f')), float(format(val_AUC,'.4f')),\
                 float(format(test_AUC,'.4f')), float(format(test_acc,'.4f'))]

                f.write('\t'.join(map(str, train_results))+'\n')
            f.close()
        
        result = [best_test_AUC, best_test_acc]
        
        print('The best performance of' + str(fold) + ' fold is epoch ' + str(n_epoch))
        print('==========================================')
        print('The best performace is:')
        print('AUC:' + str(result[0]))  
        print('ACC:' + str(result[1]))
        print('==========================================')
        results_df.loc[fold] = [result[0], result[1]]
        
    results_df.to_csv(save_results_path + 'results_df.csv',index=True, header=True)
