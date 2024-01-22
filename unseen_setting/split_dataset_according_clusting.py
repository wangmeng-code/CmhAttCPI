import numpy as np
import pandas as pd
import os
import pickle
from sklearn.metrics import mean_squared_error, precision_score, roc_auc_score
from sklearn.model_selection import KFold
import random

# load data
def data_from_index(data_pack, idx_list):
    SMILES, cid, Sequence, pid = [data_pack[i][idx_list] for i in range(4)]
    label = data_pack[4][idx_list].astype(int).reshape(-1,1)
    data = pd.concat([pd.DataFrame(SMILES), pd.DataFrame(cid), pd.DataFrame(Sequence), pd.DataFrame(pid), pd.DataFrame(label)],axis=1)
    data.columns = ['SMILES','Ligand_name','Sequence','Target_name','Label']
    return data

def split_train_test_clusters(cluster_threshold, data_path, Kfolds, SEED):
    with open(data_path + 'compound_cluster_dict_'+str(cluster_threshold), 'rb') as f:
        C_cluster_dict = pickle.load(f)
    
    with open(data_path + 'protein_cluster_dict_'+str(cluster_threshold), 'rb') as f:
        P_cluster_dict = pickle.load(f)
    
    C_cluster_set = set(list(C_cluster_dict.values()))
    P_cluster_set = set(list(P_cluster_dict.values()))
    C_cluster_list = np.array(list(C_cluster_set))
    P_cluster_list = np.array(list(P_cluster_set))

    # n-fold split
    c_kf = KFold(Kfolds, random_state=SEED, shuffle=True)
    p_kf = KFold(Kfolds, random_state=SEED, shuffle=True)

    c_train_clusters, c_test_clusters = [], []
    for train_idx, test_idx in c_kf.split(C_cluster_list):

        c_train_clusters.append(C_cluster_list[train_idx])
        c_test_clusters.append(C_cluster_list[test_idx])
    
    p_train_clusters, p_test_clusters = [], []
    for train_idx, test_idx in p_kf.split(P_cluster_list):
        p_train_clusters.append(P_cluster_list[train_idx])
        p_test_clusters.append(P_cluster_list[test_idx])
    
    pair_list = []
    for i_d in C_cluster_list:
        for i_p in P_cluster_list:
            pair_list.append('drug'+str(i_d)+'--' + 'protein'+str(i_p))
    pair_list = np.array(pair_list)
    
    pair_kf = KFold(Kfolds, random_state=SEED, shuffle=True)
    pair_train_clusters, pair_test_clusters = [], []
    for train_idx, test_idx in pair_kf.split(pair_list):
        pair_train_clusters.append(pair_list[train_idx])
        pair_test_clusters.append(pair_list[test_idx])

    return pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict

def load_data(cluster_cross_validation_setting, file_path, cluster_threshold, Kfolds, SEED):
    
    with open(file_path + 'all_combined_input','rb') as f:
        data_pack = pickle.load(f)
       
    cid_list = data_pack[1]
    pid_list = data_pack[3]
    n_sample = len(cid_list)
    
    # train-test split
    train_idx_list, test_idx_list = [], []
    print('cluster_cross_validation_setting:', cluster_cross_validation_setting)
    
    if cluster_cross_validation_setting == 'protein_clustering_unseen':
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict = split_train_test_clusters(cluster_threshold, file_path, Kfolds, SEED)
        for fold in range(Kfolds):
            p_train, p_test = p_train_clusters[fold], p_test_clusters[fold]
            Train_idx, Test_idx = [], []
            for ele in range(n_sample): 
                if P_cluster_dict[pid_list[ele]] in p_train:
                    Train_idx.append(ele)
                elif P_cluster_dict[pid_list[ele]] in p_test:
                    Test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(Train_idx)
            test_idx_list.append(Test_idx)
            
    elif cluster_cross_validation_setting == 'compound_clustering_unseen':
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict = split_train_test_clusters(cluster_threshold, file_path, Kfolds, SEED)
        for fold in range(Kfolds):
            c_train, c_test = c_train_clusters[fold], c_test_clusters[fold]
            Train_idx, Test_idx = [], []
            for ele in range(n_sample):
                if C_cluster_dict[cid_list[ele]] in c_train:
                    Train_idx.append(ele)
                elif C_cluster_dict[cid_list[ele]] in c_test:
                    Test_idx.append(ele)
                else:
                    print('error')
            train_idx_list.append(Train_idx)
            test_idx_list.append(Test_idx)

    elif cluster_cross_validation_setting == 'both_unseen':
        pair_train_clusters, pair_test_clusters, c_train_clusters, c_test_clusters, p_train_clusters, p_test_clusters, C_cluster_dict, P_cluster_dict = split_train_test_clusters(cluster_threshold, file_path, int(Kfolds**0.5), SEED)    
        for fold_1 in range(int(Kfolds**0.5)):
            for fold_2 in range(int(Kfolds**0.5)):
                c_train, p_train = c_train_clusters[fold_1], p_train_clusters[fold_2]
                c_test, p_test = c_test_clusters[fold_1], p_test_clusters[fold_2]

                Train_idxs, Test_idxs = [], []
                for ele in range(n_sample):
                    if C_cluster_dict[cid_list[ele]] in c_train and P_cluster_dict[pid_list[ele]] in p_train:
                        Train_idxs.append(ele)
                    elif C_cluster_dict[cid_list[ele]] in c_test and P_cluster_dict[pid_list[ele]] in p_test:
                        Test_idxs.append(ele)
                train_idx_list.append(Train_idxs)
                test_idx_list.append(Test_idxs)
            
    return data_pack, train_idx_list, test_idx_list

def split_train_val(train_val_data, train_rate, SEED):
    np.random.seed(SEED)
    train_val_data = train_val_data.reindex(np.random.permutation(train_val_data.index))

    num = len(train_val_data)
    train_len = int(train_rate*num)
    
    train_data = train_val_data.iloc[:train_len,:]
    val_data = train_val_data.iloc[train_len:,:]
    
    return train_data, val_data

if __name__ == '__main__':
    
    cluster_cross_validation_setting = 'compound_clustering_unseen' #  ['compound_clustering_unseen', 'protein_clustering_unseen', 'both_unseen']
    DATASET = 'BindingDB'
    cluster_threshold = 0.3
    train_rate = 0.89  
    root_path = './' + DATASET + '/'

    file_path = root_path + 'files_preparation/'
    save_path = root_path + cluster_cross_validation_setting + '/'
    os.makedirs(save_path, exist_ok = True)
    Kfolds = 5
    SEED = 1234
    
    data_pack, Train_idx_list, Test_idx_list = load_data(cluster_cross_validation_setting, file_path, cluster_threshold, Kfolds, SEED)
    
    for fold in range(Kfolds):
        print('fold ' + str(fold) + ' spliting')
        if fold < len(Train_idx_list):
            
            train_idx, test_idx = Train_idx_list[fold], Test_idx_list[fold]
            
            train_val_data = data_from_index(data_pack, train_idx)
            test_data = data_from_index(data_pack, test_idx)

            train_data, val_data = split_train_val(train_val_data, train_rate, SEED)
            
            train_data[['SMILES','Sequence','Label']].to_csv(save_path + str(fold) + 'fold_' +  str(cluster_threshold) + '_train.txt',sep=' ',index=False,header=None)
            val_data[['SMILES','Sequence','Label']].to_csv(save_path + str(fold) + 'fold_'  + str(cluster_threshold) +'_val.txt',sep=' ',index=False,header=None)
            test_data[['SMILES','Sequence','Label']].to_csv(save_path + str(fold) + 'fold_' + str(cluster_threshold) +'_test.txt',sep=' ',index=False,header=None)
