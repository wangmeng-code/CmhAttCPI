#!/usr/bin/env python
# coding=utf-8
import pandas as pd
from sklearn.model_selection import KFold
import numpy as np
import os

def split_train_val(train_val_data, train_rate, SEED):
    np.random.seed(SEED)
    train_val_data = train_val_data.reindex(np.random.permutation(train_val_data.index))

    num = len(train_val_data)
    train_len = int(train_rate*num)
    
    train_data = train_val_data.iloc[:train_len,:]
    val_data = train_val_data.iloc[train_len:,:]
    
    return train_data, val_data

def split_train_test(data, n_fold, SEED):
    kfolder = KFold(n_splits=n_fold, random_state=SEED, shuffle=True)
    
    train_index = []
    test_index = [] 

    for train, test in kfolder.split(data.iloc[:,:2], data.iloc[:,-1]):
        train_index.append(train)
        test_index.append(test)
    return train_index, test_index

if __name__ == '__main__':
    DATASET = 'Davis'   # ['BindingDB', 'DrugBank', 'GPCR', 'Davis']
    SEED=1234
    n_fold=5
    train_rate = 0.875

    save_path = './' + DATASET + '/5fold_data/'
    os.makedirs(save_path, exist_ok=True)

    Data = pd.read_csv( ATASET + '.txt', sep=' ', header=None)
    
    print('split dataset...')
    # train_index, test_index = split_train_test(Data, 5, SEED)
    train_index = np.load('./' + DATASET + '/5fold_data/kfold_train_index.npy', allow_pickle=True)
    test_index = np.load('./' + DATASET + '/5fold_data/kfold_test_index.npy', allow_pickle=True)
    
    for fold in range(n_fold):
        print(str(fold) + 'fold...')
        train_val_data = Data.iloc[train_index[fold], :]
        test_data = Data.iloc[test_index[fold], :]
        
        train_data, val_data = split_train_val(train_val_data, train_rate, SEED)
        
        train_data.to_csv(save_path + str(fold) + '_train.txt', sep=' ', index=False, header=None)
        val_data.to_csv(save_path + str(fold) + '_val.txt', sep=' ', index=False, header=None)
        test_data.to_csv(save_path + str(fold) + '_test.txt', sep=' ', index=False, header=None)
       
        # np.save(save_path + 'kfold_train_index.npy', train_index)
        # np.save(save_path + 'kfold_test_index.npy', test_index)
   
    
    
