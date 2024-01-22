#!/usr/bin/env python
# coding=utf-8
import os
import pandas as pd
import numpy as np
import pickle
import rdkit
from rdkit import Chem
from ssw import *

## load data
def data_formula(data_path, DATASET):
    df = pd.read_csv(data_path + DATASET + '.txt',sep=' ',header=None)
    df.columns = ['SMILES', 'Sequence', 'Label']
    df = df.drop_duplicates()
    df = df.reset_index(drop=True)

    compounds = pd.DataFrame(set(list(df['SMILES'])))
    compounds.columns = ['SMILES']
    compounds['Ligand_name'] = pd.DataFrame(['Ligand_'+str(x+1) for x in range(len(compounds))])

    proteins = pd.DataFrame(set(list(df['Sequence'])))
    proteins.columns = ['Sequence']
    proteins['Target_name'] = pd.DataFrame(['Target_'+str(x+1) for x in range(len(proteins))])

    data = pd.merge(df, compounds, left_on='SMILES', right_on='SMILES')
    data = pd.merge(data, proteins, left_on='Sequence', right_on='Sequence')
    data.to_csv(data_path + DATASET + '_formulated.csv', index=False)
    return data

def mol_convert(x):
    mol = Chem.MolFromSmiles(x)
    return mol

def ligand_files_preparation(save_path, formulated_data):
    """create files of ligands"""
    print('files of ligands preparating...')
    print('1/2 creating dict...')
    compounds_dict = formulated_data.set_index(['Ligand_name'])['SMILES'].to_dict()
    formulated_data['mol'] = formulated_data['SMILES'].apply(lambda x: mol_convert(x))
    mols_dict = formulated_data.set_index(['Ligand_name'])['mol'].to_dict()

    # save dict
    print('2/2 saving dict...')
    with open(save_path + 'compounds_dict','wb') as f:
        pickle.dump(compounds_dict,f)

    with open(save_path + 'mols_dict','wb') as f:
        pickle.dump(mols_dict,f)

    print('files of ligands have finished!')

def protein_files_preparation(save_path, formulated_data, gap_open_penalty, gap_extend_penalty, match_score, mismatch_score):
    """ create files of proteins """
    print('files of proteins preparating...')
    print('1/3 creating dict...')
    
    proteins_dict = formulated_data.set_index(['Target_name'])['Sequence'].to_dict() 
   
    # create similarity matrix
    print('2/3 creating similarity matrix...')
    scores = np.zeros(shape=(len(proteins_dict), len(proteins_dict)))
    proteins = formulated_data[['Target_name', 'Sequence']].drop_duplicates()

    for i in range(len(proteins)):
        score_ls = []
        for j in range(len(proteins)):
            sw_score = calculate_sw(proteins.iloc[i,1],proteins.iloc[j,1], gap_open_penalty, gap_extend_penalty, match_score, mismatch_score)
            score_ls.append(sw_score)
        scores[i] = score_ls
        print(i)
    
    print('3/3 saving all files of proteins...')
    with open(save_path + 'proteins_dict','wb') as f:
        pickle.dump(proteins_dict,f)
    
    np.save(save_path+'proteins_list.npy', list(proteins['Target_name']))
    np.save(save_path+'sim_mat.npy',scores)

    print('files of proteins has finished!')

if __name__ == '__main__':
    DATASET = 'BindingDB'  # ['BindingDB', 'DrugBank', 'GPCR', 'Davis','user-dataset']
    
    gap_open_penalty = 2
    gap_extend_penalty = 1
    match_score = 3
    mismatch_score = -1
    data_path = '../dataset/' + DATASET + '/'
    
    save_cluster_path = './' +  DATASET + '/files_preparation/'
    
    os.makedirs(save_cluster_path, exist_ok=True)

    formulated_data = data_formula(data_path, DATASET)
    formulated_data.to_csv(save_cluster_path +  DATASET + '_formulated.csv',index=False)
    
    ligand_files_preparation(save_cluster_path, formulated_data)
    protein_files_preparation(save_cluster_path, formulated_data, gap_open_penalty, gap_extend_penalty, match_score, mismatch_score)
    

