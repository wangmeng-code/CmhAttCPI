from collections import defaultdict
import pandas as pd
import os
import pickle
import sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist

def get_mol_dict(mol_dict_path):
    with open(mol_dict_path + 'files_preparation/mols_dict','rb') as f:
        mol_dict = pickle.load(f)   
    return mol_dict


def get_fps(mol_list):
    fps = []
    for mol in mol_list:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=1024,useChirality=True)
        fps.append(fp)
    return fps


def calculate_sims(fps1,fps2,simtype='tanimoto'):
    sim_mat = np.zeros((len(fps1),len(fps2))) 
    for i in range(len(fps1)):
        fp_i = fps1[i]
        if simtype == 'tanimoto':
            sims = DataStructs.BulkTanimotoSimilarity(fp_i,fps2)
        elif simtype == 'dice':
            sims = DataStructs.BulkDiceSimilarity(fp_i,fps2)
        sim_mat[i,:] = sims
    return sim_mat


def compound_cluster(ligand_ls, mol_ls, thred_ls, root_path):
    print('compound clustering...')
    compound_cluster_path = root_path + 'files_preparation/'
    os.makedirs(compound_cluster_path, exist_ok = True)

    fps = get_fps(mol_ls)
    sim_mat = calculate_sims(fps, fps)
    
    C_dist = pdist(fps, 'jaccard')
    C_link = single(C_dist)

    for thred in thred_ls:
        print('thred:'+str(thred))
        C_clusters = fcluster(C_link, thred, 'distance') 
        len_list = []
        for i in range(1,max(C_clusters)+1):
            len_list.append(C_clusters.tolist().count(i))
        print('threshold:', thred, ' total num of compounds:', len(ligand_ls), ' num of clusters:', max(C_clusters), ' max length:', max(len_list))
        C_cluster_dict = {ligand_ls[i]:C_clusters[i] for i in range(len(ligand_ls))}

        with open(compound_cluster_path + 'compound_cluster_dict_' + str(thred),'wb') as f:
            pickle.dump(C_cluster_dict, f)

def protein_cluster(protein_ls, idx_ls, thred_ls, prot_sim_mat, root_path):
    print('protein clustering...')
    
    protein_cluster_path = root_path + 'files_preparation/'
    os.makedirs(protein_cluster_path, exist_ok = True)
    
    sim_mat = prot_sim_mat[idx_ls, :]
    sim_mat = sim_mat[:, idx_ls]

    print('original protein sim_mat:', prot_sim_mat.shape)
    
    P_dist = []
    for i in range(sim_mat.shape[0]):
        P_dist += (1-sim_mat[i,(i+1):]).tolist()
    P_dist = np.array(P_dist)
    P_link = single(P_dist)
    
    for thred in thred_ls:
        P_clusters = fcluster(P_link, thred, 'distance')
        len_list = []
        for i in range(1,max(P_clusters)+1):
            len_list.append(P_clusters.tolist().count(i))
        print('threshold:', thred, ' total num of proteins:', len(protein_ls), ' num of clusters:', max(P_clusters), ' max length:', max(len_list))
        P_cluster_dict = {protein_ls[i]:P_clusters[i] for i in range(len(protein_ls))}
        
        with open(protein_cluster_path + 'protein_cluster_dict_' + str(thred),'wb') as f:
            pickle.dump(P_cluster_dict, f)

def pickle_dump(dictionary, file_name):
    pickle.dump(dict(dictionary), open(file_name, 'wb'))

if __name__ == "__main__":
    DATASET = 'BindingDB'
    root_path = './data/'
    
    print('Step 1/3, loading preparation files...')
    mol_dict = get_mol_dict(root_path)

    print('Step 2/3, generating cpi information...')
    CPI_info_dict = {}
    CPI_data = pd.read_csv(root_path + DATASET +'_formulated.csv')
    
    for i in range(len(CPI_data)):
        print('processed sample {}/{}'.format(i, len(CPI_data)))
        SMILES, ligand_name, sequence, target_name,  label = \
        CPI_data['SMILES'][i], CPI_data['Ligand_name'][i], CPI_data['Sequence'][i], CPI_data['Target_name'][i], CPI_data['Label'][i]

        if ligand_name not in mol_dict:
            print('ligand not in mol_dict')
            continue
        mol = mol_dict[ligand_name]
        
        if ligand_name +'--' + target_name not in CPI_info_dict:
            CPI_info_dict[ligand_name +'--'+target_name] = [SMILES, ligand_name, mol, sequence, target_name, label]
            
    label_ls = []
    ligand_name_ls = []
    target_name_ls = []
    SMILES_ls = []
    Sequence_ls = []
       
    # get inputs
    for item in CPI_info_dict:
        SMILES, ligand_name, mol, sequence, target_name, label = CPI_info_dict[item]

        label_ls.append(label)
        ligand_name_ls.append(ligand_name)
        target_name_ls.append(target_name)
        SMILES_ls.append(SMILES)
        Sequence_ls.append(sequence)
    
    data_pack = [np.array(SMILES_ls).astype(object), np.array(ligand_name_ls).astype(object), np.array(Sequence_ls).astype(object), np.array(target_name_ls).astype(object), np.array(label_ls)]

    with open(root_path + 'files_preparation/all_combined_input', 'wb') as f:
        pickle.dump(data_pack, f)
    
    print('Step 3/3, clustering...')
    ligand_ls = list(set(ligand_name_ls))
    protein_ls = list(set(target_name_ls))
    
    # clustering
    thred_ls = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    # compound clustering
    mol_ls = [mol_dict[ligand] for ligand in ligand_ls]
    compound_cluster(ligand_ls, mol_ls, thred_ls, root_path)

    # protein clustering
    prot_sim_mat = np.load(root_path + 'files_preparation/sim_mat.npy').astype(np.float32)
    ori_protein_list = np.load(root_path + 'files_preparation/proteins_list.npy', allow_pickle=True).tolist()
    idx_ls = [ori_protein_list.index(pid) for pid in protein_ls]
    protein_cluster(protein_ls, idx_ls, thred_ls, prot_sim_mat, root_path)
    