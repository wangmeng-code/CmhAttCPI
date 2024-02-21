from collections import defaultdict
import os
import pickle
import sys
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina
from scipy.cluster.hierarchy import fcluster, linkage, single
from scipy.spatial.distance import pdist

63+18+1
elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
aa_list = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1
bond_fdim = 6
max_nb = 6


def onek_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom):
    return np.array(onek_encoding_unk(atom.GetSymbol(), elem_list) 
            + onek_encoding_unk(atom.GetDegree(), [0,1,2,3,4,5]) 
            + onek_encoding_unk(atom.GetExplicitValence(), [1,2,3,4,5,6])
            + onek_encoding_unk(atom.GetImplicitValence(), [0,1,2,3,4,5])
            + [atom.GetIsAromatic()], dtype=np.float32)


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, \
    bt == Chem.rdchem.BondType.AROMATIC, bond.GetIsConjugated(), bond.IsInRing()], dtype=np.float32)


def Mol2Graph(mol):
    # convert molecule to GNN input
    
    idxfunc=lambda x:x.GetIdx()

    n_atoms = mol.GetNumAtoms()
    assert mol.GetNumBonds() >= 0
    
    n_bonds = max(mol.GetNumBonds(), 1)
    fatoms = np.zeros((n_atoms,), dtype=np.int32)  
    fbonds = np.zeros((n_bonds,), dtype=np.int32) 
    atom_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)   
    bond_nb = np.zeros((n_atoms, max_nb), dtype=np.int32)    
    num_nbs = np.zeros((n_atoms,), dtype=np.int32)
    num_nbs_mat = np.zeros((n_atoms,max_nb), dtype=np.int32) 
    
    for atom in mol.GetAtoms():
        idx = idxfunc(atom)
        fatoms[idx] = atom_dict[''.join(str(x) for x in atom_features(atom).astype(int).tolist())] 
    
    for bond in mol.GetBonds():
        
        a1 = idxfunc(bond.GetBeginAtom())
        a2 = idxfunc(bond.GetEndAtom())
        idx = bond.GetIdx()
        fbonds[idx] = bond_dict[''.join(str(x) for x in bond_features(bond).astype(int).tolist())] 
        try:
            atom_nb[a1,num_nbs[a1]] = a2
            atom_nb[a2,num_nbs[a2]] = a1
        except:
            return [], [], [], [], []
        bond_nb[a1,num_nbs[a1]] = idx
        bond_nb[a2,num_nbs[a2]] = idx
        num_nbs[a1] += 1
        num_nbs[a2] += 1
    
    for i in range(len(num_nbs)):
        num_nbs_mat[i,:num_nbs[i]] = 1

    return fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat


def seq_code(sequence, ngram):
    sequence = '-' + sequence + '='
    words = [word_dict[sequence[i:i+ngram]] for i in range(len(sequence)-ngram+1)]
    
    return np.array(words, np.int32)


def generate_input(data):
    mol_inputs, seq_inputs, label_list = [], [], []
    
    # get inputs
    for i in range(len(data)):
        print('converting graph {}/{}'.format(i+1, len(data)))
        smile, seq, label = data.iloc[i, :].tolist()
        fa, fb, anb, bnb, nbs_mat = Mol2Graph(Chem.MolFromSmiles(smile))
        
        mol_inputs.append([fa, fb, anb, bnb, nbs_mat])
        seq_inputs.append(seq_code(seq,ngram=3))
        label_list.append(label)
    
    # get data pack
    fa_list, fb_list, anb_list, bnb_list, nbs_mat_list = zip(*mol_inputs)
    data_pack = [np.array(fa_list), np.array(fb_list), np.array(anb_list), np.array(bnb_list), np.array(nbs_mat_list), \
                np.array(seq_inputs), np.array(label_list)]
    
    return data_pack

def train_val_split(train_val_data, SEED):
    np.random.seed(SEED)

    train_val_data = train_val_data.reindex(np.random.permutation(train_val_data.index))
    # train_rate = 0.875
    train_rate = 0.89
    num = len(train_val_data)
    train_len = int(train_rate*num)
    
    train_data = train_val_data.iloc[:train_len,:]
    val_data = train_val_data.iloc[train_len:,:]
    
    return train_data, val_data


def pickle_dump(dictionary, file_name):
    pickle.dump(dict(dictionary), open(file_name, 'wb'), protocol=0)

if __name__ == "__main__":
    print('*'*20 + ' Preprocessing data... ' + '*'*20)
    
    scenario = sys.argv[1]
    if scenario == 'cross_validation':
        kfold_data_path, save_preprocessed_data_path = sys.argv[2:]

    elif scenario == 'unknown_setting':
        kfold_data_path, save_preprocessed_data_path, cluster_threshold = sys.argv[2:]

    Kfolds = 5
    SEED = 1234

    for fold in range(Kfolds):
        if scenario == 'cross_validation':
            os.makedirs(save_preprocessed_data_path, exist_ok=True)
            print(str(fold) + 'fold data preparing...' )

            train_data = pd.read_csv(kfold_data_path  + str(fold) +  '_train.txt',sep=' ',header=None)
            train_data.columns = ['compound_iso_smiles','target_sequence','label']
        
            val_data = pd.read_csv(kfold_data_path  + str(fold) + '_val.txt',sep=' ',header=None)
            val_data.columns = ['compound_iso_smiles','target_sequence','label']
            
            test_data = pd.read_csv(kfold_data_path  + str(fold) + '_test.txt',sep=' ',header=None)
            test_data.columns = ['compound_iso_smiles','target_sequence','label']

        elif scenario == 'unknown_setting':
            threshold = cluster_threshold
            os.makedirs(save_preprocessed_data_path, exist_ok=True)

            print(str(fold) + 'fold data preparing...' )
            train_data = pd.read_csv(kfold_data_path + str(fold) +  'fold_' + str(threshold) + '_train.txt',sep=' ',header=None)
            train_data.columns = ['compound_iso_smiles','target_sequence','label']
        
            val_data = pd.read_csv(kfold_data_path + str(fold) +  'fold_' + str(threshold) + '_val.txt',sep=' ',header=None)
            val_data.columns = ['compound_iso_smiles','target_sequence','label']
            
            test_data = pd.read_csv(kfold_data_path + str(fold) +  'fold_' + str(threshold) + '_test.txt',sep=' ',header=None)
            test_data.columns = ['compound_iso_smiles','target_sequence','label']

        # initialize feature dicts
        atom_dict = defaultdict(lambda: len(atom_dict))
        bond_dict = defaultdict(lambda: len(bond_dict))
        word_dict = defaultdict(lambda: len(word_dict))

        print('training dataset generating inputs...')
        train_data_pack = generate_input(train_data)
        
        print('validation dataset generating inputs...')
        val_data_pack = generate_input(val_data)

        print('testing dataset generating inputs...')
        test_data_pack = generate_input(test_data)
        
        with open(save_preprocessed_data_path + str(fold) + 'fold_train_input', 'wb') as f:
            pickle.dump(train_data_pack, f, protocol=0)

        with open(save_preprocessed_data_path + str(fold) + 'fold_val_input', 'wb') as f:
            pickle.dump(val_data_pack, f, protocol=0)

        with open(save_preprocessed_data_path + str(fold) + 'fold_test_input', 'wb') as f:
            pickle.dump(test_data_pack, f, protocol=0)
        
        pickle_dump(atom_dict, save_preprocessed_data_path + str(fold) + 'fold_atom_dict')
        pickle_dump(bond_dict, save_preprocessed_data_path + str(fold) + 'fold_bond_dict')
        
        word_dict['unk'] = len(word_dict)
        pickle_dump(word_dict, save_preprocessed_data_path + str(fold) + 'fold_word_dict')
