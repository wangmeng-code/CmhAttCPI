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
    
    atom_symbols = []
    for atom in mol.GetAtoms():
        atom_symbols.append(atom.GetSymbol())
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

    return fatoms, fbonds, atom_nb, bond_nb, num_nbs_mat, atom_symbols


def seq_code(sequence, ngram):
    sequence = '-' + sequence + '='
    words1 = []
    words2 = []
    for i in range(len(sequence)-ngram+1):
        if sequence[i:i+ngram] not in word_dict.keys():
            words1.append(0)
        else:
            words2.append(word_dict[sequence[i:i+ngram]])
    words = words1+words2
    
    return np.array(words, np.int32)


def generate_input(data):
    mol_inputs, seq_inputs, label_list = [], [], []
    Atom_Symbols = []
    Residue_Symbols = []
    
    # get inputs
    for i in range(len(data)):
        print('converting graph {}/{}'.format(i+1, len(data)))
        smile, seq, label = data.iloc[i, :].tolist()
        fa, fb, anb, bnb, nbs_mat, atom_symbols = Mol2Graph(Chem.MolFromSmiles(smile))
        
        mol_inputs.append([fa, fb, anb, bnb, nbs_mat])
        seq_inputs.append(seq_code(seq,ngram=3))
        label_list.append(label)
        Atom_Symbols.append(atom_symbols)
        Residue_Symbols.append([x for x in seq])
    # get data pack
    fa_list, fb_list, anb_list, bnb_list, nbs_mat_list = zip(*mol_inputs)
    data_pack = [np.array(fa_list), np.array(fb_list), np.array(anb_list), np.array(bnb_list), np.array(nbs_mat_list), \
                np.array(seq_inputs), np.array(label_list)]
    
    return data_pack, Atom_Symbols, Residue_Symbols


def pickle_dump(dictionary, file_name):
    pickle.dump(dict(dictionary), open(file_name, 'wb'), protocol=0)

if __name__ == "__main__":
    
    save_path = './analysis_result/'
    os.makedirs(save_path, exist_ok=True)
    
    f = open('atom_dict','rb')
    atom_dict = pickle.load(f)
    f = open('bond_dict','rb')
    bond_dict = pickle.load(f)
    f = open('word_dict','rb')
    word_dict = pickle.load(f)
    # word = defaultdict(lambda: len(word_dict)+len(word)) 

    test_data = pd.read_csv('test.txt', sep=' ', header=None)

    test_data.columns = ['compound_iso_smiles','target_sequence','label']
    
    print('testing dataset generating inputs...')
    test_data_pack, Atom_Symbols, Residue_Symbols = generate_input(test_data)
    
    with open('./test_input', 'wb') as f:
        pickle.dump(test_data_pack, f, protocol=0)
    
    
    atom_df = pd.DataFrame(Atom_Symbols).T
    seq_df = pd.DataFrame(Residue_Symbols).T

    atom_df.to_csv(save_path+'atom_symbol.csv',index=False, header=None)
    seq_df.to_csv(save_path+'residue_symbol.csv',index=False, header=None)
    # import ipdb; ipdb.set_trace()
    
