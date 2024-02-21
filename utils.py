import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, precision_score, roc_auc_score
from sklearn.model_selection import KFold
import time
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
#setup_seed(100)

elem_list = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'W', 'Ru', 'Nb', 'Re', 'Te', 'Rh', 'Tc', 'Ba', 'Bi', 'Hf', 'Mo', 'U', 'Sm', 'Os', 'Ir', 'Ce','Gd','Ga','Cs', 'unknown']
atom_fdim = len(elem_list) + 6 + 6 + 6 + 1   # 82   atom.GetSymbol(), elem_list len(elem_list)
                                                    # atom.GetDegree(), [0,1,2,3,4,5]  6
                                                    # atom.GetExplicitValence(), [1,2,3,4,5,6]  6
                                                    # atom.GetImplicitValence(), [0,1,2,3,4,5]  6
                                                    # [atom.GetIsAromatic()]  1
bond_fdim = 6
max_nb = 6

test_atom_fdim = atom_fdim
test_bond_fdim = bond_fdim

#padding functions
def pad_label(arr_list, ref_list):
    N = ref_list.shape[1]
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        n = len(arr)
        a[i,0:n] = arr
    return a


def pad_label_2d(label, vertex, sequence):
    dim1 = vertex.size(1)
    dim2 = sequence.size(1)
    a = np.zeros((len(label), dim1, dim2))
    for i, arr in enumerate(label):
        a[i, :arr.shape[0], :arr.shape[1]] = arr
    return a


def pack1D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        a[i,0:n] = arr
    return a

def pack2D(arr_list):
    N = max([x.shape[0] for x in arr_list])
    M = max_nb#max([x.shape[1] for x in arr_list])
    a = np.zeros((len(arr_list), N, M))
    for i, arr in enumerate(arr_list):
        n = arr.shape[0]
        m = arr.shape[1]
        a[i,0:n,0:m] = arr
    return a

def get_mask(arr_list):
    N = max([x.shape[0] for x in arr_list])
    a = np.zeros((len(arr_list), N))
    for i, arr in enumerate(arr_list):
        a[i,:arr.shape[0]] = 1
    return a

# embedding selection function
def add_index(input_array, ebd_size):
    # import ipdb; ipdb.set_trace()
    batch_size, n_vertex, n_nbs = np.shape(input_array)
    #add_idx = np.array(range(0,(ebd_size)*batch_size,ebd_size)*(n_nbs*n_vertex))
    add_idx = np.array(list(range(0,(ebd_size)*batch_size,ebd_size))*(n_nbs*n_vertex))
    
    add_idx = np.transpose(add_idx.reshape(-1,batch_size))  # [30, 6*49]
    add_idx = add_idx.reshape(-1)   # (30*49*6,)
    new_array = input_array.reshape(-1)+add_idx
    return new_array


def batch_data_process(data, max_atom_num, max_prot_len, device):

    vertex, edge, atom_adj, bond_adj, nbs, sequence = data   
    vertex_mask = get_mask(vertex, 'compound', max_atom_num, max_prot_len)
    vertex = pack1D(vertex, 'compound', max_atom_num, max_prot_len)    # [30, 49]
    edge = pack1D(edge, 'compound', max_atom_num, max_prot_len)  # [30, 49]
    atom_adj = pack2D(atom_adj, max_atom_num)   # [30, 49, 6]
    bond_adj = pack2D(bond_adj, max_atom_num)   # [30, 49, 6]
    nbs_mask = pack2D(nbs, max_atom_num)
    
    #pad proteins and make masks
    seq_mask = get_mask(sequence, 'protein', max_atom_num, max_prot_len)
    sequence = pack1D(sequence, 'protein', max_atom_num, max_prot_len)
    
    #add index
    atom_adj = add_index(atom_adj, np.shape(atom_adj)[1])
    bond_adj = add_index(bond_adj, np.shape(edge)[1])
    
    #convert to torch cuda data type
    vertex_mask = torch.FloatTensor(vertex_mask).to(device)
    vertex = torch.LongTensor(vertex).to(device)
    edge = torch.LongTensor(edge).to(device)
    atom_adj = torch.LongTensor(atom_adj).to(device)
    bond_adj = torch.LongTensor(bond_adj).to(device)
    nbs_mask = torch.FloatTensor(nbs_mask).to(device)
    
    seq_mask = torch.FloatTensor(seq_mask).to(device)
    sequence = torch.LongTensor(sequence).to(device)
    
    return vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence

# load data
def data_from_pack(data_pack):
    fa, fb, anb, bnb, nbs_mat, seq_input = [data_pack[i] for i in range(6)]
    label = data_pack[6]
    return [fa, fb, anb, bnb, nbs_mat, seq_input, label]


def load_data(fold, preprocessed_data_path, dataset):
    with open(preprocessed_data_path +  str(fold) +'fold_' + dataset + '_input','rb') as f:
        data_pack = pickle.load(f)

    return data_pack

def load_test_data(preprocessed_testdata_path):
    with open(preprocessed_testdata_path + 'test_input','rb') as f:
        test_data_pack = pickle.load(f)
    return test_data_pack

def load_dict(fold, preprocessed_data_path):
    f = open(preprocessed_data_path + str(fold) + 'fold_atom_dict','rb')
    atom_dict = pickle.load(f)
    f.close()
    
    f = open(preprocessed_data_path + str(fold) + 'fold_bond_dict','rb')
    bond_dict = pickle.load(f)
    f.close()

    f = open(preprocessed_data_path + str(fold) + 'fold_word_dict','rb')
    word_dict = pickle.load(f)
    f.close()
    
    init_atom_features = np.zeros((len(atom_dict), atom_fdim))
    init_bond_features = np.zeros((len(bond_dict), bond_fdim))
    
    for key,value in atom_dict.items():
        init_atom_features[value] = np.array(list(map(int, key)))
    
    for key,value in bond_dict.items():
        init_bond_features[value] = np.array(list(map(int, key)))
    
    init_atom_features = torch.FloatTensor(init_atom_features)
    init_bond_features = torch.FloatTensor(init_bond_features)
    
    return init_atom_features, init_bond_features, word_dict

def load_test_dict(preprocessed_testdata_path):
    f = open(preprocessed_testdata_path + 'test_atom_dict','rb')
    test_atom_dict = pickle.load(f)
    f.close()
    
    f = open(preprocessed_testdata_path + 'test_bond_dict','rb')
    test_bond_dict = pickle.load(f)
    f.close()

    f = open(preprocessed_testdata_path + 'test_word_dict','rb')
    test_word_dict = pickle.load(f)
    f.close()

    test_init_atom_features = np.zeros((len(test_atom_dict), test_atom_fdim))
    test_init_bond_features = np.zeros((len(test_bond_dict), test_bond_fdim))

    for key,value in test_atom_dict.items():
        test_init_atom_features[value] = np.array(list(map(int, key)))
    
    for key,value in test_bond_dict.items():
        test_init_bond_features[value] = np.array(list(map(int, key)))
    
    test_init_atom_features = torch.FloatTensor(test_init_atom_features)
    test_init_bond_features = torch.FloatTensor(test_init_bond_features)
    
    return test_init_atom_features, test_init_bond_features, test_word_dict
