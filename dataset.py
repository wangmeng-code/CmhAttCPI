
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from utils import *

class ProcessDataSet(Dataset):
    def __init__(self, data_pack):
        self.data_pack = data_pack
    
    def __len__(self):
        return len(self.data_pack[0])

    def __getitem__(self, idx):
        vertex, edge, atom_adj, bond_adj, num_nbs, seq, label = [self.data_pack[data_idx][idx] for data_idx in range(7)]    
        
        return [vertex, edge, atom_adj, bond_adj, num_nbs, seq, label]
        
def Collate_fn(batch_data):
    
    vertex, edge, atom_adj, bond_adj, nbs, sequence, label = [[data[data_idx] for data in batch_data] for data_idx in range(7)]
    vertex_mask = get_mask(vertex)
    
    vertex = pack1D(vertex)    # [30, 49]
    edge = pack1D(edge)  # [30, 49]

    atom_adj = pack2D(atom_adj)   # [30, 49, 6]
    bond_adj = pack2D(bond_adj)   # [30, 49, 6]
    nbs_mask = pack2D(nbs)

    # pad proteins and make masks
    seq_mask = get_mask(sequence)
    sequence = pack1D(sequence)
    
    #add index
    atom_adj = add_index(atom_adj, np.shape(atom_adj)[1])
    bond_adj = add_index(bond_adj, np.shape(edge)[1])

    #convert to torch cuda data type
    vertex_mask = torch.FloatTensor(vertex_mask)
    vertex = torch.LongTensor(vertex)
    edge = torch.LongTensor(edge)
    atom_adj = torch.LongTensor(atom_adj)
    bond_adj = torch.LongTensor(bond_adj)
    nbs_mask = torch.FloatTensor(nbs_mask)
    
    seq_mask = torch.FloatTensor(seq_mask)
    sequence = torch.LongTensor(sequence)
    label = torch.LongTensor(label)

    return (vertex_mask, vertex, edge, atom_adj, bond_adj, nbs_mask, seq_mask, sequence, label)
   




