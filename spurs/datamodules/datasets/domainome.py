import torch
from torch.utils.data import ConcatDataset
import pandas as pd
import numpy as np
import pickle
import os
from Bio import pairwise2
from math import isnan
from tqdm import tqdm
from dataclasses import dataclass
from typing import Optional
from .utils import fermi_transform,tied_featurize,get_pdb,parse_pdb,alt_parse_PDB
import lmdb
import glob
from spurs import utils
log = utils.get_logger(__name__)
import json
from .LMDBDataset import LMDBDataset
from .batch import CoordBatchConverter
from .data_utils import Alphabet
from collections import defaultdict
import math
ALPAHBET = 'ACDEFGHIKLMNPQRSTVWYX'
class domainome(torch.utils.data.Dataset):
    """A dataset class for handling domain-level protein stability data.

    This dataset provides domain-specific stability measurements and is used for
    analyzing and predicting stability changes at the protein domain level.
    It includes domain-level annotations and corresponding stability measurements.

    Args:
        pdb_dir (str): Directory containing PDB structure files.
        csv_fname (str): Path to the CSV file containing domain mutation data.
        dataset_name (str): Name of the dataset.
        stage (str, optional): Dataset stage ('full', 'train', 'test'). Defaults to 'full'.
        mut_seq (bool, optional): Whether to include mutated sequences. Defaults to False.
        train_size (float, optional): Training data size ratio. Defaults to 1.
    """

    def __init__(self, pdb_dir, csv_fname, dataset_name, stage='full',mut_seq=False,train_size=1):

        self.pdb_dir = pdb_dir
        df = pd.read_csv(csv_fname)
        df = df.dropna(subset=['aPCA_fitness'])
        self.df = df
        self.dataset_name = dataset_name

        self.wt_seqs = {}
        self.mut_rows = {}

        self.wt_names = df.domain_ID.unique()


        self.wt_names = [x for x in self.wt_names if str(x) != 'nan']
    
        for wt_name in self.wt_names:
            wt_name_query = wt_name
            self.mut_rows[wt_name] = df.query('domain_ID == @wt_name_query').reset_index(drop=True)

        len_arr = [len(self.mut_rows[wt_name]) for wt_name in self.wt_names]

        self.json_dataset = defaultdict(lambda: defaultdict(lambda: -1))


    def __len__(self):
        return len(self.wt_names)

    
    def _get_wt_item(self, index):

        """Batch retrieval fxn - each batch is a single protein"""

        wt_name = self.wt_names[index]
        chain = 'A'
        
        mut_data = self.mut_rows[wt_name]


        # modified PDB parser returns list of residue IDs so we can align them easier
  
        if isinstance(self.json_dataset[wt_name][chain[0]],int):
            pdb = alt_parse_PDB(os.path.join(self.pdb_dir,wt_name+".pdb"),chain)
            self.json_dataset[wt_name][chain[0]] = pdb
        pdb = self.json_dataset[wt_name][chain[0]]
        resn_list = pdb[0]["resn_list"]


        protein = get_pdb(pdb[0], wt_name, wt_name, check_assert=False)
        
        
        for i, row in mut_data.iterrows():
            mut_info = row.variant_ID.split('_')[-1]
            wtAA, mutAA = mut_info[0], mut_info[-1]

            pdb_idx = row.pdb_pos

            try:

                assert pdb[0]['seq'][pdb_idx] == wtAA
                
            except AssertionError:  # contingency for mis-alignments
                assert False       
                
            wt = wtAA
            mut = mutAA
            ddG = torch.tensor([row.aPCA_normalized_fitness * -1.], dtype=torch.float32)
            wt_onehot = torch.zeros((21))
            wt_onehot[ALPAHBET.index(wt)] = 1
            mt_onehot = torch.zeros((21))
            mt_onehot[ALPAHBET.index(mut)] = 1
            append_tensor = torch.cat([wt_onehot,mt_onehot])
            append_tensor = append_tensor.float()

            protein['mut_ids'].append(pdb_idx)
            protein['ddG'].append(ddG)
            protein['append_tensors'].append(append_tensor)
            
        if len(protein['ddG'])==0:
            protein['mut_ids'] = [1]
            protein['ddG'] = [torch.tensor([0.0])]
            protein['append_tensors'] = [torch.zeros((42)).float()]
        protein['mut_ids'] = torch.LongTensor(protein['mut_ids'])
        protein['ddG'] = torch.stack(protein['ddG'])
        protein['append_tensors'] = torch.stack(protein['append_tensors'])

        protein['dataset'] = self.dataset_name+wt_name

        return protein
        
    def __getitem__(self, index):
        return self._get_wt_item(index)