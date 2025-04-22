
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
from collections import defaultdict
import math
import threading
ALPAHBET = 'ACDEFGHIKLMNPQRSTVWYX'

class ddgGeo(torch.utils.data.Dataset):

    def __init__(self, pdb_dir, csv_fname, dataset_name, stage='full',mut_seq=False,train_size=1):

        self.pdb_dir = pdb_dir
        df = pd.read_csv(csv_fname)
        self.df = df
        self.dataset_name = dataset_name

        self.wt_seqs = {}
        self.mut_rows = {}
        df.PDB = df.PDB+df.chain

        self.wt_names = df.PDB.unique()
        print(len(self.wt_names))
        # del nan
        self.wt_names = [x for x in self.wt_names if str(x) != 'nan']
    
        for wt_name in self.wt_names:
            wt_name_query = wt_name
            self.mut_rows[wt_name] = df.query('PDB == @wt_name_query').reset_index(drop=True)
            if 'ssym' in self.pdb_dir:
                self.wt_seqs[wt_name] = self.mut_rows[wt_name].SEQ[0]

        len_arr = [len(self.mut_rows[wt_name]) for wt_name in self.wt_names]
        wt_len = sum(len_arr)
        log.info(f"Dataset {dataset_name} has {len(self.wt_names)} proteins and {wt_len} mutations")
        self.json_dataset = defaultdict(lambda: defaultdict(lambda: -1))
        
        self.mut_seq = mut_seq
        self.fake_bs = 32 if stage=='train' else 10000
        if self.mut_seq:
            self.index_list = []
            self.start_index = []
            self.proteins = [self._get_wt_item(i) for i in tqdm(range(len(self.wt_names)))]
            mut_numbers = []
            for i in range(len(self.wt_names)):
                mut_numbers.append(len(self.mut_rows[self.wt_names[i][:4]]))
            mut_numbers = [math.ceil(i / self.fake_bs) for i in mut_numbers]
            for cur_protein,num in enumerate(mut_numbers):
                self.index_list+=[cur_protein]*num
                self.start_index+=[i*self.fake_bs for i in range(num)]
                
            self.dataset_len = sum(mut_numbers)

            self.mut_numbers = mut_numbers
        self.protein_index_list = [np.arange(i) for i in len_arr]

        
    def __len__(self):
        return len(self.wt_names) if not self.mut_seq else self.dataset_len
    
    
    def _get_wt_item(self, index):

        """Batch retrieval fxn - each batch is a single protein"""

        wt_name = self.wt_names[index]
        chain = [wt_name[-1]]

        wt_name = wt_name.split(".pdb")[0]
        mut_data = self.mut_rows[wt_name]
        wt_name = wt_name[:-1]

        # modified PDB parser returns list of residue IDs so we can align them easier
  
        if isinstance(self.json_dataset[wt_name][chain[0]],int):
            pdb = alt_parse_PDB(os.path.join(self.pdb_dir,wt_name+".pdb"),chain)
            self.json_dataset[wt_name][chain[0]] = pdb
        pdb = self.json_dataset[wt_name][chain[0]]
        resn_list = pdb[0]["resn_list"]


        protein = get_pdb(pdb[0], wt_name, wt_name, check_assert=False)
    
        if  self.mut_seq:
            protein['S'] = [protein['S']]
        
        
        for i, row in mut_data.iterrows():
            mut_info = row.MUT
            wtAA, mutAA = mut_info[0], mut_info[-1]
            try:
                pos = mut_info[1:-1]
                pdb_idx = resn_list.index(pos)
            except ValueError:  # skip positions with insertion codes for now - hard to parse

                continue
            try:
                assert pdb[0]['seq'][pdb_idx] == wtAA
            except AssertionError:  # contingency for mis-alignments
                # if gaps are present, add these to idx (+10 to get any around the mutation site, kinda a hack)
                if 'S669' in self.pdb_dir:
                    gaps = [g for g in pdb[0]['seq'] if g == '-']
                else:
                    gaps = [g for g in pdb[0]['seq'][:pdb_idx + 10] if g == '-']                

                if len(gaps) > 0:
                    pdb_idx += len(gaps)
                else:
                    pdb_idx += 1

                if pdb_idx is None:
                    continue
                if pdb[0]['seq'][pdb_idx] != wtAA :

                    continue

            if  self.mut_seq:
                pdb_seq_old = pdb[0]['seq']
                pdb[0]['seq'] = pdb[0]['seq'][:pdb_idx] + mutAA + pdb[0]['seq'][pdb_idx + 1:]
                protein['mut_seq'].append(pdb[0]['seq'])
                mut_protein = get_pdb(pdb[0], wt_name, wt_name, check_assert=False)
                protein['S'].append(mut_protein['S'])
                pdb[0]['seq'] = pdb_seq_old
                
                
                
            wt = wtAA
            mut = mutAA
            if 'DTM' in row:
                ddG = torch.tensor([row.DTM * -1.], dtype=torch.float32)
            else:
                ddG = None if row.DDG is None or isnan(row.DDG) else torch.tensor([row.DDG * -1.], dtype=torch.float32)
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
            
        if  self.mut_seq:
            protein['S'] = torch.cat(protein['S'],dim=0).clone()
            protein['X'] = protein['X'].expand(len(protein['S']),-1,-1,-1).clone()
            protein['mask'] = protein['mask'].expand(len(protein['S']),-1).clone()
            protein['chain_M'] = protein['chain_M'].expand(len(protein['S']),-1).clone()
            protein['chain_M_chain_M_pos'] = protein['chain_M_chain_M_pos'].expand(len(protein['S']),-1).clone()
            protein['residue_idx'] = protein['residue_idx'].expand(len(protein['S']),-1).clone()
            protein['chain_encoding_all'] = protein['chain_encoding_all'].expand(len(protein['S']),-1).clone()
            protein['randn_1'] = protein['randn_1'].expand(len(protein['S']),-1).clone()
            
        protein['ddG'] = torch.stack(protein['ddG'])
        protein['append_tensors'] = torch.stack(protein['append_tensors'])

        protein['dataset'] = self.dataset_name

        return protein
        
    def __getitem__(self, index):
        return self._get_wt_item(index) 