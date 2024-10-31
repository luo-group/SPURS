
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
from .utils import fermi_transform,tied_featurize,get_pdb,parse_pdb
import lmdb
import glob
from spurs import utils
log = utils.get_logger(__name__)
import json

ALPAHBET = 'ACDEFGHIKLMNPQRSTVWYX'

class FireProtDataset(torch.utils.data.Dataset):

    def __init__(self,split):

        self.split = split
        root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),'../../../')
        filename = os.path.join(root_path,"data/dataset/fireprot/fireprot_upload/csvs/4_fireprotDB_bestpH.csv")

        df = pd.read_csv(filename).dropna(subset=['ddG'])
        df = df.where(pd.notnull(df), None)

        self.seq_to_data = {}
        seq_key = "pdb_sequence"

        for wt_seq in df[seq_key].unique():
            self.seq_to_data[wt_seq] = df.query(f"{seq_key} == @wt_seq").reset_index(drop=True)

        self.df = df

        # load splits produced by mmseqs clustering
        with open(os.path.join(root_path,"data/dataset/fireprot/fireprot_upload/csvs/fireprot_splits.pkl"), 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split
            
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "homologue-free": [],
            "all": []
        }

        self.wt_seqs = {}
        self.mut_rows = {}

        if self.split == 'all':
            all_names = list(splits.values())
            all_names = [j for sub in all_names for j in sub]
            self.split_wt_names[self.split] = all_names
        else:
            self.split_wt_names[self.split] = splits[self.split]

        self.wt_names = self.split_wt_names[self.split]

        for wt_name in self.wt_names:
            self.mut_rows[wt_name] = df.query('pdb_id_corrected == @wt_name').reset_index(drop=True)
            self.wt_seqs[wt_name] = self.mut_rows[wt_name].pdb_sequence[0]

        structure_path = os.path.join(root_path,"data/dataset/fireprot/fireprot_upload/pdbs/")
        # structure_path_lmdb = os.path.join(structure_path,"../parsed_structure.lmdb")
        structure_path_json = os.path.join(structure_path,"../parsed_structure.json")
        self.structure_path = structure_path
        
        parse_pdb(structure_path,structure_path_json)
                
        log.info("loading structure dataset")
        with open(structure_path_json, 'r') as file:
            self.json_dataset = json.load(file)
    
    def __len__(self):
        return len(self.wt_names)

    def __getitem__(self, index):

        wt_name = self.wt_names[index]
        seq = self.wt_seqs[wt_name]
        data = self.seq_to_data[seq]

        pdb = self.json_dataset[data.pdb_id_corrected[0]]
        protein = get_pdb(pdb,seq,wt_name,check_assert=False)

        mutations = []
        for i, row in data.iterrows():

            pdb_idx = row.pdb_position
            assert pdb['seq'][pdb_idx] == row.wild_type == row.pdb_sequence[row.pdb_position]
            # there might be '-' in  pdb['seq']
            # replace them with X
            pdb['seq'] = pdb['seq'].replace('-','X')
            
            wt = row.wild_type
            mut = row.mutation
            
            ddG = None if row.ddG is None or isnan(row.ddG) else torch.tensor([row.ddG], dtype=torch.float32)
            wt_onehot = torch.zeros((21))
            wt_onehot[ALPAHBET.index(wt)] = 1
            mt_onehot = torch.zeros((21))
            mt_onehot[ALPAHBET.index(mut)] = 1
            append_tensor = torch.cat([wt_onehot,mt_onehot])
            append_tensor = append_tensor.float()
            
            protein['mut_ids'].append(pdb_idx)
            protein['ddG'].append(ddG)
            protein['append_tensors'].append(append_tensor)
        protein['ddG'] = torch.stack(protein['ddG'])
        protein['append_tensors'] = torch.stack(protein['append_tensors'])
        protein['pdb_path'] = self.structure_path
        protein['dataset'] = 'fHF'

        return protein