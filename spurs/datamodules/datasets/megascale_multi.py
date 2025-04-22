
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
from .LMDBDataset import LMDBDataset
from .batch import CoordBatchConverter
from .data_utils import Alphabet
from .fireport import FireProtDataset
from .ddgbench import ddgBenchDataset
from .ddggeo import ddgGeo
from .domainome import domainome
ALPAHBET = 'ACDEFGHIKLMNPQRSTVWYX'
from joblib import Parallel, delayed
from collections import defaultdict
import math
import random
from IPython import embed

class MegaScaleDoubleDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 reduce: str = '',
                 split: str = 'train',
                 ):

        self.split = split
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        root_path = os.path.join(current_dir,'../../../')
        fname = os.path.join(root_path,'data/dataset/megascale/Tsuboyama2023_Dataset2_Dataset3_20230416.csv')
        # only load rows needed to save memory
        df = pd.read_csv(fname, usecols=["ddG_ML", "mut_type", "WT_name", "aa_seq", "dG_ML"])
        # remove unreliable data and more complicated mutations
        df = df.loc[df.ddG_ML != '-', :].reset_index(drop=True)
        # print(len(df.mut_type == 'wt'))
        old_df = df
        df = df.loc[(~df.mut_type.str.contains("ins") & ~df.mut_type.str.contains("del") & df.mut_type.str.contains(":")) | (df.mut_type == 'wt'), :].reset_index(drop=True)
        self.df = df

        
        # dont remove seqs
        if  self.split!='test':
            mmseq_wt_search = os.path.join(root_path,'data/dataset/megascale/mmseq_mut_search_0.25.m8')
            ret = []
            with open(mmseq_wt_search, 'r') as f:
                for line in f.readlines():
                    second_column_value = int(line.split("\t")[1])
                    ret.append(second_column_value)
            # we dont want the rows in the ret
            previous_len = len(df)
            df = df.loc[~df.index.isin(ret), :].reset_index(drop=True)
            cur_len = len(df)
            # log.info(f"removed {previous_len - cur_len} rows from the dataset")
            
        # load splits produced by mmseqs clustering
        with open( os.path.join(root_path,'data/dataset/megascale/mega_splits.pkl'), 'rb') as f:
            splits = pickle.load(f)  # this is a dict with keys train/val/test and items holding FULL PDB names for a given split
        
        self.split_wt_names = {
            "val": [],
            "test": [],
            "train": [],
            "train_s669": [],
            "all": [], 
            "cv_train_0": [],
            "cv_train_1": [],
            "cv_train_2": [],
            "cv_train_3": [],
            "cv_train_4": [],
            "cv_val_0": [],
            "cv_val_1": [],
            "cv_val_2": [],
            "cv_val_3": [],
            "cv_val_4": [],
            "cv_test_0": [],
            "cv_test_1": [],
            "cv_test_2": [],
            "cv_test_3": [],
            "cv_test_4": [],
        }

        self.wt_seqs = {}
        self.mut_rows = {}
        
        if self.split == 'all':
            all_names = np.concatenate([splits['train'],splits['val'],splits['test']])
            self.split_wt_names[self.split] = all_names
        else:
            if reduce == 'prot' and split == 'train':
                n_prots_reduced = 58
                self.split_wt_names[self.split] = np.random.choice(splits["train"], n_prots_reduced)
            else:
                self.split_wt_names[self.split] = splits[self.split]
                # self.split_wt_names[self.split] = ['2K28.pdb', '3DKM.pdb', '2L33.pdb', '1PSE.pdb', '2KWH.pdb', '1GYZ.pdb']
                # self.split_wt_names[self.split] = ['2KWH.pdb']
        self.wt_names = self.split_wt_names[self.split]
        
        removed_wt_names = []
        for wt_name in tqdm(self.wt_names):
            wt_rows = df.query('WT_name == @wt_name and mut_type == "wt"').reset_index(drop=True)
            self.mut_rows[wt_name] = df.query('WT_name == @wt_name and mut_type != "wt"').reset_index(drop=True)
            if type(reduce) is float and self.split == 'train':
                self.mut_rows[wt_name] = self.mut_rows[wt_name].sample(frac=float(reduce), replace=False)
            if len(wt_rows) == 0 or len(self.mut_rows[wt_name])==0:
                # log.info(f'remove {wt_name}')
                removed_wt_names.append(wt_name)
            else:
                self.wt_seqs[wt_name] = wt_rows.aa_seq[0]
        previous_len = len(self.wt_names)
        self.wt_names = list(set(self.wt_names) - set(removed_wt_names))
        cur_len = len(self.wt_names)


        structure_path = os.path.join(root_path,'data/dataset/megascale/AlphaFold_model_PDBs/')
        # structure_path_lmdb = os.path.join(structure_path,"../parsed_structure.lmdb")
        structure_path_json = os.path.join(structure_path,"../parsed_structure.json")
        
        self.structure_path = structure_path
        
        if not os.path.exists(structure_path_json):
            parse_pdb(structure_path,structure_path_json)
                
        # log.info("loading structure dataset")
        with open(structure_path_json, 'r') as file:
            self.json_dataset = json.load(file)
        
        def process_mutation_row(row, y):
            mut1, mut2 = row['mut_type'].split(":")
            ddg1 = y[y.mut_type == mut1]['ddG_ML'].values
            ddg2 = y[y.mut_type == mut2]['ddG_ML'].values

            row['ddg1'] = -float(ddg1[0]) if len(ddg1) > 0 else np.nan
            row['ddg2'] = -float(ddg2[0]) if len(ddg2) > 0 else np.nan

            return row


    def __len__(self):

        return len(self.wt_names) 
    def _get_wt_item(self, index):

        # wt_name, mut_seq, wt_seq = self.cal_index2mt(index)
        
        wt_name = self.wt_names[index]
        wt_seq = self.wt_seqs[wt_name]
        mut_data = self.mut_rows[wt_name]

        wt_name = wt_name.split(".pdb")[0].replace("|",":")

        pdb = self.json_dataset[wt_name]
        protein = get_pdb(pdb,wt_seq,wt_name)
        
        protein['ground_truth'] = []
        for i in range(len(mut_data)):
            mut_seq = mut_data.iloc[i]

            
            if "ins" in mut_seq.mut_type or "del" in mut_seq.mut_type:
                return None
            
            assert mut_seq.mut_type.count(":") == 1
            assert len(mut_seq.aa_seq) == len(wt_seq)

            mut_info1, mut_info2 = mut_seq.mut_type.split(":")

            wt1 = mut_info1[0]
            mut1 = mut_info1[-1]
            mut_id1 = int(mut_info1[1:-1]) - 1

            wt2 = mut_info2[0]
            mut2 = mut_info2[-1]
            mut_id2 = int(mut_info2[1:-1]) - 1


            assert wt_seq[mut_id1] == wt1 and wt_seq[mut_id2] == wt2
            assert mut_seq.aa_seq[mut_id1] == mut1 and mut_seq.aa_seq[mut_id2] == mut2
            
            if mut_seq.ddG_ML == '-':
                return None
            ddG = -torch.tensor([float(mut_seq.ddG_ML)], dtype=torch.float32)

            append_tensor = torch.LongTensor([
                ALPAHBET.index(wt1),
                ALPAHBET.index(mut1),
                ALPAHBET.index(wt2),
                ALPAHBET.index(mut2)])
            append_tensor = append_tensor.int()
            # split = fing_split_in_proteingym(mut_seq.aa_seq)
            
            
            protein['mut_ids'].append([mut_id1,mut_id2])
            protein['ddG'].append(ddG)
            protein['append_tensors'].append(append_tensor)
            protein['mut_seq'].append(mut_seq.aa_seq)
            # dataset_name.append('megascale'+str(split))
            if self.split!='test' and False:
                protein['ground_truth'].append([mut_seq.ddg1,mut_seq.ddg2])

        protein['ddG'] = torch.stack(protein['ddG']).to(protein['X'].device,non_blocking=True)
        protein['append_tensors'] = torch.stack(protein['append_tensors'])
        # protein['dataset'] = dataset_name
        protein['dataset'] = 'megascale'
        protein['pdb_path'] = self.structure_path
        return protein
    
    def __getitem__(self, index):
        return self._get_wt_item(index)


    
class Featurizer(object):
    def __init__(self, alphabet: Alphabet, 
                 to_pifold_format=False, 
                 coord_nan_to_zero=True,
                 atoms=('N', 'CA', 'C', 'O'),
                 single_mut = False,
                 mut_seq= False
                 ):
        self.alphabet = alphabet
        self.batcher = CoordBatchConverter(
            alphabet=alphabet,
            coord_pad_inf=alphabet.add_special_tokens,
            to_pifold_format=to_pifold_format, 
            coord_nan_to_zero=coord_nan_to_zero
        )
        self.single_mut = single_mut
        self.atoms = atoms
        self.cache = defaultdict(lambda: -1)
        self.mut_seq = mut_seq


    def __call__(self, raw_batch: dict):
        
        if not self.single_mut:
            
            raw_batch = raw_batch[0]
            if not self.mut_seq:
                seqs = [raw_batch['seq']]
                coords = [np.stack([raw_batch['coords'][atom] for atom in self.atoms], 1)]
            else:
                seqs = [raw_batch['seq']]+raw_batch['mut_seq']
                coords = [np.stack([raw_batch['coords'][atom] for atom in self.atoms], 1)]*len(seqs)
            coords, confidence, strs, tokens, lengths, coord_mask = self.batcher.from_lists(
                coords_list=coords, confidence_list=None, seq_list=seqs
            ) 
            # print(seqs,raw_batch['mut_seq'])
            if not self.mut_seq:
                raw_batch['tokens'] = tokens
                raw_batch['mut_tokens'] = None 
            else:
                raw_batch['tokens'] = tokens
                raw_batch['mut_tokens'] = None
            
            if True:
                ddg = raw_batch['ddG']
                ddg = fermi_transform(ddg)
                raw_batch['ddG'] = ddg
            raw_batch['ddG'] = raw_batch['ddG'].reshape(-1)
            
            # ddG_order = torch.argsort(raw_batch['ddG'])
            # raw_batch['ddG_order'] = ddG_order
            return raw_batch
            
        # seqs = [raw_batch['seq']]+raw_batch['mut_seq']
        
        # coords = [np.stack([raw_batch['coords'][atom] for atom in self.atoms], 1)]*len(seqs)
        for protein in raw_batch:
            name = protein['name']
            if isinstance(self.cache[name], int):
                seqs = [protein['seq']]
                coords = [np.stack([protein['coords'][atom] for atom in self.atoms], 1)]
                
                coords, confidence, strs, tokens, lengths, coord_mask = self.batcher.from_lists(
                    coords_list=coords, confidence_list=None, seq_list=seqs
                )
                self.cache[name] = {
                    'tokens': tokens,
                    'mut_tokens': None
                }
            else:
                tokens = self.cache[name]['tokens']
                mut_tokens = self.cache[name]['mut_tokens']
            protein['tokens'] = tokens
            protein['mut_tokens'] = None  
        
        ddg = torch.stack([protein['ddG'] for protein in raw_batch])
        if False:
            ddg = fermi_transform(ddg)
            print(ddg.min(),ddg.max())
        return {
            'raw_batch': raw_batch,
            'mut_ids': [protein['mut_ids'] for protein in raw_batch],
            'append_tensors' : torch.stack([protein['append_tensors'] for protein in raw_batch]),
            'ddG': ddg,
            'name': [protein['name']+protein['chain_ids'] for protein in raw_batch],
            'dataset': [protein['dataset'] for protein in raw_batch],
        }
    
