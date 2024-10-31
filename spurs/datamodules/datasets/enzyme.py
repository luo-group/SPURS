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

class EngymeDataset(torch.utils.data.Dataset):
    def __init__(self,):
        
        self.pdb_dir = './data/enzyme/1qlh.pdb'
        self.sequence = 'MSTAGKVIKCKAAVLWEEKKPFSIEEVEVAPPKAHEVRIKMVATGICRSDDHVVSGTLVTPLPVIAGHEAAGIVESIGEGVTTVRPGDKVIPLFTPQCGKCRVCKHPEGNFCLKNDLSMPRGTMQDGTSRFTCRGKPIHHFLGTSTFSQYTVVDEISVAKIDAASPLEKVCLIGCGFSTGYGSAVKVAKVTQGSTCAVFGLGGVGLSVIMGCKAAGAARIIGVDINKDKFAKAKEVGATECVNPQDYKKPIQEVLTEMSNGGVDFSFEVIGRLDTMVTALSCCQEAYGVSVIVGVPPDSQNLSMNPMLLLSGRTWKGAIFGGFKSKDSVPKLVADFMAKKFALDPLITHVLPFEKINEGFDLLRSGESIRTILTF'
        self.sequence = self.sequence[1:]# 2-376
        
        ret = self._get_wt_item(0)
    def __len__(self):
        return 1
    
    def _get_wt_item(self, index):
        wt_name = self.pdb_dir.split('/')[-1].split('.')[0]
        
        chain = 'A'
        pdb = alt_parse_PDB(self.pdb_dir,chain)
        
        pdb[0]['resn_list'] = pdb[0]['resn_list']
        pdb[0]['seq_chain_A'] = pdb[0]['seq_chain_A']
        
        for k in pdb[0]['coords_chain_A'].keys():
            pdb[0]['coords_chain_A'][k] = pdb[0]['coords_chain_A'][k]
        pdb[0]['seq'] = self.sequence
        
        
        protein = get_pdb(pdb[0], wt_name, wt_name, check_assert=False)
        
        protein['mut_ids'] = torch.tensor([0])
        protein['ddG'] = torch.tensor([[0]])
        protein['append_tensors'] = torch.tensor([0])

        protein['dataset'] = ['0']
        
        return protein
    def __getitem__(self, index):
        return self._get_wt_item(index)