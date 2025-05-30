from dataclasses import dataclass, field
from typing import List

import torch
from spurs.models import register_model
from spurs.models.stability.basemodel import BaseModel
from spurs.models.stability.protein_mpnn import ProteinMPNNConfig

from spurs.models.stability.modules.esm2 import ESM2
from spurs import utils
from spurs.models.stability.org_transfer_model import get_protein_mpnn
import torch.nn.functional as F

import torch.nn as nn

log = utils.get_logger(__name__)
from .mlp import MLP, MLPConfig

@dataclass
class MPNN:
    encoder: ProteinMPNNConfig = field(default=ProteinMPNNConfig())
    adapter_layer_indices: List = field(default_factory=lambda: [-1, ])
    separate_loss: bool = True
    name: str = 'proteinmpnn'
    dropout: float = 0.1
    mlp: MLPConfig = field(default=MLPConfig()) # mlp is not used in this model

    


@register_model('mpnn_unsupervised')
class MPNN(BaseModel):
    _default_cfg = MPNN()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.tune = cfg.encoder.tune
        self.use_input_decoding_order = cfg.encoder.use_input_decoding_order
        self.encoder = get_protein_mpnn(tune=cfg.encoder.tune)  
        

        self.input_dim = self.cfg.mlp.input_dim  
        self.mlp = MLP(self.cfg.mlp)

    def forward(self, batch, **kwargs):
        if self.mlp is not None:
            self.mlp = None
            
            
        if not self.tune:
            with torch.no_grad():
                batch['feats'] = self.forward_encoder(batch)
        else:   
            batch['feats'] = self.forward_encoder(batch)
        representation = batch['feats'][:,:,:self.input_dim]
        
        shifed_mut_ids = torch.LongTensor(batch['mut_ids']).to(representation.device)

        muted_id_representation = representation[:, shifed_mut_ids.long()] # [B, H]
        batch['muted_id_representation'] = muted_id_representation
        
        ddg_out = muted_id_representation.squeeze(0)
        ddg_out = F.log_softmax(ddg_out,dim=-1)
        ddg_out_aa = (ddg_out*batch['append_tensors'][:,21:]).sum(-1)
        ddg_out_wt_aa = (ddg_out*batch['append_tensors'][:,:21]).sum(-1)
        ddg_out = ddg_out_aa - ddg_out_wt_aa

        
        ddg_out = -ddg_out.cpu()
        import numpy as np
        if np.isinf(ddg_out).sum()>0:
            assert False
        ddg_out = ddg_out.to(representation.device)
        
        return ddg_out

    def forward_encoder(self,batch):
        X = batch['X']
        S = batch['S']
        mask = batch['mask']
        chain_M = batch['chain_M']
        chain_M_chain_M_pos = batch['chain_M_chain_M_pos']
        residue_idx = batch['residue_idx']
        chain_encoding_all = batch['chain_encoding_all']
        randn_1 = batch['randn_1']
        all_mpnn_hid, mpnn_embed, logistics = self.encoder(X, S, mask, chain_M, residue_idx, chain_encoding_all, None,self.use_input_decoding_order)
        return logistics

        