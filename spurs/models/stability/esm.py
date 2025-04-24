from dataclasses import dataclass, field
from typing import List

import torch
from spurs.models import register_model
from spurs.models.stability.basemodel import BaseModel
from spurs.models.stability.protein_mpnn import ProteinMPNNConfig

from spurs.models.stability.modules.esm2 import ESM2
from spurs import utils
import torch.nn.functional as F

import torch.nn as nn

import torch.nn as nn

log = utils.get_logger(__name__)
from .mlp import MLP, MLPConfig
@dataclass
class ESM:
    encoder: ProteinMPNNConfig = field(default=ProteinMPNNConfig())
    adapter_layer_indices: List = field(default_factory=lambda: [-1, ])
    separate_loss: bool = True
    name: str = 'esm2_t33_650M_UR50D'
    dropout: float = 0.1
    mlp: MLPConfig = field(default=MLPConfig())


@register_model('esm_reg')
class ESM(BaseModel):
    """
    ESM-based model for protein stability prediction.
    
    This model leverages the ESM2 language model to understand protein sequences
    and predict stability changes. Features include:
    
    1. Pre-trained ESM2 language model for sequence encoding
    2. MLP-based regression head for stability predictions
    
    Args:
        cfg (ESM): Configuration object containing:
            - name: ESM2 model name (e.g. 'esm2_t33_650M_UR50D')
            - dropout: Dropout rate
            - mlp: MLP configuration
    """
    _default_cfg = ESM()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.decoder = ESM2.from_pretrained(args=self.cfg, name=self.cfg.name)
        self.mlp = MLP(self.cfg.mlp)
        
        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx
        

    def forward(self, batch, **kwargs):
        init_pred = batch['tokens']

        with torch.no_grad():
            decoder_out = self.decoder(
                tokens=init_pred,
                encoder_out=None,
            )

        representation = decoder_out['representations'][-1]

        
        shifed_mut_ids = torch.LongTensor(batch['mut_ids']).to(representation.device)+1

        muted_id_representation = representation[:, shifed_mut_ids.long()] # [B, H]
        batch['muted_id_representation'] = muted_id_representation
        pre_output = self.mlp(batch)
        
        ddg_out = pre_output.squeeze()
        
        ddg_out_aa = (ddg_out*batch['append_tensors'][:,21:]).sum(-1)
        ddg_out_wt_aa = (ddg_out*batch['append_tensors'][:,:21]).sum(-1)
        ddg = ddg_out_aa - ddg_out_wt_aa
        
        return ddg

    def forward_encoder(self,batch):
        X = batch['X']
        S = batch['S']
        mask = batch['mask']
        chain_M = batch['chain_M']
        chain_M_chain_M_pos = batch['chain_M_chain_M_pos']
        residue_idx = batch['residue_idx']
        chain_encoding_all = batch['chain_encoding_all']
        randn_1 = batch['randn_1']
        all_mpnn_hid, mpnn_embed, _ = self.encoder(X, S, mask, chain_M, residue_idx, chain_encoding_all, None)

        return all_mpnn_hid[0]
        