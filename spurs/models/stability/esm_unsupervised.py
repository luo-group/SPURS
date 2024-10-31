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


log = utils.get_logger(__name__)
from .mlp import MLP, MLPConfig

@dataclass
class MPNNESMCAT:

    separate_loss: bool = True
    name: str = 'esm2_t33_650M_UR50D'
    dropout: float = 0.1
    mlp: MLPConfig = field(default=MLPConfig())


@register_model('esm_unsupervised')
class ESM(BaseModel):
    _default_cfg = MPNNESMCAT()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.decoder,self.decoder_alphabet = ESM2.from_pretrained(args=self.cfg, name=self.cfg.name,ret_alphabet=True)
        
        self.mlp = MLP(self.cfg.mlp)
        proteinmpnn_alphabet = 'ACDEFGHIKLMNPQRSTVWYX'
        tok_to_idx = self.decoder_alphabet.tok_to_idx
        esm_idx_to_output_idx = [tok_to_idx[tok] for tok in proteinmpnn_alphabet]
        self.esm_idx_to_output_idx = torch.LongTensor(esm_idx_to_output_idx)
        
        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx
        

    def forward(self, batch, **kwargs):
        if self.mlp is not None:
            self.mlp = None
        init_pred = batch['tokens']
        with torch.no_grad():
            decoder_out = self.decoder(
                tokens=init_pred,
                encoder_out=None,
            )

        representation = decoder_out['logits']

        representation = representation[:,:,self.esm_idx_to_output_idx]
        shifed_mut_ids = torch.LongTensor(batch['mut_ids']).to(representation.device)+1

        muted_id_representation = representation[:, shifed_mut_ids.long()] # [B, H]
        
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
        return
        