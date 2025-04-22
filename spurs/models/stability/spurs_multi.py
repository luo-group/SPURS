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
from spurs.models.stability.modules.esm2_adapter import ESM2WithStructuralAdatper
import torch.nn as nn
from .mlp import MLP, MLPConfig



log = utils.get_logger(__name__)

@dataclass
class SPURSMultiConfig:
    encoder: ProteinMPNNConfig = field(default=ProteinMPNNConfig())
    adapter_layer_indices: List = field(default_factory=lambda: [-1, ])
    separate_loss: bool = True
    name: str = 'esm2_t33_650M_UR50D'
    dropout: float = 0.1
    mlp: MLPConfig = field(default=MLPConfig())
    multi: MLPConfig = field(default=MLPConfig())
    agg_type: str = 'sum'
    agg_dim: int = 128


@register_model('spurs_multi')
class SPURSMulti(BaseModel):
    _default_cfg = SPURSMultiConfig()

    def __init__(self, cfg) -> None:
        super().__init__(cfg)

        self.tune = cfg.encoder.tune
        self.use_input_decoding_order = cfg.encoder.use_input_decoding_order
        self.encoder = get_protein_mpnn(tune=cfg.encoder.tune) 
        
        self.cfg.encoder.d_model = self.cfg.mlp.input_dim
        self.decoder = ESM2WithStructuralAdatper.from_pretrained(args=self.cfg, name=self.cfg.name)
        self.input_dim = self.cfg.mlp.input_dim


        self.cfg.mlp.input_dim = self.cfg.mlp.input_dim+1280 if self.cfg.mlp.flat_dim < 0 else self.cfg.mlp.input_dim+self.cfg.mlp.flat_dim
        self.mlp = MLP(self.cfg.mlp)

        if self.cfg.mlp.flat_dim > 0:
            self.flat_layers = nn.Linear(1280, self.cfg.mlp.flat_dim)
            self.dp = nn.Dropout(self.cfg.dropout)
        
        self.padding_idx = self.decoder.padding_idx
        self.mask_idx = self.decoder.mask_idx
        self.cls_idx = self.decoder.cls_idx
        self.eos_idx = self.decoder.eos_idx
        
        agg_dim = self.cfg.agg_dim if self.cfg.agg_type == 'concate' else self.cfg.mlp.input_dim

        self.aa_embedding = nn.Embedding(20, agg_dim)
        self.agg_type = self.cfg.agg_type
        if self.agg_type == 'concate':
            self.cfg.multi.input_dim = self.cfg.mlp.input_dim + agg_dim
        self.multi = MLP(self.cfg.multi)
        
    def forward(self, batch, **kwargs):
        batch['feats'] = self.forward_encoder(batch)

        batch['feats'] = batch['feats'][:,:,:self.input_dim]
        encoder_out = {'feats':F.pad(batch['feats'], (0, 0, 1, 1))}
        
        init_pred = batch['tokens']
        # need to rethink
        decoder_out = self.decoder(
            tokens=init_pred,
            encoder_out=encoder_out,
        )
        
        representation = decoder_out['representations'][-1]
        if self.cfg.mlp.flat_dim > 0:
        
            flat_representation = self.flat_layers(representation)
            flat_representation = self.dp(flat_representation)
            flat_representation = F.gelu(flat_representation)
            representation = flat_representation

        representation = torch.cat([representation, encoder_out['feats']], dim=-1)

    

        batch['mut_ids'] = batch['mut_ids'] if isinstance(batch['mut_ids'], torch.Tensor) else torch.tensor(batch['mut_ids'])
        mask = batch['mut_ids']>-1
        batch['mut_ids'] = torch.clamp(batch['mut_ids'], min=0)
        

        shifed_mut_ids = batch['mut_ids'].to(representation.device)+1
        muted_id_representation = representation[:, shifed_mut_ids.long()] # [B, H]
        batch['muted_id_representation'] = muted_id_representation
        pre_output = self.mlp(batch)
        
        ddg_out = pre_output.squeeze(0)
        batch['append_tensors'] = batch['append_tensors'].reshape(batch['append_tensors'].size(0),-1,2)
        
        
        indices = batch['append_tensors'].long()
        indices = torch.clamp(indices, min=0)  

        ddg_out = torch.take_along_dim(ddg_out, indices, dim=-1)  # [B, 2, 2]
        ddg_out = ddg_out[:, :, 1] - ddg_out[:, :, 0]  # [B, 2]
        batch['append_tensors'] = torch.clamp(batch['append_tensors'], min=0)

        mut_aa_representation = self.aa_embedding(batch['append_tensors'][:,:,1].long())
        
        if self.agg_type == 'sum':
            batch['muted_id_representation'] = batch['muted_id_representation']+ mut_aa_representation.unsqueeze(0)
        elif self.agg_type == 'concate':
            batch['muted_id_representation'] = torch.cat([batch['muted_id_representation'], mut_aa_representation.unsqueeze(0)], dim=-1)

        d_1,b,d_10,dim = batch['muted_id_representation'].shape
        batch['muted_id_representation'] = batch['muted_id_representation']*mask.unsqueeze(0).unsqueeze(-1).expand(-1,-1,-1,dim).to(batch['muted_id_representation'].device)

        ddg_out = ddg_out*mask.to(ddg_out.device)
        batch['muted_id_representation'] = batch['muted_id_representation'].sum(2)

        dddg_out = self.multi(batch)
        ddg = dddg_out.squeeze()+ddg_out.sum(1)
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
        all_mpnn_hid, mpnn_embed, _ = self.encoder(X, S, mask, chain_M, residue_idx, chain_encoding_all, None,self.use_input_decoding_order)
        
        all_mpnn_hid = torch.cat([all_mpnn_hid[0],mpnn_embed,all_mpnn_hid[1],all_mpnn_hid[2]],dim=-1)
        
        return all_mpnn_hid