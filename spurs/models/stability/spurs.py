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
log = utils.get_logger(__name__)
from .mlp import MLP, MLPConfig


@dataclass
class SPURSConfig:
    encoder: ProteinMPNNConfig = field(default=ProteinMPNNConfig())
    adapter_layer_indices: List = field(default_factory=lambda: [-1, ])
    separate_loss: bool = True
    name: str = 'esm2_t33_650M_UR50D'
    dropout: float = 0.1
    mlp: MLPConfig = field(default=MLPConfig())


@register_model('spurs')
class SPURS(BaseModel):
    """
    SPURS (Structure-based Protein Understanding and Recognition System) model for protein stability prediction.
    
    This model combines protein structure information (from ProteinMPNN) and sequence information (from ESM2)
    to predict protein stability changes. The architecture consists of three main components:
    
    1. Encoder (ProteinMPNN): Processes protein structure information
    2. Decoder (ESM2): Processes sequence information with structural prior
    3. MLP: Final stability prediction layer
    
    The model uses a structural adapter to effectively combine structural and sequence information,
    allowing for more accurate stability predictions.
    
    Args:
        cfg (SPURSConfig): Configuration object containing model parameters
            - encoder: ProteinMPNN configuration
            - adapter_layer_indices: List of ESM2 layer indices to adapt
            - name: ESM2 model name
            - dropout: Dropout rate
            - mlp: MLP configuration
    """
    _default_cfg = SPURSConfig()

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
        
    def forward(self, batch, **kwargs):
        if not self.tune:
            with torch.no_grad():
                batch['feats'] = self.forward_encoder(batch)
        else:   
            batch['feats'] = self.forward_encoder(batch)
        
        batch['feats'] = batch['feats'][:,:,:self.input_dim]
        encoder_out = {'feats':F.pad(batch['feats'], (0, 0, 1, 1))}
        
        init_pred = batch['tokens']

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

        if 'return_logist' in kwargs and kwargs['return_logist']:
            wt=batch['seq']
            shifted_mut_ids = torch.repeat_interleave(torch.arange(1, 1+len(wt)), 20)
            muted_id_representation = representation[:, shifted_mut_ids.long()]
            batch['muted_id_representation'] = muted_id_representation
            pre_output = self.mlp(batch)
            
            ddg_out = pre_output.squeeze()
            mt_aa = torch.arange(20).repeat(len(wt))
            
            
            ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
            wt_aa = torch.repeat_interleave(torch.tensor([ALPHABET.index(s) for s in wt]), 20)
            num_classes = 21
            
            mt_aa_one_hot = F.one_hot(mt_aa, num_classes=num_classes).to(representation.device)
            wt_aa_one_hot = F.one_hot(wt_aa, num_classes=num_classes).to(representation.device)
            
            ddg_out_aa = (ddg_out*mt_aa_one_hot).sum(-1)
            ddg_out_wt_aa = (ddg_out*wt_aa_one_hot).sum(-1)
            ddg = ddg_out_aa - ddg_out_wt_aa
            return ddg.reshape(-1,20)

        batch['mut_ids'] = batch['mut_ids'] if isinstance(batch['mut_ids'], torch.Tensor) else torch.tensor(batch['mut_ids'])
        shifed_mut_ids = batch['mut_ids'].to(representation.device)+1
        muted_id_representation = representation[:, shifed_mut_ids.long()] # [B, H]
        batch['muted_id_representation'] = muted_id_representation
        pre_output = self.mlp(batch)
        
        ddg_out = pre_output.squeeze()
        
        ddg_out_aa = (ddg_out*batch['append_tensors'][:,21:]).sum(-1)
        ddg_out_wt_aa = (ddg_out*batch['append_tensors'][:,:21]).sum(-1)
        ddg = ddg_out_aa - ddg_out_wt_aa
        ddg[torch.isnan(ddg)] = 10000
        
        return ddg

    def forward_encoder(self,batch):
        """
        Forward pass through the encoder (ProteinMPNN) component of the SPURS model.
        
        This function processes the input protein structure data (X, S, mask, chain_M, residue_idx, chain_encoding_all, randn_1)
        and returns the encoded features from the ProteinMPNN encoder.

        Args:
            batch (dict): Input batch containing protein structure data
                - X: Protein structure coordinates
                - S: Protein structure mask
                - mask: Mask indicating valid positions
                - chain_M: Chain mask
                - residue_idx: Residue indices
                - chain_encoding_all: Chain encoding
        
        Returns:
            torch.Tensor: Encoded features from the ProteinMPNN encoder     
        """

        
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