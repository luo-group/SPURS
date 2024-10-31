# form spurs
from dataclasses import dataclass

import torch
from spurs.models import register_model
from spurs.models.stability.basemodel import BaseModel

from spurs.datamodules.datasets.data_utils import Alphabet

# from .decoder import MPNNSequenceDecoder
# from .encoder import MPNNEncoder


@dataclass
class ProteinMPNNConfig:
    d_model: int = 128
    d_node_feats: int = 128
    d_edge_feats: int = 128
    k_neighbors: int = 48
    augment_eps: float = 0.0
    n_enc_layers: int = 3
    dropout: float = 0.1
    
    tune: bool = False
    use_input_decoding_order: bool = False
    # decoder-only
    n_vocab: int = 22
    n_dec_layers: int = 3
    random_decoding_order: bool = True
    nar: bool = True
    crf: bool = False
    use_esm_alphabet: bool = False


# @register_model('protein_mpnn_cmlm')
# class ProteinMPNNCMLM(BaseModel):
#     _default_cfg = ProteinMPNNConfig()

#     def __init__(self, cfg) -> None:
#         super().__init__(cfg)

#         self.encoder = MPNNEncoder(
#             node_features=self.cfg.d_node_feats,
#             edge_features=self.cfg.d_edge_feats,
#             hidden_dim=self.cfg.d_model,
#             num_encoder_layers=self.cfg.n_enc_layers,
#             k_neighbors=self.cfg.k_neighbors,
#             augment_eps=self.cfg.augment_eps,
#             dropout=self.cfg.dropout
#         )

#         if self.cfg.use_esm_alphabet:
#             alphabet = Alphabet('esm', 'cath')
#             self.padding_idx = alphabet.padding_idx
#             self.mask_idx = alphabet.mask_idx
#         else:
#             alphabet = None
#             self.padding_idx = 0
#             self.mask_idx = 1

#         self.decoder = MPNNSequenceDecoder(
#             n_vocab=self.cfg.n_vocab,
#             d_model=self.cfg.d_model,
#             n_layers=self.cfg.n_dec_layers,
#             random_decoding_order=self.cfg.random_decoding_order,
#             dropout=self.cfg.dropout,
#             nar=self.cfg.nar,
#             crf=self.cfg.crf,
#             alphabet=alphabet
#         )

#     def forward(self, batch, return_feats=False, **kwargs):
#         coord_mask = batch['coord_mask'].float()

#         residue_idx = batch.get('residue_idx', None)
#         chain_idx = batch.get('chain_idx', None)
#         residue_idx = None
#         chain_idx = None
#         encoder_out = self.encoder(
#             X=batch['coords'],
#             mask=coord_mask,
#             residue_idx=residue_idx,
#             chain_idx=chain_idx
#         )

#         logits, feats = self.decoder(
#             prev_tokens=batch['prev_tokens'],
#             memory=encoder_out, 
#             memory_mask=coord_mask,
#             target_tokens=batch.get('tokens'),
#             **kwargs
#         )

#         if return_feats:
#             return logits, feats
#         return logits