import torch
from spurs.models import register_model

import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass,field
from spurs.models.stability.basemodel import BaseModel
from typing import List, Union

@dataclass
class MLPConfig:
    num_layers: int = 3
    input_dim: int = 2560+42
    # hidden_dim: Union[List[int], int] = field(default_factory=lambda: [512, 128])
    hidden_dim: int = 512
    output_dim: int = 1
    dropout: float = 0.1
    ckpt_path: str = ''
    append_tensors: bool = True
    flat_dim: int = -1
    
    
@register_model('mlp')
class MLP(BaseModel):
    _default_cfg = MLPConfig()
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        # self.fc1 = nn.Linear(input_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)
        # self.relu = nn.ReLU()
        
        # self.fcs = nn.ModuleList(
        #     [nn.Linear(input_dim, hidden_dim, bias=True)] + [nn.Linear(hidden_dim, hidden_dim, bias=True) for _ in range(num_layers-2)] + [nn.Linear(hidden_dim, output_dim, bias=True)]
        # )
        # num_layers = 3, input_dim = 128, hidden_dim=[128,128], output_dim = 128, dropout=0.1, ckpt_path=None
        # num_layers = self.cfg.num_layers
        input_dim = self.cfg.input_dim
        hidden_dim = self.cfg.hidden_dim if isinstance(self.cfg.hidden_dim, list) else [self.cfg.hidden_dim]*3
        num_layers = len(hidden_dim)+1
        output_dim = self.cfg.output_dim
        dropout = self.cfg.dropout
        ckpt_path = self.cfg.ckpt_path
        self.append_tensors = self.cfg.append_tensors
        assert len(hidden_dim)==num_layers-1
        
        self.fcs = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim[0], bias=True)] + [nn.Linear(hidden_dim[i], hidden_dim[i+1], bias=True) for i in range(num_layers-2)] + [nn.Linear(hidden_dim[-1], output_dim, bias=True)]
        )
        self.dropouts = nn.ModuleList( 
            [nn.Dropout(dropout) for _ in range(num_layers-1)]
        ) 
        
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        self.to(self.device)
        self.initialize_weights(ckpt_path)
        
    def initialize_weights(self,ckpt_path=''):
        if ckpt_path is '':
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        else:
            checkpoint = torch.load(ckpt_path, map_location=self.device) 
            self.load_state_dict(checkpoint)
            # print("MLP model loaded from checkpoint: {}".format(ckpt_path))

    def forward(self, batch,return_embed = False):
        muted_id_representation = batch.get('muted_id_representation',batch.get('mpnn_outputs',None))
        x = muted_id_representation
        for i in range(len(self.fcs)-1):
            x = self.fcs[i](x)
            x = self.dropouts[i](x)
            x = F.gelu(x)
        if return_embed:
            return x
        x = self.fcs[-1](x)
        return x