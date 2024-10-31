import spurs
from omegaconf import OmegaConf
from spurs.utils import seed_everything
import torch
import os
from spurs.models.stability.spurs import SPURS
from spurs.datamodules.datasets.utils import alt_parse_PDB
from spurs.datamodules.datasets.utils import get_pdb
from spurs.datamodules.datasets.data_utils import Alphabet

def get_SPURS(ckpt_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> torch.nn.Module:
    cfg = OmegaConf.load(os.path.join(ckpt_path,'.hydra/config.yaml'))

    del cfg['model']['_target_']
    seed_everything(cfg['train']['seed'])

    model = SPURS(cfg['model']).to(device)

    ckpt = torch.load(os.path.join(ckpt_path,'checkpoints/best.ckpt'), map_location=torch.device('cpu'))['state_dict']
    ckpt_remove_model = {k[6:]:v for k, v in ckpt.items() if 'model.' in k}
    model.load_state_dict(ckpt_remove_model, strict=False)    
    
    return model, cfg

def parse_pdb(pdb_path: str, pdb_name: str, chain: str, cfg, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        pdb = alt_parse_PDB(pdb_path, chain)
        resn_list_int = [int(i) for i in pdb[0]['resn_list']]
        # assert min(resn_list_int) == 1 and max(resn_list_int) == len(resn_list_int) for af2 structure only

        pdb = get_pdb(pdb[0], pdb_name, pdb_name, check_assert=False)
        pdb['mut_ids'] = torch.tensor([0])
        pdb['ddG'] = torch.tensor([[0]])
        pdb['append_tensors'] = torch.tensor([0])

        pdb['dataset'] = ['0']
        alphabet = Alphabet(**cfg['datamodule']['alphabet'])
        pdb = alphabet.featurize([pdb])
        def move_tensors_to_device(d, device):
            """
            """
            if isinstance(d, dict):
                return {k: move_tensors_to_device(v, device) for k, v in d.items()}
            elif isinstance(d, list):
                return [move_tensors_to_device(v, device) for v in d]
            elif isinstance(d, torch.Tensor):
                return d.to(device)
            else:
                return d
        pdb = move_tensors_to_device(pdb,device)
        # result = model(pdb,return_logist=True)
        # result = result.detach().cpu()
        # result_dict[pdb_name] = result
        return pdb