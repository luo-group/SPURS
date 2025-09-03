import spurs
from omegaconf import OmegaConf
from spurs.utils import seed_everything
import torch
import os
from spurs.models.stability.spurs import SPURS
from spurs.datamodules.datasets.utils import alt_parse_PDB
from spurs.datamodules.datasets.utils import get_pdb
from spurs.datamodules.datasets.data_utils import Alphabet
from huggingface_hub import hf_hub_download
from spurs.models.stability.spurs_multi import SPURSMulti


def get_SPURS(ckpt_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a SPURS model from a local checkpoint file.

    This function loads a SPURS model and its configuration from a local checkpoint directory.
    The checkpoint should contain both the model weights and the configuration file.

    Args:
        ckpt_path (str): Path to the checkpoint directory containing .hydra/config.yaml and checkpoints/best.ckpt
        device (str, optional): Device to load the model on. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        tuple: A tuple containing:
        - model (torch.nn.Module): The loaded SPURS model
        - cfg (OmegaConf): The model configuration

    Example:
        >>> model, cfg = get_SPURS("path/to/checkpoint")
    """
    cfg = OmegaConf.load(os.path.join(ckpt_path,'.hydra/config.yaml'))

    del cfg['model']['_target_']
    seed_everything(cfg['train']['seed'])

    model = SPURS(cfg['model']).to(device)

    ckpt = torch.load(os.path.join(ckpt_path,'checkpoints/best.ckpt'), map_location=torch.device('cpu'))['state_dict']
    ckpt_remove_model = {k[6:]:v for k, v in ckpt.items() if 'model.' in k}
    model.load_state_dict(ckpt_remove_model, strict=False)    
    
    return model, cfg


def get_SPURS_from_hub(repo_id: str = "cyclization9/SPURS", device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a pre-trained SPURS model directly from the Hugging Face model hub.

    This function downloads and loads a pre-trained SPURS model and its configuration
    from the Hugging Face model hub. This is the recommended way to get started with SPURS.

    Args:
        repo_id (str, optional): Hugging Face model repository ID. Defaults to "cyclization9/SPURS".
        device (str, optional): Device to load the model on. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The loaded SPURS model
            - cfg (OmegaConf): The model configuration

    Example:
        >>> model, cfg = get_SPURS_from_hub()
    """
    # Download files from Hugging Face Hub
    config_path = hf_hub_download(repo_id=repo_id, filename="spurs/.hydra/config.yaml")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="spurs/checkpoints/best.ckpt")

    # Load config
    cfg = OmegaConf.load(config_path)
    del cfg['model']['_target_']
    seed_everything(cfg['train']['seed'])

    # Instantiate model and load checkpoint
    model = SPURS(cfg['model']).to(device)
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
    ckpt_remove_model = {k[6:]: v for k, v in ckpt.items() if 'model.' in k}
    model.load_state_dict(ckpt_remove_model, strict=False)

    return model, cfg

def get_SPURS_multi_from_hub(repo_id: str = "cyclization9/SPURS", device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Load a pre-trained SPURS multi-mutation model from the Hugging Face model hub.

    This function downloads and loads a pre-trained SPURS model specialized for
    predicting the effects of multiple mutations simultaneously.

    Args:
        repo_id (str, optional): Hugging Face model repository ID. Defaults to "cyclization9/SPURS".
        device (str, optional): Device to load the model on. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        tuple: A tuple containing:
            - model (torch.nn.Module): The loaded SPURS multi-mutation model
            - cfg (OmegaConf): The model configuration

    Example:
        >>> model, cfg = get_SPURS_multi_from_hub()
    """
    # Download files from Hugging Face Hub
    config_path = hf_hub_download(repo_id=repo_id, filename="spurs_multi/.hydra/config.yaml")
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="spurs_multi/checkpoints/best.ckpt")

    # Load config
    cfg = OmegaConf.load(config_path)
    del cfg['model']['_target_']
    seed_everything(cfg['train']['seed'])

    # Instantiate model and load checkpoint
    model = SPURSMulti(cfg['model']).to(device)
    ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))['state_dict']
    ckpt_remove_model = {k[6:]: v for k, v in ckpt.items() if 'model.' in k}
    model.load_state_dict(ckpt_remove_model, strict=False)

    return model, cfg


def parse_pdb(pdb_path: str, pdb_name: str, chain: str, cfg, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Parse a PDB file and prepare it for SPURS model input.

    This function processes a PDB file and converts it into the format required by SPURS models.
    It handles coordinate extraction, feature generation, and device placement.

    Args:
        pdb_path (str): Path to the PDB file
        pdb_name (str): Name of the protein
        chain (str): Chain identifier to analyze
        cfg: Model configuration containing alphabet settings
        device (str, optional): Device to place the tensors on. Defaults to 'cuda' if available, else 'cpu'.

    Returns:
        dict: A dictionary containing processed PDB data including:
            - coordinates
            - sequence information
            - features required by the model
            All tensors are placed on the specified device.

    """
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
        """Move all tensors in a nested dictionary or list to the specified device.

        Args:
            d: Dictionary, list, or tensor to process
            device: Target device for tensors

        Returns:
            The input with all tensors moved to the specified device
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
    return pdb

def parse_pdb_for_mutation(mut_info_list):
    """Parse mutation information into tensor format required by model.
    
    This function converts a list of mutation specifications into the tensor format
    required by SPURS models for mutation analysis.
    
    Args:
        mut_info_list (List[List[str]]): List of lists containing mutation strings.
            Each inner list represents a set of mutations to analyze together.
            Format: [['V2C','P3T'], ['W1A','V2Y']] where each mutation string is
            formatted as 'OriginalAAPositionNewAA'.
        
    Returns:
        tuple: A tuple containing:
            - mut_ids (torch.Tensor): Tensor of mutation positions (0-indexed)
            - append_tensors (torch.Tensor): Tensor of amino acid indices for wild-type and mutant residues

    Example:
        >>> mut_ids, append_tensors = parse_pdb_for_mutation([['V2C', 'P3T']])
        >>> print(mut_ids)  # Shows positions 1,2 (0-indexed)
        >>> print(append_tensors)  # Shows amino acid indices for V->C, P->T
    """
    mut_ids = []
    append_tensors = []
    ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
    
    for mut_info in mut_info_list:
        mut_ids_ = []
        append_tensors_ = []
        for mut in mut_info:
            # Extract position (convert to 0-based indexing)
            mut_ids_.append(int(mut[1:-1]) - 1)
            # Get indices for wild-type and mutant amino acids
            append_tensors_.append(ALPHABET.index(mut[0]))    # Wild-type
            append_tensors_.append(ALPHABET.index(mut[-1]))   # Mutant
        mut_ids.append(mut_ids_)
        append_tensors.append(append_tensors_)
        
    mut_ids = torch.tensor(mut_ids)
    append_tensors = torch.tensor(append_tensors)
    
    return mut_ids, append_tensors


