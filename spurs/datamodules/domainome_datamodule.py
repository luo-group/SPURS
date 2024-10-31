from .datasets.megascale import MegaScaleDataset,MegaScaleTestDatasets
from spurs import utils
from spurs.datamodules import register_datamodule
from pytorch_lightning import LightningDataModule
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from torch.utils.data import DataLoader, Dataset
from .datasets.data_utils import Alphabet
from spurs import utils
from .datasets.domainome import domainome

import os


current_dir = os.path.dirname(os.path.abspath(__file__))

log = utils.get_logger(__name__)


@register_datamodule('domainome')
class DomainomeModule(LightningDataModule):
    def __init__(self,
        alphabet: None,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
        single_mut: bool=False,
        mut_seq: bool=False,
        std_ratio: float=0.75,
        loss_ratio: float=1.,
        train_ratio: float=1,
                 ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.alphabet = None

        self.train_dataset: Optional[Dataset] = None
        self.valid_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None
        
    def setup(self, stage: Optional[str] = None):
        self.alphabet = Alphabet(**self.hparams.alphabet)
        if stage == 'test':
            root_path = os.path.join(current_dir,'../../')
            self.test_dataset = domainome(os.path.join(root_path,'data/dataset/Domainome/files_to_reproduce_analyses/analysis_files/pdb_files'),
                                       os.path.join(root_path,'data/dataset/Domainome/Supplementary_Table_5_aPCA_vs_variant_effect_predictors.csv'),
                                       'Domainome', stage='full',mut_seq=False,train_size=1)
            


            self.collate_batch = self.alphabet.featurize
        else:
            assert False, 'Not implemented'
    def train_dataloader(self):
        # True for rasp
        return 
    def val_dataloader(self):
        return 
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
            )