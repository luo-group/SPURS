from .datasets.megascale import MegaScaleDataset,MegaScaleTestDatasets
from .datasets.megascale_multi import MegaScaleDoubleDataset
from spurs import utils
from spurs.datamodules import register_datamodule
from pytorch_lightning import LightningDataModule
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from torch.utils.data import DataLoader, Dataset
from .datasets.data_utils import Alphabet
from spurs import utils
log = utils.get_logger(__name__)

@register_datamodule('megascale_multi')
class MegaScaleDoubleModule(LightningDataModule):
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
        if stage == 'fit':
            self.train_dataset = MegaScaleDoubleDataset(
                reduce = '',
                split = 'train',
            )
            self.valid_dataset = MegaScaleDoubleDataset(
                reduce = '',
                split = 'val',
  
            )
            self.collate_batch = self.alphabet.featurize
        elif stage == 'test':
            
            self.test_dataset = MegaScaleDoubleDataset(
                reduce = '',
                split = 'test',
   
            )
            # self.test_dataset = PtmulDataset()
            self.collate_batch = self.alphabet.featurize
        
    def train_dataloader(self):
        # True for rasp
        return DataLoader(
            self.train_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=True, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
            )
    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
            )
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, 
            batch_size=self.hparams.batch_size, 
            shuffle=False, 
            num_workers=self.hparams.num_workers, 
            pin_memory=self.hparams.pin_memory,
            collate_fn=self.collate_batch
            )