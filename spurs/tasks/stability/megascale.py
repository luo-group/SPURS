import os
from typing import Any, Callable, List, Union
from pathlib import Path
import numpy as np
import torch
from spurs import utils
from spurs.modules import metrics
from spurs.tasks import TaskLitModule, register_task
from spurs.utils.config import compose_config as Cfg, merge_config

from omegaconf import DictConfig
from torch import nn
from torch.nn import functional as F
from torchmetrics import CatMetric, MaxMetric, MeanMetric, MinMetric

from spurs.datamodules.datasets.data_utils import Alphabet
from spurs.modules.rho import cal_roh,cal_rho_by_chain
# import esm

log = utils.get_logger(__name__)


def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()


@register_task('stability/megascale')
class MegaScale(TaskLitModule):

    def __init__(
        self,
        model: Union[nn.Module, DictConfig],
        criterion: Union[nn.Module, DictConfig],
        optimizer: DictConfig,
        lr_scheduler: DictConfig = None,
    ):
        super().__init__(model, criterion, optimizer, lr_scheduler)
        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        # self.save_hyperparameters(ignore=['model', 'criterion'], logger=False)
        self.save_hyperparameters(logger=True)
        self.build_model() 
        
        # self.build_generator()

    def setup(self, stage=None) -> None:
        super().setup(stage)

        self.build_criterion()
        self.build_torchmetric()

        if self.stage == 'fit':
            log.info(f'\n{self.model}')

    def build_model(self):
        log.info(f"Instantiating neural model <{self.hparams.model._target_}>")
        self.model = utils.instantiate_from_config(cfg=self.hparams.model, group='model')

    def build_generator(self):
        pass

    def build_criterion(self):
        self.criterion = utils.instantiate_from_config(cfg=self.hparams.criterion) 

    def build_torchmetric(self):
        self.eval_loss = MeanMetric()
        self.eval_loss1 = None
        self.eval_loss2 = None

        self.fermi_scores = CatMetric()
        self.pred_scores = CatMetric()
        self.pdb_chain = []
        self.dataset_name = []
        
        self.mut_ids = CatMetric()


        self.mpnn_output = CatMetric()
        self.append_tensors = CatMetric()
        
        self.rho_avg = MeanMetric()
        self.use_rho_avg = False


    def load_from_ckpt(self, ckpt_path):
        state_dict = torch.load(ckpt_path, map_location='cpu')['state_dict']

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        print(f"Restored from {ckpt_path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def on_epoch_start(self) -> None:
        pass

    # -------# Training #-------- #
    @torch.no_grad()
    def inject_noise(self, tokens, coord_mask, noise=None, sel_mask=None, mask_by_unk=False):
        pass

    def step(self, batch,stage='train_val'):
        """
        batch is a Dict containing:
            - corrds: FloatTensor [bsz, len, n_atoms, 3], coordinates of proteins
            - corrd_mask: BooltTensor [bsz, len], where valid coordinates
                are set True, otherwise False
            - lengths: int [bsz, len], protein sequence lengths
            - tokens: LongTensor [bsz, len], sequence of amino acids  
            - mut_ids: LongTensor [bsz], mutation ids 
            - score_fermis: FloatTensor [bsz], fermi energy scores 
            - append_tensors: FloatTensor [bsz, 42], appended tensors
               
        """
        pre_ddg = self.model(batch)
        pre = None
        if not isinstance(pre_ddg,torch.Tensor):
            pre_ddg, pre = pre_ddg
            pre = pre.squeeze()
            if self.eval_loss1 is None:
                self.eval_loss1 = MeanMetric().to(self.eval_loss.device)
                self.eval_loss2 = MeanMetric().to(self.eval_loss.device)
        if len(pre_ddg.size())==2:
            pre_ddg = pre_ddg.squeeze(1)
        batch['ddG'] = batch['ddG'].reshape(-1)
        if stage == 'train_val':
            if self.criterion.__class__.__name__ == 'RANKLoss':
                loss, logging_output = self.criterion(pre_ddg, batch['ddG'])
            elif self.criterion.__class__.__name__ == 'BINLoss':
                # breakpoint()
                loss, logging_output = self.criterion(pre_ddg,batch['ddG'],batch['loss_type'],batch['std_ratio'],batch['loss_ratio'])
            elif pre is None:
                    loss, logging_output = self.criterion(pre_ddg, batch['ddG'])
            else:
                loss, logging_output = self.criterion(pre_ddg, batch['ddG'],pre,batch['append_tensors'][:,:21])
            # log.info(f"Loss: {loss.item()}")
            if loss>100:
                breakpoint()
            return loss, logging_output
        elif stage == 'test':
            loss = 0
            loss, logging_output = self.criterion(pre_ddg, batch['ddG'])
            assert logging_output['pred_value'].shape==logging_output['y'].shape
            logging_output = {
                'pred_value': logging_output['pred_value'],
                'y': batch['ddG']
            }

            return loss, logging_output

    def training_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)
        
        self.log('lr', self.lrate, on_step=True, on_epoch=False, prog_bar=True)
        if 'regularization_strength' in logging_output:
            self.log('regularization_strength', logging_output['regularization_strength'], on_step=True, on_epoch=False, prog_bar=True)

        return {"loss": loss}
    
    def training_epoch_end(self, outputs: List[Any]):
        return 

        
            
    # -------# Evaluating #-------- #
    def on_test_epoch_start(self) -> None:
        return

    def validation_step(self, batch: Any, batch_idx: int):
        loss, logging_output = self.step(batch)
        self.eval_loss.update(loss)
        if self.eval_loss1 is not None:
            self.eval_loss1.update(logging_output['loss_mse'])
            self.eval_loss2.update(logging_output['loss_crossentropy'])
        if 'rho' in logging_output:
            self.rho_avg.update(logging_output['rho'])
            self.use_rho_avg = True
        self.fermi_scores.update(logging_output['y'].cpu())
        self.pred_scores.update(logging_output['pred_value'].cpu())
        self.pdb_chain += [batch['name']]*len(batch['ddG']) if isinstance(batch['name'],str) else batch['name']
        self.dataset_name += [batch['dataset']]*len(batch['ddG']) if isinstance(batch['dataset'],str) else batch['dataset']

        return {"loss": loss}

    def validation_epoch_end(self, outputs: List[Any]):
        log_key = 'test' if self.stage == 'test' else 'val'

        # compute metrics averaged over the whole dataset
        eval_loss = self.eval_loss.compute()
        self.eval_loss.reset()
        
        if self.eval_loss1 is not None:
            eval_loss1 = self.eval_loss1.compute()
            self.eval_loss1.reset()
            eval_loss2 = self.eval_loss2.compute()
            self.eval_loss2.reset()
            self.log(f"{log_key}/loss_mse", eval_loss1, on_step=False, on_epoch=True, prog_bar=True)
            self.log(f"{log_key}/loss_crossentropy", eval_loss2, on_step=False, on_epoch=True, prog_bar=True)
        if self.use_rho_avg:
            rho_avg = self.rho_avg.compute()
            self.rho_avg.reset()
            self.log(f"{log_key}/rho_avg", rho_avg, on_step=False, on_epoch=True, prog_bar=True)
        
        fermi_scores = self.fermi_scores.compute()
        self.fermi_scores.reset()
        
        pred_scores = self.pred_scores.compute()
        self.pred_scores.reset()

        pdb_chain = self.pdb_chain
        self.pdb_chain = []
        
        dataset_name = self.dataset_name
        self.dataset_name = []
        
        rhos = cal_roh(pred_scores,fermi_scores,pdb_chain,dataset_name)
        # rhos = cal_rho_by_chain(pred_scores,fermi_scores,pdb_chain,dataset_name)
        self.log(f"{log_key}/loss", eval_loss, on_step=False, on_epoch=True, prog_bar=True)
        for k1,v1 in list(reversed(list(rhos.items()))):
            for k2,v2 in v1.items():

                self.log(f"{log_key}/rho_{k1.split('_')[0]}_{k2}", v2, on_step=False, on_epoch=True, prog_bar=False)


        if self.stage == 'fit':

            self.predict_epoch_end(results=None)

        super().validation_epoch_end(outputs)

    # -------# Inference/Prediction #-------- #
    def forward(self, batch, return_ids=False):
        pass

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0, log_metrics=True) -> Any:
        loss, logging_output = self.step(batch,stage='test')
        # log other metrics
        # self.eval_loss.update(loss)
        # if self.eval_loss1 is not None:
        #     self.eval_loss1.update(logging_output['loss_mse'])
        #     self.eval_loss2.update(logging_output['loss_crossentropy'])
        # if 'rho' in logging_output:
        #     self.rho_avg.update(logging_output['rho'])
        #     self.use_rho_avg = True
        assert logging_output['y'].shape==logging_output['pred_value'].shape
        
        self.fermi_scores.update(logging_output['y'].cpu())
        self.pred_scores.update(logging_output['pred_value'].cpu())
        
        self.mut_ids.update(batch['mut_ids'])
        self.append_tensors.update(batch['append_tensors'])
        if len(self.fermi_scores.compute())!=len(self.pred_scores.compute()):
            breakpoint()
        self.pdb_chain += [batch['name']]*len(batch['ddG']) if isinstance(batch['name'],str) else batch['name']
        self.dataset_name += [batch['dataset']]*len(batch['ddG']) if isinstance(batch['dataset'],str) else batch['dataset']
        
        return {"loss": loss}

    def predict_epoch_end(self, results: List[Any]) -> None:
        if self.stage == 'fit':
            return
        log_key = 'test'

        # # compute metrics averaged over the whole dataset
        # eval_loss = self.eval_loss.compute()
        # self.eval_loss.reset()
        # if self.eval_loss1 is not None:
        #     eval_loss1 = self.eval_loss1.compute()
        #     self.eval_loss1.reset()
        #     eval_loss2 = self.eval_loss2.compute()
        #     self.eval_loss2.reset()
        #     self.log(f"{log_key}/loss_mse", eval_loss1, on_step=False, on_epoch=True, prog_bar=True)
        #     self.log(f"{log_key}/loss_crossentropy", eval_loss2, on_step=False, on_epoch=True, prog_bar=True)
        
        if self.use_rho_avg:
            rho_avg = self.rho_avg.compute()
            self.rho_avg.reset()
            self.log(f"{log_key}/rho_avg", rho_avg, on_step=False, on_epoch=True, prog_bar=True)
        
        fermi_scores = self.fermi_scores.compute()
        self.fermi_scores.reset()
        
        pred_scores = self.pred_scores.compute()
        self.pred_scores.reset()
        
        pdb_chain = self.pdb_chain
        self.pdb_chain = []
        
        dataset_name = self.dataset_name
        self.dataset_name = []
        
        pdb_chain_set = set(pdb_chain)
        pdb_chain = np.array(pdb_chain)
        
        mut_ids = self.mut_ids.compute().cpu()
        append_tensors = self.append_tensors.compute().cpu()
        indices = torch.argmax(append_tensors[:,21:], dim=1)

        if len(np.unique(pdb_chain))==522: # domainome, return median rho
            # use cal_rho_by_chain
            rhos = cal_rho_by_chain(pred_scores,fermi_scores,pdb_chain,dataset_name,mut_ids,indices)
        else:
            rhos = cal_roh(pred_scores,fermi_scores,pdb_chain,dataset_name)

        save_result = [pred_scores,fermi_scores,pdb_chain,dataset_name]
        torch.save(save_result, 'pred_result.pt')
        
        for k,v in rhos.items():
            if k[:3] == 'avg' or k[:3] == 'med':
                self.log(f"{log_key}/rho_{k}", v, on_step=False, on_epoch=True, prog_bar=False)

    def save_prediction(self, results, saveto=None):
        pass

    def esm_refine(self, pred_ids, only_mask=False):
        pass

