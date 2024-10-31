import torch
from torch import Tensor, nn
from torch.nn import functional as F

class MSELoss(nn.MSELoss):
    def forward(self, pred_value: Tensor, y: Tensor) -> Tensor:
        """
        pred_value: [N, ...]
        y: [N, ...]
        """
        loss_avg = super().forward(pred_value, y)
        logging_output = {
            'loss_sum': loss_avg.data,
            'pred_value': pred_value,
            'y': y
        }
        return loss_avg, logging_output