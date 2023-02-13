#!/usr/bin/env python3
from typing import Sequence, Union
import pdb
import torch
import torch.nn as nn


from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


class ForwardLoss(Metric):
    r"""Loss metric that simply records the loss calculated
    in forward pass
    """

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum_of_squared_errors = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        self._sum_of_squared_errors += torch.sum(output).to(self._device)
        self._num_examples += 1

    @sync_all_reduce("_sum_of_squared_errors", "_num_examples")
    def compute(self) -> Union[float, torch.Tensor]:
        if self._num_examples == 0:
            raise NotComputableError("MeanSquaredError must have at least one example before it can be computed.")
        return self._sum_of_squared_errors.item() / self._num_examples

class QuantileRegressionLoss(nn.Module):
    def __init__(self, params):
        super(QuantileRegressionLoss, self).__init__()

        self.q_lo_loss = PinballLoss(quantile=params["q_lo"])
        self.q_hi_loss = PinballLoss(quantile=params["q_hi"])
        self.mse_loss = nn.MSELoss()

        self.q_lo_weight = params['q_lo_weight']
        self.q_hi_weight = params['q_hi_weight']
        self.mse_weight = params['mse_weight']

    def forward(self, pred, target):
        loss = self.q_lo_weight * self.q_lo_loss(pred[:,0,:,:,:].squeeze(), target.squeeze()) + \
               self.q_hi_weight * self.q_hi_loss(pred[:,2,:,:,:].squeeze(), target.squeeze()) + \
               self.mse_weight * self.mse_loss(pred[:,1,:,:,:].squeeze(), target.squeeze())

        return loss

class qqquantile_regression_loss_fn(Metric):
    def __init__(self, params):
        super(quantile_regression_loss_fn, self).__init__()

        self.params = params
        self.q_lo_loss = PinballLoss(quantile=params["q_lo"])
        self.q_hi_loss = PinballLoss(quantile=params["q_hi"])
        self.mse_loss = nn.MSELoss()

    @reinit__is_reduced
    def reset(self):
        self._sum = 0
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output
        loss = self._quantile_regression_loss_fn(y_pred, y)
        self._sum += loss.item()
        self._num_examples += 1

    @sync_all_reduce("_sum", "_num_examples")
    def compute(self):
        return self._sum / self._num_examples

    def _quantile_regression_loss_fn(self, y_pred, y):
        # implement custom loss function here
        loss = self.params['q_lo_weight'] * self.q_lo_loss(y_pred[:,0,:,:,:].squeeze(), y.squeeze()) + \
        self.params['q_hi_weight'] * self.q_hi_loss(y_pred[:,2,:,:,:].squeeze(), y.squeeze()) + \
        self.params['mse_weight'] * self.mse_loss(y_pred[:,1,:,:,:].squeeze(), y.squeeze())
        return loss

class PinballLoss():

  def __init__(self, quantile=0.10, reduction='mean'):
      self.quantile = quantile
      assert 0 < self.quantile
      assert self.quantile < 1
      self.reduction = reduction

  def __call__(self, output, target):
      assert output.shape == target.shape
      loss = torch.zeros_like(target, dtype=torch.float)
      error = output - target
      smaller_index = error < 0
      bigger_index = 0 < error
      loss[smaller_index] = self.quantile * (abs(error)[smaller_index])
      loss[bigger_index] = (1-self.quantile) * (abs(error)[bigger_index])

      if self.reduction == 'sum':
        loss = loss.sum()
      if self.reduction == 'mean':
        loss = loss.mean()

      return loss