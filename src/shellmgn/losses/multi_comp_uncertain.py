import importlib

import torch
from torch import nn


class MultiComponentLossWithUncertainty(nn.Module):
	def __init__(self, base_loss_module, base_loss_fn, num_components=3):
		super(MultiComponentLossWithUncertainty, self).__init__()
		self.log_vars = nn.Parameter(torch.zeros(num_components))
		base_loss_module = importlib.import_module(base_loss_module)
		self.components = num_components
		self.base_loss_fn = getattr(base_loss_module, base_loss_fn)()

	def forward(self, pred, target):
		assert pred.shape == target.shape, "Mismatch in shape between pred and target"

		total_loss = 0.0
		for i in range(self.components):
			losses = self.base_loss_fn(pred[:, i], target[:, i])
			precision = torch.exp(-self.log_vars[i])
			weighted_loss = precision * losses + self.log_vars[i]
			total_loss += weighted_loss

		return total_loss
