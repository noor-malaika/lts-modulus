import torch
from torch import nn


class WeightedLogCoshLoss(nn.Module):
	def __init__(self, weights):
		super().__init__()
		self.weights = weights

	def forward(self, preds, trues):
		loss = 0
		for i, weight in enumerate(self.weights):
			diff = preds[:, i] - trues[:, i]
			loss += weight * torch.mean(torch.log(torch.cosh(diff)))
		return loss
