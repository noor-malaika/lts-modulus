import torch


class MRAELoss(torch.nn.Module):
	def __init__(self):
		super(MRAELoss, self).__init__()

	def forward(self, inputs, targets):
		return torch.mean(torch.abs(inputs - targets) / (torch.abs(targets) + 1e-8))
