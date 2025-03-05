import torch

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, pred, true):
        loss = torch.mean(torch.log(torch.cosh(pred - true)))
        return loss