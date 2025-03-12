import torch
from torch import nn
class LogCoshLoss_old(torch.nn.Module): ## small weights scaled, high become negligible
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, pred, true, epsilon=1e-8):
        log_cosh_loss = torch.log(torch.cosh(pred - true))
        running_losses = torch.mean(log_cosh_loss, dim=0)
        raw_weights = 1 / (running_losses + epsilon)
        weights = torch.softmax(raw_weights, dim=0)
        loss = torch.sum(weights * running_losses)

        return loss

class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, pred, true):
        loss = torch.mean(torch.log(torch.cosh(pred - true)))
        return loss

class WeightedLogCoshLoss(nn.Module): ## weights r explicitly provided
    def __init__(self, weights):
        super().__init__()
        self.weights = weights

    def forward(self, preds, trues):
        loss = 0
        for i, weight in enumerate(self.weights):
            diff = preds[:, i] - trues[:, i]
            loss += weight * torch.mean(torch.log(torch.cosh(diff)))
        return loss

class MultiComponentLoss(nn.Module):
    def __init__(self, num_components=3):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_components))

    def forward(self, preds, trues):
        epsilon = 1e-8
        losses = []
        for i in range(len(preds)):
            log_cosh_loss = torch.log(torch.cosh(preds[:, i] - trues[:, i] + epsilon))
            losses.append(torch.mean(log_cosh_loss))

        total_loss = 0
        for i, loss_component in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            total_loss += precision * loss_component + self.log_vars[i]

        return total_loss

