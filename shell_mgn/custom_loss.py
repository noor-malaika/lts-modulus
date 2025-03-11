import torch
from torch import nn
class LogCoshLoss(torch.nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, pred, true, epsilon=1e-8):
        log_cosh_loss = torch.log(torch.cosh(pred - true))
        running_losses = torch.mean(log_cosh_loss, dim=0)
        raw_weights = 1 / (running_losses + epsilon)
        weights = torch.softmax(raw_weights, dim=0)
        loss = torch.sum(weights * running_losses)

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

