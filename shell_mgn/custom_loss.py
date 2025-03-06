import torch

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