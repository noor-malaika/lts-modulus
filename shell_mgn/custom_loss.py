import torch
from torch import nn
import importlib


class LogCoshLoss_old(torch.nn.Module):  ## small weights scaled, high become negligible
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


class WeightedLogCoshLoss(nn.Module):  ## weights r explicitly provided
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


class MultiComponentLossWithUncertainty(nn.Module):
    def __init__(self, base_loss_module, base_loss_fn, num_components=3):
        """
        Args:
            num_components (int): Number of loss components (e.g., outputs).
            base_loss_fn (function): The base loss function (MSE, MAE, etc.).
        """
        super(MultiComponentLossWithUncertainty, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_components))  # Learnable weights
        base_loss_module = importlib.import_module(base_loss_module)
        self.components = num_components
        self.base_loss_fn = getattr(
            base_loss_module, base_loss_fn
        )()  # Single loss function for all components

    def forward(self, pred, target):
        """
        Computes weighted loss for multiple components.

        Args:
            pred (Tensor): Predicted values [batch, num_components]
            target (Tensor): Ground truth values [batch, num_components]

        Returns:
            Tensor: Weighted total loss.
        """
        assert pred.shape == target.shape, "Mismatch in shape between pred and target"

        total_loss = 0.0
        for i in range(self.components):  # Iterate over components
            losses = self.base_loss_fn(pred[:, i], target[:, i])  # Compute loss per component
            precision = torch.exp(-self.log_vars[i])  # Precision term (1/σ²)
            weighted_loss = (
                precision * losses + self.log_vars[i]
            )  # Weighted loss
            total_loss += weighted_loss

        return total_loss


class MRAELoss(torch.nn.Module):
    def __init__(self):
        super(MRAELoss, self).__init__()

    def forward(self, inputs, targets):
        return torch.mean(torch.abs((inputs - targets) / (torch.abs(targets) + 1e-8)))
