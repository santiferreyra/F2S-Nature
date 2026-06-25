import torch
import torch.nn as nn

class DataDrivenLoss(nn.Module):
    def __init__(self, alpha=0.05):
        super(DataDrivenLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred, y_true):
        # Compute squared error
        squared_error = (y_pred - y_true)**2
        
        # Apply scaling factor for targets that are zero
        scaled_error = torch.where(y_true == 0, squared_error * self.alpha, squared_error)
        scaled_error = scaled_error.sum()
        
        return scaled_error

class DataDrivenLossWithL1(nn.Module):
    def __init__(self, alpha=0.05, delta=1e-5):
        super(DataDrivenLossWithL1, self).__init__()
        self.alpha = alpha
        self.delta = delta

    def forward(self, y_pred, y_true, drugs, side_effects):
        scaled_error = 0

        # Compute squared error
        squared_error = (y_pred - y_true) ** 2

        # Scale the error for targets that are 0
        scaled_error = torch.where(y_true == 0, squared_error * self.alpha, squared_error)
        scaled_error = scaled_error.mean()

        # L1 loss for signatures
        signature_loss = drugs.abs().mean() + side_effects.abs().mean()

        return scaled_error + self.delta * signature_loss