import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    
    Args:
        alpha (float): Weighting factor [0, 1]
        gamma (float): Focusing parameter
        reduction (str): Specifies the reduction to apply
    """
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss
    
    Args:
        num_classes (int): Number of classes
        smoothing (float): Smoothing parameter
    """
    def __init__(self, num_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.smoothing = smoothing
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(inputs, dim=-1)
        targets = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.smoothing) * targets + self.smoothing / self.num_classes
        loss = (-targets * log_probs).sum(dim=-1).mean()
        return loss

class MixupLoss(nn.Module):
    """
    Mixup Loss for data augmentation
    
    Args:
        criterion (nn.Module): Base loss function
    """
    def __init__(self, criterion: nn.Module):
        super().__init__()
        self.criterion = criterion
        
    def forward(self, pred: torch.Tensor, y_a: torch.Tensor, y_b: torch.Tensor, lam: float) -> torch.Tensor:
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

def get_loss_function(loss_name: str, num_classes: int, **kwargs) -> nn.Module:
    """
    Get loss function by name
    
    Args:
        loss_name (str): Name of the loss function
        num_classes (int): Number of classes
        **kwargs: Additional arguments for the loss function
        
    Returns:
        nn.Module: Loss function
    """
    loss_name = loss_name.lower()
    
    if loss_name == 'crossentropy' or loss_name == 'ce':
        return nn.CrossEntropyLoss(**kwargs)
    elif loss_name == 'focal':
        return FocalLoss(**kwargs)
    elif loss_name == 'label_smoothing':
        return LabelSmoothingLoss(num_classes=num_classes, **kwargs)
    elif loss_name == 'mse':
        return nn.MSELoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")
