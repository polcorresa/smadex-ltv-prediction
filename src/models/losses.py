"""
Custom loss functions for multi-task LTV prediction
Implements ODMN order-preserving loss + Dynamic Huber
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class OrderPreservingHuberLoss(nn.Module):
    """
    Combined loss for multi-horizon revenue prediction
    - Huber loss for robustness to outliers
    - Order-preserving constraints (D1 ≤ D7 ≤ D14)
    
    Based on: Kuaishou ODMN (CIKM 2022)
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        lambda_order: float = 0.1,
        lambda_d1: float = 0.3,
        lambda_d7: float = 0.5,
        lambda_d14: float = 0.2
    ):
        """
        Args:
            delta: Huber loss threshold
            lambda_order: Weight for order-preserving penalty
            lambda_d1/d7/d14: Weights for each horizon
        """
        super().__init__()
        self.delta = delta
        self.lambda_order = lambda_order
        self.lambda_d1 = lambda_d1
        self.lambda_d7 = lambda_d7
        self.lambda_d14 = lambda_d14
    
    def huber_loss(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Compute Huber loss"""
        diff = torch.abs(y_true - y_pred)
        
        quadratic = torch.where(
            diff <= self.delta,
            0.5 * diff ** 2,
            torch.zeros_like(diff)
        )
        
        linear = torch.where(
            diff > self.delta,
            self.delta * (diff - 0.5 * self.delta),
            torch.zeros_like(diff)
        )
        
        return quadratic + linear
    
    def forward(
        self, 
        y_pred: torch.Tensor, 
        y_true: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            y_pred: (N, 3) predictions for [D1, D7, D14]
            y_true: (N, 3) ground truth for [D1, D7, D14]
        
        Returns:
            Combined loss scalar
        """
        # Ensure correct shape
        if y_pred.dim() == 1:
            y_pred = y_pred.unsqueeze(1)
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)
        
        # Multi-horizon Huber loss
        loss_d1 = self.huber_loss(y_true[:, 0], y_pred[:, 0]).mean()
        loss_d7 = self.huber_loss(y_true[:, 1], y_pred[:, 1]).mean()
        loss_d14 = self.huber_loss(y_true[:, 2], y_pred[:, 2]).mean()
        
        huber_total = (
            self.lambda_d1 * loss_d1 + 
            self.lambda_d7 * loss_d7 + 
            self.lambda_d14 * loss_d14
        )
        
        # Order-preserving penalty
        order_penalty = (
            F.relu(y_pred[:, 0] - y_pred[:, 1]) ** 2 +  # D1 ≤ D7
            F.relu(y_pred[:, 1] - y_pred[:, 2]) ** 2    # D7 ≤ D14
        ).mean()
        
        total_loss = huber_total + self.lambda_order * order_penalty
        
        return total_loss


def lgbm_order_preserving_objective(y_pred: np.ndarray, dtrain) -> Tuple[np.ndarray, np.ndarray]:
    """
    Custom LightGBM objective for order-preserving regression
    
    Returns:
        grad: First-order gradients
        hess: Second-order gradients (Hessian diagonal)
    """
    y_true = dtrain.get_label()
    
    # Reshape for multi-output
    n_samples = len(y_true) // 3
    y_pred = y_pred.reshape(n_samples, 3)
    y_true = y_true.reshape(n_samples, 3)
    
    delta = 1.0
    lambda_order = 0.1
    
    # Huber loss gradients
    diff = y_true - y_pred
    
    # Gradient for Huber
    grad_huber = np.where(
        np.abs(diff) <= delta,
        -diff,  # Quadratic region
        -delta * np.sign(diff)  # Linear region
    )
    
    # Hessian for Huber
    hess_huber = np.where(
        np.abs(diff) <= delta,
        np.ones_like(diff),  # Quadratic region
        np.zeros_like(diff)  # Linear region (set to small value for stability)
    ) + 1e-6
    
    # Order constraint gradients
    grad_order = np.zeros_like(y_pred)
    
    # D1 ≤ D7 constraint
    violation_d1_d7 = y_pred[:, 0] > y_pred[:, 1]
    grad_order[violation_d1_d7, 0] += 2 * lambda_order * (y_pred[violation_d1_d7, 0] - y_pred[violation_d1_d7, 1])
    grad_order[violation_d1_d7, 1] -= 2 * lambda_order * (y_pred[violation_d1_d7, 0] - y_pred[violation_d1_d7, 1])
    
    # D7 ≤ D14 constraint
    violation_d7_d14 = y_pred[:, 1] > y_pred[:, 2]
    grad_order[violation_d7_d14, 1] += 2 * lambda_order * (y_pred[violation_d7_d14, 1] - y_pred[violation_d7_d14, 2])
    grad_order[violation_d7_d14, 2] -= 2 * lambda_order * (y_pred[violation_d7_d14, 1] - y_pred[violation_d7_d14, 2])
    
    # Combined gradients
    grad = grad_huber + grad_order
    hess = hess_huber
    
    # Flatten for LightGBM
    return grad.flatten(), hess.flatten()
