import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
from collections import OrderedDict



# Настройка логирования
logger = logging.getLogger(__name__)


class OptimizationError(Exception):
    """Кастомное исключение для ошибок оптимизации."""
    pass


def cosine_similarity_loss(x: Tensor, y: Tensor) -> Tensor:
    """
    Compute cosine similarity loss between two tensors.
    
    Args:
        x: First tensor
        y: Second tensor
        
    Returns:
        Negative cosine similarity loss
    """
    if not isinstance(x, Tensor) or not isinstance(y, Tensor):
        raise TypeError("Both inputs must be torch.Tensors")
    
    if x.shape != y.shape:
        raise ValueError("Input tensors must have the same shape")
    
    x_norm = torch.norm(x)
    y_norm = torch.norm(y)
    
    if x_norm == 0 or y_norm == 0:
        return x_norm
    
    cosine_sim = x @ y / (x_norm * y_norm)
    return -cosine_sim


def create_convolution_kernel(width: int, kernel_type: str = "lorenz") -> Tensor:
    """
    Create convolution kernel for spectrum smoothing.
    
    Args:
        width: Kernel width parameter
        kernel_type: Type of kernel ('lorenz' or 'gauss')
        
    Returns:
        Convolution kernel tensor
    """
    if width <= 0:
        raise ValueError("Width must be positive")
    
    kernel_range = torch.arange(-4 * width, 4 * width + 1)
    
    if kernel_type == "lorenz":
        kernel = 0.5 / width * ((0.5 * width) ** 2 / ((0.5 * width) ** 2 + kernel_range ** 2))
    elif kernel_type == "gauss":
        kernel = 0.1 * torch.exp(-(kernel_range ** 2) / width ** 2)
    else:
        raise ValueError("kernel_type must be 'lorenz' or 'gauss'")
    
    return kernel.view(1, 1, -1)


def optimize_model(
    model: nn.Module,
    target: Tensor,
    learning_rate: float = 1e-2,
    num_epochs: int = 500,
    use_regularization: bool = False,
    lambda_reg: float = 0.5,
    use_only_center_of_mass_loss: bool = False
) -> Tuple[nn.Module, Tensor, Tensor]:
    """
    Optimize model parameters to fit target spectrum.
    
    Args:
        model: Model to optimize
        target: Target spectrum tensor [frequency, intensity]
        learning_rate: Optimization learning rate
        num_epochs: Number of training epochs
        use_regularization: Whether to use regularization
        lambda_reg: Regularization strength
        use_only_center_of_mass_loss: Use only center of mass loss for pretuning
        
    Returns:
        Tuple of (optimized_model, initial_metric, final_metric)
    """
    if not isinstance(model, nn.Module):
        raise TypeError("model must be a nn.Module")
    
    if target.shape[1] != 2:
        raise ValueError("Target must have shape [n_points, 2]")
    
    # Setup optimization
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = cosine_similarity_loss
    # Calculate initial metric
    with torch.no_grad():
        initial_output = model(target[:, 0])
        initial_metric = torch.log(1 + loss_fn(initial_output[:, 1], target[:, 1]))
    
    # Training loop
    loss_history = []
    interrupted = False
    final_epoch = 0
    try:
        for epoch in range(num_epochs):
            final_epoch = epoch
            optimizer.zero_grad()
            
            # Forward pass
            output = model(target[:, 0])
            if use_only_center_of_mass_loss:
                main_loss = _compute_center_of_mass_loss(output, target)
            else:
                main_loss = torch.log(1 + loss_fn(output[:, 1], target[:, 1]))
            
            reg_loss = 0.0
            # Add regularization if requested
            if use_regularization:
                reg_loss = _compute_regularization_loss(model, lambda_reg)
            total_loss = main_loss + reg_loss
            
            # Backward pass
            total_loss.backward(retain_graph=True)
            optimizer.step()
            
            loss_history.append(total_loss.item())
            
            if epoch % 50 == 0:
                logger.info(f"Epoch {epoch}, Loss: {total_loss.item():.6f}")
    except KeyboardInterrupt:
        interrupted = True
        logger.info(f"Optimization interrupted by user at epoch {final_epoch}")
                
    # Calculate final metric
    with torch.no_grad():
        final_output = model(target[:, 0])
        final_metric = torch.log(1 + loss_fn(final_output[:, 1], target[:, 1]))
    
    for module in model.models:
        module.ppm0.data.copy_(module.v0 / module.freq)
    
    status = "complited"
    if interrupted:
        status = "stoped"
    logger.info(f"Optimization {status}: initial_metric={initial_metric.item():.6f}, "
                f"final_metric={final_metric.item():.6f}")
    
    return model, initial_metric, final_metric


def pretune_model(model: nn.Module,
    target: Tensor,
    width: int = 100,
    learning_rate: float = 1e-2,
    num_epochs: int = 500,
    use_regularization: bool = False,
    lambda_reg: float = 0.5
)  -> Tuple[nn.Module, Tensor, Tensor]:

    """
    Preoptimize model parameters to fit chemical shifts to broadened target spectrum.
    
    Args:
        model: Model to optimize
        target: Target spectrum tensor [frequency, intensity]
        width: Smoothing kernel width
        learning_rate: Optimization learning rate
        num_epochs: Number of training epochs
        use_regularization: Whether to use regularization
        lambda_reg: Regularization strength
        
    Returns:
        Tuple of (optimized_model, initial_metric, final_metric)
    """
    # Prepare target spectrum
    if width > 0:
        kernel = create_convolution_kernel(width, "lorenz")
        target_smoothed = F.conv1d(
            target[:, 1].clone().view(1, 1, -1), 
            kernel, 
            padding=width * 4
        ).squeeze()
    else:
        target_smoothed = target[:, 1]

    #save initial widths
    widths_dict = {i: model.w0.clone() for i, model in enumerate(model.models)}
    #Prepare model
    for name, param in model.named_parameters():
        if 'w0' in name:
            for p in param.data:
                p.copy_(width/4.5)
        if 'J0' in name or 'r0' in name or 'w0' in name or 'r_global' in name:
            param.requires_grad = False

    
    model, initial_metric, final_metric = optimize_model(model = model, 
                                                         target = torch.stack([target[:, 0], target_smoothed], dim=1), 
                                                         learning_rate = learning_rate,
                                                         num_epochs = num_epochs,
                                                         use_regularization = use_regularization,
                                                         lambda_reg = lambda_reg                                                   )
        
    #Restore model
    for name, param in model.named_parameters():
        if 'J0' in name or 'r0' in name or 'w0' in name or 'r_global' in name:
            param.requires_grad = True

    for model_idx, saved_width in widths_dict.items():
        model.models[model_idx].w0.data.copy_(saved_width)


    return model, initial_metric, final_metric


def _compute_regularization_loss(model: nn.Module, lambda_reg: float) -> Tensor:
    """
    Compute regularization loss for model parameters.
    
    Args:
        model: Model to regularize
        lambda_reg: Regularization strength
        
    Returns:
        Regularization loss
    """
    reg_loss = 0.0
    
    for module in model.models:
        if hasattr(module, 'J0') and module.J0 is not None:
            # Regularize off-diagonal elements of J-coupling matrix
            j_matrix = module.J0
            if j_matrix.dim() == 2 and j_matrix.shape[0] >= 2 and j_matrix.shape[1] >= 2:
                reg_loss += (j_matrix[0, 1] ** 2) + (j_matrix[1, 0] ** 2)
    
    return lambda_reg * reg_loss



