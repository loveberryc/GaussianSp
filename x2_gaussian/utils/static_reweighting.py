"""
s1: Projection Reliability-Weighted Static Warm-Up

This module implements two approaches for static warm-up reweighting:

1. StaticReweightingManager (方案B: residual-based)
   - Maintains per-projection EMA residuals
   - Down-weights projections with high residuals (motion-affected)
   - No learnable parameters, uses fixed functional mapping

2. LearnableStaticReweighting (方案A: learnable weights)
   - Introduces learnable parameters α_j for each projection
   - w_j = sigmoid(α_j) ∈ (0,1)
   - Gradients naturally push α_j down for high-residual projections
   - Includes L_mean regularization to prevent all weights from collapsing

Core idea:
- During static warm-up, some projections are more consistent with the static assumption
  (e.g., at end-inspiration/end-expiration phases) while others conflict with it
- By reweighting, canonical Gaussians focus on static-consistent views
- This results in a cleaner canonical representation for the dynamic phase

Usage (方案B - residual):
    manager = StaticReweightingManager(
        num_projections=N,
        burnin_steps=1500,
        ema_beta=0.1,
        tau=0.3,
        weight_type="exp"
    )
    weight = manager.get_weight_and_update(proj_idx, loss_value, iteration)

Usage (方案A - learnable):
    learnable_weights = LearnableStaticReweighting(
        num_projections=N,
        target_mean=0.85,
        burnin_steps=1000
    )
    # Add to optimizer: optimizer.add_param_group({'params': learnable_weights.parameters(), 'lr': 0.01})
    w_j = learnable_weights.get_weight(proj_idx, iteration)
    L_mean = learnable_weights.compute_mean_regularization_loss()
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Optional, Literal, Tuple


class StaticReweightingManager:
    """
    Manages per-projection EMA residuals and computes weights for static warm-up.
    
    Attributes:
        num_projections: Total number of training projections
        burnin_steps: Number of steps to collect EMA residuals before applying weights
        ema_beta: EMA momentum factor (0 < beta <= 1)
        tau: Temperature parameter for weight computation
        weight_type: "exp" for exp(-E/tau), "inv" for 1/(1+E/tau)
        ema_residuals: Tensor of shape [N] storing EMA residuals for each projection
        update_counts: Tensor of shape [N] tracking how many times each projection was sampled
    """
    
    def __init__(
        self,
        num_projections: int,
        burnin_steps: int = 1500,
        ema_beta: float = 0.1,
        tau: float = 0.3,
        weight_type: Literal["exp", "inv"] = "exp",
        device: str = "cuda"
    ):
        """
        Initialize the StaticReweightingManager.
        
        Args:
            num_projections: Total number of training projections (N)
            burnin_steps: Steps to collect EMA before applying weights
            ema_beta: EMA update rate (higher = faster adaptation)
            tau: Temperature for weight computation (lower = more aggressive down-weighting)
            weight_type: Weight function type ("exp" or "inv")
            device: Device for tensors
        """
        self.num_projections = num_projections
        self.burnin_steps = burnin_steps
        self.ema_beta = ema_beta
        self.tau = tau
        self.weight_type = weight_type
        self.device = device
        
        # Initialize EMA residuals to 0 (will be updated during training)
        self.ema_residuals = torch.zeros(num_projections, device=device)
        
        # Track how many times each projection has been sampled
        self.update_counts = torch.zeros(num_projections, dtype=torch.long, device=device)
        
        # Cache for computed weights (updated periodically)
        self._cached_weights = torch.ones(num_projections, device=device)
        self._cache_valid = False
        
        # Statistics for logging
        self.total_updates = 0
        
    def update_ema(self, proj_idx: int, loss_value: float) -> None:
        """
        Update the EMA residual for a specific projection.
        
        Args:
            proj_idx: Index of the projection (usually viewpoint_cam.uid)
            loss_value: Current reconstruction loss for this projection
        """
        # Handle first update for this projection (initialize to actual value)
        if self.update_counts[proj_idx] == 0:
            self.ema_residuals[proj_idx] = loss_value
        else:
            # EMA update: E_j = (1 - beta) * E_j + beta * L_j
            self.ema_residuals[proj_idx] = (
                (1 - self.ema_beta) * self.ema_residuals[proj_idx] + 
                self.ema_beta * loss_value
            )
        
        self.update_counts[proj_idx] += 1
        self.total_updates += 1
        self._cache_valid = False  # Invalidate weight cache
        
    def compute_weight(self, proj_idx: int) -> float:
        """
        Compute the weight for a specific projection based on its EMA residual.
        
        The weight is computed as:
        - exp(-E_j / tau) if weight_type == "exp"
        - 1 / (1 + E_j / tau) if weight_type == "inv"
        
        Args:
            proj_idx: Index of the projection
            
        Returns:
            Computed weight (not normalized)
        """
        E_j = self.ema_residuals[proj_idx].item()
        
        if self.weight_type == "exp":
            w_j = np.exp(-E_j / self.tau)
        else:  # "inv"
            w_j = 1.0 / (1.0 + E_j / self.tau)
            
        return w_j
    
    def compute_all_weights(self) -> torch.Tensor:
        """
        Compute weights for all projections.
        
        Returns:
            Tensor of shape [N] with weights for all projections
        """
        if self.weight_type == "exp":
            weights = torch.exp(-self.ema_residuals / self.tau)
        else:  # "inv"
            weights = 1.0 / (1.0 + self.ema_residuals / self.tau)
        
        return weights
    
    def get_normalized_weight(self, proj_idx: int) -> float:
        """
        Get the normalized weight for a projection.
        
        Normalization ensures that the mean weight across all projections is 1.0,
        so the overall loss scale is preserved.
        
        Args:
            proj_idx: Index of the projection
            
        Returns:
            Normalized weight
        """
        if not self._cache_valid:
            self._cached_weights = self.compute_all_weights()
            mean_weight = self._cached_weights.mean()
            if mean_weight > 0:
                self._cached_weights = self._cached_weights / mean_weight
            self._cache_valid = True
            
        return self._cached_weights[proj_idx].item()
    
    def get_weight_and_update(
        self,
        proj_idx: int,
        loss_value: float,
        iteration: int
    ) -> float:
        """
        Main interface: update EMA and return the weight for this projection.
        
        During burn-in (iteration < burnin_steps), returns 1.0 but still updates EMA.
        After burn-in, returns the normalized weight based on EMA residuals.
        
        Args:
            proj_idx: Index of the projection (viewpoint_cam.uid)
            loss_value: Current reconstruction loss (L1 + lambda_dssim * D-SSIM)
            iteration: Current training iteration
            
        Returns:
            Weight to apply to the loss (1.0 during burn-in, normalized weight after)
        """
        # Always update EMA residual
        self.update_ema(proj_idx, loss_value)
        
        # During burn-in, return 1.0 (uniform weighting)
        if iteration < self.burnin_steps:
            return 1.0
        
        # After burn-in, return normalized weight
        return self.get_normalized_weight(proj_idx)
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the current state of the reweighting manager.
        
        Returns:
            Dictionary with statistics for logging/debugging
        """
        weights = self.compute_all_weights()
        mean_weight = weights.mean().item()
        
        # Normalize for statistics
        if mean_weight > 0:
            norm_weights = weights / mean_weight
        else:
            norm_weights = weights
            
        # Find projections with extreme weights
        min_idx = weights.argmin().item()
        max_idx = weights.argmax().item()
        
        return {
            "ema_residual_mean": self.ema_residuals.mean().item(),
            "ema_residual_std": self.ema_residuals.std().item(),
            "ema_residual_min": self.ema_residuals.min().item(),
            "ema_residual_max": self.ema_residuals.max().item(),
            "weight_mean": mean_weight,
            "weight_std": weights.std().item(),
            "weight_min": weights.min().item(),
            "weight_max": weights.max().item(),
            "norm_weight_min": norm_weights.min().item(),
            "norm_weight_max": norm_weights.max().item(),
            "min_weight_proj_idx": min_idx,
            "max_weight_proj_idx": max_idx,
            "total_updates": self.total_updates,
            "projections_seen": (self.update_counts > 0).sum().item(),
        }
    
    def get_weight_distribution(self) -> tuple:
        """
        Get the weight distribution for visualization.
        
        Returns:
            Tuple of (projection_indices, normalized_weights)
        """
        weights = self.compute_all_weights()
        mean_weight = weights.mean()
        if mean_weight > 0:
            weights = weights / mean_weight
        
        indices = torch.arange(self.num_projections, device=self.device)
        return indices.cpu().numpy(), weights.cpu().numpy()


class LearnableStaticReweighting(nn.Module):
    """
    方案A: Learnable per-projection weights for static warm-up.
    
    Introduces learnable parameters α_j for each projection j, where:
    - w_j = sigmoid(α_j) ∈ (0,1)
    - Static loss becomes: L_static = [Σ_j w_j * L_j] / [Σ_j w_j]
    - Plus mean regularization: L_mean = (mean(w_j) - ρ)²
    
    The gradient naturally pushes α_j down for projections with high L_j,
    causing the canonical Gaussians to focus on static-consistent projections.
    
    Attributes:
        num_projections: Total number of training projections (N)
        target_mean: Target mean weight ρ (typically 0.8-0.9)
        burnin_steps: Steps before applying learnable weights (use w_j=1 during burn-in)
        alpha: nn.Parameter of shape [N], the learnable logits α_j
    """
    
    def __init__(
        self,
        num_projections: int,
        target_mean: float = 0.85,
        burnin_steps: int = 1000,
        device: str = "cuda"
    ):
        """
        Initialize learnable per-projection weights.
        
        Args:
            num_projections: Total number of training projections (N)
            target_mean: Target mean weight ρ for L_mean regularization
            burnin_steps: Steps before applying learnable weights
            device: Device for parameters
        """
        super().__init__()
        
        self.num_projections = num_projections
        self.target_mean = target_mean
        self.burnin_steps = burnin_steps
        self.device = device
        
        # Initialize α_j such that initial w_j ≈ target_mean
        # w_j = sigmoid(α_j) = target_mean
        # α_j = log(target_mean / (1 - target_mean))
        alpha_init = math.log(target_mean / (1 - target_mean + 1e-8))
        
        # Learnable parameters: α_j for each projection
        self.alpha = nn.Parameter(
            torch.full((num_projections,), alpha_init, device=device)
        )
        
        # Statistics tracking
        self.total_calls = 0
        
    def get_all_weights(self) -> torch.Tensor:
        """
        Compute weights for all projections.
        
        Returns:
            Tensor of shape [N] with w_j = sigmoid(α_j) for all projections
        """
        return torch.sigmoid(self.alpha)
    
    def get_weight(self, proj_idx: int, iteration: int) -> torch.Tensor:
        """
        Get the weight for a specific projection.
        
        During burn-in, returns 1.0 (uniform weighting).
        After burn-in, returns w_j = sigmoid(α_j).
        
        Args:
            proj_idx: Index of the projection (viewpoint_cam.uid)
            iteration: Current training iteration
            
        Returns:
            Weight w_j as a scalar tensor (differentiable)
        """
        self.total_calls += 1
        
        # During burn-in, use uniform weights
        if iteration < self.burnin_steps:
            return torch.ones(1, device=self.device)
        
        # After burn-in, use learnable weights
        return torch.sigmoid(self.alpha[proj_idx:proj_idx+1])
    
    def get_weight_for_batch(self, proj_indices: torch.Tensor, iteration: int) -> torch.Tensor:
        """
        Get weights for a batch of projections.
        
        Args:
            proj_indices: Tensor of projection indices [B]
            iteration: Current training iteration
            
        Returns:
            Tensor of weights [B] (differentiable)
        """
        # During burn-in, use uniform weights
        if iteration < self.burnin_steps:
            return torch.ones(len(proj_indices), device=self.device)
        
        # After burn-in, use learnable weights
        return torch.sigmoid(self.alpha[proj_indices])
    
    def compute_mean_regularization_loss(self) -> torch.Tensor:
        """
        Compute the mean regularization loss L_mean.
        
        L_mean = (mean(w_j) - ρ)²
        
        This prevents all weights from collapsing to 0 or staying at 1.
        
        Returns:
            L_mean as a scalar tensor (differentiable)
        """
        all_weights = self.get_all_weights()
        mean_weight = all_weights.mean()
        return (mean_weight - self.target_mean) ** 2
    
    def compute_weighted_loss(
        self,
        proj_idx: int,
        loss_value: torch.Tensor,
        iteration: int,
        lambda_mean: float = 0.05
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute the weighted loss and mean regularization for a single sample.
        
        For a single sample, the weighted loss is simply: w_j * L_j
        The mean regularization is computed over all α_j.
        
        Args:
            proj_idx: Index of the projection
            loss_value: The render loss L_j (tensor, must be differentiable)
            iteration: Current training iteration
            lambda_mean: Weight for L_mean regularization
            
        Returns:
            Tuple of (weighted_loss, L_mean, w_j)
        """
        w_j = self.get_weight(proj_idx, iteration)
        weighted_loss = w_j * loss_value
        
        # Only compute L_mean after burn-in
        if iteration < self.burnin_steps:
            L_mean = torch.zeros(1, device=self.device)
        else:
            L_mean = self.compute_mean_regularization_loss()
        
        return weighted_loss, lambda_mean * L_mean, w_j.item() if w_j.numel() == 1 else w_j.mean().item()
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the learnable weights.
        
        Returns:
            Dictionary with statistics for logging/debugging
        """
        with torch.no_grad():
            weights = self.get_all_weights()
            alpha = self.alpha
            
            return {
                "alpha_mean": alpha.mean().item(),
                "alpha_std": alpha.std().item(),
                "alpha_min": alpha.min().item(),
                "alpha_max": alpha.max().item(),
                "weight_mean": weights.mean().item(),
                "weight_std": weights.std().item(),
                "weight_min": weights.min().item(),
                "weight_max": weights.max().item(),
                "target_mean": self.target_mean,
                "mean_deviation": abs(weights.mean().item() - self.target_mean),
                "total_calls": self.total_calls,
                "min_weight_proj_idx": weights.argmin().item(),
                "max_weight_proj_idx": weights.argmax().item(),
            }
    
    def get_weight_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the weight distribution for visualization.
        
        Returns:
            Tuple of (projection_indices, weights)
        """
        with torch.no_grad():
            weights = self.get_all_weights()
            indices = torch.arange(self.num_projections, device=self.device)
            return indices.cpu().numpy(), weights.cpu().numpy()


def apply_s1_preset(opt):
    """
    Print s1 static reweighting configuration info.
    
    Args:
        opt: OptimizationParams object
    """
    if not opt.use_static_reweighting:
        return
    
    print("=" * 60)
    print("S1 STATIC REWEIGHTING ACTIVATED")
    print("=" * 60)
    print("Projection Reliability-Weighted Static Warm-Up:")
    print(f"  - Method: {opt.static_reweight_method}")
    print(f"  - Burn-in steps: {opt.static_reweight_burnin_steps}")
    
    if opt.static_reweight_method == "residual":
        # 方案B: Residual-based reweighting
        print(f"  - EMA beta: {opt.static_reweight_ema_beta}")
        print(f"  - Temperature tau: {opt.static_reweight_tau}")
        print(f"  - Weight type: {opt.static_reweight_weight_type}")
        print("")
        print("During static warm-up (coarse stage):")
        print(f"  1. First {opt.static_reweight_burnin_steps} steps: collect EMA residuals with uniform weights")
        print("  2. After burn-in: apply w_j based on EMA residuals")
        print("     - w_j = exp(-E_j / tau) or 1/(1+E_j/tau)")
        print("     - Projections with high residuals (motion-affected) -> lower weight")
        print("     - Projections with low residuals (static-consistent) -> higher weight")
        print("  3. Weights are normalized so mean(w_j) = 1")
    
    elif opt.static_reweight_method == "learnable":
        # 方案A: Learnable per-projection weights
        print(f"  - Target mean (ρ): {opt.static_reweight_target_mean}")
        print(f"  - λ_mean: {opt.lambda_static_reweight_mean}")
        print(f"  - Learning rate: {opt.static_reweight_lr}")
        print("")
        print("During static warm-up (coarse stage):")
        print(f"  1. First {opt.static_reweight_burnin_steps} steps: use uniform weights (w_j=1)")
        print("  2. After burn-in: use learnable weights")
        print("     - w_j = sigmoid(α_j) ∈ (0,1)")
        print("     - α_j is optimized jointly with Gaussians")
        print("     - High-residual projections naturally get lower α_j -> lower w_j")
        print(f"  3. L_mean = (mean(w_j) - {opt.static_reweight_target_mean})² regularization")
        print(f"     - Weighted by λ_mean = {opt.lambda_static_reweight_mean}")
        print("     - Prevents weights from collapsing to 0 or all staying at 1")
    
    print("")
    print("Dynamic stage (fine) is NOT affected by s1.")
    print("=" * 60)
