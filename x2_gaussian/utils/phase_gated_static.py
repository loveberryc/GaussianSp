"""
X2-Gaussian s2: Phase-Gated Static Canonical Warm-Up

This module implements phase-gated weighting for the static warm-up stage.
The core idea is to learn an implicit breathing period and canonical phase,
then use a circular Gaussian window to weight projections based on their
phase proximity to the canonical phase.

Only affects the static warm-up stage (coarse); dynamic stage is unchanged.

Key concepts:
- tau_s2: Log-period parameter, T_s2 = exp(tau_s2)
- psi_s2: Phase offset for acquisition time alignment
- phi_c_s2: Canonical phase center in [-π, π)
- xi_s2: Log-window width parameter, sigma_phi = exp(xi_s2)

For each projection with acquisition time t_j:
1. Compute raw phase: phi_raw = 2π * t_j / T_s2 + psi_s2
2. Wrap to (-π, π]: phi_j = wrap(phi_raw)
3. Compute circular distance: delta = wrap(phi_j - phi_c_s2)
4. Compute weight: w_j = exp(-delta² / (2 * sigma_phi²))

The phase-gated static loss is:
  L_static_s2 = [Σ w_j * L_j] / [Σ w_j] + λ_win * (log(sigma_phi) - log(sigma_target))²
"""

import torch
import torch.nn as nn
import math


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    """
    Wrap angle to (-π, π].
    
    Args:
        x: Angle in radians (can be any value)
        
    Returns:
        Wrapped angle in (-π, π]
    """
    return ((x + math.pi) % (2 * math.pi)) - math.pi


class PhaseGatedStatic(nn.Module):
    """
    Phase-Gated Static Canonical Warm-Up (s2).
    
    Learns an implicit breathing period and canonical phase to weight projections
    during static warm-up, allowing the canonical Gaussians to focus on projections
    near a specific breathing phase rather than averaging across all phases.
    
    Args:
        sigma_target: Target window width σ_target (in radians), default 0.6 (~34°, covers ~20% of cycle)
        burnin_steps: Number of burn-in steps with uniform weights (w_j=1)
        init_period: Initial period estimate T_s2 (default 1.0, will be learned)
        device: Device to place parameters on
    """
    
    def __init__(
        self,
        sigma_target: float = 0.6,
        burnin_steps: int = 1500,
        init_period: float = 1.0,
        device: str = "cuda"
    ):
        super().__init__()
        
        self.sigma_target = sigma_target
        self.burnin_steps = burnin_steps
        self.device = device
        
        # Initialize learnable parameters
        # tau_s2: log-period, T_s2 = exp(tau_s2)
        # Initialize such that T_s2 = init_period
        tau_init = math.log(init_period)
        self.tau_s2 = nn.Parameter(torch.tensor(tau_init, device=device))
        
        # psi_s2: phase offset for acquisition time alignment
        # Initialize to 0
        self.psi_s2 = nn.Parameter(torch.tensor(0.0, device=device))
        
        # phi_c_s2: canonical phase center in [-π, π)
        # Initialize to 0 (mid-cycle)
        self.phi_c_s2 = nn.Parameter(torch.tensor(0.0, device=device))
        
        # xi_s2: log-window width, sigma_phi = exp(xi_s2)
        # Initialize such that sigma_phi = sigma_target
        xi_init = math.log(sigma_target)
        self.xi_s2 = nn.Parameter(torch.tensor(xi_init, device=device))
        
        # Statistics tracking
        self._last_weight = 1.0
        self._last_phase = 0.0
        self._weight_sum = 0.0
        self._weight_count = 0
        
    @property
    def T_s2(self) -> torch.Tensor:
        """Breathing period T_s2 = exp(tau_s2)"""
        return torch.exp(self.tau_s2)
    
    @property
    def sigma_phi(self) -> torch.Tensor:
        """Phase window width sigma_phi = exp(xi_s2)"""
        return torch.exp(self.xi_s2)
    
    def compute_phase(self, t_j: float) -> torch.Tensor:
        """
        Compute the wrapped phase φ_j for a given acquisition time t_j.
        
        Args:
            t_j: Acquisition time of the projection
            
        Returns:
            Phase φ_j in (-π, π]
        """
        # Raw phase: phi_raw = 2π * t_j / T_s2 + psi_s2
        phi_raw = 2 * math.pi * t_j / self.T_s2 + self.psi_s2
        
        # Wrap to (-π, π]
        phi_j = wrap_to_pi(phi_raw)
        
        return phi_j
    
    def compute_weight(self, t_j: float, iteration: int) -> torch.Tensor:
        """
        Compute the phase-gated weight w_j for a projection.
        
        During burn-in (iteration < burnin_steps), returns 1.0.
        After burn-in, returns exp(-delta² / (2 * sigma_phi²)).
        
        Args:
            t_j: Acquisition time of the projection
            iteration: Current training iteration
            
        Returns:
            Weight w_j (tensor for gradient flow, or float 1.0 during burn-in)
        """
        # During burn-in, use uniform weights
        if iteration < self.burnin_steps:
            self._last_weight = 1.0
            self._last_phase = 0.0
            return torch.tensor(1.0, device=self.device, requires_grad=False)
        
        # Compute phase
        phi_j = self.compute_phase(t_j)
        
        # Compute circular distance to canonical phase
        delta = wrap_to_pi(phi_j - self.phi_c_s2)
        
        # Compute weight: w_j = exp(-delta² / (2 * sigma_phi²))
        sigma_sq = self.sigma_phi ** 2 + 1e-8
        w_j = torch.exp(-delta ** 2 / (2 * sigma_sq))
        
        # Track statistics
        self._last_weight = w_j.item()
        self._last_phase = phi_j.item()
        self._weight_sum += w_j.item()
        self._weight_count += 1
        
        return w_j
    
    def compute_window_regularization_loss(self) -> torch.Tensor:
        """
        Compute the window width regularization loss L_win.
        
        L_win = (log(sigma_phi) - log(sigma_target))²
              = (xi_s2 - log(sigma_target))²
        
        Returns:
            L_win loss tensor
        """
        log_sigma_target = math.log(self.sigma_target)
        L_win = (self.xi_s2 - log_sigma_target) ** 2
        return L_win
    
    def get_statistics(self) -> dict:
        """
        Get statistics about the phase-gated weighting.
        
        Returns:
            Dictionary with statistics
        """
        T_s2 = self.T_s2.item()
        sigma_phi = self.sigma_phi.item()
        phi_c = self.phi_c_s2.item()
        psi = self.psi_s2.item()
        
        # Window coverage: fraction of cycle covered by ~2σ window
        # 2σ covers ~95% of Gaussian, so window_coverage ≈ 2*sigma_phi / (2π)
        window_coverage = 2 * sigma_phi / (2 * math.pi)
        
        # Mean weight (if any samples collected)
        mean_weight = self._weight_sum / max(self._weight_count, 1)
        
        return {
            "T_s2": T_s2,
            "sigma_phi": sigma_phi,
            "phi_c": phi_c,
            "psi": psi,
            "window_coverage": window_coverage,
            "last_weight": self._last_weight,
            "last_phase": self._last_phase,
            "mean_weight": mean_weight,
        }
    
    def reset_statistics(self):
        """Reset accumulated statistics."""
        self._weight_sum = 0.0
        self._weight_count = 0


def apply_s2_preset(opt):
    """
    Print s2 phase-gated static configuration info.
    
    Args:
        opt: OptimizationParams object
    """
    if not opt.use_phase_gated_static:
        return
    
    print("=" * 60)
    print("S2 PHASE-GATED STATIC CANONICAL WARM-UP ACTIVATED")
    print("=" * 60)
    print("Phase-Gated Static Warm-Up:")
    print(f"  - Burn-in steps: {opt.static_phase_burnin_steps}")
    print(f"  - Target sigma (σ_target): {opt.static_phase_sigma_target:.3f} rad")
    print(f"  - λ_win (window reg weight): {opt.lambda_static_phase_window}")
    print(f"  - Learning rate for phase params: {opt.static_phase_lr}")
    print("")
    print("Learnable parameters:")
    print("  - τ_s2: log-period (T_s2 = exp(τ_s2))")
    print("  - ψ_s2: phase offset for time alignment")
    print("  - φ_c: canonical phase center ∈ [-π, π)")
    print("  - ξ_s2: log-window width (σ_φ = exp(ξ_s2))")
    print("")
    print("During static warm-up (coarse stage):")
    print(f"  1. First {opt.static_phase_burnin_steps} steps: use uniform weights (w_j=1)")
    print("  2. After burn-in: use phase-gated weights")
    print("     - φ_j = wrap(2π * t_j / T_s2 + ψ_s2)")
    print("     - w_j = exp(-d_circ(φ_j, φ_c)² / (2σ_φ²))")
    print("     - Projections near canonical phase φ_c get higher weights")
    print("  3. L_win = (log(σ_φ) - log(σ_target))² regularization")
    print("")
    print("Dynamic stage (fine) is NOT affected by s2.")
    print("=" * 60)
