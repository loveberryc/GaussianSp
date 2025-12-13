import functools
import math
import os
import time
from tkinter import W

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from x2_gaussian.gaussian.graphics_utils import apply_rotation, batch_quaternion_multiply
from x2_gaussian.gaussian.hexplane import HexPlaneField
from x2_gaussian.gaussian.grid import DenseGrid
# from scene.grid import HashHexPlane
class Deformation(nn.Module):
    def __init__(self, D=8, W=256, input_ch=27, input_ch_time=9, grid_pe=0, skips=[], args=None):
        super(Deformation, self).__init__()
        self.D = D
        self.W = W
        self.input_ch = input_ch
        self.input_ch_time = input_ch_time
        self.skips = skips
        self.grid_pe = grid_pe
        self.no_grid = args.no_grid
        self.grid = HexPlaneField(args.bounds, args.kplanes_config, args.multires)
        # breakpoint()
        self.args = args
        # self.args.empty_voxel=True
        if self.args.empty_voxel:
            self.empty_voxel = DenseGrid(channels=1, world_size=[64,64,64])
        if self.args.static_mlp:
            self.static_mlp = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        
        # Velocity field parameters
        self.use_velocity_field = getattr(args, 'use_velocity_field', False)
        self.velocity_num_steps = getattr(args, 'velocity_num_steps', 4)
        
        # v6: Shared velocity inverse parameters
        # When enabled, uses the same velocity field for both forward and backward mappings
        self.use_shared_velocity_inverse = getattr(args, 'use_shared_velocity_inverse', False)
        # Only valid when use_velocity_field is True
        if self.use_shared_velocity_inverse and not self.use_velocity_field:
            print("Warning: use_shared_velocity_inverse requires use_velocity_field=True. Forcing use_velocity_field=True.")
            self.use_velocity_field = True
        
        # v8: Phase-Conditioned Deformation parameters
        # When enabled, uses SSRML's learned period T_hat to compute phase embedding
        # that is concatenated to trunk output before D_f/D_b heads
        self.use_phase_conditioned_deformation = getattr(args, 'use_phase_conditioned_deformation', False)
        
        # v9: Low-Rank Motion Modes parameters
        # When enabled, D_f/D_b are computed as: D(μ_i, t) = Σ_m coeff_m(t) * u_{i,m}
        # where u_{i,m} are per-Gaussian motion modes and coeff(t) are phase-dependent
        self.use_low_rank_motion_modes = getattr(args, 'use_low_rank_motion_modes', False)
        self.num_motion_modes = getattr(args, 'num_motion_modes', 3)

        # v10: Adaptive Gating parameters
        # When enabled, uses a learned gate to adaptively fuse base and low-rank displacements
        self.use_adaptive_gating = getattr(args, 'use_adaptive_gating', False)
        self.gating_hidden_size = getattr(args, 'gating_hidden_size', 32)
        self.gating_num_layers = getattr(args, 'gating_num_layers', 2)
        
        # External references container - stored in a dict to prevent PyTorch from
        # registering them as parameters (which would cause duplicate parameter errors)
        # These hold references to parameters owned by GaussianModel
        self._external_refs = {
            'period': None,        # Reference to SSRML period parameter (v8)
            'motion_modes': None,  # Reference to motion modes U (v9)
            'last_gate_forward': None,  # v10: Store last forward gating values
            'last_gate_backward': None,  # v10: Store last backward gating values
        }
        
        self.ratio=0
        self.create_net()
    @property
    def get_aabb(self):
        return self.grid.get_aabb
    def set_aabb(self, xyz_max, xyz_min):
        print("Deformation Net Set aabb",xyz_max, xyz_min)
        self.grid.set_aabb(xyz_max, xyz_min)
        if self.args.empty_voxel:
            self.empty_voxel.set_aabb(xyz_max, xyz_min)
    def create_net(self):
        mlp_out_dim = 0
        if self.grid_pe !=0:
            
            grid_out_dim = self.grid.feat_dim+(self.grid.feat_dim)*2 
        else:
            grid_out_dim = self.grid.feat_dim
        if self.no_grid:
            self.feature_out = [nn.Linear(4,self.W)]
        else:
            self.feature_out = [nn.Linear(mlp_out_dim + grid_out_dim ,self.W)]
        
        for i in range(self.D-1):
            self.feature_out.append(nn.ReLU())
            self.feature_out.append(nn.Linear(self.W,self.W))
        self.feature_out = nn.Sequential(*self.feature_out)
        
        # ============================================================
        # V7/V8: Shared Trunk + Two Heads Architecture
        # ============================================================
        # Network structure for bidirectional displacement:
        #   1. Shared K-Planes encoder: E(x, t) = grid(x, t)
        #   2. Shared trunk MLP: z(x, t) = feature_out(E(x, t))
        #   3. Forward head F_f: D_f(x, t) = pos_deform(z) or pos_deform_phase([z, phase_embed])
        #   4. Backward head F_b: D_b(y, t) = pos_deform_backward(z) or pos_deform_backward_phase([z, phase_embed])
        #
        # In v8 mode (use_phase_conditioned_deformation=True):
        #   - Phase embedding p(t) = [sin(2π*t/T_hat), cos(2π*t/T_hat)] is computed
        #   - Trunk output z is concatenated with phase embedding: [z, p(t)]
        #   - Heads have input dimension W+2 instead of W
        # ============================================================
        
        # Forward deformation head D_f (original pos_deform) - used in v7 mode
        self.pos_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        # Backward deformation head D_b for inverse consistency - used in v7 mode
        self.pos_deform_backward = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        
        # V7.4: Additional backward heads for canonical decode of shape and density
        # These heads decode canonical log-scale and density residuals from the backward trunk
        self.use_v7_4_canonical_decode = getattr(self.args, 'use_v7_4_canonical_decode', False)
        # D_b_shape: backward shape decode head (outputs D_scale = 3 for per-axis log-scale residual)
        self.backward_shape_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W // 2),
            nn.ReLU(),
            nn.Linear(self.W // 2, 3)  # 3D log-scale residual
        )
        # D_b_dens: backward density decode head (outputs 1D density residual)
        self.backward_dens_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, self.W // 2),
            nn.ReLU(),
            nn.Linear(self.W // 2, 1)  # 1D density residual
        )
        
        # v8: Phase-conditioned heads (input dim = W + 2 for phase embedding)
        if self.use_phase_conditioned_deformation:
            phase_input_dim = self.W + 2  # trunk output + [sin(phi), cos(phi)]
            # Forward deformation head D_f with phase conditioning
            self.pos_deform_phase = nn.Sequential(
                nn.ReLU(),
                nn.Linear(phase_input_dim, self.W),
                nn.ReLU(),
                nn.Linear(self.W, 3)
            )
            # Backward deformation head D_b with phase conditioning
            self.pos_deform_backward_phase = nn.Sequential(
                nn.ReLU(),
                nn.Linear(phase_input_dim, self.W),
                nn.ReLU(),
                nn.Linear(self.W, 3)
            )
        
        # ============================================================
        # V9: Low-Rank Motion Modes - Coefficient Networks F_a, F_b
        # ============================================================
        # These small MLPs take phase embedding [sin(φ), cos(φ)] as input
        # and output M coefficients for combining per-Gaussian motion modes.
        #   D_f(μ_i, t) = Σ_m a_m(t) * u_{i,m}
        #   D_b(μ_i, t) = Σ_m b_m(t) * u_{i,m}
        # ============================================================
        if self.use_low_rank_motion_modes:
            M = self.num_motion_modes
            # F_a: R^2 → R^M (forward coefficients)
            self.coeff_forward = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, M)
            )
            # F_b: R^2 → R^M (backward coefficients)
            self.coeff_backward = nn.Sequential(
                nn.Linear(2, 32),
                nn.ReLU(),
                nn.Linear(32, 32),
                nn.ReLU(),
                nn.Linear(32, M)
            )
            print(f"[v9] Created coefficient networks F_a, F_b: R^2 -> R^{M}")
        
        # ================================================================================
        # V10: Adaptive Gating Networks G_f, G_b
        # ================================================================================
        # These networks take (mu_i, phase_embed) and output a scalar gate g in (0,1)
        # The gate controls the fusion of base and low-rank displacements:
        #   D_f_total = g_f * D_f_lr + (1 - g_f) * D_f_base
        #   D_b_total = g_b * D_b_lr + (1 - g_b) * D_b_base
        # ================================================================================
        if self.use_adaptive_gating:
            # Input: canonical position (3) + phase embedding (2) = 5
            gate_input_dim = 3 + 2
            hidden_size = self.gating_hidden_size
            
            # Build gating network layers
            gate_layers_f = [nn.Linear(gate_input_dim, hidden_size), nn.ReLU()]
            gate_layers_b = [nn.Linear(gate_input_dim, hidden_size), nn.ReLU()]
            for _ in range(self.gating_num_layers - 1):
                gate_layers_f.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
                gate_layers_b.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])
            gate_layers_f.append(nn.Linear(hidden_size, 1))  # Output scalar
            gate_layers_b.append(nn.Linear(hidden_size, 1))
            
            self.gate_forward = nn.Sequential(*gate_layers_f)
            self.gate_backward = nn.Sequential(*gate_layers_b)
            print(f"[v10] Created gating networks G_f, G_b: input_dim={gate_input_dim}, hidden={hidden_size}, layers={self.gating_num_layers}")
        
        self.scales_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 3))
        self.rotations_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 4))
        
        # ============================================================
        # M1: Eulerian Uncertainty Head for Uncertainty-Gated Fusion
        # ============================================================
        # Outputs log-variance s_E(x,t) = log(σ_E²) for each Gaussian position
        # This is used in PhysX-Boosted M1 to compute adaptive gate β(x,t)
        #
        # Paper notation (M1 fusion):
        #   Φ(x,t) = Φ_L(x,t) + β(x,t) · Φ_E(x,t)
        #   β = σ_L² / (σ_L² + σ_E²)  [Bayes mode]
        #   β = sigmoid((τ - s_E) / λ)  [Sigmoid mode]
        #
        # High s_E (uncertain) → low β → trust Lagrangian more
        # Low s_E (confident) → high β → Eulerian contributes more
        # ============================================================
        self.fusion_mode = getattr(self.args, 'fusion_mode', 'fixed_alpha')
        eulerian_uncertainty_hidden = getattr(self.args, 'eulerian_uncertainty_hidden_dim', 32)
        s_E_init = getattr(self.args, 'eulerian_s_E_init', 0.0)
        
        # Uncertainty head: takes trunk features hidden [N, W] → s_E [N, 1]
        self.uncertainty_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(self.W, eulerian_uncertainty_hidden),
            nn.ReLU(),
            nn.Linear(eulerian_uncertainty_hidden, 1)  # Output: s_E = log(σ²)
        )
        # Initialize to output s_E_init (default 0 → σ² = 1)
        nn.init.zeros_(self.uncertainty_head[-1].weight)
        nn.init.constant_(self.uncertainty_head[-1].bias, s_E_init)
        
        # Cache for M1: store last computed s_E for loss/logging
        self._last_s_E = None
        # self.density_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 1))
        # self.shs_deform = nn.Sequential(nn.ReLU(),nn.Linear(self.W,self.W),nn.ReLU(),nn.Linear(self.W, 16*3))

    def set_period_ref(self, period_param):
        """
        Set reference to the SSRML period parameter (tau_hat stored as log(T_hat)).
        This is called by GaussianModel after initialization.
        
        Args:
            period_param: nn.Parameter storing log(T_hat), so T_hat = exp(period_param)
        """
        self._external_refs['period'] = period_param
    
    def set_motion_modes_ref(self, motion_modes_param):
        """
        Set reference to the per-Gaussian motion modes U: [K, M, 3].
        This is called by GaussianModel after initialization or densification.
        
        Args:
            motion_modes_param: nn.Parameter storing U [K, M, 3]
        """
        self._external_refs['motion_modes'] = motion_modes_param
    
    def compute_phase_embedding(self, time_emb):
        """
        Compute phase embedding for v8 phase-conditioned deformation.
        
        Phase embedding encodes the breathing phase using SSRML's learned period T_hat:
            phi(t) = 2π * t / T_hat
            phase_embed = [sin(phi), cos(phi)]
        
        This makes the deformation network explicitly aware of where in the 
        breathing cycle the current time t is located.
        
        Args:
            time_emb: Time tensor (shape: [N, 1])
        
        Returns:
            phase_embed: Phase embedding tensor (shape: [N, 2]) = [sin(phi), cos(phi)]
        """
        period_ref = self._external_refs.get('period')
        if period_ref is None:
            raise RuntimeError("Period reference not set. Call set_period_ref() first.")
        
        # T_hat = exp(period), where period stores log(T_hat)
        T_hat = torch.exp(period_ref)  # scalar
        
        # Compute normalized phase: phi = 2π * t / T_hat
        # time_emb is typically in range [0, 1] representing normalized time
        phi = 2.0 * math.pi * time_emb / T_hat  # [N, 1]
        
        # Compute phase embedding: [sin(phi), cos(phi)]
        sin_phi = torch.sin(phi)  # [N, 1]
        cos_phi = torch.cos(phi)  # [N, 1]
        phase_embed = torch.cat([sin_phi, cos_phi], dim=-1)  # [N, 2]
        
        return phase_embed
    
    def compute_low_rank_forward_displacement(self, gaussian_indices, time_emb):
        """
        Compute forward displacement D_f using low-rank motion modes (v9).
        
        D_f(μ_i, t) = Σ_m a_m(t) * u_{i,m}
        
        where:
            - u_{i,m} are per-Gaussian motion modes from motion_modes reference
            - a(t) = F_a(phase_embed) are time-dependent coefficients
        
        Args:
            gaussian_indices: Indices of Gaussians (for selecting motion modes), or None for all
            time_emb: Time tensor (shape: [N, 1])
        
        Returns:
            D_f: Forward displacement [N, 3]
        """
        motion_modes_ref = self._external_refs.get('motion_modes')
        if motion_modes_ref is None:
            raise RuntimeError("Motion modes reference not set. Call set_motion_modes_ref() first.")
        
        # Compute phase embedding: [N, 2]
        phase_embed = self.compute_phase_embedding(time_emb)
        
        # Compute forward coefficients a(t): [N, M]
        # F_a broadcasts the same coefficients for all samples if time is scalar
        # If time_emb has unique values per sample, we get per-sample coefficients
        # However, typically all samples share the same time, so a(t) is [1, M] or [N, M]
        a_t = self.coeff_forward(phase_embed)  # [N, M]
        
        # Get motion modes: [K, M, 3] where K is total Gaussians
        U = motion_modes_ref
        
        # If gaussian_indices is provided, select the corresponding modes
        # Otherwise, assume we're using all Gaussians in order
        # U_i: [N, M, 3]
        if gaussian_indices is not None:
            U_i = U[gaussian_indices]  # [N, M, 3]
        else:
            U_i = U  # [K, M, 3], K should equal N
        
        # D_f(μ_i, t) = Σ_m a_m(t) * u_{i,m}
        # a_t: [N, M], U_i: [N, M, 3]
        # Use einsum: D_f[n, d] = Σ_m a_t[n, m] * U_i[n, m, d]
        D_f = torch.einsum('nm,nmd->nd', a_t, U_i)  # [N, 3]
        
        return D_f
    
    def compute_low_rank_backward_displacement(self, gaussian_indices, time_emb):
        """
        Compute backward displacement D_b using low-rank motion modes (v9).
        
        D_b(μ_i, t) = Σ_m b_m(t) * u_{i,m}
        
        where:
            - u_{i,m} are per-Gaussian motion modes from motion_modes reference
            - b(t) = F_b(phase_embed) are time-dependent coefficients
        
        Args:
            gaussian_indices: Indices of Gaussians (for selecting motion modes), or None for all
            time_emb: Time tensor (shape: [N, 1])
        
        Returns:
            D_b: Backward displacement [N, 3]
        """
        motion_modes_ref = self._external_refs.get('motion_modes')
        if motion_modes_ref is None:
            raise RuntimeError("Motion modes reference not set. Call set_motion_modes_ref() first.")
        
        # Compute phase embedding: [N, 2]
        phase_embed = self.compute_phase_embedding(time_emb)
        
        # Compute backward coefficients b(t): [N, M]
        b_t = self.coeff_backward(phase_embed)  # [N, M]
        
        # Get motion modes: [K, M, 3] where K is total Gaussians
        U = motion_modes_ref
        
        # If gaussian_indices is provided, select the corresponding modes
        if gaussian_indices is not None:
            U_i = U[gaussian_indices]  # [N, M, 3]
        else:
            U_i = U  # [K, M, 3], K should equal N
        
        # D_b(μ_i, t) = Σ_m b_m(t) * u_{i,m}
        D_b = torch.einsum('nm,nmd->nd', b_t, U_i)  # [N, 3]
        
        return D_b


    def compute_gating_values(self, canonical_pts, time_emb):
        """
        Compute gating values g_f and g_b for adaptive fusion of base and low-rank displacements (v10).
        
        The gating networks take (mu_i, phase_embed) as input and output scalar gates in (0, 1).
        
        Args:
            canonical_pts: Canonical positions mu_i [N, 3]
            time_emb: Time tensor [N, 1]
        
        Returns:
            g_f: Forward gating values [N, 1] in (0, 1)
            g_b: Backward gating values [N, 1] in (0, 1)
        """
        # Compute phase embedding: [N, 2]
        phase_embed = self.compute_phase_embedding(time_emb)
        
        # Concatenate canonical position with phase embedding: [N, 5]
        gate_input = torch.cat([canonical_pts[:, :3], phase_embed], dim=-1)
        
        # Compute gating values using sigmoid for (0, 1) range
        g_f = torch.sigmoid(self.gate_forward(gate_input))  # [N, 1]
        g_b = torch.sigmoid(self.gate_backward(gate_input))  # [N, 1]
        
        return g_f, g_b


    def get_last_gating_values(self):
        """
        Get the last computed gating values for regularization (v10).
        
        Returns:
            g_f: Last forward gating values [N, 1] or None
            g_b: Last backward gating values [N, 1] or None
        """
        g_f = self._external_refs.get('last_gate_forward')
        g_b = self._external_refs.get('last_gate_backward')
        return g_f, g_b

    def query_time(self, rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb):

        if self.no_grid:
            h = torch.cat([rays_pts_emb[:,:3],time_emb[:,:1]],-1)
        else:

            grid_feature = self.grid(rays_pts_emb[:,:3], time_emb[:,:1])
            # breakpoint()
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature,self.grid_pe)
            hidden = torch.cat([grid_feature],-1) 
        
        
        hidden = self.feature_out(hidden)      # jsut nn.Linear(128, 256)
 

        return hidden

    def query_velocity(self, pts, time_emb):
        """
        Query the velocity field v(x, t) at given positions and time.
        This uses the same network as D_f but interprets the output as velocity.
        
        Args:
            pts: Position tensor [N, 3]
            time_emb: Time tensor [N, 1]
        
        Returns:
            velocity: Velocity vector v(x, t) [N, 3]
        """
        if self.no_grid:
            h = torch.cat([pts[:,:3], time_emb[:,:1]], -1)
        else:
            grid_feature = self.grid(pts[:,:3], time_emb[:,:1])
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.grid_pe)
            h = torch.cat([grid_feature], -1)
        
        hidden = self.feature_out(h)
        velocity = self.pos_deform(hidden)  # [N, 3]
        return velocity

    def forward_velocity_integration(self, canonical_pts, time_emb, density_emb):
        """
        Compute phi_f(x, t) using velocity field integration with K-step Euler method.
        
        For given (x, t), compute:
            pos_0 = x
            For k = 0 ... K-1:
                tau_k = (k + 0.5) / K
                t_k = tau_k * t
                v_k = v(pos_k, t_k)
                pos_{k+1} = pos_k + (t / K) * v_k
            phi_f(x, t) = pos_K
        
        Args:
            canonical_pts: Canonical positions x [N, 3]
            time_emb: Target time t [N, 1]
            density_emb: Density for mask computation [N, 1]
        
        Returns:
            phi_f: Deformed positions phi_f(x, t) [N, 3]
        """
        K = self.velocity_num_steps
        pos = canonical_pts.clone()  # [N, 3]
        t = time_emb  # [N, 1]
        
        # Compute mask (same as in forward_dynamic)
        if self.args.static_mlp:
            # Query at canonical position for mask
            if self.no_grid:
                h = torch.cat([canonical_pts[:,:3], time_emb[:,:1]], -1)
            else:
                grid_feature = self.grid(canonical_pts[:,:3], time_emb[:,:1])
                if self.grid_pe > 1:
                    grid_feature = poc_fre(grid_feature, self.grid_pe)
                h = torch.cat([grid_feature], -1)
            hidden = self.feature_out(h)
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(canonical_pts[:,:3])
        else:
            mask = torch.ones_like(density_emb[:,0]).unsqueeze(-1)
        
        # K-step Euler integration
        dt = t / K  # Time step [N, 1]
        
        for k in range(K):
            # Compute normalized time tau_k = (k + 0.5) / K
            tau_k = (k + 0.5) / K
            # Actual query time t_k = tau_k * t
            t_k = tau_k * t  # [N, 1]
            
            # Query velocity at current position and time
            v_k = self.query_velocity(pos, t_k)  # [N, 3]
            
            # Euler step: pos = pos + dt * v_k
            pos = pos + dt * v_k
        
        # Apply mask: final position is canonical * mask + integrated displacement
        # Note: the integration gives us the full trajectory, so we apply mask to the displacement
        displacement = pos - canonical_pts
        pts = canonical_pts * mask + displacement
        
        return pts

    def backward_velocity_integration(self, deformed_pts, time_emb):
        """
        Compute phi_b(y, t) using shared velocity field with BACKWARD integration.
        This is the v6 implementation that uses the same velocity field v(x, τ) to
        construct the backward mapping by integrating in the reverse direction.
        
        For given (y, t), compute:
            x^{(0)} = y
            For k = 0 ... K-1:
                tau_k = t - (k + 0.5) / K * t    # From t walking back towards 0
                v_k = v(x^{(k)}, tau_k)
                x^{(k+1)} = x^{(k)} - (t / K) * v_k
            phi_b(y, t) = x^{(K)}
        
        Args:
            deformed_pts: Deformed positions y (at time t) [N, 3]
            time_emb: Target time t [N, 1]
        
        Returns:
            phi_b: Reconstructed canonical positions phi_b(y, t) [N, 3]
            D_b_eff: Effective backward displacement D_b^eff = phi_b - y [N, 3]
        """
        K = self.velocity_num_steps
        pos = deformed_pts.clone()  # [N, 3], start from y
        t = time_emb  # [N, 1]
        
        # Time step (negative direction)
        dt = t / K  # [N, 1]
        
        for k in range(K):
            # Compute tau_k = t - (k + 0.5) / K * t
            # This walks from t back towards 0
            tau_k = t - (k + 0.5) / K * t  # [N, 1]
            
            # Query velocity at current position and time
            v_k = self.query_velocity(pos, tau_k)  # [N, 3]
            
            # Backward Euler step: pos = pos - dt * v_k
            pos = pos - dt * v_k
        
        # phi_b(y, t) = pos (the reconstructed canonical position)
        phi_b = pos
        
        # D_b^eff = phi_b - y (the effective backward displacement)
        D_b_eff = phi_b - deformed_pts
        
        return phi_b, D_b_eff

    @property
    def get_empty_ratio(self):
        return self.ratio
    def forward(self, rays_pts_emb, scales_emb=None, rotations_emb=None, density = None, time_feature=None, time_emb=None):
        if time_emb is None:
            return self.forward_static(rays_pts_emb[:,:3])
        else:
            return self.forward_dynamic(rays_pts_emb, scales_emb, rotations_emb, density, time_feature, time_emb)

    def forward_static(self, rays_pts_emb):
        grid_feature = self.grid(rays_pts_emb[:,:3])
        dx = self.static_mlp(grid_feature)
        return rays_pts_emb[:, :3] + dx
    def forward_dynamic(self,rays_pts_emb, scales_emb, rotations_emb, density_emb, time_feature, time_emb):
        hidden = self.query_time(rays_pts_emb, scales_emb, rotations_emb, time_feature, time_emb)
        
        # ============================================================
        # M1: Compute Eulerian uncertainty s_E(x,t) = log(σ_E²)
        # This is cached for use in uncertainty-gated fusion
        # ============================================================
        if self.fusion_mode == 'uncertainty_gated':
            self._last_s_E = self.uncertainty_head(hidden)  # [N, 1]
        else:
            self._last_s_E = None
        
        if self.args.static_mlp:
            mask = self.static_mlp(hidden)
        elif self.args.empty_voxel:
            mask = self.empty_voxel(rays_pts_emb[:,:3])
        else:
            mask = torch.ones_like(density_emb[:,0]).unsqueeze(-1)
        # breakpoint()
        if self.args.no_dx:
            pts = rays_pts_emb[:,:3]
        else:
            # Check if velocity field mode is enabled
            if self.use_velocity_field:
                # Use velocity field integration to compute phi_f(x, t)
                pts = self.forward_velocity_integration(rays_pts_emb[:,:3], time_emb, density_emb)
            # v9/v10: Low-rank motion modes with optional adaptive gating
            elif self.use_low_rank_motion_modes:
                # Compute low-rank displacement D_f_lr
                D_f_lr = self.compute_low_rank_forward_displacement(None, time_emb)  # [N, 3]
                
                # v10: Adaptive gating - fuse base and low-rank displacements
                if self.use_adaptive_gating:
                    # Compute base displacement D_f_base using phase-conditioned or standard head
                    if self.use_phase_conditioned_deformation:
                        phase_embed = self.compute_phase_embedding(time_emb)  # [N, 2]
                        hidden_aug = torch.cat([hidden, phase_embed], dim=-1)  # [N, W+2]
                        D_f_base = self.pos_deform_phase(hidden_aug)  # [N, 3]
                    else:
                        D_f_base = self.pos_deform(hidden)  # [N, 3]
                    
                    # Compute gating values g_f in (0, 1)
                    g_f, _ = self.compute_gating_values(rays_pts_emb[:, :3], time_emb)  # [N, 1]
                    
                    # Adaptive fusion: D_f_total = g_f * D_f_lr + (1 - g_f) * D_f_base
                    dx = g_f * D_f_lr + (1 - g_f) * D_f_base  # [N, 3]
                    
                    # Store gating value for potential regularization
                    self._external_refs['last_gate_forward'] = g_f
                else:
                    # v9: Pure low-rank displacement
                    dx = D_f_lr
                
                pts = torch.zeros_like(rays_pts_emb[:,:3])
                pts = rays_pts_emb[:,:3]*mask + dx
            else:
                # v8: Phase-conditioned displacement D_f(x, t) with phase embedding
                if self.use_phase_conditioned_deformation:
                    # Compute phase embedding: [sin(2π*t/T_hat), cos(2π*t/T_hat)]
                    phase_embed = self.compute_phase_embedding(time_emb)  # [N, 2]
                    # Concatenate trunk output with phase embedding
                    hidden_aug = torch.cat([hidden, phase_embed], dim=-1)  # [N, W+2]
                    # Use phase-conditioned head
                    dx = self.pos_deform_phase(hidden_aug)  # [N, 3]
                else:
                    # Original v7 behavior: direct displacement D_f(x, t)
                    dx = self.pos_deform(hidden)  # [N, 3]
                # breakpoint()
                pts = torch.zeros_like(rays_pts_emb[:,:3])
                pts = rays_pts_emb[:,:3]*mask + dx
        if self.args.no_ds :
            
            scales = scales_emb[:,:3]
        else:
            ds = self.scales_deform(hidden)

            scales = torch.zeros_like(scales_emb[:,:3])
            scales = scales_emb[:,:3]*mask + ds
            
        if self.args.no_dr :
            rotations = rotations_emb[:,:4]
        else:
            dr = self.rotations_deform(hidden)

            rotations = torch.zeros_like(rotations_emb[:,:4])
            if self.args.apply_rotation:
                rotations = batch_quaternion_multiply(rotations_emb, dr)
            else:
                rotations = rotations_emb[:,:4] + dr

        # if self.args.no_do :
        #     density = density_emb[:,:1] 
        # else:
        #     do = self.density_deform(hidden) 
          
        #     density = torch.zeros_like(density_emb[:,:1])
        #     density = density_emb[:,:1]*mask + do
        return pts, scales, rotations #, density

    def forward_backward_position(self, deformed_pts, time_emb):
        """
        Compute backward deformation D_b for inverse consistency.
        Given deformed position y = x + D_f(x, t), compute x_hat = y + D_b(y, t).
        
        In v6 mode (use_shared_velocity_inverse=True):
            Uses the shared velocity field v(x, t) with backward integration to compute phi_b.
            phi_b(y, t) is computed by integrating backwards from t to 0 using the same v.
        
        In v9 mode (use_low_rank_motion_modes=True):
            D_b(μ_i, t) = Σ_m b_m(t) * u_{i,m}
            Uses the same per-Gaussian motion modes but with backward coefficients b(t).
        
        In v8 mode (use_phase_conditioned_deformation=True):
            Uses phase embedding [sin(2π*t/T_hat), cos(2π*t/T_hat)] concatenated to trunk output.
        
        In v7 mode (default):
            Uses the separate D_b network (pos_deform_backward head) as before.
        
        Args:
            deformed_pts: Deformed positions y (shape: [N, 3])
            time_emb: Time embedding (shape: [N, 1])
        
        Returns:
            reconstructed_pts: Reconstructed canonical positions x_hat (shape: [N, 3])
            backward_deform: The backward deformation D_b(y, t) or D_b^eff(y, t) (shape: [N, 3])
        """
        # v6 mode: Use shared velocity field with backward integration
        if self.use_shared_velocity_inverse:
            reconstructed_pts, backward_deform = self.backward_velocity_integration(
                deformed_pts, time_emb
            )
            return reconstructed_pts, backward_deform
        
        # v9/v10 mode: Low-rank motion modes with optional adaptive gating
        if self.use_low_rank_motion_modes:
            # Compute low-rank backward displacement D_b_lr
            D_b_lr = self.compute_low_rank_backward_displacement(None, time_emb)  # [N, 3]
            
            # v10: Adaptive gating - fuse base and low-rank backward displacements
            if self.use_adaptive_gating:
                # Compute base backward displacement D_b_base
                if self.no_grid:
                    h = torch.cat([deformed_pts[:,:3], time_emb[:,:1]], -1)
                else:
                    grid_feature = self.grid(deformed_pts[:,:3], time_emb[:,:1])
                    if self.grid_pe > 1:
                        grid_feature = poc_fre(grid_feature, self.grid_pe)
                    h = torch.cat([grid_feature], -1)
                
                hidden = self.feature_out(h)
                
                if self.use_phase_conditioned_deformation:
                    phase_embed = self.compute_phase_embedding(time_emb)  # [N, 2]
                    hidden_aug = torch.cat([hidden, phase_embed], dim=-1)  # [N, W+2]
                    D_b_base = self.pos_deform_backward_phase(hidden_aug)  # [N, 3]
                else:
                    D_b_base = self.pos_deform_backward(hidden)  # [N, 3]
                
                # Compute backward gating values g_b in (0, 1)
                # Note: Use deformed_pts as the "canonical" position for gating
                # since we're computing D_b at the deformed location
                _, g_b = self.compute_gating_values(deformed_pts[:, :3], time_emb)  # [N, 1]
                
                # Adaptive fusion: D_b_total = g_b * D_b_lr + (1 - g_b) * D_b_base
                db = g_b * D_b_lr + (1 - g_b) * D_b_base  # [N, 3]
                
                # Store gating value for potential regularization
                self._external_refs['last_gate_backward'] = g_b
            else:
                # v9: Pure low-rank displacement
                db = D_b_lr
            
            reconstructed_pts = deformed_pts[:,:3] + db
            return reconstructed_pts, db
        
        # v7/v8 mode: Use separate D_b network
        if self.no_grid:
            h = torch.cat([deformed_pts[:,:3], time_emb[:,:1]], -1)
        else:
            grid_feature = self.grid(deformed_pts[:,:3], time_emb[:,:1])
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.grid_pe)
            h = torch.cat([grid_feature], -1)
        
        hidden = self.feature_out(h)
        
        # v8: Phase-conditioned backward deformation D_b(y, t) with phase embedding
        if self.use_phase_conditioned_deformation:
            # Compute phase embedding: [sin(2π*t/T_hat), cos(2π*t/T_hat)]
            phase_embed = self.compute_phase_embedding(time_emb)  # [N, 2]
            # Concatenate trunk output with phase embedding
            hidden_aug = torch.cat([hidden, phase_embed], dim=-1)  # [N, W+2]
            # Use phase-conditioned backward head
            db = self.pos_deform_backward_phase(hidden_aug)  # [N, 3]
        else:
            # v7: Apply original backward deformation head
            db = self.pos_deform_backward(hidden)  # [N, 3]
        
        reconstructed_pts = deformed_pts[:,:3] + db
        
        return reconstructed_pts, db

    def backward_deform_full(self, deformed_pts, time_emb):
        """
        V7.4: Full backward decode including position, shape, and density residuals.
        
        Uses the shared backward trunk to compute features, then applies separate
        heads for position (existing), shape (new), and density (new) decode.
        
        Args:
            deformed_pts: Dynamic positions at time t (shape: [N, 3])
            time_emb: Time embedding (shape: [N, 1])
        
        Returns:
            D_b_pos: Position residual for canonical center decode (N, 3)
            D_b_shape: Log-scale residual for canonical shape decode (N, 3)
            D_b_dens: Density residual for canonical density decode (N, 1)
        """
        # Compute shared trunk features (same as in forward_backward_position)
        if self.no_grid:
            h = torch.cat([deformed_pts[:,:3], time_emb[:,:1]], -1)
        else:
            grid_feature = self.grid(deformed_pts[:,:3], time_emb[:,:1])
            if self.grid_pe > 1:
                grid_feature = poc_fre(grid_feature, self.grid_pe)
            h = torch.cat([grid_feature], -1)
        
        hidden = self.feature_out(h)
        
        # Position head (existing behavior)
        if self.use_phase_conditioned_deformation:
            phase_embed = self.compute_phase_embedding(time_emb)  # [N, 2]
            hidden_aug = torch.cat([hidden, phase_embed], dim=-1)  # [N, W+2]
            D_b_pos = self.pos_deform_backward_phase(hidden_aug)  # [N, 3]
        else:
            D_b_pos = self.pos_deform_backward(hidden)  # [N, 3]
        
        # V7.4: Shape and density heads (use base hidden, not phase-augmented)
        # These heads decode canonical residuals from the shared trunk
        D_b_shape = self.backward_shape_head(hidden)  # [N, 3]
        D_b_dens = self.backward_dens_head(hidden)    # [N, 1]
        
        return D_b_pos, D_b_shape, D_b_dens
    
    def get_mlp_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" not in name:
                parameter_list.append(param)
        return parameter_list
    
    def get_grid_parameters(self):
        parameter_list = []
        for name, param in self.named_parameters():
            if  "grid" in name:
                parameter_list.append(param)
        return parameter_list
    
    # ============================================================
    # M1: Uncertainty-Gated Fusion Getters
    # ============================================================
    def get_last_s_E(self):
        """
        Get the last computed Eulerian log-variance s_E(x,t).
        
        Returns:
            s_E: Tensor [N, 1] of log-variance values, or None if not computed
        """
        return self._last_s_E
    
    def get_uncertainty_parameters(self):
        """
        Get parameters of the uncertainty head for optimizer.
        
        Returns:
            List of parameters from uncertainty_head
        """
        return list(self.uncertainty_head.parameters())
    
class deform_network(nn.Module):
    def __init__(self, args) :
        super(deform_network, self).__init__()
        net_width = args.net_width
        timebase_pe = args.timebase_pe
        defor_depth= args.defor_depth
        posbase_pe= args.posebase_pe
        scale_rotation_pe = args.scale_rotation_pe
        density_pe = args.density_pe
        timenet_width = args.timenet_width
        timenet_output = args.timenet_output
        grid_pe = args.grid_pe
        times_ch = 2*timebase_pe+1
        self.timenet = nn.Sequential(
        nn.Linear(times_ch, timenet_width), nn.ReLU(),
        nn.Linear(timenet_width, timenet_output))
        self.deformation_net = Deformation(W=net_width, D=defor_depth, input_ch=(3)+(3*(posbase_pe))*2, grid_pe=grid_pe, input_ch_time=timenet_output, args=args)
        self.register_buffer('time_poc', torch.FloatTensor([(2**i) for i in range(timebase_pe)]))
        self.register_buffer('pos_poc', torch.FloatTensor([(2**i) for i in range(posbase_pe)]))
        self.register_buffer('rotation_scaling_poc', torch.FloatTensor([(2**i) for i in range(scale_rotation_pe)]))
        # self.register_buffer('density_poc', torch.FloatTensor([(2**i) for i in range(density_pe)]))
        self.apply(initialize_weights)
        # print(self)
    
    def set_period_ref(self, period_param):
        """
        Set reference to the SSRML period parameter for phase-conditioned deformation (v8).
        This is called by GaussianModel after initialization.
        
        Args:
            period_param: nn.Parameter storing log(T_hat), so T_hat = exp(period_param)
        """
        self.deformation_net.set_period_ref(period_param)
    
    def set_motion_modes_ref(self, motion_modes_param):
        """
        Set reference to per-Gaussian motion modes for low-rank deformation (v9).
        This is called by GaussianModel after initialization or densification.
        
        Args:
            motion_modes_param: nn.Parameter storing U [K, M, 3]
        """
        self.deformation_net.set_motion_modes_ref(motion_modes_param)

    def forward(self, point, scales=None, rotations=None, density=None, times_sel=None):
        return self.forward_dynamic(point, scales, rotations, density, times_sel)
    
    @property
    def get_aabb(self):
        
        return self.deformation_net.get_aabb
    @property
    def get_empty_ratio(self):
        return self.deformation_net.get_empty_ratio
        
    def forward_static(self, points):
        points = self.deformation_net(points)
        return points
    
    def forward_dynamic(self, point, scales=None, rotations=None, density=None, times_sel=None):
        # times_emb = poc_fre(times_sel, self.time_poc)
        point_emb = poc_fre(point,self.pos_poc)
        # breakpoint()
        scales_emb = poc_fre(scales,self.rotation_scaling_poc)
        rotations_emb = poc_fre(rotations,self.rotation_scaling_poc)
        # time_emb = poc_fre(times_sel, self.time_poc)
        # times_feature = self.timenet(time_emb)
        means3D, scales, rotations = self.deformation_net( point_emb,
                                                  scales_emb,
                                                rotations_emb,
                                                density,
                                                None,
                                                times_sel)
    
       
        return means3D, scales, rotations #, density

    def forward_backward_position(self, deformed_pts, times_sel):
        """
        Compute backward deformation for inverse consistency loss.
        
        Args:
            deformed_pts: Deformed positions y = x + D_f(x, t) (shape: [N, 3])
            times_sel: Time values (shape: [N, 1])
        
        Returns:
            reconstructed_pts: Reconstructed canonical positions x_hat = y + D_b(y, t)
            backward_deform: The backward deformation D_b(y, t)
        """
        return self.deformation_net.forward_backward_position(deformed_pts, times_sel)

    def backward_deform_full(self, deformed_pts, times_sel):
        """
        V7.4: Full backward decode including position, shape, and density residuals.
        
        Args:
            deformed_pts: Dynamic positions at time t (shape: [N, 3])
            times_sel: Time values (shape: [N, 1])
        
        Returns:
            D_b_pos: Position residual for canonical center decode (N, 3)
            D_b_shape: Log-scale residual for canonical shape decode (N, 3)
            D_b_dens: Density residual for canonical density decode (N, 1)
        """
        return self.deformation_net.backward_deform_full(deformed_pts, times_sel)
    
    def get_mlp_parameters(self):
        return self.deformation_net.get_mlp_parameters() + list(self.timenet.parameters())
    
    def get_grid_parameters(self):
        return self.deformation_net.get_grid_parameters()
    
    # ============================================================
    # M1: Uncertainty-Gated Fusion Getters (wrapper)
    # ============================================================
    def get_last_s_E(self):
        """
        Get the last computed Eulerian log-variance s_E(x,t) from the underlying network.
        
        Returns:
            s_E: Tensor [N, 1] of log-variance values, or None if not computed
        """
        return self.deformation_net.get_last_s_E()
    
    def get_uncertainty_parameters(self):
        """
        Get parameters of the uncertainty head for optimizer.
        
        Returns:
            List of parameters from uncertainty_head
        """
        return self.deformation_net.get_uncertainty_parameters()

def initialize_weights(m):
    if isinstance(m, nn.Linear):
        # init.constant_(m.weight, 0)
        init.xavier_uniform_(m.weight,gain=1)
        if m.bias is not None:
            init.xavier_uniform_(m.weight,gain=1)
            # init.constant_(m.bias, 0)
def poc_fre(input_data,poc_buf):

    input_data_emb = (input_data.unsqueeze(-1) * poc_buf).flatten(-2)
    input_data_sin = input_data_emb.sin()
    input_data_cos = input_data_emb.cos()
    input_data_emb = torch.cat([input_data, input_data_sin,input_data_cos], -1)
    return input_data_emb