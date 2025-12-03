#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import os.path as osp
from argparse import ArgumentParser, Namespace

sys.path.append("./")
from x2_gaussian.utils.argument_utils import ParamGroup


class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self._source_path = ""
        self._model_path = ""
        self.data_device = "cuda"
        self.ply_path = ""  # Path to initialization point cloud (if None, we will try to find `init_*.npy`.)
        self.scale_min =  0.0  # percent of volume size  0.0005
        self.scale_max =  0.5  # percent of volume size
        self.eval = True
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = osp.abspath(g.source_path)
        return g


class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")


class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_init = 0.0002
        self.position_lr_final = 0.00002
        self.position_lr_max_steps = 30_000
        self.density_lr_init = 0.01
        self.density_lr_final = 0.001
        self.density_lr_max_steps = 30_000
        self.scaling_lr_init = 0.005
        self.scaling_lr_final = 0.0005
        self.scaling_lr_max_steps = 30_000
        self.rotation_lr_init = 0.001
        self.rotation_lr_final = 0.0001
        self.rotation_lr_max_steps = 30_000

        self.deformation_lr_init = 0.0002     # 0.0002   4
        self.deformation_lr_final = 0.00002     # 0.00002   4
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.002  # 0.002   4
        self.grid_lr_final = 0.0002   # 0.0002   4

        self.period_lr_init = 0.0002
        self.period_lr_final = 0.00002
        self.period_lr_max_steps = 30_000

        self.deformation_low_lr_init = 0.0002     # 0.0002   4
        self.deformation_low_lr_final = 0.00002     # 0.00002   4
        self.deformation_low_lr_delay_mult = 0.01
        self.grid_low_lr_init = 0.002  # 0.002   4
        self.grid_low_lr_final = 0.0002   # 0.0002   4

        self.deformation_high_lr_init = 0.0002     # 0.0002   4
        self.deformation_high_lr_final = 0.00002     # 0.00002   4
        self.deformation_high_lr_delay_mult = 0.01
        self.grid_high_lr_init = 0.002  # 0.002   4
        self.grid_high_lr_final = 0.0002   # 0.0002   4

        self.hf_weights_lr_init = 0.0002
        self.hf_weights_lr_final = 0.00002
        self.hf_weights_lr_max_steps = 30_000

        self.lambda_dssim = 0.25
        self.lambda_tv = 0.05
        self.lambda_prior = 1.0
        self.lambda_prior_3d = 0.01 # useless
        self.tv_vol_size = 32
        self.density_min_threshold = 0.00001
        self.densification_interval = 100
        self.densify_from_iter = 500
        self.densify_until_iter = 15000
        self.densify_grad_threshold = 5.0e-5
        self.densify_scale_threshold = 0.1  # percent of volume size
        self.max_screen_size = None
        self.max_scale = None  # percent of volume size
        self.max_num_gaussians = 500_000
        
        # Inverse consistency parameters
        self.use_inverse_consistency = False  # Enable inverse consistency loss for position deformation
        self.lambda_inv = 0.01  # Weight for inverse consistency loss L_inv_pos
        self.inv_sample_ratio = 1.0  # Ratio of Gaussians to sample for inverse consistency (1.0 = all)
        
        # Symmetry regularization parameters (D_b ≈ -D_f)
        self.use_symmetry_reg = False  # Enable weak symmetry regularization L_sym
        self.lambda_sym = 0.01  # Weight for symmetry loss L_sym = ||D_b + D_f||_1
        
        # Cycle motion consistency parameters (motion periodicity constraint)
        self.use_cycle_motion = False  # Enable cycle motion loss L_cycle_motion
        self.lambda_cycle = 0.1  # Weight for cycle motion loss: ||D_f(mu, t+T) - D_f(mu, t)||_1
        self.no_period = False  # Disable period learning and L_cycle (for non-periodic motion)
        
        # Jacobian regularization parameters (local folding prevention)
        self.use_jacobian_reg = False  # Enable Jacobian regularization L_jac
        self.lambda_jac = 0.01  # Weight for Jacobian loss: penalizes det(J) < 0 (folding)
        self.jacobian_num_samples = 64  # Number of points to sample for Jacobian estimation per batch
        self.jacobian_step_size = 1e-3  # Finite difference step size h for Jacobian approximation
        
        # v7 Main Model: Bidirectional Displacement + L_inv + L_cycle (recommended for main experiments)
        # When enabled, this activates the best-performing configuration from ablation studies:
        # - Uses displacement-based D_f, D_b (NOT velocity field)
        # - Enables L_inv (inverse consistency) and L_cycle (motion periodicity)
        # - Force-disables L_sym, L_jac, velocity_field, and shared_velocity_inverse
        self.use_v7_bidirectional_displacement = False  # Enable v7 main model preset
        
        # v9: Low-Rank Motion Modes regularization parameters
        # When use_low_rank_motion_modes=True, this adds a regularization on motion modes U:
        #   L_mode = mean_{i,m} || U[i,m,:] ||_2^2
        # This prevents U from growing too large and causing numerical instability.
        self.use_mode_regularization = False  # Enable L_mode regularization
        self.lambda_mode = 1e-4  # Weight for mode regularization (very small by default)
        
        # v10: Adaptive Gating and Trajectory Smoothing parameters
        # When use_adaptive_gating=True, uses a learned gate to adaptively fuse base and low-rank displacements:
        #   D_f_total = g_f * D_f_lr + (1 - g_f) * D_f_base
        #   D_b_total = g_b * D_b_lr + (1 - g_b) * D_b_base
        # This allows the network to decide per-Gaussian/per-time whether to use low-rank or base displacement.
        self.use_adaptive_gating = False  # Enable v10 mode: adaptive gating of base and low-rank displacements
        self.lambda_gate = 0.0  # Weight for gate regularization L_gate = mean(g*(1-g)) to encourage binary gates
        
        # When use_trajectory_smoothing=True, adds trajectory smoothing regularization:
        #   L_traj = mean_{i,k} || x_i(t_{k+1}) - 2*x_i(t_k) + x_i(t_{k-1}) ||^2
        # This penalizes acceleration (second-order difference) to smooth Gaussian trajectories over time.
        self.use_trajectory_smoothing = False  # Enable trajectory smoothing regularization
        self.lambda_traj = 1e-3  # Weight for trajectory smoothing loss L_traj
        self.traj_num_time_samples = 5  # Number of consecutive time triplets to sample for L_traj
        
        # s1: Projection Reliability-Weighted Static Warm-Up parameters
        # During static warm-up (coarse stage), some projections are more consistent with the static
        # assumption (e.g., at end-inspiration/end-expiration) while others show significant motion blur.
        # s1 adaptively reweights projections based on their reconstruction residuals, giving higher
        # weight to projections that are easier to fit (more static-consistent) and lower weight to
        # projections with large residuals (likely affected by motion).
        #
        # Method:
        # 1. Maintain per-projection EMA residuals E_j for all N projections
        # 2. During burn-in (first static_reweight_burnin_steps), collect residuals with w_j = 1
        # 3. After burn-in, compute w_j = exp(-E_j / tau) or 1 / (1 + E_j / tau)
        # 4. Normalize weights so mean(w_j) = 1 within each batch
        # 5. Apply w_j to the render loss: L_static = w_j * L_render_j
        #
        # This only affects the static warm-up stage (coarse); dynamic stage (fine) is unchanged.
        self.use_static_reweighting = False  # Enable s1 projection reliability-weighted static warm-up
        self.static_reweight_method = "residual"  # Method: "residual" (EMA-based) or "learnable" (方案A)
        self.static_reweight_burnin_steps = 1500  # Steps before applying weights (for residual: EMA collection; for learnable: use w_j=1)
        self.static_reweight_ema_beta = 0.1  # EMA momentum: E_j = (1-beta)*E_j + beta*L_j (residual method only)
        self.static_reweight_tau = 0.3  # Temperature for weight computation: w_j = exp(-E_j / tau) (residual method only)
        self.static_reweight_weight_type = "exp"  # Weight function: "exp" for exp(-E/tau), "inv" for 1/(1+E/tau) (residual method only)
        
        # s1 方案A: Learnable per-projection weights parameters
        # When static_reweight_method="learnable", introduces learnable parameters α_j for each projection:
        #   w_j = sigmoid(α_j) ∈ (0,1)
        #   L_static = [Σ_j w_j * L_j] / [Σ_j w_j] + λ_mean * (mean(w_j) - ρ)²
        #
        # The gradient naturally pushes α_j down for projections with high residuals,
        # causing the canonical Gaussians to focus on static-consistent projections.
        self.static_reweight_target_mean = 0.85  # Target mean weight ρ (typically 0.8-0.9)
        self.lambda_static_reweight_mean = 0.05  # Weight for L_mean regularization
        self.static_reweight_lr = 0.01  # Learning rate for learnable α_j parameters
        
        # s2: Phase-Gated Static Canonical Warm-Up parameters
        # During static warm-up, s2 learns an implicit breathing period and canonical phase,
        # then uses a circular Gaussian window to weight projections based on their phase
        # proximity to the canonical phase. This allows the canonical Gaussians to focus on
        # projections near a specific breathing phase rather than averaging across all phases.
        #
        # Method:
        # 1. Learn τ_s2 (log-period), ψ_s2 (phase offset), φ_c (canonical phase), ξ_s2 (log-window width)
        # 2. For each projection with time t_j:
        #    - φ_j = wrap(2π * t_j / T_s2 + ψ_s2), where T_s2 = exp(τ_s2)
        #    - w_j = exp(-d_circ(φ_j, φ_c)² / (2σ_φ²)), where σ_φ = exp(ξ_s2)
        # 3. Phase-gated loss: L_s2 = [Σ w_j * L_j] / [Σ w_j] + λ_win * (log(σ_φ) - log(σ_target))²
        #
        # This only affects the static warm-up stage (coarse); dynamic stage (fine) is unchanged.
        self.use_phase_gated_static = False  # Enable s2 phase-gated static canonical warm-up
        self.static_phase_burnin_steps = 1500  # Burn-in steps with uniform weights before phase gating
        self.static_phase_sigma_target = 0.6  # Target window width σ_target in radians (~34°, covers ~20% cycle)
        self.lambda_static_phase_window = 0.01  # Weight for window regularization L_win
        self.static_phase_lr = 0.005  # Learning rate for phase parameters (τ_s2, ψ_s2, φ_c, ξ_s2)
        
        # s3: Dual-Representation Static Warm-Up (Gaussian + Voxel Co-Training) parameters
        # During static warm-up, s3 co-trains canonical Gaussians with a low-resolution 3D voxel volume V.
        # Both representations are supervised by projection loss, and a 3D distillation loss constrains
        # Gaussians to align with the smoother V structure.
        #
        # Method:
        # 1. Gaussians render projection Î_G → L_G = L_render(Î_G, I)
        # 2. Voxel V renders projection Î_V via ray marching → L_V = L_render(Î_V, I)
        # 3. 3D distillation: L_distill = mean|σ_G(x) - V(x)| at random 3D samples
        # 4. 3D smoothness: L_VTV = TV(V) (total variation on voxel volume)
        # 5. Total: L_s3 = λ_G*L_G + λ_V*L_V + λ_distill*L_distill + λ_VTV*L_VTV
        #
        # After static warm-up, V is discarded; dynamic stage is unchanged.
        self.use_s3_dual_static_volume = False  # Enable s3 dual-representation static warm-up
        self.s3_voxel_resolution = 64  # Voxel resolution D=H=W (e.g., 64 or 96)
        self.lambda_s3_G = 1.0  # Weight for Gaussian projection loss (default 1.0)
        self.lambda_s3_V = 0.2  # Weight for Voxel projection loss (e.g., 0.1-0.5)
        self.lambda_s3_distill = 1.0  # Weight for Gaussian↔Voxel distillation loss
        self.lambda_s3_VTV = 1e-4  # Weight for 3D TV regularization on V
        self.s3_num_distill_samples = 5000  # Number of 3D samples for distillation per step (reduced for memory)
        self.s3_num_ray_samples = 64  # Number of samples along each ray for voxel rendering
        self.s3_voxel_lr = 0.01  # Learning rate for voxel volume parameters
        self.s3_downsample_factor = 2  # Downsample factor for voxel projection (for efficiency)
        
        # s4: Temporal-Ensemble Guided Static Warm-Up parameters
        # During static warm-up, s4 uses externally provided pseudo-supervision:
        # - Average CT volume (V_avg): time-averaged 3D volume from traditional reconstruction
        # - Average projections (optional): time-averaged 2D projections per view
        #
        # Method:
        # 1. Gaussians render projection Î_G → L_G = L_render(Î_G, I)
        # 2. 3D distillation with V_avg: L_vol = |σ_G(x) - V_avg(x)|
        # 3. (Optional) Projection distillation: L_proj_avg = L_render(Î_G, I_avg)
        # 4. Total: L_s4 = λ_G*L_G + λ_vol*L_vol + λ_proj_avg*L_proj_avg
        #
        # After static warm-up, V_avg and avg projections are no longer used.
        self.use_s4_temporal_ensemble_static = False  # Enable s4 temporal-ensemble static warm-up
        self.s4_avg_ct_path = ""  # Path to average CT volume (.npy, .npz, .pt)
        self.s4_avg_proj_path = ""  # Path to average projections (.npy, .npz, .pt) [optional]
        self.lambda_s4_G = 1.0  # Weight for Gaussian projection loss (default 1.0)
        self.lambda_s4_vol = 0.3  # Weight for volume distillation loss (e.g., 0.1-0.5)
        self.lambda_s4_proj_avg = 0.0  # Weight for projection distillation (default 0, disabled)
        self.s4_num_vol_samples = 10000  # Number of 3D samples for volume distillation per step
        
        # s4_2: Use average CT for Gaussians initialization (optional enhancement)
        # Instead of random or FDK-based init, use V_avg to initialize Gaussians:
        # 1. Select high-density voxels from V_avg (> threshold)
        # 2. Use FPS or random sampling to select N centers as initial μ_i
        # 3. Set ρ_i from V_avg values
        self.use_s4_avg_ct_init = False  # Enable initialization from avg CT
        self.s4_avg_ct_init_thresh = 0.1  # Threshold for selecting high-density voxels (0-1)
        self.s4_avg_ct_init_num_gaussians = 50000  # Number of Gaussians to initialize
        self.s4_avg_ct_init_method = "random"  # Sampling method: "fps" (better but slower) or "random"
        self.s4_avg_ct_init_density_rescale = 0.15  # Rescaling factor for densities
        
        # =====================================================================
        # =====================================================================
        # V7.1: Consistency-Aware Rendering parameters
        # =====================================================================
        # V7.1 extends V7 by using the backward field D_b to "correct" the forward
        # deformed centers during rendering. The corrected center is computed as:
        #   y_i = mu_i + D_f(mu_i, t)            # V7 forward deformation
        #   x_hat_i = y_i + D_b(y_i, t)          # Round-trip back to canonical
        #   r_i = x_hat_i - mu_i                 # Round-trip residual
        #   y_corrected = y_i - alpha * r_i     # Corrected center
        #
        # When alpha=0, this reduces to V7 (no correction).
        # When alpha>0, centers with large round-trip residuals are "pulled back"
        # toward more consistent positions.
        self.use_v7_1_correction = False  # Enable V7.1 consistency-aware rendering
        self.correction_alpha = 0.3  # Alpha coefficient for correction (0=V7, >0=V7.1)
        
        # =====================================================================
        # TTO-α: Test-Time Optimization parameters
        # =====================================================================
        # TTO-α performs per-case calibration by optimizing a single scalar alpha
        # at test time. This allows the model to adapt to case-specific motion
        # patterns without retraining.
        #
        # TTO process:
        # 1. Load trained V7 model (all weights frozen)
        # 2. Initialize alpha = tto_alpha_init
        # 3. For each TTO step:
        #    - Sample tto_num_views_per_step views from test set
        #    - Render with V7.1 correction using current alpha
        #    - Compute L_TTO = L_render + lambda_alpha * (alpha - alpha_init)^2
        #    - Update alpha via gradient descent
        # 4. Use optimized alpha for final evaluation
        self.use_tto_alpha = False  # Enable TTO-α test-time optimization
        self.tto_alpha_init = 0.3  # Initial alpha value for TTO
        self.tto_alpha_lr = 1e-2  # Learning rate for TTO optimization
        self.tto_alpha_steps = 100  # Number of TTO optimization steps
        self.tto_alpha_reg = 1e-3  # Regularization weight lambda_alpha
        self.tto_num_views_per_step = 8  # Number of views to sample per TTO step
        
        # =====================================================================
        # V7.1-trainable-α: Train-time learnable alpha (方式 B)
        # =====================================================================
        # Instead of using a fixed alpha or TTO, train a global learnable alpha
        # together with the network during training.
        # The alpha parameter is added to the optimizer and regularized with L2.
        self.use_trainable_alpha = False  # Enable train-time learnable alpha
        self.trainable_alpha_init = 0.0  # Initial value for trainable alpha
        self.trainable_alpha_reg = 1e-3  # L2 regularization weight for alpha
        
        # =====================================================================
        # V7.1-α(t): Time-dependent alpha (扩展版本 2.3)
        # =====================================================================
        # Instead of a global scalar alpha, use a small network g_theta(t) to
        # predict time-dependent alpha values. This allows different correction
        # strength for different breathing phases.
        # Options: 'none' (scalar), 'mlp' (small MLP), 'fourier' (Fourier basis)
        self.use_time_dependent_alpha = False  # Enable time-dependent alpha
        self.alpha_network_type = 'fourier'  # Type: 'mlp' or 'fourier'
        self.alpha_fourier_freqs = 4  # Number of Fourier frequencies (if fourier)
        self.alpha_mlp_hidden = 32  # Hidden layer size (if mlp)
        
        # =====================================================================
        # V7.2: End-to-End Consistency-Aware Bidirectional Deformation
        # =====================================================================
        # V7.2 uses the consistency-aware forward mapping DURING TRAINING, not just
        # at test time like V7.1. The key difference is:
        # - V7.1: Training uses y = mu + D_f, test-time applies correction
        # - V7.2: BOTH training and test use y_corrected = y - alpha * r
        #
        # This allows D_b to receive gradients from L_render through the main
        # rendering path, not just from L_inv. As a result:
        # - L_inv is NOT used in V7.2 (lambda_inv forced to 0)
        # - Instead, a lightweight L_b regularization controls D_b magnitude
        #
        # Formula:
        #   y = mu + D_f(mu, t)           # forward deformation
        #   x_hat = y + D_b(y, t)         # round-trip reconstruction
        #   r = x_hat - mu                # round-trip residual
        #   y_corrected = y - alpha * r   # V7.2 effective center
        self.use_v7_2_consistency = False  # Enable V7.2 end-to-end consistency-aware forward
        self.v7_2_alpha_init = 0.3  # Initial value for V7.2 alpha
        self.v7_2_alpha_learnable = True  # Whether alpha is learnable (global scalar)
        self.v7_2_lambda_b_reg = 1e-3  # L_b: regularization weight for |D_b(y,t)|
        self.v7_2_lambda_alpha_reg = 1e-3  # L_alpha: regularization for (alpha - alpha_init)^2
        # V7.2 time-dependent alpha: use a small network g_theta(t) to predict alpha
        # This allows different correction strength for different breathing phases
        self.v7_2_use_time_dependent_alpha = False  # Enable time-dependent alpha for V7.2
        self.v7_2_alpha_network_type = 'fourier'  # Type: 'mlp' or 'fourier'
        self.v7_2_alpha_fourier_freqs = 4  # Number of Fourier frequencies (if fourier)
        self.v7_2_alpha_mlp_hidden = 32  # Hidden layer size (if mlp)
        
        # V7.2.1: Canonical-Space Cycle Consistency (L_cycle-canon)
        # =====================================================================
        # V7.2.1 adds a canonical-space cycle consistency loss on top of V7.2.
        # Instead of requiring D_b to be a perfect inverse (L_inv), we require:
        #   "The same point μ_i decoded back to canonical at t and t+T̂ should be consistent"
        # Formula:
        #   x_canon(t)   = ϕ̃_f(μ, t)   + D_b(ϕ̃_f(μ, t),   t)
        #   x_canon(t+T) = ϕ̃_f(μ, t+T) + D_b(ϕ̃_f(μ, t+T), t+T)
        #   L_cycle_canon = |x_canon(t+T) - x_canon(t)|
        # This provides a softer temporal constraint on D_b without forcing r→0.
        self.use_v7_2_1_cycle_canon = False  # Enable canonical-space cycle consistency
        self.v7_2_1_lambda_cycle_canon = 1e-3  # Weight for L_cycle_canon
        
        # V7.3: Temporal Bidirectional Warp Consistency
        # =====================================================================
        # V7.3 adds explicit time-forward/time-backward warp constraints on top of V7.2.1.
        # Instead of just canonical↔t bidirectionality, we define true temporal warps:
        #   - Time-forward warp (t1→t2): x(t1) → canonical → x'(t2) ≈ x(t2)
        #   - Time-backward warp (t2→t1): x(t2) → canonical → x'(t1) ≈ x(t1)
        # Uses canonical as a bridge to warp between different time points.
        # Formula:
        #   μ^(t1) = x(t1) + D_b(x(t1), t1)  # decode to canonical at t1
        #   x_fw(t2|t1) = ϕ̃_f(μ^(t1), t2)   # forward to t2 from decoded canonical
        #   L_fw = |x_fw(t2|t1) - x(t2)|     # should match direct position at t2
        # Similarly for backward warp.
        # Note: Requires use_v7_2_consistency=True (depends on corrected centers).
        self.use_v7_3_timewarp = False  # Enable temporal bidirectional warp losses
        self.v7_3_lambda_fw = 1e-3  # Weight for time-forward consistency loss L_fw
        self.v7_3_lambda_bw = 1e-3  # Weight for time-backward consistency loss L_bw
        self.v7_3_lambda_fw_bw = 0.0  # Weight for round-trip closure L_fw_bw (optional)
        self.v7_3_timewarp_delta_fraction = 0.1  # Δ = fraction * T̂, time gap for warp pairs
        
        # V7.3.1: Temporal Regularization for Covariance and Density
        # =====================================================================
        # V7.3.1 extends V7.3 by adding temporal smoothness and periodicity losses
        # for per-Gaussian covariance (scale) and density, in addition to centers:
        #   - Temporal TV: penalizes rapid changes in scale/density along time
        #   - Periodic consistency: enforces cycle consistency over learned period T̂
        # Formula:
        #   L_tv_sigma = E[|log(S(t+δ)) - log(S(t))|]   # log-scale for stability
        #   L_tv_rho = E[|ρ(t+δ) - ρ(t)|]
        #   L_cycle_sigma = E[|log(S(t+T̂)) - log(S(t))|]
        #   L_cycle_rho = E[|ρ(t+T̂) - ρ(t)|]
        # Note: If scale/density are static (not time-dependent), losses auto no-op.
        self.use_v7_3_1_sigma_rho = False  # Enable temporal regularization for Σ/ρ
        self.v7_3_1_lambda_tv_rho = 1e-4  # Weight for density temporal TV loss
        self.v7_3_1_lambda_tv_sigma = 1e-4  # Weight for scale temporal TV loss
        self.v7_3_1_lambda_cycle_rho = 1e-4  # Weight for density periodic consistency
        self.v7_3_1_lambda_cycle_sigma = 1e-4  # Weight for scale periodic consistency
        self.v7_3_1_sigma_rho_delta_fraction = 0.1  # δ = fraction * T̂ for TV step
        
        # V7.4: Canonical Decode of Shape and Density
        # =====================================================================
        # V7.4 extends backward field to decode not just canonical center, but also
        # canonical shape (log-scale) and density via learnable decode heads:
        #   D_b_pos(x,t): position residual (already exists)
        #   D_b_shape(x,t): canonical log-scale residual (new)
        #   D_b_dens(x,t): canonical density residual (new)
        # Canonical decoded: ŝ_canon(t) = s_base + D_b_shape(x(t), t)
        #                    ρ̂_canon(t) = ρ_base + D_b_dens(x(t), t)
        # Losses:
        #   L_cycle_canon_sigma = |ŝ_canon(t+T̂) - ŝ_canon(t)| (periodic consistency)
        #   L_cycle_canon_rho = |ρ̂_canon(t+T̂) - ρ̂_canon(t)|
        #   L_prior_sigma = |ŝ_canon(t) - s_base| (anchor to base)
        #   L_prior_rho = |ρ̂_canon(t) - ρ_base|
        # This gives backward field a 4D→3D factorization role for Σ/ρ.
        self.use_v7_4_canonical_decode = False  # Enable canonical decode for Σ/ρ
        self.v7_4_lambda_cycle_canon_sigma = 1e-4  # Weight for canonical shape cycle loss
        self.v7_4_lambda_cycle_canon_rho = 1e-4  # Weight for canonical density cycle loss
        self.v7_4_lambda_prior_sigma = 1e-4  # Weight for shape prior (close to base)
        self.v7_4_lambda_prior_rho = 1e-4  # Weight for density prior (close to base)
        
        # V7.5: Full Time-Warp Consistency for Center, Covariance and Density
        # =====================================================================
        # V7.5 extends time-warp consistency (V7.3) to shape and density:
        #   G(t1) -> Backward decode -> Canonical state -> Forward timewarp -> G_hat(t2|t1)
        # We require G_hat(t2|t1) matches G(t2) for center, shape, and density.
        #
        # New forward timewarp decoders (used only for loss, not rendering):
        #   shape_timewarp_head: canonical log-scale + time -> predicted log-scale
        #   dens_timewarp_head: canonical density + time -> predicted density
        #
        # Losses:
        #   L_fw_Sigma = |s_hat_fw(t2|t1) - s(t2)|  (forward warp shape)
        #   L_fw_rho = |rho_hat_fw(t2|t1) - rho(t2)|  (forward warp density)
        #   L_bw_Sigma = |s_hat_bw(t1|t2) - s(t1)|  (backward warp shape)
        #   L_bw_rho = |rho_hat_bw(t1|t2) - rho(t1)|  (backward warp density)
        #   L_rt_Sigma, L_rt_rho: round-trip consistency (optional)
        self.use_v7_5_full_timewarp = False  # Enable full time-warp for Σ/ρ
        self.v7_5_lambda_fw_sigma = 1e-5  # Weight for forward warp shape loss
        self.v7_5_lambda_fw_rho = 1e-5  # Weight for forward warp density loss
        self.v7_5_lambda_bw_sigma = 0.0  # Weight for backward warp shape loss (start disabled)
        self.v7_5_lambda_bw_rho = 0.0  # Weight for backward warp density loss (start disabled)
        self.v7_5_lambda_rt_sigma = 0.0  # Weight for round-trip shape loss (optional)
        self.v7_5_lambda_rt_rho = 0.0  # Weight for round-trip density loss (optional)
        self.v7_5_timewarp_delta_fraction = 0.25  # Δt = fraction * T̂ for time-warp
        
        # V7.5.1: Full-State Bidirectional Time-Warp Consistency
        # =====================================================================
        # V7.5.1 extends the bidirectional time-warp consistency to the full
        # Gaussian state (center, covariance, density).
        #
        # Key changes from V7.5:
        #   - Center: Ensures forward + backward losses are both active
        #   - Σ/ρ: Enables backward warp in addition to forward warp
        #
        # When use_v7_5_1_roundtrip_full_state=True:
        #   - Center uses L_fw + L_bw from V7.3 (both active)
        #   - Σ/ρ uses L_fw + L_bw (both active)
        #
        # Backward for Σ/ρ:
        #   L_bw_Sigma = |s_bw(t1|t2) - s(t1)|, canonical(t2) warped to t1
        #   L_bw_rho = |ρ_bw(t1|t2) - ρ(t1)|, canonical(t2) warped to t1
        self.use_v7_5_1_roundtrip_full_state = False  # Enable full-state bidirectional consistency
        
        # V7.6: Period-Free Mode (extension of V7.5.1)
        # =====================================================================
        # V7.6 disables all losses that enforce periodicity using the learned period T̂.
        # This is useful when the motion is NOT strictly periodic (e.g., irregular breathing).
        #
        # When use_v7_6_no_period=True:
        #   - L_cycle (motion periodicity): DISABLED
        #   - L_cycle_canon (V7.2.1 canonical cycle): DISABLED  
        #   - L_cycle_sigma/rho (V7.3.1 periodic σ/ρ): DISABLED
        #   - L_cycle_canon_Sigma/rho (V7.4 canonical cycle): DISABLED
        #   - period learning: DISABLED (period parameter frozen)
        #
        # Losses that are KEPT (they use T̂ only for time sampling, not periodicity):
        #   - L_fw, L_bw (V7.3 time-warp for centers)
        #   - L_fw_Sigma/rho, L_bw_Sigma/rho (V7.5 time-warp for σ/ρ)
        #   - L_tv_sigma/rho (V7.3.1 temporal smoothness)
        #   - All other losses
        self.use_v7_6_no_period = False  # Enable period-free mode
        
        # V7.7: Freeze Period Only (extension of V7.5.1)
        # =====================================================================
        # V7.7 keeps ALL losses from V7.5.1 (including periodicity losses),
        # but freezes the period parameter learning (lr=0).
        # This tests whether a fixed period (init=2.8) works as well as learned.
        #
        # Difference from V7.6:
        #   - V7.6: Disables periodicity losses + freezes period
        #   - V7.7: Keeps periodicity losses + freezes period (uses init period=2.8)
        self.use_v7_7_freeze_period_only = False  # Freeze period but keep all losses
        
        # s5: 4D Dynamic-Aware Multi-Phase FDK Point Cloud Initialization
        # =====================================================================
        # s5 is an alternative initialization strategy that uses multi-phase FDK
        # to generate 4D-aware Gaussian initialization:
        # - Divides projections into P phases based on acquisition time
        # - Computes V_ref (reference phase), V_avg (temporal mean), V_var (temporal variance)
        # - Uses V_var to identify dynamic vs static regions
        # - Samples more Gaussians in dynamic regions with smaller initial scales
        # - Samples fewer Gaussians in static regions with larger initial scales
        #
        # s5 only affects initialization, not training. It can be combined with
        # any dynamic model (v7, v9, etc.) and static warm-up methods (s1-s4).
        #
        # To use s5:
        # 1. Generate init file: python tools/build_init_s5_4d.py --input data/case.pickle --output data/init_s5_4d_case.npy
        # 2. Train with: python train.py -s data/case.pickle --use_s5_4d_init --s5_init_path data/init_s5_4d_case.npy
        self.use_s5_4d_init = False  # Enable s5 4D-aware initialization
        self.s5_init_path = ""  # Path to s5 init file (init_s5_4d_*.npy)
        # The following are for reference/documentation; actual generation uses build_init_s5_4d.py
        self.s5_num_phases = 3  # Number of phases for multi-phase FDK
        self.s5_ref_phase_index = -1  # Reference phase index (-1 = P//2)
        self.s5_num_points = 50000  # Number of Gaussians to initialize
        self.s5_static_weight = 0.7  # Weight for static/density component
        self.s5_dynamic_weight = 0.3  # Weight for dynamic/variance component
        self.s5_density_exponent = 1.5  # Exponent for density term
        self.s5_var_exponent = 1.0  # Exponent for variance term
        
        super().__init__(parser, "Optimization Parameters")

class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64 #64 # width of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.timebase_pe = 4 # useless
        self.defor_depth = 1 # 1 # depth of deformation MLP, larger will increase the rendering quality and decrase the training/rendering speed.
        self.posebase_pe = 10 # useless
        self.scale_rotation_pe = 2 # useless
        self.density_pe = 2 # useless
        self.timenet_width = 64 # useless
        self.timenet_output = 32 # useless
        self.bounds = 1.6 
        self.plane_tv_weight = 0.0001 # TV loss of spatial grid
        self.time_smoothness_weight = 0.001 # TV loss of temporal grid  0.01
        self.l1_time_planes = 0.0001  # TV loss of temporal grid
        self.period_regulation_weight = 1.0   # useless
        self.period_construction_weight = 1e-5  # useless
        self.kplanes_config = {
                             'grid_dimensions': 2,
                             'input_coordinate_dim': 4,
                             'output_coordinate_dim': 32,   # 32
                             'resolution': [64, 64, 64, 150]  # [64,64,64]: resolution of spatial grid. 25: resolution of temporal grid, better to be half length of dynamic frames
                            }    # 150
        self.multires = [1, 2, 4, 8] # multi resolution of voxel grid
        self.no_dx=False # cancel the deformation of Gaussians' position
        self.no_grid=False # cancel the spatial-temporal hexplane.
        self.no_ds=False # cancel the deformation of Gaussians' scaling
        self.no_dr=False # cancel the deformation of Gaussians' rotations
        self.no_do=True # cancel the deformation of Gaussians' opacity     # True
        # self.no_dshs=True # cancel the deformation of SH colors.
        self.empty_voxel=False # useless
        self.grid_pe=0 # useless, I was trying to add positional encoding to hexplane's features
        self.static_mlp=False # useless
        self.apply_rotation=False # useless
        
        # Velocity field parameters (diffeomorphic-like forward motion)
        self.use_velocity_field = False  # Enable velocity field mode: interpret network output as velocity v(x,t) instead of displacement D_f(x,t)
        self.velocity_num_steps = 4  # Number of Euler integration steps K for velocity field integration
        
        # v6: Shared velocity inverse parameters
        # When enabled, uses the same velocity field v(x,t) to construct both forward and backward mappings
        # via numerical integration, removing the need for a separate D_b network and L_sym
        self.use_shared_velocity_inverse = False  # Enable v6 mode: shared velocity for phi_f and phi_b
        
        # v8: Phase-Conditioned Deformation parameters
        # When enabled, uses SSRML's learned period T_hat to compute a phase embedding:
        #   phi(t) = 2π * t / T_hat
        #   phase_embed = [sin(phi), cos(phi)]
        # This phase embedding is concatenated to the trunk output before the D_f/D_b heads,
        # making the deformation network explicitly aware of the breathing phase.
        self.use_phase_conditioned_deformation = False  # Enable v8 mode: phase-conditioned D_f/D_b
        
        # v9: Low-Rank Motion Modes parameters
        # When enabled, reparameterizes D_f/D_b as low-rank decomposition:
        #   D_f(μ_i, t) = Σ_m a_m(t) * u_{i,m}
        #   D_b(μ_i, t) = Σ_m b_m(t) * u_{i,m}
        # where u_{i,m} are per-Gaussian motion modes and a(t)/b(t) are phase-dependent coefficients.
        # This leverages the inherently low-dimensional nature of respiratory motion.
        self.use_low_rank_motion_modes = False  # Enable v9 mode: low-rank motion decomposition
        self.num_motion_modes = 3  # Number of motion modes M (typically 2-4)
        
        # v10: Adaptive Gating Network parameters
        # Hidden size and number of layers for the gating networks G_f and G_b
        self.gating_hidden_size = 32  # Hidden dimension for gating MLP
        self.gating_num_layers = 2  # Number of hidden layers in gating MLP
        
        super().__init__(parser, "ModelHiddenParams")


def get_combined_args(parser: ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = osp.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k, v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
