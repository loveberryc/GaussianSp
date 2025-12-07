import os
import os.path as osp
import torch
# torch.autograd.set_detect_anomaly(True)  # DEBUG: Disabled - may cause issues
import torch.nn.functional as F
from random import randint
import sys
from tqdm import tqdm
from argparse import ArgumentParser
import numpy as np
import yaml
import math

sys.path.append("./")
from x2_gaussian.arguments import ModelParams, OptimizationParams, PipelineParams, ModelHiddenParams
from x2_gaussian.gaussian import GaussianModel, render, query, initialize_gaussian, render_prior_oneT
from x2_gaussian.utils.general_utils import safe_state
from x2_gaussian.utils.cfg_utils import load_config
from x2_gaussian.utils.log_utils import prepare_output_and_logger
from x2_gaussian.dataset import Scene
from x2_gaussian.utils.loss_utils import l1_loss, ssim, tv_3d_loss
from x2_gaussian.utils.image_utils import metric_vol, metric_proj
from x2_gaussian.utils.plot_utils import show_two_slice
from x2_gaussian.utils.static_reweighting import StaticReweightingManager, LearnableStaticReweighting, apply_s1_preset
from x2_gaussian.utils.phase_gated_static import PhaseGatedStatic, apply_s2_preset
from x2_gaussian.utils.dual_static_volume import DualStaticVolume, apply_s3_preset
from x2_gaussian.utils.temporal_ensemble_static import TemporalEnsembleStatic, apply_s4_preset, initialize_gaussians_from_avg_ct

def apply_physx_preset(opt, hyper):
    """
    Apply PhysX-Gaussian preset: Anchor-based Spacetime Transformer.
    
    PhysX-Gaussian replaces the HexPlane + MLP deformation field with an
    anchor-based transformer that learns physical traction relationships
    between anatomical structures via masked modeling (BERT-style).
    
    When use_anchor_deformation is enabled:
    1. Disables HexPlane+MLP in favor of anchor transformer
    2. Enables physics completion loss L_phys
    3. Optionally reduces period consistency weight (not fully relying on periodicity)
    4. Enables anchor motion smoothness regularization
    """
    if not getattr(hyper, 'use_anchor_deformation', False):
        return
    
    print("=" * 60)
    print("PHYSX-GAUSSIAN: ANCHOR-BASED SPACETIME TRANSFORMER ACTIVATED")
    print("=" * 60)
    print("Replacing HexPlane + MLP with Anchor Transformer:")
    print(f"  - num_anchors: {hyper.num_anchors}")
    print(f"  - anchor_k: {hyper.anchor_k}")
    print(f"  - mask_ratio: {hyper.mask_ratio}")
    print(f"  - transformer_dim: {hyper.transformer_dim}")
    print(f"  - transformer_heads: {hyper.transformer_heads}")
    print(f"  - transformer_layers: {hyper.transformer_layers}")
    print("")
    print("PhysX-Gaussian Losses:")
    lambda_phys = getattr(hyper, 'lambda_phys', 0.1)
    lambda_anchor_smooth = getattr(hyper, 'lambda_anchor_smooth', 0.01)
    phys_warmup_steps = getattr(hyper, 'phys_warmup_steps', 2000)
    print(f"  - L_phys (physics completion): λ={lambda_phys}")
    print(f"  - L_anchor_smooth (anchor smoothness): λ={lambda_anchor_smooth}")
    print(f"  - Physics warmup steps: {phys_warmup_steps}")
    print("")
    print("Key Innovation:")
    print("  - Learn physical traction relationships via masked self-attention")
    print("  - Generalize to irregular breathing patterns without periodic assumption")
    print("=" * 60)


def apply_v7_preset(opt, hyper):
    """
    Apply v7 main model preset: Bidirectional Displacement + L_inv + L_cycle.
    
    v7 is the recommended configuration based on ablation study results showing that:
    - v1 (D_f + D_b + L_inv) + v3 (L_cycle) provides the best performance
    - v2 (L_sym), v4 (L_jac), v5 (velocity field), v6 (shared velocity) do not improve results
    
    When use_v7_bidirectional_displacement is enabled, this function:
    1. Force-enables L_inv (inverse consistency) with default lambda_inv=0.1
    2. Force-enables L_cycle (motion periodicity) with default lambda_cycle=0.01
    3. Force-disables L_sym (symmetry regularization)
    4. Force-disables L_jac (Jacobian regularization)
    5. Force-disables velocity_field mode
    6. Force-disables shared_velocity_inverse mode
    
    v8 (Phase-Conditioned) extends v7 by adding phase embedding to D_f/D_b heads:
    - When use_phase_conditioned_deformation is True, the deformation heads receive
      [sin(2π*t/T_hat), cos(2π*t/T_hat)] as additional input, making them explicitly
      aware of the breathing phase learned by SSRML.
    
    v9 (Low-Rank Motion Modes) extends v7/v8 by reparameterizing D_f/D_b as:
    - D_f(μ_i, t) = Σ_m a_m(t) * u_{i,m}
    - D_b(μ_i, t) = Σ_m b_m(t) * u_{i,m}
    where u_{i,m} are per-Gaussian motion modes and a(t)/b(t) are phase-dependent coefficients.
    """
    if not opt.use_v7_bidirectional_displacement:
        return
    
    # Check if v8 (phase conditioning) or v9 (low-rank modes) is also enabled
    use_phase = getattr(hyper, 'use_phase_conditioned_deformation', False)
    use_low_rank = getattr(hyper, 'use_low_rank_motion_modes', False)
    use_adaptive_gating = getattr(opt, 'use_adaptive_gating', False)
    use_traj_smoothing = getattr(opt, 'use_trajectory_smoothing', False)
    
    # Determine version string based on enabled features
    if use_low_rank and (use_adaptive_gating or use_traj_smoothing):
        version_str = "V10 ADAPTIVE GATED MOTION"
    elif use_low_rank:
        version_str = "V9 LOW-RANK MOTION MODES"
    elif use_phase:
        version_str = "V8 PHASE-ALIGNED"
    else:
        version_str = "V7"
    
    # Check if V7.2 end-to-end consistency is enabled
    use_v7_2 = getattr(opt, 'use_v7_2_consistency', False)
    # Check if V7.2.1 canonical cycle consistency is enabled
    use_v7_2_1 = getattr(opt, 'use_v7_2_1_cycle_canon', False) and use_v7_2
    # Check if V7.3 temporal bidirectional warp is enabled
    use_v7_3 = getattr(opt, 'use_v7_3_timewarp', False) and use_v7_2
    # Check if V7.3.1 sigma/rho temporal regularization is enabled
    use_v7_3_1 = getattr(opt, 'use_v7_3_1_sigma_rho', False) and use_v7_2
    # Check if V7.4 canonical decode is enabled
    use_v7_4 = getattr(opt, 'use_v7_4_canonical_decode', False) and use_v7_2
    # Check if V7.5 full time-warp is enabled
    use_v7_5 = getattr(opt, 'use_v7_5_full_timewarp', False) and use_v7_2 and use_v7_4
    # Check if V7.5.1 full-state bidirectional is enabled
    use_v7_5_1 = getattr(opt, 'use_v7_5_1_roundtrip_full_state', False) and use_v7_5
    
    if use_v7_5_1:
        version_str = "V7.5.1 FULL-STATE BIDIRECTIONAL TIME-WARP"
    elif use_v7_5:
        version_str = "V7.5 FULL TIME-WARP SIGMA/RHO"
    elif use_v7_4:
        version_str = "V7.4 CANONICAL DECODE SIGMA/RHO"
    elif use_v7_3_1:
        version_str = "V7.3.1 SIGMA/RHO TEMPORAL REGULARIZATION"
    elif use_v7_3:
        version_str = "V7.3 TEMPORAL BIDIRECTIONAL WARP"
    elif use_v7_2_1:
        version_str = "V7.2.1 END-TO-END + CANONICAL CYCLE"
    elif use_v7_2:
        version_str = "V7.2 END-TO-END CONSISTENCY-AWARE"
    
    print("=" * 60)
    print(f"{version_str} MAIN MODEL PRESET ACTIVATED")
    print("=" * 60)
    print("Using bidirectional displacement (D_f, D_b) with:")
    if use_v7_2:
        print("  - L_inv (inverse consistency): DISABLED (V7.2 uses D_b in rendering path)")
        print("  - L_b (backward magnitude reg): ENABLED (V7.2)")
        alpha_learnable = getattr(opt, 'v7_2_alpha_learnable', False)
        alpha_init = getattr(opt, 'v7_2_alpha_init', 0.3)
        print(f"  - V7.2 alpha: {alpha_init} ({'learnable' if alpha_learnable else 'fixed'})")
        if use_v7_2_1:
            lambda_cycle_canon = getattr(opt, 'v7_2_1_lambda_cycle_canon', 1e-3)
            print(f"  - L_cycle_canon (canonical cycle): ENABLED (V7.2.1, λ={lambda_cycle_canon})")
        if use_v7_3:
            lambda_fw = getattr(opt, 'v7_3_lambda_fw', 1e-3)
            lambda_bw = getattr(opt, 'v7_3_lambda_bw', 1e-3)
            lambda_fw_bw = getattr(opt, 'v7_3_lambda_fw_bw', 0.0)
            delta_frac = getattr(opt, 'v7_3_timewarp_delta_fraction', 0.1)
            print(f"  - L_fw (time-forward): ENABLED (V7.3, λ={lambda_fw})")
            print(f"  - L_bw (time-backward): ENABLED (V7.3, λ={lambda_bw})")
            if lambda_fw_bw > 0:
                print(f"  - L_fw_bw (round-trip): ENABLED (V7.3, λ={lambda_fw_bw})")
            print(f"  - Timewarp Δ fraction: {delta_frac}")
        # V7.3.1: Sigma/Rho temporal regularization
        use_v7_3_1 = getattr(opt, 'use_v7_3_1_sigma_rho', False) and use_v7_2
        if use_v7_3_1:
            lambda_tv_sigma = getattr(opt, 'v7_3_1_lambda_tv_sigma', 1e-4)
            lambda_tv_rho = getattr(opt, 'v7_3_1_lambda_tv_rho', 1e-4)
            lambda_cycle_sigma = getattr(opt, 'v7_3_1_lambda_cycle_sigma', 1e-4)
            lambda_cycle_rho = getattr(opt, 'v7_3_1_lambda_cycle_rho', 1e-4)
            delta_frac_sigma_rho = getattr(opt, 'v7_3_1_sigma_rho_delta_fraction', 0.1)
            print(f"  - L_tv_sigma (scale temporal TV): ENABLED (V7.3.1, λ={lambda_tv_sigma})")
            print(f"  - L_cycle_sigma (scale periodic): ENABLED (V7.3.1, λ={lambda_cycle_sigma})")
            print(f"  - L_tv_rho (density temporal TV): λ={lambda_tv_rho} (active if density is dynamic)")
            print(f"  - L_cycle_rho (density periodic): λ={lambda_cycle_rho} (active if density is dynamic)")
            print(f"  - Sigma/Rho Δ fraction: {delta_frac_sigma_rho}")
        # V7.4: Canonical decode for Sigma/Rho
        if use_v7_4:
            lambda_cycle_canon_sigma = getattr(opt, 'v7_4_lambda_cycle_canon_sigma', 1e-4)
            lambda_cycle_canon_rho = getattr(opt, 'v7_4_lambda_cycle_canon_rho', 1e-4)
            lambda_prior_sigma = getattr(opt, 'v7_4_lambda_prior_sigma', 1e-4)
            lambda_prior_rho = getattr(opt, 'v7_4_lambda_prior_rho', 1e-4)
            print(f"  - L_cycle_canon_Sigma (canonical shape cycle): ENABLED (V7.4, λ={lambda_cycle_canon_sigma})")
            print(f"  - L_cycle_canon_rho (canonical density cycle): ENABLED (V7.4, λ={lambda_cycle_canon_rho})")
            print(f"  - L_prior_Sigma (shape anchor to base): ENABLED (V7.4, λ={lambda_prior_sigma})")
            print(f"  - L_prior_rho (density anchor to base): ENABLED (V7.4, λ={lambda_prior_rho})")
            print(f"  - Backward shape/density heads: ENABLED (V7.4)")
        # V7.5: Full time-warp for Sigma/Rho
        if use_v7_5:
            lambda_fw_sigma = getattr(opt, 'v7_5_lambda_fw_sigma', 1e-5)
            lambda_fw_rho = getattr(opt, 'v7_5_lambda_fw_rho', 1e-5)
            lambda_bw_sigma = getattr(opt, 'v7_5_lambda_bw_sigma', 0.0)
            lambda_bw_rho = getattr(opt, 'v7_5_lambda_bw_rho', 0.0)
            lambda_rt_sigma = getattr(opt, 'v7_5_lambda_rt_sigma', 0.0)
            lambda_rt_rho = getattr(opt, 'v7_5_lambda_rt_rho', 0.0)
            delta_frac_v7_5 = getattr(opt, 'v7_5_timewarp_delta_fraction', 0.25)
            print(f"  - L_fw_Sigma (forward warp shape): ENABLED (V7.5, λ={lambda_fw_sigma})")
            print(f"  - L_fw_rho (forward warp density): ENABLED (V7.5, λ={lambda_fw_rho})")
            if lambda_bw_sigma > 0:
                print(f"  - L_bw_Sigma (backward warp shape): ENABLED (V7.5, λ={lambda_bw_sigma})")
            if lambda_bw_rho > 0:
                print(f"  - L_bw_rho (backward warp density): ENABLED (V7.5, λ={lambda_bw_rho})")
            if lambda_bw_sigma > 0 or lambda_bw_rho > 0 or use_v7_5_1:
                # V7.5.1 enables backward warp with default weights
                bw_sigma_eff = lambda_bw_sigma if lambda_bw_sigma > 0 else 1e-5
                bw_rho_eff = lambda_bw_rho if lambda_bw_rho > 0 else 1e-5
                print(f"  - L_bw_Sigma/rho (backward): ENABLED (V7.5.1, λ_σ={bw_sigma_eff}, λ_ρ={bw_rho_eff})")
            if lambda_rt_sigma > 0 or lambda_rt_rho > 0:
                print(f"  - L_rt_Sigma/rho (round-trip): ENABLED (λ_σ={lambda_rt_sigma}, λ_ρ={lambda_rt_rho})")
            print(f"  - Forward timewarp decoders: ENABLED (V7.5)")
            print(f"  - Timewarp Δ fraction: {delta_frac_v7_5}")
        # V7.5.1: Full-state bidirectional consistency
        if use_v7_5_1:
            print(f"  - V7.5.1: Full-state bidirectional enabled for centers + Σ/ρ")
            print(f"  - Center: L_fw + L_bw (both active)")
            print(f"  - Σ/ρ: L_fw + L_bw (both active)")
    else:
        print("  - L_inv (inverse consistency): ENABLED")
    print("  - L_cycle (motion periodicity): ENABLED")
    if use_low_rank:
        num_modes = getattr(hyper, 'num_motion_modes', 3)
        print(f"  - Low-rank motion modes U[K, M, 3]: ENABLED (v9, M={num_modes})")
        print(f"  - Coefficient networks F_a, F_b: ENABLED (v9)")
    if use_adaptive_gating:
        print(f"  - Adaptive gating networks G_f, G_b: ENABLED (v10)")
        print(f"  - Gate regularization (lambda_gate={opt.lambda_gate}): {'ENABLED' if opt.lambda_gate > 0 else 'DISABLED'}")
    if use_traj_smoothing:
        print(f"  - Trajectory smoothing (lambda_traj={opt.lambda_traj}): ENABLED (v10)")
    if use_phase:
        print("  - Phase embedding [sin(2π*t/T_hat), cos(2π*t/T_hat)]: ENABLED (v8)")
    print("Disabling ablation-only features:")
    print("  - L_sym (symmetry regularization): DISABLED")
    print("  - L_jac (Jacobian regularization): DISABLED")
    print("  - Velocity field integration: DISABLED")
    print("  - Shared velocity inverse: DISABLED")
    print("=" * 60)
    
    # Force-enable v1 + v3 components (but skip L_inv for V7.2)
    if use_v7_2:
        # V7.2: L_inv is disabled, D_b participates through rendering path
        opt.use_inverse_consistency = False
        opt.lambda_inv = 0.0
        print(f"  -> V7.2: lambda_inv = 0 (L_inv disabled, using L_b instead)")
    else:
        opt.use_inverse_consistency = True
        if opt.lambda_inv <= 0:
            opt.lambda_inv = 0.1  # Default value from best experiments
            print(f"  -> Setting lambda_inv to {opt.lambda_inv} (default)")
        else:
            print(f"  -> Using user-specified lambda_inv = {opt.lambda_inv}")
    
    opt.use_cycle_motion = True
    if opt.lambda_cycle <= 0:
        opt.lambda_cycle = 0.01  # Default value from best experiments
        print(f"  -> Setting lambda_cycle to {opt.lambda_cycle} (default)")
    else:
        print(f"  -> Using user-specified lambda_cycle = {opt.lambda_cycle}")
    
    # Force-disable v2 (symmetry regularization)
    if opt.use_symmetry_reg:
        print("  -> Warning: use_symmetry_reg was True but v7/v8/v9 forces it OFF")
    opt.use_symmetry_reg = False
    opt.lambda_sym = 0.0
    
    # Force-disable v4 (Jacobian regularization)
    if opt.use_jacobian_reg:
        print("  -> Warning: use_jacobian_reg was True but v7/v8/v9 forces it OFF")
    opt.use_jacobian_reg = False
    opt.lambda_jac = 0.0
    
    # Force-disable v5 (velocity field)
    if hyper.use_velocity_field:
        print("  -> Warning: use_velocity_field was True but v7/v8/v9 forces it OFF")
    hyper.use_velocity_field = False
    
    # Force-disable v6 (shared velocity inverse)
    if hyper.use_shared_velocity_inverse:
        print("  -> Warning: use_shared_velocity_inverse was True but v7/v8/v9 forces it OFF")
    hyper.use_shared_velocity_inverse = False
    
    print("=" * 60)
    phase_str = ", phase_conditioned=True" if use_phase else ""
    low_rank_str = f", low_rank_modes={num_modes}" if use_low_rank else ""
    gating_str = f", adaptive_gating=True, lambda_gate={opt.lambda_gate}" if use_adaptive_gating else ""
    traj_str = f", traj_smoothing=True, lambda_traj={opt.lambda_traj}" if use_traj_smoothing else ""
    v7_2_str = f", v7_2_alpha={getattr(opt, 'v7_2_alpha_init', 0.3)}, L_b_reg={getattr(opt, 'v7_2_lambda_b_reg', 1e-3)}" if use_v7_2 else ""
    v7_2_1_str = f", L_cycle_canon_reg={getattr(opt, 'v7_2_1_lambda_cycle_canon', 1e-3)}" if use_v7_2_1 else ""
    v7_3_str = f", L_fw={getattr(opt, 'v7_3_lambda_fw', 1e-3)}, L_bw={getattr(opt, 'v7_3_lambda_bw', 1e-3)}, Δ_frac={getattr(opt, 'v7_3_timewarp_delta_fraction', 0.1)}" if use_v7_3 else ""
    v7_3_1_str = f", L_tv_σ={getattr(opt, 'v7_3_1_lambda_tv_sigma', 1e-4)}, L_cycle_σ={getattr(opt, 'v7_3_1_lambda_cycle_sigma', 1e-4)}" if use_v7_3_1 else ""
    v7_4_str = f", L_cycle_canon_σ={getattr(opt, 'v7_4_lambda_cycle_canon_sigma', 1e-4)}, L_prior_σ={getattr(opt, 'v7_4_lambda_prior_sigma', 1e-4)}" if use_v7_4 else ""
    v7_5_str = f", L_fw_σ={getattr(opt, 'v7_5_lambda_fw_sigma', 1e-5)}, L_fw_ρ={getattr(opt, 'v7_5_lambda_fw_rho', 1e-5)}" if use_v7_5 else ""
    print(f"{version_str} Final Config: lambda_inv={opt.lambda_inv}, lambda_cycle={opt.lambda_cycle}{phase_str}{low_rank_str}{gating_str}{traj_str}{v7_2_str}{v7_2_1_str}{v7_3_str}{v7_3_1_str}{v7_4_str}{v7_5_str}")
    print("=" * 60)


def training(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    hyper: ModelHiddenParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    coarse_iter,
    load_model_path=None,
    start_iteration=None,
):
    # Apply v7 preset if enabled (must be done before setting up GaussianModel)
    apply_v7_preset(opt, hyper)
    
    # Apply PhysX-Gaussian preset if enabled
    apply_physx_preset(opt, hyper)
    
    # Set up dataset
    scene = Scene(dataset, shuffle=False)

    # Set up some parameters
    scanner_cfg = scene.scanner_cfg
    volume_to_world = max(scanner_cfg["sVoxel"])
   
    scale_bound = None
    if dataset.scale_min > 0 and dataset.scale_max > 0:
        scale_bound = np.array([dataset.scale_min, dataset.scale_max]) * volume_to_world

    # Set up Gaussians
    gaussians = GaussianModel(scale_bound, hyper)
    
    # Determine initialization method (priority: s5 > s4_2 > standard)
    init_method_used = "standard"
    
    # s5: 4D Dynamic-Aware Multi-Phase FDK Initialization
    # s5 generates init files offline via tools/build_init_s5_4d.py
    # The init file is in standard format [N, 4] with [x, y, z, density]
    if opt.use_s5_4d_init:
        s5_init_path = opt.s5_init_path
        if s5_init_path and os.path.exists(s5_init_path):
            print("=" * 60)
            print("S5: 4D DYNAMIC-AWARE INITIALIZATION")
            print("=" * 60)
            print(f"  Using s5 init file: {s5_init_path}")
            # Override ply_path to use s5 init file
            dataset.ply_path = s5_init_path
            initialize_gaussian(gaussians, dataset, None)
            init_method_used = "s5"
            print("=" * 60)
        else:
            print(f"[WARNING] s5 enabled but init file not found: {s5_init_path}")
            print(f"  Please generate it first with:")
            print(f"    python tools/build_init_s5_4d.py --input <pickle> --output <s5_init.npy>")
            print(f"  Falling back to standard initialization.")
    
    # s4_2: Use average CT for Gaussians initialization if enabled
    if init_method_used == "standard" and opt.use_s4_avg_ct_init and opt.s4_avg_ct_path and os.path.exists(opt.s4_avg_ct_path):
        print(f"[s4_2] Using average CT for Gaussians initialization")
        positions, densities = initialize_gaussians_from_avg_ct(
            avg_ct_path=opt.s4_avg_ct_path,
            scanner_cfg=scanner_cfg,
            bbox=scene.bbox,
            num_gaussians=opt.s4_avg_ct_init_num_gaussians,
            density_thresh=opt.s4_avg_ct_init_thresh,
            density_rescale=opt.s4_avg_ct_init_density_rescale,
            method=opt.s4_avg_ct_init_method,
            save_path=None
        )
        # Initialize from custom positions and densities
        initialize_gaussian(gaussians, dataset, None, 
                           custom_positions=positions, 
                           custom_densities=densities)
        init_method_used = "s4_2"
    
    # Continue training from saved model (if load_model_path is provided)
    if load_model_path is not None and os.path.exists(load_model_path):
        print("=" * 60)
        print("CONTINUE TRAINING FROM SAVED MODEL")
        print("=" * 60)
        print(f"  Loading from: {load_model_path}")
        if start_iteration is None:
            raise ValueError("--start_iteration is required when using --load_model_path")
        print(f"  Starting from iteration: {start_iteration}")
        
        # Load model from saved path
        gaussians.load_from_model_path(load_model_path, opt)
        init_method_used = "continue"
        
        # V7.4: Load base canonical params if they exist
        # (saved during original training after coarse stage)
        use_v7_4 = getattr(opt, 'use_v7_4_canonical_decode', False)
        if use_v7_4:
            # Re-save base params from current state (they should be similar)
            gaussians.save_base_canonical_params()
        
        print("=" * 60)
    elif init_method_used == "standard":
        # Standard initialization (if no special init method used)
        initialize_gaussian(gaussians, dataset, None)
    
    scene.gaussians = gaussians

    # Skip coarse stage if continuing from a later iteration
    skip_coarse = (load_model_path is not None and start_iteration is not None and start_iteration >= coarse_iter)
    
    if not skip_coarse:
        scene_reconstruction(
            dataset,
            opt,
            pipe,
            hyper,
            tb_writer,
            testing_iterations,
            saving_iterations,
            checkpoint_iterations,
            checkpoint,
            coarse_iter,
            gaussians,
            scene,
            'coarse',
        )

        # V7.4: Save base canonical parameters after warm-up (coarse stage)
        # These serve as the "anchor" for canonical decode losses
        use_v7_4 = getattr(opt, 'use_v7_4_canonical_decode', False)
        if use_v7_4:
            gaussians.save_base_canonical_params()
    else:
        print(f"[Continue Training] Skipping coarse stage (start_iteration={start_iteration} >= coarse_iter={coarse_iter})")

    # Fine stage - pass start_iteration for continue training
    scene_reconstruction(
        dataset,
        opt,
        pipe,
        hyper,
        tb_writer,
        testing_iterations,
        saving_iterations,
        checkpoint_iterations,
        checkpoint,
        coarse_iter,
        gaussians,
        scene,
        'fine',
        start_iteration if skip_coarse else None,  # Pass start_iteration only if skipping coarse
    )


def scene_reconstruction(
    dataset: ModelParams,
    opt: OptimizationParams,
    pipe: PipelineParams,
    hyper: ModelHiddenParams,
    tb_writer,
    testing_iterations,
    saving_iterations,
    checkpoint_iterations,
    checkpoint,
    coarse_iter,
    gaussians,
    scene,
    stage,
    continue_from_iteration=None,  # For continue training
):
    
    scanner_cfg = scene.scanner_cfg
    bbox = scene.bbox
    volume_to_world = max(scanner_cfg["sVoxel"])
    max_scale = opt.max_scale * volume_to_world if opt.max_scale else None
    densify_scale_threshold = (
        opt.densify_scale_threshold * volume_to_world
        if opt.densify_scale_threshold
        else None
    )
    queryfunc = lambda x, y, z: query(
        x,
        scanner_cfg["offOrigin"],
        scanner_cfg["nVoxel"],
        scanner_cfg["sVoxel"],
        pipe,
        y,
        z,
    )
    
    # s1: Initialize static reweighting for coarse (static warm-up) stage
    static_reweight_manager = None  # For residual method (方案B)
    learnable_static_weights = None  # For learnable method (方案A)
    
    if stage == 'coarse' and opt.use_static_reweighting:
        num_train_projections = len(scene.getTrainCameras())
        apply_s1_preset(opt)
        
        if opt.static_reweight_method == "residual":
            # 方案B: Residual-based reweighting (EMA)
            static_reweight_manager = StaticReweightingManager(
                num_projections=num_train_projections,
                burnin_steps=opt.static_reweight_burnin_steps,
                ema_beta=opt.static_reweight_ema_beta,
                tau=opt.static_reweight_tau,
                weight_type=opt.static_reweight_weight_type,
                device="cuda"
            )
            print(f"  -> Initialized StaticReweightingManager (residual) for {num_train_projections} projections")
            
        elif opt.static_reweight_method == "learnable":
            # 方案A: Learnable per-projection weights
            learnable_static_weights = LearnableStaticReweighting(
                num_projections=num_train_projections,
                target_mean=opt.static_reweight_target_mean,
                burnin_steps=opt.static_reweight_burnin_steps,
                device="cuda"
            )
            print(f"  -> Initialized LearnableStaticReweighting (learnable) for {num_train_projections} projections")
            print(f"     - α_j initialized to {learnable_static_weights.alpha[0].item():.4f} (w_j ≈ {opt.static_reweight_target_mean})")

    # s2: Initialize PhaseGatedStatic for coarse (static warm-up) stage
    phase_gated_static = None
    
    if stage == 'coarse' and opt.use_phase_gated_static:
        apply_s2_preset(opt)
        
        phase_gated_static = PhaseGatedStatic(
            sigma_target=opt.static_phase_sigma_target,
            burnin_steps=opt.static_phase_burnin_steps,
            init_period=1.0,  # Will be learned
            device="cuda"
        )
        print(f"  -> Initialized PhaseGatedStatic")
        print(f"     - σ_target = {opt.static_phase_sigma_target:.3f} rad")
        print(f"     - burn-in steps = {opt.static_phase_burnin_steps}")

    # s3: Initialize DualStaticVolume for coarse (static warm-up) stage
    dual_static_volume = None
    
    if stage == 'coarse' and opt.use_s3_dual_static_volume:
        apply_s3_preset(opt)
        
        dual_static_volume = DualStaticVolume(
            resolution=opt.s3_voxel_resolution,
            bbox=bbox,
            num_ray_samples=opt.s3_num_ray_samples,
            init_value=0.0,
            device="cuda"
        )
        print(f"  -> Initialized DualStaticVolume")
        print(f"     - resolution = {opt.s3_voxel_resolution}³")
        print(f"     - bbox = [{bbox[0].tolist()}, {bbox[1].tolist()}]")
        print(f"     - ray samples = {opt.s3_num_ray_samples}")

    # s4: Initialize TemporalEnsembleStatic for coarse (static warm-up) stage
    temporal_ensemble_static = None
    
    if stage == 'coarse' and opt.use_s4_temporal_ensemble_static:
        apply_s4_preset(opt)
        
        temporal_ensemble_static = TemporalEnsembleStatic(
            avg_ct_path=opt.s4_avg_ct_path,
            avg_proj_path=opt.s4_avg_proj_path,
            bbox=bbox,
            device="cuda"
        )
        print(f"  -> Initialized TemporalEnsembleStatic")
        print(f"     - avg_ct_loaded = {temporal_ensemble_static.avg_ct_loaded}")
        print(f"     - avg_proj_loaded = {temporal_ensemble_static.avg_proj_loaded}")
        print(f"     - bbox = [{bbox[0].tolist()}, {bbox[1].tolist()}]")

    gaussians.training_setup(opt)
    
    # s1 方案A: Create separate optimizer for learnable weights
    # NOTE: We use a separate optimizer to avoid conflicts with Gaussian densification/pruning
    # The Gaussian optimizer prunes parameters based on Gaussian count, but α_j has fixed size [N_projections]
    learnable_weights_optimizer = None
    if learnable_static_weights is not None:
        learnable_weights_optimizer = torch.optim.Adam(
            learnable_static_weights.parameters(),
            lr=opt.static_reweight_lr
        )
        print(f"  -> Created separate optimizer for learnable weights (α_j) with lr={opt.static_reweight_lr}")
    
    # s2: Create separate optimizer for phase-gated static parameters
    # NOTE: We use a separate optimizer to avoid conflicts with Gaussian densification/pruning
    phase_gated_optimizer = None
    if phase_gated_static is not None:
        phase_gated_optimizer = torch.optim.Adam(
            phase_gated_static.parameters(),
            lr=opt.static_phase_lr
        )
        print(f"  -> Created separate optimizer for phase-gated params (τ, ψ, φ_c, ξ) with lr={opt.static_phase_lr}")
    
    # s3: Create separate optimizer for dual static volume
    # NOTE: We use a separate optimizer to avoid conflicts with Gaussian densification/pruning
    dual_volume_optimizer = None
    if dual_static_volume is not None:
        dual_volume_optimizer = torch.optim.Adam(
            dual_static_volume.parameters(),
            lr=opt.s3_voxel_lr
        )
        print(f"  -> Created separate optimizer for voxel volume V with lr={opt.s3_voxel_lr}")
    
    # Checkpoint loading: restore model state and iteration
    checkpoint_first_iter = None
    if checkpoint is not None:
        (model_params, checkpoint_first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        print(f"Load checkpoint {osp.basename(checkpoint)} at iteration {checkpoint_first_iter}.")

    # Set up loss
    use_tv = opt.lambda_tv > 0
    if use_tv:
        print("Use total variation loss")
        tv_vol_size = opt.tv_vol_size
        tv_vol_nVoxel = torch.tensor([tv_vol_size, tv_vol_size, tv_vol_size])
        tv_vol_sVoxel = torch.tensor(scanner_cfg["dVoxel"]) * tv_vol_nVoxel

    if stage == 'coarse':
        train_iterations = coarse_iter
        # If checkpoint was loaded, resume from checkpoint iteration
        if checkpoint_first_iter is not None:
            first_iter = checkpoint_first_iter
            print(f"Resuming coarse stage from iteration {first_iter}")
        else:
            first_iter = 0
    else:
        train_iterations = opt.iterations
        # Priority: continue_from_iteration > checkpoint_first_iter > coarse_iter
        if continue_from_iteration is not None:
            first_iter = continue_from_iteration
            print(f"[Continue Training] Resuming fine stage from iteration {first_iter}")
        elif checkpoint_first_iter is not None:
            first_iter = checkpoint_first_iter
            print(f"Resuming fine stage from iteration {first_iter}")
        else:
            first_iter = coarse_iter

    # Train
    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)
    ckpt_save_path = osp.join(scene.model_path, "ckpt")
    os.makedirs(ckpt_save_path, exist_ok=True)
    viewpoint_stack = None
    progress_bar = tqdm(range(0, train_iterations), desc="Train", leave=False)
    progress_bar.update(first_iter)
    first_iter += 1

    for iteration in range(first_iter, train_iterations + 1):
        iter_start.record()

        # ================================================================
        # V11: Pretrain-Finetune Stage Control
        # ================================================================
        use_pretrain_finetune = getattr(hyper, 'use_pretrain_finetune', False)
        pretrain_steps = getattr(hyper, 'pretrain_steps', 3000)
        finetune_anchor_lr_scale = getattr(hyper, 'finetune_anchor_lr_scale', 0.1)
        
        in_pretrain_stage = False
        if use_pretrain_finetune and stage == 'fine':
            fine_start = coarse_iter + 1 if 'coarse_iter' in dir() else 1
            steps_in_fine = iteration - fine_start
            in_pretrain_stage = steps_in_fine < pretrain_steps
            
            # Set flag in anchor module
            if gaussians.use_anchor_deformation and gaussians._deformation_anchor is not None:
                gaussians._deformation_anchor._in_pretrain_stage = in_pretrain_stage
                
                # Switch LR when transitioning from pretrain to finetune
                if steps_in_fine == pretrain_steps:
                    print(f"\n[V11] Pretrain complete. Switching to finetune stage.")
                    print(f"[V11] Scaling anchor LR by {finetune_anchor_lr_scale}")

        # Update learning rate
        gaussians.update_learning_rate(iteration)

        # Get one camera for training
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack) - 1))

        # Render X-ray projection
        # V7.1-trainable-α: Use correction during training if enabled
        use_train_correction = opt.use_trainable_alpha and stage == 'fine'
        
        # Compute iteration_ratio for PhysX-Gaussian mask decay scheduler
        # iteration_ratio = 0.0 at start of fine stage, 1.0 at end
        if stage == 'fine':
            # Fine stage iterations: from (coarse_iter + 1) to train_iterations
            fine_start = coarse_iter + 1
            fine_total = train_iterations - coarse_iter
            iteration_ratio = (iteration - fine_start) / max(fine_total - 1, 1)
            iteration_ratio = max(0.0, min(1.0, iteration_ratio))
        else:
            iteration_ratio = 0.0
        
        # V16 Fix 2: If st_coupled_render=True, compute L_lagbert BEFORE rendering
        # so that forward_anchors() uses the cached dx_center from compute_lagbert_loss()
        # This ensures rendering and L_lagbert share the same forward pass.
        _v16_lagbert_cached = None
        if stage == 'fine' and gaussians.is_st_coupled_render():
            time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device)
            _, _v16_lagbert_cached = gaussians.compute_lagbert_loss(time_tensor, is_training=True)
        
        if use_train_correction:
            # Get current time for this view
            current_time = viewpoint_cam.time
            # Get trainable alpha (scalar or time-dependent)
            train_alpha = gaussians.get_alpha(current_time)
            if train_alpha is None:
                train_alpha = 0.0
            render_pkg = render(
                viewpoint_cam, gaussians, pipe, stage,
                use_v7_1_correction=True,
                correction_alpha=train_alpha if isinstance(train_alpha, float) else train_alpha.item() if not train_alpha.requires_grad else train_alpha,
                iteration_ratio=iteration_ratio
            )
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, stage, iteration_ratio=iteration_ratio)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Compute loss
        gt_image = viewpoint_cam.original_image.cuda()
        loss = {"total": 0.0}
        render_loss = l1_loss(image, gt_image)
        loss["render"] = render_loss
        
        # Compute combined render loss for s1 reweighting (L1 + D-SSIM)
        combined_render_loss = render_loss.clone()
        if opt.lambda_dssim > 0:
            loss_dssim = 1.0 - ssim(image, gt_image)
            loss["dssim"] = loss_dssim
            combined_render_loss = combined_render_loss + opt.lambda_dssim * loss_dssim
        
        # s1: Apply projection reliability-weighted reweighting in static warm-up stage
        s1_weight = 1.0
        s1_L_mean = 0.0
        
        if stage == 'coarse' and opt.use_static_reweighting:
            proj_idx = viewpoint_cam.uid
            
            if opt.static_reweight_method == "residual" and static_reweight_manager is not None:
                # 方案B: Residual-based reweighting (non-differentiable weight)
                s1_weight = static_reweight_manager.get_weight_and_update(
                    proj_idx=proj_idx,
                    loss_value=combined_render_loss.detach().item(),
                    iteration=iteration
                )
                loss["s1_weight"] = s1_weight
                
                # Apply weight to render loss and dssim loss
                loss["total"] += s1_weight * loss["render"]
                if opt.lambda_dssim > 0:
                    loss["total"] = loss["total"] + s1_weight * opt.lambda_dssim * loss_dssim
                    
            elif opt.static_reweight_method == "learnable" and learnable_static_weights is not None:
                # 方案A: Learnable per-projection weights (differentiable)
                # Get learnable weight w_j = sigmoid(α_j)
                w_j = learnable_static_weights.get_weight(proj_idx, iteration)
                s1_weight = w_j.item() if isinstance(w_j, torch.Tensor) else w_j
                loss["s1_weight"] = s1_weight
                
                # Apply weight to render loss: w_j * L_j
                # Note: w_j is differentiable, so gradients flow to α_j
                weighted_render_loss = w_j * loss["render"]
                loss["total"] += weighted_render_loss
                
                if opt.lambda_dssim > 0:
                    weighted_dssim_loss = w_j * opt.lambda_dssim * loss_dssim
                    loss["total"] = loss["total"] + weighted_dssim_loss
                
                # Compute L_mean regularization (only after burn-in)
                if iteration >= opt.static_reweight_burnin_steps:
                    L_mean = learnable_static_weights.compute_mean_regularization_loss()
                    s1_L_mean = L_mean.item()
                    loss["s1_L_mean"] = s1_L_mean
                    loss["total"] = loss["total"] + opt.lambda_static_reweight_mean * L_mean
            else:
                # Fallback: no s1 reweighting
                loss["total"] += loss["render"]
                if opt.lambda_dssim > 0:
                    loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
        else:
            # Original behavior (no s1 reweighting or not in coarse stage)
            # V11: Skip render loss during pretrain stage (only use L_phys)
            if not in_pretrain_stage:
                loss["total"] += loss["render"]
                if opt.lambda_dssim > 0:
                    loss["total"] = loss["total"] + opt.lambda_dssim * loss_dssim
            else:
                # V11 Pretrain: Only L_phys will be added later
                loss["v11_pretrain"] = True
        
        # s2: Apply phase-gated weighting in static warm-up stage
        # s2 can be combined with s1 - it multiplies an additional phase-based weight
        s2_weight = 1.0
        s2_L_win = 0.0
        
        if stage == 'coarse' and opt.use_phase_gated_static and phase_gated_static is not None:
            # Get acquisition time t_j from camera
            t_j = viewpoint_cam.time
            
            # Compute phase-gated weight w_j
            w_j_s2 = phase_gated_static.compute_weight(t_j, iteration)
            s2_weight = w_j_s2.item() if isinstance(w_j_s2, torch.Tensor) else w_j_s2
            loss["s2_weight"] = s2_weight
            
            # Apply phase-gated weighting to the current total loss
            # This multiplies the already computed loss by w_j_s2
            # Note: If s1 is also enabled, the loss is already s1-weighted, and s2 adds another layer
            if iteration >= opt.static_phase_burnin_steps:
                # After burn-in: apply phase-gated weight
                # Scale the render-related losses (already in loss["total"]) by w_j_s2
                # We do this by scaling the entire current total loss contribution
                # Recompute: instead of modifying existing total, we track s2 contribution separately
                
                # Get the render-related loss that was just added to total
                render_related_loss = loss["render"].clone()
                if opt.lambda_dssim > 0:
                    render_related_loss = render_related_loss + opt.lambda_dssim * loss["dssim"]
                
                # If s1 is enabled, it already weighted the loss, so we apply s2 on top
                if opt.use_static_reweighting:
                    # s1 already applied its weight, now multiply by (w_j_s2 / 1.0)
                    # This is equivalent to scaling by w_j_s2
                    # We need to adjust: subtract old contribution, add s2-weighted contribution
                    # Old contribution = s1_weight * render_related_loss (already added)
                    # New contribution = s1_weight * w_j_s2 * render_related_loss
                    # Adjustment = (w_j_s2 - 1) * s1_weight * render_related_loss
                    adjustment = (w_j_s2 - 1.0) * s1_weight * render_related_loss
                    loss["total"] = loss["total"] + adjustment
                else:
                    # No s1, just apply s2 weight
                    # Old contribution = render_related_loss (already added)
                    # New contribution = w_j_s2 * render_related_loss
                    # Adjustment = (w_j_s2 - 1) * render_related_loss
                    adjustment = (w_j_s2 - 1.0) * render_related_loss
                    loss["total"] = loss["total"] + adjustment
                
                # Add window regularization loss L_win = (log(σ_φ) - log(σ_target))²
                L_win = phase_gated_static.compute_window_regularization_loss()
                s2_L_win = L_win.item()
                loss["s2_L_win"] = s2_L_win
                loss["total"] = loss["total"] + opt.lambda_static_phase_window * L_win

        # s3: Dual-representation static warm-up (Gaussian + Voxel Co-Training)
        # Only active in coarse stage when enabled
        s3_L_V = 0.0
        s3_L_distill = 0.0
        s3_L_VTV = 0.0
        
        if stage == 'coarse' and opt.use_s3_dual_static_volume and dual_static_volume is not None:
            # 1. Scale Gaussian projection loss by λ_G (already computed as render_loss + dssim)
            # The Gaussian loss is already in loss["total"], we just need to scale it
            # For simplicity, we apply λ_G by scaling the existing render contribution
            # Note: If λ_G != 1.0, we need to adjust the contribution
            if opt.lambda_s3_G != 1.0:
                # Current loss["total"] already includes render_loss (with s1/s2 weights if applied)
                # We need to scale the render-related portion by λ_G
                # Get the render-related loss
                render_related = loss["render"].clone()
                if opt.lambda_dssim > 0:
                    render_related = render_related + opt.lambda_dssim * loss["dssim"]
                
                # Calculate adjustment factor
                # Current contribution: 1.0 * render_related (already in total)
                # Desired contribution: λ_G * render_related
                # Adjustment: (λ_G - 1.0) * render_related
                # But we also need to account for s1/s2 weights if they were applied
                effective_weight = 1.0
                if opt.use_static_reweighting:
                    effective_weight *= s1_weight
                if opt.use_phase_gated_static:
                    effective_weight *= s2_weight
                
                adjustment = (opt.lambda_s3_G - 1.0) * effective_weight * render_related
                loss["total"] = loss["total"] + adjustment
            
            # 2. Render projection from Voxel volume V and compute L_V
            image_V = dual_static_volume.render_projection(
                camera=viewpoint_cam,
                image_height=viewpoint_cam.image_height,
                image_width=viewpoint_cam.image_width,
                downsample_factor=opt.s3_downsample_factor
            )
            
            # Downsample ground truth if needed
            if opt.s3_downsample_factor > 1:
                gt_image_ds = F.interpolate(
                    gt_image.unsqueeze(0),
                    scale_factor=1.0 / opt.s3_downsample_factor,
                    mode='bilinear',
                    align_corners=False
                ).squeeze(0)
            else:
                gt_image_ds = gt_image
            
            # Compute L_V (voxel projection loss)
            L_V_render = l1_loss(image_V, gt_image_ds)
            loss["s3_L_V_render"] = L_V_render
            s3_L_V = L_V_render.item()
            
            if opt.lambda_dssim > 0:
                L_V_dssim = 1.0 - ssim(image_V, gt_image_ds)
                loss["s3_L_V_dssim"] = L_V_dssim
                L_V_total = L_V_render + opt.lambda_dssim * L_V_dssim
            else:
                L_V_total = L_V_render
            
            loss["s3_L_V"] = L_V_total
            loss["total"] = loss["total"] + opt.lambda_s3_V * L_V_total
            
            # 3. Compute distillation loss L_distill = mean|σ_G(x) - V(x)|
            L_distill = dual_static_volume.compute_distillation_loss(
                gaussians=gaussians,
                num_samples=opt.s3_num_distill_samples,
                bbox=bbox
            )
            s3_L_distill = L_distill.item()
            loss["s3_L_distill"] = L_distill
            loss["total"] = loss["total"] + opt.lambda_s3_distill * L_distill
            
            # 4. Compute 3D TV loss on V: L_VTV
            L_VTV = dual_static_volume.compute_tv_loss()
            s3_L_VTV = L_VTV.item()
            loss["s3_L_VTV"] = L_VTV
            loss["total"] = loss["total"] + opt.lambda_s3_VTV * L_VTV
            
            # Update statistics
            dual_static_volume._last_L_V = s3_L_V
            dual_static_volume._last_L_distill = s3_L_distill
            dual_static_volume._last_L_VTV = s3_L_VTV

        # s4: Temporal-ensemble guided static warm-up
        # Only active in coarse stage when enabled
        s4_L_vol = 0.0
        s4_L_proj_avg = 0.0
        
        if stage == 'coarse' and opt.use_s4_temporal_ensemble_static and temporal_ensemble_static is not None:
            # 1. Scale Gaussian projection loss by λ_s4_G (if different from 1.0)
            if opt.lambda_s4_G != 1.0:
                render_related = loss["render"].clone()
                if opt.lambda_dssim > 0:
                    render_related = render_related + opt.lambda_dssim * loss["dssim"]
                
                # Calculate adjustment factor (accounting for s1/s2 weights if applied)
                effective_weight = 1.0
                if opt.use_static_reweighting:
                    effective_weight *= s1_weight
                if opt.use_phase_gated_static:
                    effective_weight *= s2_weight
                
                adjustment = (opt.lambda_s4_G - 1.0) * effective_weight * render_related
                loss["total"] = loss["total"] + adjustment
            
            # 2. Compute volume distillation loss L_vol = |σ_G(x) - V_avg(x)|
            if temporal_ensemble_static.avg_ct_loaded and opt.lambda_s4_vol > 0:
                L_vol = temporal_ensemble_static.compute_volume_distillation_loss(
                    gaussians=gaussians,
                    num_samples=opt.s4_num_vol_samples,
                    bbox=bbox
                )
                s4_L_vol = L_vol.item()
                loss["s4_L_vol"] = L_vol
                loss["total"] = loss["total"] + opt.lambda_s4_vol * L_vol
            
            # 3. (Optional) Compute projection distillation loss L_proj_avg
            # This requires matching view indices and average projection data
            if temporal_ensemble_static.avg_proj_loaded and opt.lambda_s4_proj_avg > 0:
                avg_proj = temporal_ensemble_static.get_avg_projection(viewpoint_cam.uid)
                if avg_proj is not None:
                    # Resize avg_proj to match rendered image size if needed
                    if avg_proj.shape[-2:] != image.shape[-2:]:
                        avg_proj = F.interpolate(
                            avg_proj.unsqueeze(0),
                            size=image.shape[-2:],
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)
                    
                    L_proj_avg = l1_loss(image, avg_proj)
                    s4_L_proj_avg = L_proj_avg.item()
                    loss["s4_L_proj_avg"] = L_proj_avg
                    loss["total"] = loss["total"] + opt.lambda_s4_proj_avg * L_proj_avg
            
            # Update statistics
            temporal_ensemble_static._last_L_vol = s4_L_vol
            temporal_ensemble_static._last_L_proj_avg = s4_L_proj_avg

        # Prior loss (L_pc) - Original X²-Gaussian SSRML core loss
        # Renders I(t + n*T̂) and compares with I_gt to learn optimal breathing period T̂
        # NOTE: For PhysX-Boosted, we enable this loss since it has full HexPlane baseline
        # The period parameter gradient flows through new_time = t + n*exp(period)
        use_anchor_deform_prior = getattr(hyper, 'use_anchor_deformation', False) and gaussians.use_anchor_deformation
        use_boosted_prior = getattr(hyper, 'use_boosted', False)
        # Skip only for pure PhysX-Gaussian (no HexPlane), enable for baseline and PhysX-Boosted
        prior_anchor_skip = use_anchor_deform_prior and not use_boosted_prior
        # Skip if lambda_prior=0 to avoid wasted computation
        if stage=='fine' and iteration > 7000 and not prior_anchor_skip and opt.lambda_prior > 0:
            render_pkg_prior = render_prior_oneT(viewpoint_cam, gaussians, pipe, stage)
            image_prior = render_pkg_prior["render"]    
            render_loss_prior = l1_loss(image_prior, gt_image)
            loss["render_prior"] = render_loss_prior
            loss["total"] += opt.lambda_prior * loss["render_prior"]
            if opt.lambda_dssim > 0:
                loss_dssim_prior = 1.0 - ssim(image_prior, gt_image)
                loss["dssim_prior"] = loss_dssim_prior
                loss["total"] = loss["total"] + opt.lambda_prior * opt.lambda_dssim * loss_dssim_prior

        # 3D TV loss
        # NOTE: For PhysX-Boosted, we enable this loss with torch.no_grad() to avoid graph conflicts
        # The query() function creates a second forward pass, but we only need the volume for TV regularization
        use_anchor_mode_tv = getattr(hyper, 'use_anchor_deformation', False) and gaussians.use_anchor_deformation
        use_boosted_tv = getattr(hyper, 'use_boosted', False)
        # Skip only for pure PhysX-Gaussian, enable for baseline and PhysX-Boosted
        skip_tv = use_anchor_mode_tv and not use_boosted_tv
        if use_tv and not skip_tv:
            # Randomly get the tiny volume center
            tv_vol_center = (bbox[0] + tv_vol_sVoxel / 2) + (
                bbox[1] - tv_vol_sVoxel - bbox[0]
            ) * torch.rand(3)
            vol_pred = query(
                gaussians,
                tv_vol_center,
                tv_vol_nVoxel,
                tv_vol_sVoxel,
                pipe,
                viewpoint_cam.time,
                stage,
            )["vol"]
            loss_tv = tv_3d_loss(vol_pred, reduction="mean")
            loss["tv"] = loss_tv
            loss["total"] = loss["total"] + opt.lambda_tv * loss_tv

        # 4D TV loss (HexPlane regularization)
        # NOTE: Skipped when using pure PhysX-Gaussian (HexPlane not used)
        # BUT: Enabled for PhysX-Boosted (full HexPlane baseline + Anchor)
        # V4: Can be explicitly disabled via --disable_4d_tv for ablation
        use_anchor_deform = getattr(hyper, 'use_anchor_deformation', False) and gaussians.use_anchor_deformation
        use_boosted = getattr(hyper, 'use_boosted', False)
        disable_4d_tv = getattr(hyper, 'disable_4d_tv', False)
        # PhysX-Boosted: Use HexPlane from anchor module (unless explicitly disabled)
        skip_4d_tv = (use_anchor_deform and not use_boosted) or disable_4d_tv
        if hyper.time_smoothness_weight != 0 and stage=='fine' and not skip_4d_tv:
            tv_loss_4d = gaussians.compute_regulation(hyper.time_smoothness_weight, hyper.l1_time_planes, hyper.plane_tv_weight)
            loss["4d_tv"] = tv_loss_4d
            loss["total"] = loss["total"] + tv_loss_4d

        # Inverse consistency loss and symmetry loss (only in fine stage when enabled)
        # NOTE: When V7.2 is enabled, L_inv is DISABLED because V7.2 uses D_b in the
        # main rendering path instead of as a regularization tool
        # NOTE: When pure PhysX-Gaussian (use_anchor_deformation without boosted) is enabled,
        # HexPlane-based losses are DISABLED. BUT: PhysX-Boosted enables them.
        use_v7_2 = getattr(opt, 'use_v7_2_consistency', False) and gaussians.use_v7_2_consistency
        use_anchor = getattr(hyper, 'use_anchor_deformation', False) and gaussians.use_anchor_deformation
        # PhysX-Boosted: Keep HexPlane losses enabled (we have full HexPlane baseline)
        skip_hexplane_losses = use_v7_2 or (use_anchor and not use_boosted)
        if stage == 'fine' and opt.use_inverse_consistency and opt.lambda_inv > 0 and not skip_hexplane_losses:
            time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device).repeat(gaussians.get_xyz.shape[0], 1)
            
            # Check if symmetry loss should also be computed
            # In v6 mode (use_shared_velocity_inverse=True), L_sym is FORCED OFF
            # because the shared velocity field inherently enforces inverse consistency
            use_shared_velocity_inverse = getattr(hyper, 'use_shared_velocity_inverse', False)
            compute_sym = opt.use_symmetry_reg and opt.lambda_sym > 0 and not use_shared_velocity_inverse
            
            result = gaussians.compute_inverse_consistency_loss(
                time_tensor, 
                sample_ratio=opt.inv_sample_ratio,
                compute_symmetry=compute_sym
            )
            
            if compute_sym:
                loss_inv, loss_sym = result
                loss["inv_consistency"] = loss_inv
                loss["symmetry"] = loss_sym
                loss["total"] = loss["total"] + opt.lambda_inv * loss_inv + opt.lambda_sym * loss_sym
            else:
                loss_inv = result
                loss["inv_consistency"] = loss_inv
                loss["total"] = loss["total"] + opt.lambda_inv * loss_inv

        # Cycle motion consistency loss (only in fine stage when enabled)
        # Uses the learned breathing period T_hat to enforce motion periodicity
        # NOTE: Skipped when using pure PhysX-Gaussian. Enabled for PhysX-Boosted.
        if stage == 'fine' and opt.use_cycle_motion and opt.lambda_cycle > 0 and not skip_hexplane_losses:
            time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device).repeat(gaussians.get_xyz.shape[0], 1)
            
            loss_cycle = gaussians.compute_cycle_motion_loss(
                time_tensor,
                sample_ratio=opt.inv_sample_ratio  # Reuse the same sample ratio
            )
            
            loss["cycle_motion"] = loss_cycle
            loss["total"] = loss["total"] + opt.lambda_cycle * loss_cycle

        # Jacobian regularization loss (only in fine stage when enabled)
        # Penalizes negative determinants of the Jacobian to prevent local folding
        # NOTE: Skipped when using pure PhysX-Gaussian. Enabled for PhysX-Boosted.
        if stage == 'fine' and opt.use_jacobian_reg and opt.lambda_jac > 0 and not skip_hexplane_losses:
            time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device).repeat(gaussians.get_xyz.shape[0], 1)
            
            loss_jac = gaussians.compute_jacobian_loss(
                time_tensor,
                num_samples=opt.jacobian_num_samples,
                step_size=opt.jacobian_step_size
            )
            
            loss["jacobian"] = loss_jac
            loss["total"] = loss["total"] + opt.lambda_jac * loss_jac

        # v9: Mode regularization loss (only in fine stage when using low-rank motion modes)
        # L_mode = mean_{i,m} || U[i,m,:] ||_2^2
        # Prevents motion modes from growing too large
        if stage == 'fine' and opt.use_mode_regularization and opt.lambda_mode > 0:
            if gaussians.use_low_rank_motion_modes:
                loss_mode = gaussians.compute_mode_regularization_loss()
                loss["mode_reg"] = loss_mode
                loss["total"] = loss["total"] + opt.lambda_mode * loss_mode
        
        # v10: Gate regularization loss (only in fine stage when using adaptive gating)
        # L_gate = mean(g_f * (1 - g_f) + g_b * (1 - g_b))
        # Encourages gates to be close to 0 or 1 (binary-like)
        if stage == 'fine' and opt.use_adaptive_gating and opt.lambda_gate > 0:
            if gaussians.use_low_rank_motion_modes and gaussians.use_adaptive_gating:
                loss_gate = gaussians.compute_gate_regularization_loss()
                loss["gate_reg"] = loss_gate
                loss["total"] = loss["total"] + opt.lambda_gate * loss_gate
        
        # v10: Trajectory smoothing loss (only in fine stage when enabled)
        # L_traj = mean_{i,k} || x_i(t_{k+1}) - 2*x_i(t_k) + x_i(t_{k-1}) ||^2
        # Penalizes acceleration to smooth Gaussian trajectories over time
        # NOTE: Skipped when using pure PhysX-Gaussian. Enabled for PhysX-Boosted.
        if stage == 'fine' and opt.use_trajectory_smoothing and opt.lambda_traj > 0 and not skip_hexplane_losses:
            time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device)
            loss_traj = gaussians.compute_trajectory_smoothing_loss(
                time_tensor,
                num_time_samples=opt.traj_num_time_samples,
                sample_ratio=0.1  # Sample 10% of Gaussians for efficiency
            )
            loss["traj_smooth"] = loss_traj
            loss["total"] = loss["total"] + opt.lambda_traj * loss_traj

        # V7.1-trainable-α: Alpha regularization loss (only if NOT using V7.2)
        # L_alpha_reg = lambda_alpha * (alpha - alpha_init)^2
        if stage == 'fine' and opt.use_trainable_alpha and not use_v7_2:
            loss_alpha_reg = gaussians.compute_alpha_regularization_loss()
            if loss_alpha_reg > 0:
                loss["alpha_reg"] = loss_alpha_reg
                loss["total"] = loss["total"] + loss_alpha_reg
                # Log current alpha value
                current_alpha = gaussians.get_alpha(viewpoint_cam.time)
                if current_alpha is not None:
                    if isinstance(current_alpha, torch.Tensor):
                        loss["alpha"] = current_alpha.item() if current_alpha.numel() == 1 else current_alpha.mean().item()
                    else:
                        loss["alpha"] = current_alpha

        # V7.2: End-to-End Consistency-Aware losses
        # When V7.2 is enabled:
        # - L_inv is DISABLED (handled above)
        # - L_b = |D_b(y, t)| is added as lightweight regularization on backward field
        # - L_alpha = (alpha - alpha_init)^2 keeps alpha close to initial value
        if stage == 'fine' and use_v7_2:
            time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device).repeat(gaussians.get_xyz.shape[0], 1)
            
            # L_b: Backward field magnitude regularization
            v7_2_lambda_b = getattr(opt, 'v7_2_lambda_b_reg', 1e-3)
            if v7_2_lambda_b > 0:
                loss_b = gaussians.compute_backward_magnitude_loss(time_tensor, sample_ratio=0.1)
                loss["v7_2_L_b"] = loss_b
                loss["total"] = loss["total"] + v7_2_lambda_b * loss_b
            
            # L_alpha: Alpha regularization (keep alpha close to init)
            v7_2_lambda_alpha = getattr(opt, 'v7_2_lambda_alpha_reg', 1e-3)
            if v7_2_lambda_alpha > 0 and gaussians.v7_2_alpha_learnable:
                loss_alpha_v7_2 = gaussians.compute_v7_2_alpha_regularization_loss()
                loss["v7_2_L_alpha"] = loss_alpha_v7_2
                loss["total"] = loss["total"] + v7_2_lambda_alpha * loss_alpha_v7_2
            
            # Log current V7.2 alpha value
            v7_2_alpha = gaussians.get_v7_2_alpha()
            if v7_2_alpha is not None:
                loss["v7_2_alpha"] = v7_2_alpha.item() if isinstance(v7_2_alpha, torch.Tensor) else v7_2_alpha
            
            # V7.2.1: L_cycle_canon - Canonical-space cycle consistency
            # Only enabled when use_v7_2_1_cycle_canon=True on top of V7.2
            use_v7_2_1_cycle_canon = getattr(opt, 'use_v7_2_1_cycle_canon', False)
            if use_v7_2_1_cycle_canon:
                v7_2_1_lambda_cycle_canon = getattr(opt, 'v7_2_1_lambda_cycle_canon', 1e-3)
                if v7_2_1_lambda_cycle_canon > 0:
                    loss_cycle_canon = gaussians.compute_cycle_canon_loss(time, sample_ratio=1.0)
                    loss["v7_2_1_L_cycle_canon"] = loss_cycle_canon
                    loss["total"] = loss["total"] + v7_2_1_lambda_cycle_canon * loss_cycle_canon
            
            # V7.3: Temporal Bidirectional Warp Consistency
            # Only enabled when use_v7_3_timewarp=True on top of V7.2
            use_v7_3_timewarp = getattr(opt, 'use_v7_3_timewarp', False)
            if use_v7_3_timewarp:
                v7_3_lambda_fw = getattr(opt, 'v7_3_lambda_fw', 1e-3)
                v7_3_lambda_bw = getattr(opt, 'v7_3_lambda_bw', 1e-3)
                v7_3_lambda_fw_bw = getattr(opt, 'v7_3_lambda_fw_bw', 0.0)
                v7_3_delta_fraction = getattr(opt, 'v7_3_timewarp_delta_fraction', 0.1)
                
                timewarp_losses = gaussians.compute_timewarp_loss(
                    time,
                    delta_fraction=v7_3_delta_fraction,
                    compute_fw_bw=(v7_3_lambda_fw_bw > 0),
                    sample_ratio=1.0
                )
                
                # L_fw: time-forward consistency
                if v7_3_lambda_fw > 0:
                    loss["v7_3_L_fw"] = timewarp_losses["L_fw"]
                    loss["total"] = loss["total"] + v7_3_lambda_fw * timewarp_losses["L_fw"]
                
                # L_bw: time-backward consistency
                if v7_3_lambda_bw > 0:
                    loss["v7_3_L_bw"] = timewarp_losses["L_bw"]
                    loss["total"] = loss["total"] + v7_3_lambda_bw * timewarp_losses["L_bw"]
                
                # L_fw_bw: optional round-trip closure for centers
                if v7_3_lambda_fw_bw > 0 and "L_fw_bw" in timewarp_losses:
                    loss["v7_3_L_fw_bw"] = timewarp_losses["L_fw_bw"]
                    loss["total"] = loss["total"] + v7_3_lambda_fw_bw * timewarp_losses["L_fw_bw"]
            
            # V7.3.1: Temporal regularization for covariance (scale) and density
            # Only enabled when use_v7_3_1_sigma_rho=True on top of V7.2
            use_v7_3_1_sigma_rho = getattr(opt, 'use_v7_3_1_sigma_rho', False)
            if use_v7_3_1_sigma_rho:
                sigma_rho_losses = gaussians.compute_sigma_rho_temporal_losses(
                    time, opt, sample_ratio=1.0
                )
                
                v7_3_1_lambda_tv_rho = getattr(opt, 'v7_3_1_lambda_tv_rho', 1e-4)
                v7_3_1_lambda_tv_sigma = getattr(opt, 'v7_3_1_lambda_tv_sigma', 1e-4)
                v7_3_1_lambda_cycle_rho = getattr(opt, 'v7_3_1_lambda_cycle_rho', 1e-4)
                v7_3_1_lambda_cycle_sigma = getattr(opt, 'v7_3_1_lambda_cycle_sigma', 1e-4)
                
                # L_tv_sigma: temporal smoothness for scale
                if v7_3_1_lambda_tv_sigma > 0:
                    loss["v7_3_1_L_tv_sigma"] = sigma_rho_losses["L_tv_sigma"]
                    loss["total"] = loss["total"] + v7_3_1_lambda_tv_sigma * sigma_rho_losses["L_tv_sigma"]
                
                # L_cycle_sigma: periodic consistency for scale
                if v7_3_1_lambda_cycle_sigma > 0:
                    loss["v7_3_1_L_cycle_sigma"] = sigma_rho_losses["L_cycle_sigma"]
                    loss["total"] = loss["total"] + v7_3_1_lambda_cycle_sigma * sigma_rho_losses["L_cycle_sigma"]
                
                # L_tv_rho: temporal smoothness for density (only if density is dynamic)
                if v7_3_1_lambda_tv_rho > 0:
                    loss["v7_3_1_L_tv_rho"] = sigma_rho_losses["L_tv_rho"]
                    loss["total"] = loss["total"] + v7_3_1_lambda_tv_rho * sigma_rho_losses["L_tv_rho"]
                
                # L_cycle_rho: periodic consistency for density (only if density is dynamic)
                if v7_3_1_lambda_cycle_rho > 0:
                    loss["v7_3_1_L_cycle_rho"] = sigma_rho_losses["L_cycle_rho"]
                    loss["total"] = loss["total"] + v7_3_1_lambda_cycle_rho * sigma_rho_losses["L_cycle_rho"]
            
            # V7.4: Canonical decode losses for shape (Sigma) and density
            # Only enabled when use_v7_4_canonical_decode=True on top of V7.2
            use_v7_4_canonical_decode = getattr(opt, 'use_v7_4_canonical_decode', False)
            if use_v7_4_canonical_decode:
                canon_losses = gaussians.compute_canonical_sigma_rho_losses(
                    time, opt, sample_ratio=1.0
                )
                
                v7_4_lambda_cycle_sigma = getattr(opt, 'v7_4_lambda_cycle_canon_sigma', 1e-4)
                v7_4_lambda_cycle_rho = getattr(opt, 'v7_4_lambda_cycle_canon_rho', 1e-4)
                v7_4_lambda_prior_sigma = getattr(opt, 'v7_4_lambda_prior_sigma', 1e-4)
                v7_4_lambda_prior_rho = getattr(opt, 'v7_4_lambda_prior_rho', 1e-4)
                
                # L_cycle_canon_Sigma: canonical shape periodic consistency
                if v7_4_lambda_cycle_sigma > 0:
                    loss["v7_4_L_cycle_canon_Sigma"] = canon_losses["L_cycle_canon_Sigma"]
                    loss["total"] = loss["total"] + v7_4_lambda_cycle_sigma * canon_losses["L_cycle_canon_Sigma"]
                
                # L_cycle_canon_rho: canonical density periodic consistency
                if v7_4_lambda_cycle_rho > 0:
                    loss["v7_4_L_cycle_canon_rho"] = canon_losses["L_cycle_canon_rho"]
                    loss["total"] = loss["total"] + v7_4_lambda_cycle_rho * canon_losses["L_cycle_canon_rho"]
                
                # L_prior_Sigma: canonical shape prior (close to base)
                if v7_4_lambda_prior_sigma > 0:
                    loss["v7_4_L_prior_Sigma"] = canon_losses["L_prior_Sigma"]
                    loss["total"] = loss["total"] + v7_4_lambda_prior_sigma * canon_losses["L_prior_Sigma"]
                
                # L_prior_rho: canonical density prior (close to base)
                if v7_4_lambda_prior_rho > 0:
                    loss["v7_4_L_prior_rho"] = canon_losses["L_prior_rho"]
                    loss["total"] = loss["total"] + v7_4_lambda_prior_rho * canon_losses["L_prior_rho"]
            
            # V7.5: Full time-warp consistency losses for shape and density
            # Only enabled when use_v7_5_full_timewarp=True on top of V7.2+V7.4
            use_v7_5_full_timewarp = getattr(opt, 'use_v7_5_full_timewarp', False)
            if use_v7_5_full_timewarp:
                timewarp_losses = gaussians.compute_full_timewarp_sigma_rho_losses(
                    means3D, time, opt, sample_ratio=0.1
                )
                
                v7_5_lambda_fw_sigma = getattr(opt, 'v7_5_lambda_fw_sigma', 1e-5)
                v7_5_lambda_fw_rho = getattr(opt, 'v7_5_lambda_fw_rho', 1e-5)
                v7_5_lambda_bw_sigma = getattr(opt, 'v7_5_lambda_bw_sigma', 0.0)
                v7_5_lambda_bw_rho = getattr(opt, 'v7_5_lambda_bw_rho', 0.0)
                v7_5_lambda_rt_sigma = getattr(opt, 'v7_5_lambda_rt_sigma', 0.0)
                v7_5_lambda_rt_rho = getattr(opt, 'v7_5_lambda_rt_rho', 0.0)
                
                # L_fw_Sigma: forward warp shape consistency
                if v7_5_lambda_fw_sigma > 0:
                    loss["v7_5_L_fw_Sigma"] = timewarp_losses["L_fw_Sigma"]
                    loss["total"] = loss["total"] + v7_5_lambda_fw_sigma * timewarp_losses["L_fw_Sigma"]
                
                # L_fw_rho: forward warp density consistency
                if v7_5_lambda_fw_rho > 0:
                    loss["v7_5_L_fw_rho"] = timewarp_losses["L_fw_rho"]
                    loss["total"] = loss["total"] + v7_5_lambda_fw_rho * timewarp_losses["L_fw_rho"]
                
                # L_bw_Sigma: backward warp shape consistency
                # V7.5.1: Use default weights when full-state bidirectional is enabled
                use_v7_5_1_bidirectional = getattr(opt, 'use_v7_5_1_roundtrip_full_state', False)
                v7_5_lambda_bw_sigma_eff = v7_5_lambda_bw_sigma
                v7_5_lambda_bw_rho_eff = v7_5_lambda_bw_rho
                if use_v7_5_1_bidirectional:
                    if v7_5_lambda_bw_sigma == 0:
                        v7_5_lambda_bw_sigma_eff = 1e-5  # Default weight for V7.5.1
                    if v7_5_lambda_bw_rho == 0:
                        v7_5_lambda_bw_rho_eff = 1e-5  # Default weight for V7.5.1
                
                if v7_5_lambda_bw_sigma_eff > 0:
                    loss["v7_5_L_bw_Sigma"] = timewarp_losses["L_bw_Sigma"]
                    loss["total"] = loss["total"] + v7_5_lambda_bw_sigma_eff * timewarp_losses["L_bw_Sigma"]
                
                # L_bw_rho: backward warp density consistency
                if v7_5_lambda_bw_rho_eff > 0:
                    loss["v7_5_L_bw_rho"] = timewarp_losses["L_bw_rho"]
                    loss["total"] = loss["total"] + v7_5_lambda_bw_rho_eff * timewarp_losses["L_bw_rho"]
                
                # L_rt_Sigma: round-trip shape consistency (optional, not used in V7.5.1)
                if v7_5_lambda_rt_sigma > 0:
                    loss["v7_5_L_rt_Sigma"] = timewarp_losses["L_rt_Sigma"]
                    loss["total"] = loss["total"] + v7_5_lambda_rt_sigma * timewarp_losses["L_rt_Sigma"]
                
                # L_rt_rho: round-trip density consistency (optional, not used in V7.5.1)
                if v7_5_lambda_rt_rho > 0:
                    loss["v7_5_L_rt_rho"] = timewarp_losses["L_rt_rho"]
                    loss["total"] = loss["total"] + v7_5_lambda_rt_rho * timewarp_losses["L_rt_rho"]

        # ====================================================================
        # PhysX-Gaussian / PhysX-Hybrid / PhysX-Taylor: Anchor-based deformation losses
        # ====================================================================
        # L_phys (Physics Completion) and L_anchor_smooth use RE-FORWARD strategy:
        # Do NOT reuse tensors from render pass, compute fresh forward for these losses
        use_anchor = getattr(hyper, 'use_anchor_deformation', False)
        use_hybrid = getattr(hyper, 'use_hybrid', False)
        use_taylor = getattr(hyper, 'use_taylor', False)
        
        if stage == 'fine' and use_anchor and gaussians.use_anchor_deformation:
            lambda_phys = getattr(hyper, 'lambda_phys', 0.1)
            lambda_anchor_smooth = getattr(hyper, 'lambda_anchor_smooth', 0.01)
            phys_warmup_steps = getattr(hyper, 'phys_warmup_steps', 2000)
            
            # PhysX-Hybrid: Residual warmup control
            # "Draw the skeleton first, then add texture" - freeze residual initially
            if use_hybrid:
                lambda_residual = getattr(hyper, 'lambda_residual', 0.05)
                residual_warmup_steps = getattr(hyper, 'residual_warmup_steps', 3000)
                fine_start = coarse_iter + 1
                steps_in_fine = iteration - fine_start
                
                # During warmup: lr=0 for residual (anchor learns first)
                # After warmup: restore lr and add L1 regularization
                if steps_in_fine < residual_warmup_steps:
                    gaussians.set_residual_learning_rate(0.0)
                else:
                    # Restore learning rate (use grid_lr as base)
                    gaussians.set_residual_learning_rate(
                        opt.grid_lr_init * gaussians.spatial_lr_scale
                    )
                    
                    # L_residual: L1 regularization to keep residual sparse
                    # "Only use residual when anchor truly can't explain the motion"
                    if lambda_residual > 0:
                        L_residual = gaussians.get_residual_magnitude()
                        loss["residual_reg"] = L_residual
                        loss["total"] = loss["total"] + lambda_residual * L_residual
            
            # PhysX-Taylor: Affine matrix L1 regularization
            # Forces deformation gradient A to stay sparse (most regions = rigid translation)
            # Only complex affine at sharp boundaries (blood vessel edges)
            if use_taylor:
                lambda_taylor = getattr(hyper, 'lambda_taylor', 0.01)
                if lambda_taylor > 0:
                    L_taylor = gaussians.get_affine_magnitude()
                    loss["taylor_reg"] = L_taylor
                    loss["total"] = loss["total"] + lambda_taylor * L_taylor
            
            # Physics completion loss (only after warmup)
            # Uses RE-FORWARD: compute_physics_completion_loss does its own forward pass
            # V10: Also computes masked forward separately for L_phys
            # V11: During pretrain stage, L_phys is the ONLY loss (no warmup, higher weight)
            should_compute_phys = (iteration >= phys_warmup_steps and lambda_phys > 0) or in_pretrain_stage
            if should_compute_phys:
                time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device)
                iteration_ratio = iteration / opt.iterations if opt.iterations > 0 else 0.0
                L_phys = gaussians.compute_physics_completion_loss(time_tensor, iteration_ratio)
                loss["phys_completion"] = L_phys
                
                # V11: Use higher weight during pretrain (it's the only loss)
                phys_weight = 1.0 if in_pretrain_stage else lambda_phys
                loss["total"] = loss["total"] + phys_weight * L_phys
            
            # Anchor smoothness regularization
            # Uses cached _last_anchor_displacements from render pass (same graph, safe)
            if lambda_anchor_smooth > 0:
                L_anchor_smooth = gaussians.compute_anchor_smoothness_loss()
                loss["anchor_smooth"] = L_anchor_smooth
                loss["total"] = loss["total"] + lambda_anchor_smooth * L_anchor_smooth
            
            # V13: Consistency regularization (mask as data augmentation)
            use_consistency_mask = getattr(hyper, 'use_consistency_mask', False)
            lambda_consist = getattr(hyper, 'lambda_consist', 0.1)
            if use_consistency_mask and lambda_consist > 0:
                time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device)
                L_consist = gaussians.compute_consistency_loss(time_tensor)
                loss["consistency"] = L_consist
                loss["total"] = loss["total"] + lambda_consist * L_consist
            
            # V14: Temporal smoothness (acceleration penalty)
            use_temporal_interp = getattr(hyper, 'use_temporal_interp', False)
            lambda_interp = getattr(hyper, 'lambda_interp', 0.1)
            if use_temporal_interp and lambda_interp > 0:
                time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device)
                L_temporal = gaussians.compute_temporal_smoothness_loss(time_tensor)
                loss["temporal_smooth"] = L_temporal
                loss["total"] = loss["total"] + lambda_interp * L_temporal
            
            # V16: Lagrangian Spatio-Temporal Masked Anchor Modeling
            # This is a MAJOR training objective, not a tiny regularizer
            use_spatiotemporal_mask = getattr(hyper, 'use_spatiotemporal_mask', False)
            lambda_lagbert = getattr(hyper, 'lambda_lagbert', 0.5)
            if use_spatiotemporal_mask and lambda_lagbert > 0:
                # V16 Fix 2: If st_coupled_render=True, use cached L_lagbert from before rendering
                # Otherwise compute it here (separate forward pass)
                if _v16_lagbert_cached is not None:
                    L_lagbert = _v16_lagbert_cached
                else:
                    time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device)
                    _, L_lagbert = gaussians.compute_lagbert_loss(time_tensor, is_training=True)
                loss["lagbert"] = L_lagbert
                loss["total"] = loss["total"] + lambda_lagbert * L_lagbert
            
            # V5: Learnable balance regularization
            use_learnable_balance = getattr(hyper, 'use_learnable_balance', False)
            lambda_balance = getattr(hyper, 'lambda_balance', 0.0)
            if use_learnable_balance and gaussians._deformation_anchor is not None:
                # Log current alpha value
                current_alpha = gaussians._deformation_anchor.get_balance_alpha()
                loss["balance_alpha"] = current_alpha
                
                # Balance regularization: L_balance = (α - 0.5)^2
                if lambda_balance > 0:
                    L_balance = gaussians._deformation_anchor.compute_balance_regularization_loss(alpha_target=0.5)
                    loss["balance_reg"] = L_balance.item()
                    loss["total"] = loss["total"] + lambda_balance * L_balance
            
            # V7: Uncertainty-Aware Fusion (Kendall Loss)
            use_uncertainty_fusion = getattr(hyper, 'use_uncertainty_fusion', False)
            if use_uncertainty_fusion and gaussians._deformation_anchor is not None:
                # Apply Kendall loss: L_total = L_render/(2Σ) + λ*log(Σ)
                # This modifies the total loss to incorporate uncertainty
                loss["total"] = gaussians._deformation_anchor.compute_kendall_loss(loss["total"])
                
                # Log uncertainty stats
                unc_stats = gaussians._deformation_anchor.get_uncertainty_stats()
                if unc_stats:
                    loss["unc_weight_hex"] = unc_stats.get('weight_hex', 0.5)
                    loss["unc_weight_anchor"] = unc_stats.get('weight_anchor', 0.5)
                    loss["unc_sigma_hex"] = unc_stats.get('sigma_hex', 1.0)
                    loss["unc_sigma_anchor"] = unc_stats.get('sigma_anchor', 1.0)
        
        loss["total"].backward()
        iter_end.record()
        torch.cuda.synchronize()


        with torch.no_grad():
            # Adaptive control
            # V11: Skip densification during pretrain stage (no render gradients)
            if not in_pretrain_stage:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter]
                )
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)
            if iteration < opt.densify_until_iter and not in_pretrain_stage:
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        opt.density_min_threshold,
                        opt.max_screen_size,
                        max_scale,
                        opt.max_num_gaussians,
                        densify_scale_threshold,
                        bbox,
                    )
            if gaussians.get_density.shape[0] == 0:
                raise ValueError(
                    "No Gaussian left. Change adaptive control hyperparameters!"
                )

            # Optimization
            if iteration < train_iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                # s1 方案A: Step learnable weights optimizer (separate from Gaussian optimizer)
                if learnable_weights_optimizer is not None:
                    learnable_weights_optimizer.step()
                    learnable_weights_optimizer.zero_grad(set_to_none=True)
                
                # s2: Step phase-gated optimizer (separate from Gaussian optimizer)
                if phase_gated_optimizer is not None:
                    phase_gated_optimizer.step()
                    phase_gated_optimizer.zero_grad(set_to_none=True)
                
                # s3: Step dual static volume optimizer (separate from Gaussian optimizer)
                if dual_volume_optimizer is not None:
                    dual_volume_optimizer.step()
                    dual_volume_optimizer.zero_grad(set_to_none=True)

            # Save gaussians
            if iteration in saving_iterations or iteration == train_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Gaussians")
                scene.save(iteration, queryfunc, stage)

            # Save checkpoints
            if iteration in checkpoint_iterations:
                tqdm.write(f"[ITER {iteration}] Saving Checkpoint")
                torch.save(
                    (gaussians.capture(), iteration),
                    ckpt_save_path + "/chkpnt" + str(iteration) + ".pth",
                )

            # Progress bar
            if iteration % 10 == 0:
                progress_bar.set_postfix(
                    {
                        "loss": f"{loss['total'].item():.1e}",
                        "pts": f"{gaussians.get_density.shape[0]:2.1e}",
                    }
                )
                progress_bar.update(10)
            if iteration == train_iterations:
                progress_bar.close()

            # Logging
            metrics = {}
            for l in loss:
                # Handle both tensor and float values (s1_weight, s1_L_mean are floats)
                if isinstance(loss[l], (int, float)):
                    metrics["loss_" + l] = loss[l]
                else:
                    metrics["loss_" + l] = loss[l].item()
            for param_group in gaussians.optimizer.param_groups:
                metrics[f"lr_{param_group['name']}"] = param_group["lr"]

            metrics['period'] = math.exp(gaussians.period.item())
            
            # s1: Log reweighting statistics during static warm-up
            if stage == 'coarse' and opt.use_static_reweighting:
                metrics['s1_weight'] = s1_weight
                
                if opt.static_reweight_method == "residual" and static_reweight_manager is not None:
                    # Log detailed statistics every 500 iterations
                    if iteration % 500 == 0:
                        s1_stats = static_reweight_manager.get_statistics()
                        for key, value in s1_stats.items():
                            metrics[f's1_{key}'] = value
                            
                elif opt.static_reweight_method == "learnable" and learnable_static_weights is not None:
                    metrics['s1_L_mean'] = s1_L_mean
                    # Log detailed statistics every 500 iterations
                    if iteration % 500 == 0:
                        s1_stats = learnable_static_weights.get_statistics()
                        for key, value in s1_stats.items():
                            metrics[f's1_{key}'] = value
            
            # s2: Log phase-gated statistics during static warm-up
            if stage == 'coarse' and opt.use_phase_gated_static and phase_gated_static is not None:
                metrics['s2_weight'] = s2_weight
                metrics['s2_L_win'] = s2_L_win
                
                # Log detailed statistics every 500 iterations
                if iteration % 500 == 0:
                    s2_stats = phase_gated_static.get_statistics()
                    for key, value in s2_stats.items():
                        metrics[f's2_{key}'] = value
            
            # s3: Log dual static volume statistics during static warm-up
            if stage == 'coarse' and opt.use_s3_dual_static_volume and dual_static_volume is not None:
                metrics['s3_L_V'] = s3_L_V
                metrics['s3_L_distill'] = s3_L_distill
                metrics['s3_L_VTV'] = s3_L_VTV
                
                # Log detailed statistics every 500 iterations
                if iteration % 500 == 0:
                    s3_stats = dual_static_volume.get_statistics()
                    for key, value in s3_stats.items():
                        metrics[f's3_{key}'] = value
            
            # s4: Log temporal-ensemble statistics during static warm-up
            if stage == 'coarse' and opt.use_s4_temporal_ensemble_static and temporal_ensemble_static is not None:
                metrics['s4_L_vol'] = s4_L_vol
                metrics['s4_L_proj_avg'] = s4_L_proj_avg
                
                # Log detailed statistics every 500 iterations
                if iteration % 500 == 0:
                    s4_stats = temporal_ensemble_static.get_statistics()
                    for key, value in s4_stats.items():
                        if not isinstance(value, list):  # Skip shape info
                            metrics[f's4_{key}'] = value
            
            # PhysX-Gaussian: Log anchor-based deformation statistics
            if stage == 'fine' and getattr(hyper, 'use_anchor_deformation', False) and gaussians.use_anchor_deformation:
                if "phys_completion" in loss:
                    metrics['physx_L_phys'] = loss["phys_completion"].item() if isinstance(loss["phys_completion"], torch.Tensor) else loss["phys_completion"]
                if "anchor_smooth" in loss:
                    metrics['physx_L_smooth'] = loss["anchor_smooth"].item() if isinstance(loss["anchor_smooth"], torch.Tensor) else loss["anchor_smooth"]

            training_report(
                tb_writer,
                iteration,
                metrics,
                iter_start.elapsed_time(iter_end),
                testing_iterations,
                scene,
                lambda x, y, z: render(x, y, pipe, z),
                queryfunc,
                stage,
            )

def training_report(
    tb_writer,
    iteration,
    metrics_train,
    elapsed,
    testing_iterations,
    scene: Scene,
    renderFunc,
    queryFunc,
    stage,
):
    # Add training statistics
    if tb_writer:
        for key in list(metrics_train.keys()):
            tb_writer.add_scalar(f"train/{key}", metrics_train[key], iteration)
        tb_writer.add_scalar("train/iter_time", elapsed, iteration)
        tb_writer.add_scalar(
            "train/total_points", scene.gaussians.get_xyz.shape[0], iteration
        )

    if iteration in testing_iterations:
        # Evaluate 2D rendering performance
        eval_save_path = osp.join(scene.model_path, "eval", f"iter_{iteration:06d}")
        os.makedirs(eval_save_path, exist_ok=True)
        torch.cuda.empty_cache()

        validation_configs = [
            {"name": "render_train", "cameras": scene.getTrainCameras()},
            {"name": "render_test", "cameras": scene.getTestCameras()},
        ]
        psnr_2d, ssim_2d = None, None
        for config in validation_configs:
            if config["cameras"] and len(config["cameras"]) > 0:
                images = []
                gt_images = []
                image_show_2d = []
                # Render projections
                show_idx = np.linspace(0, len(config["cameras"]), 7).astype(int)[1:-1]
                for idx, viewpoint in enumerate(config["cameras"]):
                    image = renderFunc(
                        viewpoint,
                        scene.gaussians,
                        stage,
                    )["render"]
                    gt_image = viewpoint.original_image.to("cuda")
                    images.append(image)
                    gt_images.append(gt_image)
                    if tb_writer and idx in show_idx:
                        image_show_2d.append(
                            torch.from_numpy(
                                show_two_slice(
                                    gt_image[0],
                                    image[0],
                                    f"{viewpoint.image_name} gt",
                                    f"{viewpoint.image_name} render",
                                    vmin=gt_image[0].min() if iteration != 1 else None,
                                    vmax=gt_image[0].max() if iteration != 1 else None,
                                    save=True,
                                )
                            )
                        )
                images = torch.concat(images, 0).permute(1, 2, 0)
                gt_images = torch.concat(gt_images, 0).permute(1, 2, 0)
                psnr_2d, psnr_2d_projs = metric_proj(gt_images, images, "psnr")
                ssim_2d, ssim_2d_projs = metric_proj(gt_images, images, "ssim")
                eval_dict_2d = {
                    "psnr_2d": psnr_2d,
                    "ssim_2d": ssim_2d,
                    "psnr_2d_projs": psnr_2d_projs,
                    "ssim_2d_projs": ssim_2d_projs,
                }
                with open(
                    osp.join(eval_save_path, f"eval2d_{config['name']}.yml"),
                    "w",
                ) as f:
                    yaml.dump(
                        eval_dict_2d, f, default_flow_style=False, sort_keys=False
                    )

                if tb_writer:
                    image_show_2d = torch.from_numpy(
                        np.concatenate(image_show_2d, axis=0)
                    )[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        config["name"] + f"/{viewpoint.image_name}",
                        image_show_2d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/psnr_2d", psnr_2d, iteration
                    )
                    tb_writer.add_scalar(
                        config["name"] + "/ssim_2d", ssim_2d, iteration
                    )

        # Evaluate 3D reconstruction performance
        breath_cycle = 3.0  # breath period
        num_phases = 10  # phase numbers
        phase_time = breath_cycle / num_phases
        mid_phase_time = phase_time / 2
        scanTime = 60.0
        psnr_3d_list = []
        ssim_3d_list = []
        ssim_3d_axis_x_list = []
        ssim_3d_axis_y_list = []
        ssim_3d_axis_z_list = []
        with torch.no_grad():
            for t in range(10):
                time = (mid_phase_time + phase_time * t) / scanTime
                vol_pred = queryFunc(scene.gaussians, time, stage)["vol"]
                vol_gt = scene.vol_gt[t]
                psnr_3d, _ = metric_vol(vol_gt, vol_pred, "psnr")
                ssim_3d, ssim_3d_axis = metric_vol(vol_gt, vol_pred, "ssim")
                psnr_3d_list.append(psnr_3d)
                ssim_3d_list.append(ssim_3d)
                ssim_3d_axis_x_list.append(ssim_3d_axis[0])
                ssim_3d_axis_y_list.append(ssim_3d_axis[1])
                ssim_3d_axis_z_list.append(ssim_3d_axis[2])
                if tb_writer:
                    image_show_3d = np.concatenate(
                        [
                            show_two_slice(
                                vol_gt[..., i],
                                vol_pred[..., i],
                                f"slice {i} gt",
                                f"slice {i} pred",
                                vmin=vol_gt[..., i].min(),
                                vmax=vol_gt[..., i].max(),
                                save=True,
                            )
                            for i in np.linspace(0, vol_gt.shape[2], 7).astype(int)[1:-1]
                        ],
                        axis=0,
                    )
                    image_show_3d = torch.from_numpy(image_show_3d)[None].permute([0, 3, 1, 2])
                    tb_writer.add_images(
                        f"reconstruction/slice-gt_pred_diff-T{t}",
                        image_show_3d,
                        global_step=iteration,
                    )
                    tb_writer.add_scalar(f"reconstruction/psnr_3d_T{t}", psnr_3d, iteration)
                    tb_writer.add_scalar(f"reconstruction/ssim_3d_T{t}", ssim_3d, iteration)
        psnr_3d_mean = float(np.array(psnr_3d_list).mean())
        ssim_3d_mean = float(np.array(ssim_3d_list).mean())
        eval_dict = {
                "psnr_3d": psnr_3d_list,
                "ssim_3d": ssim_3d_list,
                "ssim_3d_x": ssim_3d_axis_x_list,
                "ssim_3d_y": ssim_3d_axis_y_list,
                "ssim_3d_z": ssim_3d_axis_z_list,
                "psnr_3d_mean": psnr_3d_mean,
                "ssim_3d_mean": ssim_3d_mean,
            }
        with open(osp.join(eval_save_path, "eval3d.yml"), "w") as f:
            yaml.dump(eval_dict, f, default_flow_style=False, sort_keys=False)
        if tb_writer:
            tb_writer.add_scalar(f"reconstruction/psnr_3d_mean", psnr_3d_mean, iteration)
            tb_writer.add_scalar(f"reconstruction/ssim_3d_mean", ssim_3d_mean, iteration)
                
        tqdm.write(
            f"[ITER {iteration}] Evaluating: psnr3d {psnr_3d_mean:.3f}, ssim3d {ssim_3d_mean:.3f}, psnr2d {psnr_2d:.3f}, ssim2d {ssim_2d:.3f}"
        )

        # Record other metrics
        if tb_writer:
            tb_writer.add_histogram(
                "scene/density_histogram", scene.gaussians.get_density, iteration
            )

    torch.cuda.empty_cache()

if __name__ == "__main__":
    # fmt: off
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[5_000, 7_000, 10_000, 20_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--save_checkpoint", action="store_true",
                        help="Automatically save checkpoint at same iterations as save_iterations (syncs checkpoint_iterations with save_iterations)")
    parser.add_argument("--start_checkpoint", type=str, default=None,
                        help="Path to checkpoint file (.pth) to resume training. Contains model weights + optimizer state + iteration.")
    parser.add_argument("--load_model_path", type=str, default=None, 
                        help="Path to saved model directory (e.g., output/xxx/point_cloud/iteration_30000). Only loads model weights, not optimizer state.")
    parser.add_argument("--start_iteration", type=int, default=None,
                        help="Iteration to start from when using --load_model_path (required if --load_model_path is used)")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--coarse_iter", type=int, default=5000)
    parser.add_argument("--dirname", type=str, default="DEBUG")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    args.test_iterations.append(args.iterations)
    args.test_iterations.append(1)
    
    # --save_checkpoint: sync checkpoint_iterations with save_iterations
    if args.save_checkpoint:
        args.checkpoint_iterations = list(set(args.checkpoint_iterations + args.save_iterations))
        print(f"[save_checkpoint] Will save checkpoints at iterations: {sorted(args.checkpoint_iterations)}")
    # fmt: on

    dirname = args.dirname

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Load configuration files
    args_dict = vars(args)
    if args.config is not None:
        print(f"Loading configuration file from {args.config}")
        cfg = load_config(args.config)
        for key in list(cfg.keys()):
            args_dict[key] = cfg[key]

    # Set up logging writer
    tb_writer = prepare_output_and_logger(args, dirname)

    print("Optimizing " + args.model_path)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        lp.extract(args),
        op.extract(args),
        pp.extract(args),
        hp.extract(args),
        tb_writer,
        args.test_iterations,
        args.save_iterations,
        args.checkpoint_iterations,
        args.start_checkpoint,
        args.coarse_iter,
        args.load_model_path,
        args.start_iteration,
    )

    # All done
    print("Training complete.")
