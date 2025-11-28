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
        self.lambda_cycle = 0.01  # Weight for cycle motion loss: ||D_f(mu, t+T) - D_f(mu, t)||_1
        
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
