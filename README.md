# [ICCV 2025] X2-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction

### [Project Page](https://x2-gaussian.github.io/) | [Paper](https://arxiv.org/abs/2503.21779)

<p align="center">
  <img src="./media/gif1.gif" width="32%" style="display: inline-block; margin: 0;">
  <img src="./media/gif2.gif" width="32%" style="display: inline-block; margin: 0;">
  <img src="./media/gif3.gif" width="32%" style="display: inline-block; margin: 0;">
</p>

<p align="center">
  <img src="./media/tidal.jpg" width="80%">
</p>

<p align="center">
We achieve genuine continuous-time CT reconstruction without phase-binning. The figure illustrates temporal variations of lung volume in 4D CT reconstructed by our X2-Gaussian.
</p>

<p align="center">
  <img src="./media/teaser.jpg" width="100%">
</p>

<p align="center">
X2-Gaussian demonstrates state-of-the-art reconstruction performance.
</p>

## News

* 2025.10.27: Datasets have been released [here](https://huggingface.co/datasets/vortex778/X2GS). Welcome to have a try!
* 2025.10.17: Training code has been released.
* 2025.06.26: Our work has been accepted to ICCV 2025.
* 2025.03.27: Our paper is available on [arxiv](https://arxiv.org/abs/2503.21779).

## TODO

- [ ] Release more detailed instructions.
- [ ] Release data generation code.
- [ ] Release evaluation code.
- [ ] Release visualizaton code.

## Installation

```sh
# Download code
git clone https://github.com/yuyouxixi/x2-gaussian.git

# Install environment
conda create -n x2_gaussian python=3.9 -y
conda activate x2_gaussian

## You can choose suitable pytorch and cuda versions here on your own.
pip install torch==2.1.2+cu118 torchvision==0.16.2+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
pip install -e x2_gaussian/submodules/simple-knn
## xray-gaussian-rasterization-voxelization is from https://github.com/Ruyi-Zha/r2_gaussian/tree/main/r2_gaussian/submodules/xray-gaussian-rasterization-voxelization
pip install -e x2_gaussian/submodules/xray-gaussian-rasterization-voxelization

# Install TIGRE for data generation and initialization
wget https://github.com/CERN/TIGRE/archive/refs/tags/v2.3.zip
unzip v2.3.zip
pip install TIGRE-2.3/Python --no-build-isolation
```

## Training

### Dtaset

You can download datasets used in our paper [here](https://huggingface.co/datasets/vortex778/X2GS). We use [NAF](https://github.com/Ruyi-Zha/naf_cbct) format data (`*.pickle`) used in [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF).

### Initialization

We have included initialization files in our dataset. You can skip this step if using our dataset.

For new data, you need to use `initialize_pcd.py` to generate a `*.npy` file which stores the point cloud for Gaussian initialization.

```sh
python initialize_pcd.py --data <path to data>
```

### Start Training

Use `train.py` to train Gaussians. Make sure that the initialization file `*.npy` has been generated.

```sh
# Training
python train.py -s <path to data>

# Example
python train.py -s XXX/*.pickle  

#############################################################################
# V7 MAIN MODEL (RECOMMENDED for paper experiments)
#############################################################################
# This is the recommended configuration based on extensive ablation studies.
# v7 uses bidirectional displacement (D_f, D_b) with:
#   - L_inv: Inverse consistency regularization (phi_b(phi_f(x)) ≈ x)
#   - L_cycle: Motion cycle consistency using learned breathing period T_hat
#
# v7 automatically disables less effective components:
#   - L_sym (symmetry regularization from v2)
#   - L_jac (Jacobian regularization from v4)
#   - Velocity field integration (v5)
#   - Shared velocity inverse (v6)
#
# Default hyperparameters (can be overridden):
#   - lambda_inv = 0.1 (weight for inverse consistency loss)
#   - lambda_cycle = 0.01 (weight for motion cycle loss)

# Basic v7 main model training (uses default lambda values)
python train.py -s XXX/*.pickle --use_v7_bidirectional_displacement

# v7 main model with custom lambda values
python train.py -s XXX/*.pickle --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01

# Full v7 main model example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7_main \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case1_v7_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full v7 main model example for dir_4d_case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7_main \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case2_v7_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#############################################################################
# V8 PHASE-ALIGNED MODEL (experimental extension of v7)
#############################################################################
# v8 extends v7 by adding phase-conditioned deformation:
#   - Uses SSRML's learned breathing period T_hat to compute phase embedding
#   - Phase embedding: [sin(2π*t/T_hat), cos(2π*t/T_hat)]
#   - This embedding is concatenated to trunk output before D_f/D_b heads
#   - Makes deformation network explicitly aware of breathing phase
#
# v8 inherits all v7 settings (L_inv, L_cycle, disables L_sym/L_jac/velocity)
# The only difference is the phase-conditioned D_f/D_b heads.

# Basic v8 training with phase conditioning
python train.py -s XXX/*.pickle --use_v7_bidirectional_displacement --use_phase_conditioned_deformation

# Full v8 example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v8_phase \
  --use_v7_bidirectional_displacement --use_phase_conditioned_deformation \
  --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case1_v8_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full v8 example for dir_4d_case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v8_phase \
  --use_v7_bidirectional_displacement --use_phase_conditioned_deformation \
  --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case2_v8_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#############################################################################
# V9 LOW-RANK MOTION MODES (experimental extension of v7/v8)
#############################################################################
# v9 reparameterizes D_f/D_b as low-rank decomposition:
#   D_f(μ_i, t) = Σ_m a_m(t) * u_{i,m}
#   D_b(μ_i, t) = Σ_m b_m(t) * u_{i,m}
#
# where:
#   - u_{i,m} are per-Gaussian motion modes (M modes, each in R^3)
#   - a(t), b(t) are phase-dependent coefficients from small MLPs (F_a, F_b)
#   - Phase input: [sin(2π*t/T_hat), cos(2π*t/T_hat)]
#
# This leverages the inherently low-dimensional nature of respiratory motion.
# By factorizing the deformation into "spatial modes × time coefficients",
# v9 reduces overfitting and improves temporal coherence.
#
# v9 Parameters:
#   --use_low_rank_motion_modes     Enable v9 low-rank motion decomposition
#   --num_motion_modes M            Number of motion modes (default: 3, typically 2-4)
#   --use_mode_regularization       Enable L_mode = ||U||^2 regularization (optional)
#   --lambda_mode                   Weight for mode regularization (default: 1e-4)

# Basic v9 training with low-rank motion modes (3 modes)
python train.py -s XXX/*.pickle --use_v7_bidirectional_displacement --use_low_rank_motion_modes

# v9 with custom number of modes (2 modes)
python train.py -s XXX/*.pickle --use_v7_bidirectional_displacement --use_low_rank_motion_modes --num_motion_modes 2

# v9 with mode regularization (prevents mode vectors from growing too large)
python train.py -s XXX/*.pickle --use_v7_bidirectional_displacement --use_low_rank_motion_modes --use_mode_regularization --lambda_mode 1e-4

# Full v9 example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v9_lowrank \
  --use_v7_bidirectional_displacement --use_low_rank_motion_modes --num_motion_modes 3 \
  --use_mode_regularization --lambda_mode 1e-4 \
  --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case1_v9_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full v9 example for dir_4d_case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v9_lowrank \
  --use_v7_bidirectional_displacement --use_low_rank_motion_modes --num_motion_modes 3 \
  --use_mode_regularization --lambda_mode 1e-4 \
  --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case2_v9_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#######################################################################
# V10 ADAPTIVE GATED MOTION (experimental extension of v9)
#######################################################################
# v10 extends v9 by introducing:
#   1. Adaptive Motion Gating: G_f, G_b networks that learn per-Gaussian/per-time gates
#      to adaptively fuse base (K-Planes) and low-rank displacement:
#        D_f_total = g_f * D_f_lr + (1 - g_f) * D_f_base
#        D_b_total = g_b * D_b_lr + (1 - g_b) * D_b_base
#   2. Trajectory Smoothing: L_traj penalizes second-order differences (acceleration)
#      of Gaussian trajectories over time to reduce non-physical jitter.
#
# v10 aims to improve upon v9 by allowing the network to decide where to use
# low-rank vs base displacements, and by regularizing trajectory smoothness.
#
# v10 Parameters:
#   --use_adaptive_gating           Enable v10 adaptive gating of base and low-rank displacements
#   --lambda_gate <float>           Weight for gate regularization L_gate (default: 0, optional)
#   --use_trajectory_smoothing      Enable trajectory smoothing regularization
#   --lambda_traj <float>           Weight for trajectory smoothing L_traj (default: 1e-3)
#   --traj_num_time_samples <int>   Number of time triplets to sample for L_traj (default: 5)
#
# Basic v10 training with adaptive gating and trajectory smoothing
python train.py -s XXX/*.pickle --use_v7_bidirectional_displacement --use_low_rank_motion_modes --use_adaptive_gating --use_trajectory_smoothing

# v10 with custom lambda_traj (stronger trajectory smoothing)
python train.py -s XXX/*.pickle --use_v7_bidirectional_displacement --use_low_rank_motion_modes --use_adaptive_gating --use_trajectory_smoothing --lambda_traj 5e-3

# v10 with gate regularization (encourages binary-like gates)
python train.py -s XXX/*.pickle --use_v7_bidirectional_displacement --use_low_rank_motion_modes --use_adaptive_gating --lambda_gate 1e-4 --use_trajectory_smoothing --lambda_traj 1e-3

# Full v10 example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v10_gated \
  --use_v7_bidirectional_displacement --use_low_rank_motion_modes --num_motion_modes 3 \
  --use_adaptive_gating --use_trajectory_smoothing --lambda_traj 1e-3 \
  --use_mode_regularization --lambda_mode 1e-4 \
  --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case1_v10_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full v10 example for dir_4d_case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v10_gated \
  --use_v7_bidirectional_displacement --use_low_rank_motion_modes --num_motion_modes 3 \
  --use_adaptive_gating --use_trajectory_smoothing --lambda_traj 1e-3 \
  --use_mode_regularization --lambda_mode 1e-4 \
  --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case2_v10_$(date +%Y%m%d_%H%M%S).log 2>&1 &


#############################################################################
# BASELINE (original X2-Gaussian without bidirectional displacement)
#############################################################################

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 10000 20000 30000 --dirname dir_4d_case1_baseline \
  > train_dir_4d_case1_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#############################################################################
# ABLATION STUDY CONFIGURATIONS (for experimental purposes only)
#############################################################################
# The following configurations are provided for ablation studies and
# are NOT recommended for main experiments. Use --use_v7_bidirectional_displacement
# for the best results.

# Training with Inverse Consistency only (ablation: v1)
# This enables forward/backward deformation with inverse consistency regularization
# to encourage more physically plausible motion fields.
python train.py -s XXX/*.pickle --use_inverse_consistency --lambda_inv 0.01

# Full example with inverse consistency
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_inv_consistency \
  --use_inverse_consistency --lambda_inv 0.01 \
  > train_dir_4d_case1_inv_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Training with Inverse Consistency + Symmetry Regularization (ablation: v1 + v2)
# L_sym encourages D_b ≈ -D_f (backward deformation approximates negative of forward)
# NOTE: v2 (L_sym) showed no improvement in experiments and is NOT used in v7 main model
python train.py -s XXX/*.pickle --use_inverse_consistency --lambda_inv 0.01 --use_symmetry_reg --lambda_sym 0.01

# Full example with both inverse consistency and symmetry regularization
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_inv_sym \
  --use_inverse_consistency --lambda_inv 0.01 --use_symmetry_reg --lambda_sym 0.01 \
  > train_dir_4d_case1_inv_sym_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Training with Cycle Motion Consistency (ablation: v1 + v2 + v3)
# L_cycle_motion uses the learned breathing period T_hat to enforce motion periodicity:
# For each Gaussian, positions at t and t+T_hat should be approximately the same.
# NOTE: v3 (L_cycle) IS used in v7 main model, but v2 (L_sym) is NOT
python train.py -s XXX/*.pickle --use_inverse_consistency --lambda_inv 0.01 --use_symmetry_reg --lambda_sym 0.01 --use_cycle_motion --lambda_cycle 0.01

# Full example with inverse consistency, symmetry regularization, and cycle motion
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_inv_sym_cycle \
  --use_inverse_consistency --lambda_inv 0.05 --use_symmetry_reg --lambda_sym 0.01 --use_cycle_motion --lambda_cycle 0.01 \
  > train_dir_4d_case1_inv_sym_cycle_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_inv_cycle \
  --use_inverse_consistency --lambda_inv 0.05 --use_cycle_motion --lambda_cycle 0.01 \
  > train_dir_4d_case2_inv_cycle_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_inv_cycle \
  --use_inverse_consistency --lambda_inv 0.1 --use_cycle_motion --lambda_cycle 0.01 \
  > train_dir_4d_case2_inv01_cycle_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_inv_cycle \
  --use_inverse_consistency --lambda_inv 0.1 --use_cycle_motion --lambda_cycle 0.1 \
  > train_dir_4d_case2_inv01_cycle01_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_inv_cycle \
  --use_inverse_consistency --lambda_inv 0.1 --use_cycle_motion --lambda_cycle 0.2 \
  > train_dir_4d_case2_inv01_cycle02_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v8_phase \
  --use_v7_bidirectional_displacement --use_phase_conditioned_deformation \
  --lambda_inv 0.1 --lambda_cycle 0.1 \
  > train_dir_4d_case2_v8_$(date +%Y%m%d_%H%M%S).log 2>&1 &

  # Full v9 training for dir_4d_case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v9_lowrank \
  --use_v7_bidirectional_displacement \
  --lambda_inv 0.1 --lambda_cycle 0.1 \
  > train_dir_4d_case2_v9_lowrank_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v9_lowrank \
  --use_low_rank_motion_modes --num_motion_modes 3 \
  --use_mode_regularization --lambda_mode 1e-4 \
  --lambda_inv 0.1 --lambda_cycle 0.1 \
  > train_dir_4d_case2_v9nov8_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 \
  --use_low_rank_motion_modes --num_motion_modes 3 \
  --use_adaptive_gating \
  --use_trajectory_smoothing --lambda_traj 1e-3 \
  --use_mode_regularization --lambda_mode 1e-4 \
  --lambda_inv 0.1 --lambda_cycle 0.1 \
  --dirname dir_4d_case2_v10 \
  > train_dir_4d_case2_v10_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Training with Jacobian Regularization (ablation: v1 + v2 + v3 + v4)
# L_jac penalizes negative Jacobian determinants to prevent local folding in the deformation field.
# This encourages the motion to be locally orientation-preserving (non-folding).
# NOTE: v4 (L_jac) showed no improvement in experiments and is NOT used in v7 main model
python train.py -s XXX/*.pickle --use_inverse_consistency --lambda_inv 0.01 --use_symmetry_reg --lambda_sym 0.01 --use_cycle_motion --lambda_cycle 0.01 --use_jacobian_reg --lambda_jac 0.01

# Full example with all regularizations (inverse consistency, symmetry, cycle motion, and Jacobian)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_full_reg \
  --use_inverse_consistency --lambda_inv 0.05 --use_symmetry_reg --lambda_sym 0.01 \
  --use_cycle_motion --lambda_cycle 0.01 --use_jacobian_reg --lambda_jac 0.01 \
  --jacobian_num_samples 64 --jacobian_step_size 1e-3 \
  > train_dir_4d_case1_full_reg_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Training with Velocity Field Integration (ablation: v5)
# Uses velocity field v(x,t) with multi-step Euler integration instead of direct displacement D_f(x,t).
# This makes the forward motion more diffeomorphic-like (smoother, orientation-preserving).
# NOTE: v5 (velocity field) showed degraded performance and is NOT used in v7 main model
python train.py -s XXX/*.pickle --use_velocity_field --velocity_num_steps 4

# Full example with velocity field and all regularizations
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_velocity_field \
  --use_inverse_consistency --lambda_inv 0.05 --use_symmetry_reg --lambda_sym 0.01 \
  --use_cycle_motion --lambda_cycle 0.01 --use_jacobian_reg --lambda_jac 0.01 \
  --use_velocity_field --velocity_num_steps 4 \
  > train_dir_4d_case1_velocity_field_$(date +%Y%m%d_%H%M%S).log 2>&1 &


# Training with Shared Velocity Inverse (ablation: v6)
# Uses the SAME velocity field v(x,t) to construct both forward φ_f and backward φ_b mappings
# via numerical integration. This removes the separate D_b network and automatically disables L_sym.
# The inverse consistency is now structurally enforced by the shared flow, making the method
# closer to standard diffeomorphic registration.
# NOTE: v6 (shared velocity inverse) showed degraded performance and is NOT used in v7 main model
python train.py -s XXX/*.pickle --use_velocity_field --velocity_num_steps 4 --use_shared_velocity_inverse

# Full example with shared velocity inverse (v6 mode)
# Note: L_sym is automatically disabled in v6 mode even if --use_symmetry_reg is specified
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v6_shared_velocity \
  --use_velocity_field --velocity_num_steps 4 --use_shared_velocity_inverse \
  --use_inverse_consistency --lambda_inv 0.1 \
  --use_cycle_motion --lambda_cycle 0.01 --use_jacobian_reg --lambda_jac 0.005 \
  > train_dir_4d_case1_v6_shared_velocity_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v6_shared_velocity \
  --use_velocity_field --velocity_num_steps 4 --use_shared_velocity_inverse \
  --use_inverse_consistency --lambda_inv 0.1 \
  --use_cycle_motion --lambda_cycle 0.01 \
  > train_dir_4d_case2_v6_shared_velocity_$(date +%Y%m%d_%H%M%S).log 2>&1 &

```

## Citation

If you find this work helpful, please consider citing:

```
@article{yu2025x,
  title={X $\^{}$\{$2$\}$ $-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction},
  author={Yu, Weihao and Cai, Yuanhao and Zha, Ruyi and Fan, Zhiwen and Li, Chenxin and Yuan, Yixuan},
  journal={arXiv preprint arXiv:2503.21779},
  year={2025}
}
```

## Acknowledgement

Our code is adapted from [R2-Gaussian](https://github.com/Ruyi-Zha/r2_gaussian), [4D Gaussians](https://github.com/hustvl/4DGaussians), [X-Gaussian](https://github.com/caiyuanhao1998/X-Gaussian) and [TIGRE toolbox](https://github.com/CERN/TIGRE.git). We thank the authors for their excellent works.

