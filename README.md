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
```

### Checkpointing and Continue Training

训练过程支持保存 checkpoint 并从 checkpoint 继续训练。

#### 保存方式对比

| 保存类型 | 内容 | 保存位置 | 用途 |
|---------|------|---------|------|
| **Model Save** (`--save_iterations`) | 模型权重（Gaussians + Deformation） | `output/xxx/point_cloud/iteration_N/` | 用于推理、评估 |
| **Checkpoint** (`--checkpoint_iterations`) | 模型权重 + 优化器状态 + iteration | `output/xxx/ckpt/chkpntN.pth` | 用于**完美**继续训练 |

#### 推荐：使用 `--save_checkpoint` 自动保存 checkpoint

```sh
# --save_checkpoint 会在 save_iterations 时同时保存 checkpoint
python train.py -s XXX/*.pickle --iterations 30000 \
  --save_iterations 30000 \
  --save_checkpoint \
  --dirname my_experiment
```

等效于：
```sh
python train.py -s XXX/*.pickle --iterations 30000 \
  --save_iterations 30000 \
  --checkpoint_iterations 30000 \
  --dirname my_experiment
```

#### 继续训练方式

**方式一：从 Checkpoint 继续（推荐，完美恢复）**

```sh
# 从 checkpoint 继续训练到 50000 iterations
# checkpoint 包含：模型权重 + 优化器状态 + 当前 iteration
python train.py -s XXX/*.pickle --iterations 50000 \
  --save_iterations 50000 \
  --save_checkpoint \
  --dirname my_experiment_50k \
  --start_checkpoint output/xxx/ckpt/chkpnt30000.pth
```

**方式二：从保存的模型继续（仅恢复模型权重）**

```sh
# 从保存的模型目录继续训练
# 注意：优化器状态会重置，需要指定 --start_iteration
python train.py -s XXX/*.pickle --iterations 50000 \
  --save_iterations 50000 \
  --save_checkpoint \
  --dirname my_experiment_50k \
  --load_model_path output/xxx/point_cloud/iteration_30000 \
  --start_iteration 30000
```

#### 两种继续方式的区别

| 方面 | `--start_checkpoint` | `--load_model_path` |
|------|---------------------|---------------------|
| 模型权重 | ✓ 恢复 | ✓ 恢复 |
| 优化器状态 (momentum) | ✓ 恢复 | ✗ 重置 |
| 训练 iteration | ✓ 自动恢复 | 需指定 `--start_iteration` |
| **效果** | **完全等效继续** | 近似继续（优化器无momentum） |
| **推荐场景** | 正常继续训练 | 没有保存checkpoint时的备选方案 |

```sh
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
  --use_v7_bidirectional_displacement --lambda_inv 0 --lambda_cycle 0.1 \
  > train_dir_4d_case1_v7_inv0_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full v7 main model example for dir_4d_case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7_main \
  --use_v7_bidirectional_displacement --lambda_inv 0 --lambda_cycle 0.1 \
  > train_dir_4d_case2_v7_inv0_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#############################################################################
# V7.1 CONSISTENCY-AWARE RENDERING (evaluation-time upgrade to v7)
#############################################################################
# V7.1 is an EVALUATION-TIME modification that uses the backward field D_b
# to "correct" forward deformed centers during rendering/query:
#   y = mu + D_f(mu, t)           # V7 forward deformation
#   x_hat = y + D_b(y, t)         # Round-trip back to canonical
#   r = x_hat - mu                # Round-trip residual
#   y_corrected = y - alpha * r   # Corrected center
#
# When alpha=0, this is exactly V7. When alpha>0, centers with large
# round-trip residuals are "pulled back" toward more consistent positions.
#
# WORKFLOW:
# 1. Train V7 model (above commands)
# 2. Run grid search to find optimal alpha
# 3. Evaluate with V7.1-fixed OR V7.1+TTO-alpha

# ----- STEP 1: Grid search for optimal alpha (run after V7 training) -----
# This evaluates multiple alpha values and finds the best one for each case

# Grid search for case1
python grid_search_alpha.py \
  -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  -m output/dir_4d_case1_v7_main --iteration 30000 \
  --alpha_values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

# Grid search for case2
python grid_search_alpha.py \
  -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  -m output/dir_4d_case2_v7_main --iteration 30000 \
  --alpha_values 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0

# ----- STEP 2a: V7.1-fixed evaluation (use best alpha from grid search) -----
# Replace BEST_ALPHA with the optimal value found by grid search

# V7.1-fixed for case1 (example with alpha=0.3)
python eval_4d_x2_gaussian.py \
  -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  -m output/dir_4d_case1_v7_main --iteration 30000 \
  --use_v7_1_correction --correction_alpha 0.3

# V7.1-fixed for case2 (example with alpha=0.3)
python eval_4d_x2_gaussian.py \
  -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  -m output/dir_4d_case2_v7_main --iteration 30000 \
  --use_v7_1_correction --correction_alpha 0.3

# ----- STEP 2b: V7.1+TTO-alpha (test-time optimization per case) -----
# TTO-alpha automatically finds optimal alpha via gradient descent

# V7.1+TTO-alpha for case1
python eval_4d_x2_gaussian.py \
  -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  -m output/dir_4d_case1_v7_main --iteration 30000 \
  --use_v7_1_correction --use_tto_alpha \
  --tto_alpha_init 0.3 --tto_alpha_lr 1e-2 --tto_alpha_steps 100 \
  --tto_alpha_reg 1e-3 --tto_num_views_per_step 8

# V7.1+TTO-alpha for case2
python eval_4d_x2_gaussian.py \
  -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  -m output/dir_4d_case2_v7_main --iteration 30000 \
  --use_v7_1_correction --use_tto_alpha \
  --tto_alpha_init 0.3 --tto_alpha_lr 1e-2 --tto_alpha_steps 100 \
  --tto_alpha_reg 1e-3 --tto_num_views_per_step 8

# ----- STEP 2c: TTO-α(t) (test-time optimization with time-dependent alpha) -----
# Optimize a small α(t) network at test time (扩展版本 2.3 - test-time only)
# This learns different correction strength for different breathing phases

# TTO-α(t) with Fourier basis for case1
python eval_4d_x2_gaussian.py \
  -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  -m output/01a85696-c --iteration 30000 \
  --use_v7_1_correction --use_tto_alpha_t \
  --tto_alpha_network_type fourier --tto_alpha_fourier_freqs 4 \
  --tto_alpha_init 0.0 --tto_alpha_lr 1e-2 --tto_alpha_steps 100

# TTO-α(t) with MLP for case1
python eval_4d_x2_gaussian.py \
  -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  -m output/01a85696-c --iteration 30000 \
  --use_v7_1_correction --use_tto_alpha_t \
  --tto_alpha_network_type mlp --tto_alpha_mlp_hidden 32 \
  --tto_alpha_init 0.0 --tto_alpha_lr 1e-2 --tto_alpha_steps 100

# TTO-α(t) with Fourier basis for case2
python eval_4d_x2_gaussian.py \
  -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  -m output/9ce54723-0 --iteration 30000 \
  --use_v7_1_correction --use_tto_alpha_t \
  --tto_alpha_network_type fourier --tto_alpha_fourier_freqs 4 \
  --tto_alpha_init 0.0 --tto_alpha_lr 1e-2 --tto_alpha_steps 100

# ----- STEP 3: V7.1-trainable-α (train-time learnable alpha) -----
# Train a global learnable α together with the network (方式 B)
# Alpha is optimized during training with L2 regularization

# V7.1-trainable-α for case1 (scalar alpha)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.1_trainable_alpha \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  --use_trainable_alpha --trainable_alpha_init 0.0 --trainable_alpha_reg 1e-3 \
  > train_v7.1_trainable_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.1-trainable-α for case2 (scalar alpha)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7.1_trainable_alpha \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.1 \
  --use_trainable_alpha --trainable_alpha_init 0.0 --trainable_alpha_reg 1e-3 \
  > train_v7.1_trainable_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ----- STEP 4: V7.1-α(t) (time-dependent alpha) -----
# Use a small network g_θ(t) to predict time-dependent alpha values (扩展版本 2.3)
# Options: 'fourier' (Fourier basis) or 'mlp' (small MLP)

# V7.1-α(t) with Fourier basis for case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.1_alpha_t_fourier \
  --use_v7_bidirectional_displacement --lambda_inv 0 --lambda_cycle 0.1 \
  --use_trainable_alpha --use_time_dependent_alpha \
  --alpha_network_type fourier --alpha_fourier_freqs 4 \
  --trainable_alpha_init 0.3 --trainable_alpha_reg 1e-3 \
  > train_v7.1_alpha_t_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.1-α(t) with MLP for case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.1_alpha_t_mlp \
  --use_v7_bidirectional_displacement --lambda_inv 0 --lambda_cycle 0.1 \
  --use_trainable_alpha --use_time_dependent_alpha \
  --alpha_network_type mlp --alpha_mlp_hidden 32 \
  --trainable_alpha_init 0.3 --trainable_alpha_reg 1e-3 \
  > train_v7.1_alpha_t_mlp_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7.1_alpha_t_mlp \
  --use_v7_bidirectional_displacement --lambda_inv 0 --lambda_cycle 0.1 \
  --use_trainable_alpha --use_time_dependent_alpha \
  --alpha_network_type mlp --alpha_mlp_hidden 32 \
  --trainable_alpha_init 0.3 --trainable_alpha_reg 1e-3 \
  > train_v7.1_alpha_t_mlp_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#############################################################################
# V7.2 END-TO-END CONSISTENCY-AWARE DEFORMATION (major upgrade to v7/v7.1)
#############################################################################
# V7.2 uses the consistency-aware forward mapping DURING TRAINING, not just
# at test time like V7.1. The key differences are:
# - V7.1: Training uses y = mu + D_f, test-time optionally applies correction
# - V7.2: BOTH training and test use y_corrected = y - alpha * r
#
# This allows D_b to receive gradients from L_render through the main
# rendering path. As a result:
# - L_inv is NOT used in V7.2 (automatically disabled)
# - Instead, lightweight L_b regularization controls D_b magnitude
#
# Formula:
#   y = mu + D_f(mu, t)           # forward deformation
#   x_hat = y + D_b(y, t)         # round-trip reconstruction
#   r = x_hat - mu                # round-trip residual
#   y_corrected = y - alpha * r   # V7.2 effective center

# V7.2 with fixed alpha for case2 (no --v7_2_alpha_learnable flag)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7.2_fixed_alpha \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 \
  --v7_2_lambda_b_reg 1e-3 \
  > train_v7.2_fixed_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.2 with learnable alpha for case1 (recommended)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.2_learnable_alpha \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  > train_v7.2_learnable_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.2 with learnable alpha for case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7.2_learnable_alpha \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  > train_v7.2_learnable_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.2 with time-dependent alpha α(t) using Fourier network for case1
# Instead of a global scalar alpha, use a small network g_theta(t) to predict
# time-dependent alpha values. This allows different correction strength for
# different breathing phases.
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.2_alpha_t_fourier \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 \
  --v7_2_use_time_dependent_alpha --v7_2_alpha_network_type fourier --v7_2_alpha_fourier_freqs 4 \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  > train_v7.2_alpha_t_fourier_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.2 with time-dependent alpha α(t) using MLP network for case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7.2_alpha_t_mlp \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 \
  --v7_2_use_time_dependent_alpha --v7_2_alpha_network_type mlp --v7_2_alpha_mlp_hidden 32 \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  > train_v7.2_alpha_t_mlp_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#############################################################################
# V7.2.1 CANONICAL-SPACE CYCLE CONSISTENCY (lightweight extension of V7.2)
#############################################################################
# V7.2.1 adds a canonical-space cycle consistency loss (L_cycle-canon) on top
# of V7.2. Instead of requiring D_b to be a perfect inverse (L_inv), we require:
#   "The same point μ_i decoded back to canonical at t and t+T̂ should be consistent"
#
# Formula:
#   x_canon(t)   = ϕ̃_f(μ, t)   + D_b(ϕ̃_f(μ, t),   t)
#   x_canon(t+T) = ϕ̃_f(μ, t+T) + D_b(ϕ̃_f(μ, t+T), t+T)
#   L_cycle_canon = |x_canon(t+T) - x_canon(t)|
#
# This provides a softer temporal constraint on D_b without forcing r→0.
# Key parameters:
#   --use_v7_2_1_cycle_canon: Enable L_cycle_canon (requires V7.2)
#   --v7_2_1_lambda_cycle_canon: Weight for L_cycle_canon (default: 1e-3)

# V7.2.1 with learnable alpha for case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.2.1_cycle_canon \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  > train_v7.2.1_cycle_canon_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.2.1 with learnable alpha for case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --save_checkpoint\
  --dirname dir_4d_case2_v7.2.1_cycle_canon \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  > train_v7.2.1_cycle_canon_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#############################################################################
# V7.3 TEMPORAL BIDIRECTIONAL WARP CONSISTENCY (extension of V7.2.1)
#############################################################################
# V7.3 adds true time-forward/time-backward warp constraints on top of V7.2.1.
# Instead of just canonical↔t bidirectionality, we define temporal warps:
#   - Time-forward warp (t1→t2): x(t1) → canonical → x'(t2) ≈ x(t2)
#   - Time-backward warp (t2→t1): x(t2) → canonical → x'(t1) ≈ x(t1)
# Uses canonical as a bridge to warp between different time points.
#
# Formula:
#   μ^(t1) = x(t1) + D_b(x(t1), t1)  # decode to canonical
#   x_fw(t2|t1) = ϕ̃_f(μ^(t1), t2)   # forward to t2
#   L_fw = |x_fw(t2|t1) - x(t2)|     # time-forward consistency
#   (similarly for L_bw)
#
# Key parameters:
#   --use_v7_3_timewarp: Enable temporal bidirectional warp (requires V7.2)
#   --v7_3_lambda_fw: Weight for L_fw (default: 1e-3)
#   --v7_3_lambda_bw: Weight for L_bw (default: 1e-3)
#   --v7_3_lambda_fw_bw: Weight for optional round-trip L_fw_bw (default: 0)
#   --v7_3_timewarp_delta_fraction: Δ = fraction * T̂ (default: 0.1)

# V7.3 with temporal bidirectional warp for case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.3_timewarp \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --v7_3_timewarp_delta_fraction 0.1 \
  > train_v7.3_timewarp_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.3_timewarp \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-1 --v7_3_lambda_bw 1e-1 \
  --v7_3_timewarp_delta_fraction 0.25 \
  > train_v7.3_set2_timewarp_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.3 with temporal bidirectional warp for case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7.3_timewarp \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --v7_3_timewarp_delta_fraction 0.1 \
  > train_v7.3_timewarp_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.3 with round-trip closure loss (optional, more aggressive constraint)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.3_timewarp_roundtrip \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --v7_3_lambda_fw_bw 1e-4 --v7_3_timewarp_delta_fraction 0.1 \
  > train_v7.3_roundtrip_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7.3_timewarp_roundtrip \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --v7_3_lambda_fw_bw 1e-4 --v7_3_timewarp_delta_fraction 0.1 \
  > train_v7.3_roundtrip_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#############################################################################
# V7.3.1 SIGMA/RHO TEMPORAL REGULARIZATION (extension of V7.3)
#############################################################################
# V7.3.1 extends V7.3 by adding temporal regularization for per-Gaussian
# covariance (scale) and density, in addition to center trajectories:
#   - Temporal TV: penalizes rapid changes in scale/density along time
#   - Periodic consistency: enforces cycle consistency over learned period T̂
#
# Losses:
#   L_tv_sigma = E[|log(S(t+δ)) - log(S(t))|]   # scale temporal smoothness
#   L_tv_rho = E[|ρ(t+δ) - ρ(t)|]               # density temporal smoothness
#   L_cycle_sigma = E[|log(S(t+T̂)) - log(S(t))|] # scale periodic consistency
#   L_cycle_rho = E[|ρ(t+T̂) - ρ(t)|]           # density periodic consistency
#
# Note: If scale/density are static (not time-dependent), losses auto no-op.
# Currently density is static in this codebase, so L_tv_rho/L_cycle_rho = 0.
#
# Key parameters:
#   --use_v7_3_1_sigma_rho: Enable temporal regularization for Σ/ρ
#   --v7_3_1_lambda_tv_sigma: Weight for L_tv_sigma (default: 1e-4)
#   --v7_3_1_lambda_tv_rho: Weight for L_tv_rho (default: 1e-4)
#   --v7_3_1_lambda_cycle_sigma: Weight for L_cycle_sigma (default: 1e-4)
#   --v7_3_1_lambda_cycle_rho: Weight for L_cycle_rho (default: 1e-4)
#   --v7_3_1_sigma_rho_delta_fraction: δ = fraction * T̂ (default: 0.1)

# V7.3.1 with scale temporal regularization for case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.3.1_sigma_rho \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --use_v7_3_1_sigma_rho --v7_3_1_lambda_tv_sigma 1e-4 --v7_3_1_lambda_cycle_sigma 1e-4 \
  > train_v7.3.1_sigma_rho_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.3.1 with scale temporal regularization for case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7.3.1_sigma_rho \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --use_v7_3_1_sigma_rho --v7_3_1_lambda_tv_sigma 1e-4 --v7_3_1_lambda_cycle_sigma 1e-4 \
  > train_v7.3.1_sigma_rho_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#############################################################################
# V7.4 CANONICAL DECODE FOR SHAPE AND DENSITY (extension of V7.3.1)
#############################################################################
# V7.4 extends the backward field to decode not just canonical centers,
# but also canonical shape (log-scale) and density via learnable decode heads:
#   - D_b_pos: position residual (existing)
#   - D_b_shape: log-scale residual (new)
#   - D_b_dens: density residual (new)
#
# Canonical decoded parameters:
#   ŝ_canon(t) = s_base + D_b_shape(x(t), t)
#   ρ̂_canon(t) = ρ_base + D_b_dens(x(t), t)
#
# Losses:
#   L_cycle_canon_Sigma = |ŝ_canon(t+T̂) - ŝ_canon(t)| # periodic consistency
#   L_cycle_canon_rho = |ρ̂_canon(t+T̂) - ρ̂_canon(t)|
#   L_prior_Sigma = |ŝ_canon(t) - s_base| # anchor to base
#   L_prior_rho = |ρ̂_canon(t) - ρ_base|
#
# This gives backward field a 4D→3D factorization role for Σ/ρ.
#
# Key parameters:
#   --use_v7_4_canonical_decode: Enable canonical decode for Σ/ρ
#   --v7_4_lambda_cycle_canon_sigma: Weight for L_cycle_canon_Sigma (default: 1e-4)
#   --v7_4_lambda_cycle_canon_rho: Weight for L_cycle_canon_rho (default: 1e-4)
#   --v7_4_lambda_prior_sigma: Weight for L_prior_Sigma (default: 1e-4)
#   --v7_4_lambda_prior_rho: Weight for L_prior_rho (default: 1e-4)

# V7.4 with canonical decode for case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.4_canon_decode \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --use_v7_3_1_sigma_rho --v7_3_1_lambda_tv_sigma 1e-4 --v7_3_1_lambda_cycle_sigma 1e-4 \
  --use_v7_4_canonical_decode --v7_4_lambda_cycle_canon_sigma 1e-4 --v7_4_lambda_prior_sigma 1e-4 \
  > train_v7.4_canon_decode_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.4 with canonical decode for case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7.4_canon_decode \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --use_v7_3_1_sigma_rho --v7_3_1_lambda_tv_sigma 1e-4 --v7_3_1_lambda_cycle_sigma 1e-4 \
  --use_v7_4_canonical_decode --v7_4_lambda_cycle_canon_sigma 1e-4 --v7_4_lambda_prior_sigma 1e-4 \
  > train_v7.4_canon_decode_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#############################################################################
# V7.5 FULL TIME-WARP CONSISTENCY FOR SIGMA/RHO (extension of V7.4)
#############################################################################
# V7.5 extends time-warp consistency (V7.3) to shape and density:
#   G(t1) -> Backward decode -> Canonical state -> Forward timewarp -> G_hat(t2|t1)
# We require G_hat(t2|t1) matches G(t2) for shape and density (not just center).
#
# New forward timewarp decoders (used ONLY for loss, not rendering):
#   - shape_timewarp_head: canonical log-scale + time_embed -> predicted log-scale
#   - dens_timewarp_head: canonical density + time_embed -> predicted density
#
# Losses:
#   L_fw_Sigma = |s_hat_fw(t2|t1) - s(t2)|  (forward warp shape)
#   L_fw_rho = |rho_hat_fw(t2|t1) - rho(t2)|  (forward warp density)
#   L_bw_Sigma = |s_hat_bw(t1|t2) - s(t1)|  (backward warp shape, optional)
#   L_bw_rho = |rho_hat_bw(t1|t2) - rho(t1)|  (backward warp density, optional)
#   L_rt_Sigma, L_rt_rho: round-trip consistency (placeholder for future)
#
# Key parameters:
#   --use_v7_5_full_timewarp: Enable full time-warp for Σ/ρ
#   --v7_5_lambda_fw_sigma: Weight for forward warp shape (default: 1e-5)
#   --v7_5_lambda_fw_rho: Weight for forward warp density (default: 1e-5)
#   --v7_5_lambda_bw_sigma: Weight for backward warp shape (default: 0)
#   --v7_5_lambda_bw_rho: Weight for backward warp density (default: 0)
#   --v7_5_timewarp_delta_fraction: Δt = fraction * T̂ (default: 0.25)

# V7.5 with full time-warp for case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_v7.5_full_timewarp \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --use_v7_3_1_sigma_rho --v7_3_1_lambda_tv_sigma 1e-4 --v7_3_1_lambda_cycle_sigma 1e-4 \
  --use_v7_4_canonical_decode --v7_4_lambda_cycle_canon_sigma 1e-4 --v7_4_lambda_prior_sigma 1e-4 \
  --use_v7_5_full_timewarp --v7_5_lambda_fw_sigma 1e-5 --v7_5_lambda_fw_rho 1e-5 \
  > train_v7.5_full_timewarp_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.5 with full time-warp for case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_v7.5_full_timewarp \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --use_v7_3_1_sigma_rho --v7_3_1_lambda_tv_sigma 1e-4 --v7_3_1_lambda_cycle_sigma 1e-4 \
  --use_v7_4_canonical_decode --v7_4_lambda_cycle_canon_sigma 1e-4 --v7_4_lambda_prior_sigma 1e-4 \
  --use_v7_5_full_timewarp --v7_5_lambda_fw_sigma 1e-5 --v7_5_lambda_fw_rho 1e-5 \
  > train_v7.5_full_timewarp_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#############################################################################
# V7.5.1 FULL-STATE BIDIRECTIONAL TIME-WARP CONSISTENCY (extension of V7.5)
#############################################################################
# V7.5.1 extends V7.5 by enabling bidirectional time-warp consistency for full state:
#
# Key idea: Both forward and backward warp should be active for all state components.
#
# When use_v7_5_1_roundtrip_full_state=True:
#   - Center: Uses L_fw + L_bw (both active)
#   - Σ/ρ: Uses L_fw + L_bw (both active)
#
# Backward warp for Σ/ρ:
#   L_bw_Sigma = |s_bw(t1|t2) - s(t1)|, canonical(t2) warped to t1
#   L_bw_rho = |ρ_bw(t1|t2) - ρ(t1)|, canonical(t2) warped to t1
#
# Default weights when V7.5.1 is enabled:
#   - v7_5_lambda_bw_sigma = 1e-5 (shape backward)
#   - v7_5_lambda_bw_rho = 1e-5 (density backward)

# V7.5.1 with full-state bidirectional for case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --save_checkpoint --dirname dir_4d_case1_v7.5.1_bidirectional \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --use_v7_3_1_sigma_rho --v7_3_1_lambda_tv_sigma 1e-4 --v7_3_1_lambda_cycle_sigma 1e-4 \
  --use_v7_4_canonical_decode --v7_4_lambda_cycle_canon_sigma 1e-4 --v7_4_lambda_prior_sigma 1e-4 \
  --use_v7_5_full_timewarp --v7_5_lambda_fw_sigma 1e-5 --v7_5_lambda_fw_rho 1e-5 \
  --use_v7_5_1_roundtrip_full_state \
  > train_v7.5.1_bidirectional_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --save_checkpoint --dirname dir_4d_case1_v7.5.1_bidirectional \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --use_v7_3_1_sigma_rho --v7_3_1_lambda_tv_sigma 1e-4 --v7_3_1_lambda_cycle_sigma 1e-4 \
  --use_v7_4_canonical_decode --v7_4_lambda_cycle_canon_sigma 1e-4 --v7_4_lambda_prior_sigma 1e-4 \
  --use_v7_5_full_timewarp --v7_5_lambda_fw_sigma 1e-5 --v7_5_lambda_fw_rho 1e-5 \
  --use_v7_5_1_roundtrip_full_state \
  > train_v7.5.1_bidirectional_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V7.5.1 with full-state bidirectional for case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --save_checkpoint --dirname dir_4d_case2_v7.5.1_bidirectional \
  --use_v7_bidirectional_displacement --lambda_cycle 0.1 \
  --use_v7_2_consistency --v7_2_alpha_init 0.3 --v7_2_alpha_learnable \
  --v7_2_lambda_b_reg 1e-3 --v7_2_lambda_alpha_reg 1e-3 \
  --use_v7_2_1_cycle_canon --v7_2_1_lambda_cycle_canon 1e-3 \
  --use_v7_3_timewarp --v7_3_lambda_fw 1e-3 --v7_3_lambda_bw 1e-3 \
  --use_v7_3_1_sigma_rho --v7_3_1_lambda_tv_sigma 1e-4 --v7_3_1_lambda_cycle_sigma 1e-4 \
  --use_v7_4_canonical_decode --v7_4_lambda_cycle_canon_sigma 1e-4 --v7_4_lambda_prior_sigma 1e-4 \
  --use_v7_5_full_timewarp --v7_5_lambda_fw_sigma 1e-5 --v7_5_lambda_fw_rho 1e-5 \
  --use_v7_5_1_roundtrip_full_state \
  > train_v7.5.1_bidirectional_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

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
# S1 STATIC REWEIGHTING (orthogonal to v7/v8/v9/v10, improves static warm-up)
#############################################################################
# s1 is an orthogonal improvement that ONLY affects the static warm-up stage (coarse).
# It can be combined with ANY dynamic model (baseline, v7, v8, v9, v10).
#
# Core idea:
# - During static warm-up, some projections conflict with the "world is static" assumption
#   (e.g., projections at peak inspiration/expiration show significant motion blur)
# - s1 reweights projections to focus on static-consistent views
# - This yields a cleaner canonical Gaussian representation for the dynamic phase
#
# s1 has TWO methods:
#   1. "residual" (方案B): EMA-based, non-learnable weights based on residuals
#   2. "learnable" (方案A): Learnable per-projection weights α_j, w_j = sigmoid(α_j)
#
# s1 Common Parameters:
#   --use_static_reweighting            Enable s1 projection reliability-weighted static warm-up
#   --static_reweight_method METHOD     "residual" or "learnable" (default: residual)
#   --static_reweight_burnin_steps N    Steps before applying weights (default: 1500)
#
# s1 Residual Method (方案B) Parameters:
#   --static_reweight_ema_beta B        EMA momentum (default: 0.1, higher = faster adaptation)
#   --static_reweight_tau T             Temperature for weight computation (default: 0.3)
#   --static_reweight_weight_type exp   Weight function: "exp" for exp(-E/tau), "inv" for 1/(1+E/tau)
#
# s1 Learnable Method (方案A) Parameters:
#   --static_reweight_target_mean RHO   Target mean weight ρ (default: 0.85, range: 0.7-0.95)
#   --lambda_static_reweight_mean L     Weight for L_mean regularization (default: 0.05)
#   --static_reweight_lr LR             Learning rate for α_j (default: 0.01)

#-----------------------------------------------------------------------------
# s1 方案B (Residual-based): EMA residuals -> weights
#-----------------------------------------------------------------------------
# Basic s1 residual with default parameters
python train.py -s XXX/*.pickle --use_static_reweighting --static_reweight_method residual

# s1 + v7 main model (recommended combination)
python train.py -s XXX/*.pickle --use_static_reweighting --use_v7_bidirectional_displacement

# s1 with custom parameters
python train.py -s XXX/*.pickle --use_static_reweighting --static_reweight_burnin_steps 1500 --static_reweight_ema_beta 0.1 --static_reweight_tau 0.3

# Full s1 + v7 example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s1_v7 \
  --use_static_reweighting --static_reweight_method residual \
  --static_reweight_burnin_steps 1500 --static_reweight_ema_beta 0.1 --static_reweight_tau 0.3 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case1_s1_v7_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full s1 + v7 example for dir_4d_case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_s1_v7 \
  --use_static_reweighting --static_reweight_method residual \
  --static_reweight_burnin_steps 1500 --static_reweight_ema_beta 0.1 --static_reweight_tau 0.3 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case2_s1_v7_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# s1 + v9 example (combines static reweighting with low-rank motion modes)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s1_v9 \
  --use_static_reweighting --static_reweight_burnin_steps 1500 --static_reweight_tau 0.3 \
  --use_v7_bidirectional_displacement --use_low_rank_motion_modes --num_motion_modes 3 \
  --use_mode_regularization --lambda_mode 1e-4 --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case1_s1_v9_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# s1 + v10 example (combines static reweighting with adaptive gating)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s1_v10 \
  --use_static_reweighting --static_reweight_burnin_steps 1500 --static_reweight_tau 0.3 \
  --use_v7_bidirectional_displacement --use_low_rank_motion_modes --num_motion_modes 3 \
  --use_adaptive_gating --use_trajectory_smoothing --lambda_traj 1e-3 \
  --use_mode_regularization --lambda_mode 1e-4 --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case1_s1_v10_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#-----------------------------------------------------------------------------
# s1 方案A (Learnable): Learnable per-projection weights α_j, w_j = sigmoid(α_j)
#-----------------------------------------------------------------------------
# The learnable method jointly optimizes α_j with the Gaussians.
# Projections with high residuals naturally get lower α_j -> lower w_j.
# L_mean = (mean(w_j) - ρ)² regularization prevents weights from collapsing.

# Basic s1 learnable with default parameters
python train.py -s XXX/*.pickle --use_static_reweighting --static_reweight_method learnable

# s1 learnable with custom parameters
python train.py -s XXX/*.pickle --use_static_reweighting --static_reweight_method learnable \
  --static_reweight_target_mean 0.85 --lambda_static_reweight_mean 0.05 --static_reweight_burnin_steps 1000

# s1 learnable + v7 main model
python train.py -s XXX/*.pickle --use_static_reweighting --static_reweight_method learnable \
  --use_v7_bidirectional_displacement

# Full s1 learnable + baseline example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s1_learnable \
  --use_static_reweighting --static_reweight_method learnable \
  --static_reweight_target_mean 0.85 --lambda_static_reweight_mean 0.05 \
  --static_reweight_burnin_steps 1000 --static_reweight_lr 0.01 \
  > train_dir_4d_case1_s1_learnable_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full s1 learnable + v7 example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s1_learnable_v7 \
  --use_static_reweighting --static_reweight_method learnable \
  --static_reweight_target_mean 0.85 --lambda_static_reweight_mean 0.05 \
  --static_reweight_burnin_steps 1000 --static_reweight_lr 0.01 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case1_s1_learnable_v7_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full s1 learnable + v7 example for dir_4d_case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_s1_learnable_v7 \
  --use_static_reweighting --static_reweight_method learnable \
  --static_reweight_target_mean 0.85 --lambda_static_reweight_mean 0.05 \
  --static_reweight_burnin_steps 1000 --static_reweight_lr 0.01 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_dir_4d_case2_s1_learnable_v7_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#-----------------------------------------------------------------------------
# s1 Ablation Experiments: Compare methods
#-----------------------------------------------------------------------------
# Run these experiments to measure s1's impact on static warm-up quality:

# Experiment 1: Baseline (no s1)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_baseline_no_s1 \
  > train_baseline_no_s1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Experiment 2: Baseline + s1 residual (方案B)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_baseline_s1_residual \
  --use_static_reweighting --static_reweight_method residual \
  --static_reweight_burnin_steps 1500 --static_reweight_ema_beta 0.1 --static_reweight_tau 0.3 \
  > train_baseline_s1_residual_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Experiment 3: Baseline + s1 learnable (方案A)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_baseline_s1_learnable \
  --use_static_reweighting --static_reweight_method learnable \
  --static_reweight_target_mean 0.85 --lambda_static_reweight_mean 0.05 \
  --static_reweight_burnin_steps 1000 --static_reweight_lr 0.01 \
  > train_baseline_s1_learnable_$(date +%Y%m%d_%H%M%S).log 2>&1 &


#############################################################################
# S2 PHASE-GATED STATIC CANONICAL WARM-UP (orthogonal to v7/v8/v9/v10 and s1)
#############################################################################
# s2 is an orthogonal improvement that ONLY affects the static warm-up stage (coarse).
# It can be combined with ANY dynamic model (baseline, v7, v8, v9, v10) and with s1.
#
# Core idea:
# - During static warm-up, s2 learns an implicit breathing period and canonical phase
# - Uses a circular Gaussian window to weight projections by their phase proximity to canonical
# - Projections near the "canonical phase" contribute more to static training
# - This yields a cleaner canonical representing ONE breathing phase, not time-averaged blur
#
# s2 Learnable parameters:
#   - τ_s2 (log-period): T_s2 = exp(τ_s2)
#   - ψ_s2 (phase offset): aligns acquisition time with breathing phase
#   - φ_c (canonical phase): the target phase in [-π, π)
#   - ξ_s2 (log-window width): σ_φ = exp(ξ_s2)
#
# For each projection with time t_j:
#   - φ_j = wrap(2π * t_j / T_s2 + ψ_s2)
#   - w_j = exp(-d_circ(φ_j, φ_c)² / (2σ_φ²))
#
# s2 Parameters:
#   --use_phase_gated_static          Enable s2 phase-gated static canonical warm-up
#   --static_phase_burnin_steps N     Burn-in steps before phase gating (default: 1500)
#   --static_phase_sigma_target σ     Target window width in radians (default: 0.6, ~34°)
#   --lambda_static_phase_window λ    Weight for L_win regularization (default: 0.01)
#   --static_phase_lr LR              Learning rate for phase params (default: 0.005)

#-----------------------------------------------------------------------------
# s2 Basic Usage
#-----------------------------------------------------------------------------
# Basic s2 with default parameters
python train.py -s XXX/*.pickle --use_phase_gated_static

# s2 with custom parameters
python train.py -s XXX/*.pickle --use_phase_gated_static \
  --static_phase_burnin_steps 1500 --static_phase_sigma_target 0.6 \
  --lambda_static_phase_window 0.01 --static_phase_lr 0.005

# s2 + v7 main model
python train.py -s XXX/*.pickle --use_phase_gated_static --use_v7_bidirectional_displacement

#-----------------------------------------------------------------------------
# s2 Full Examples
#-----------------------------------------------------------------------------
# Full s2 + baseline example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s2 \
  --use_phase_gated_static --static_phase_burnin_steps 1500 \
  --static_phase_sigma_target 0.6 --lambda_static_phase_window 0.01 \
  > train_s2_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full s2 + v7 example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s2_v7 \
  --use_phase_gated_static --static_phase_burnin_steps 1500 \
  --static_phase_sigma_target 0.6 --lambda_static_phase_window 0.01 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_s2_v7_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full s2 + v7 example for dir_4d_case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_s2_v7 \
  --use_phase_gated_static --static_phase_burnin_steps 1500 \
  --static_phase_sigma_target 0.6 --lambda_static_phase_window 0.01 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_s2_v7_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#-----------------------------------------------------------------------------
# s1 + s2 Combined (both static improvements together)
#-----------------------------------------------------------------------------
# s1 residual + s2: combines EMA-based reweighting with phase gating
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s1_s2 \
  --use_static_reweighting --static_reweight_method residual \
  --static_reweight_burnin_steps 1500 --static_reweight_tau 0.3 \
  --use_phase_gated_static --static_phase_burnin_steps 1500 \
  --static_phase_sigma_target 0.6 --lambda_static_phase_window 0.01 \
  > train_s1_s2_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# s1 + s2 + v7: full combination
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s1_s2_v7 \
  --use_static_reweighting --static_reweight_method residual \
  --static_reweight_burnin_steps 1500 --static_reweight_tau 0.3 \
  --use_phase_gated_static --static_phase_burnin_steps 1500 \
  --static_phase_sigma_target 0.6 --lambda_static_phase_window 0.01 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_s1_s2_v7_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &


#############################################################################
# S3 DUAL-REPRESENTATION STATIC WARM-UP (Gaussian + Voxel Co-Training)
#############################################################################
# s3 is an orthogonal improvement that ONLY affects the static warm-up stage (coarse).
# It can be combined with ANY dynamic model (baseline, v7, v8, v9, v10) and with s1/s2.
#
# Core idea:
# - During static warm-up, s3 co-trains canonical Gaussians with a low-resolution 3D voxel volume V
# - Both representations are supervised by projection loss
# - A 3D distillation loss constrains Gaussians to align with the smoother V structure
# - 3D TV regularization ensures V remains smooth
# - After static warm-up, V is discarded; Gaussians retain improved 3D structure
#
# s3 Loss:
#   L_s3 = λ_G * L_G + λ_V * L_V + λ_distill * L_distill + λ_VTV * L_VTV
#
# s3 Parameters:
#   --use_s3_dual_static_volume        Enable s3 dual-representation static warm-up
#   --s3_voxel_resolution N            Voxel resolution D=H=W (default: 64)
#   --lambda_s3_G λ                    Weight for Gaussian projection loss (default: 1.0)
#   --lambda_s3_V λ                    Weight for Voxel projection loss (default: 0.2)
#   --lambda_s3_distill λ              Weight for distillation loss (default: 1.0)
#   --lambda_s3_VTV λ                  Weight for 3D TV regularization (default: 1e-4)
#   --s3_num_distill_samples N         3D samples for distillation per step (default: 20000)
#   --s3_num_ray_samples N             Ray samples for voxel rendering (default: 64)
#   --s3_voxel_lr LR                   Learning rate for voxel volume (default: 0.01)
#   --s3_downsample_factor N           Downsample factor for voxel projection (default: 2)

#-----------------------------------------------------------------------------
# s3 Basic Usage
#-----------------------------------------------------------------------------
# Basic s3 with default parameters
python train.py -s XXX/*.pickle --use_s3_dual_static_volume

# s3 with custom parameters
python train.py -s XXX/*.pickle --use_s3_dual_static_volume \
  --s3_voxel_resolution 64 --lambda_s3_G 1.0 --lambda_s3_V 0.2 \
  --lambda_s3_distill 1.0 --lambda_s3_VTV 1e-4

# s3 + v7 main model
python train.py -s XXX/*.pickle --use_s3_dual_static_volume --use_v7_bidirectional_displacement

#-----------------------------------------------------------------------------
# s3 Full Examples
#-----------------------------------------------------------------------------
# Full s3 + baseline example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s3 \
  --use_s3_dual_static_volume --s3_voxel_resolution 64 \
  --lambda_s3_G 1.0 --lambda_s3_V 0.2 --lambda_s3_distill 1.0 --lambda_s3_VTV 1e-4 \
  > train_s3_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full s3 + v7 example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s3_v7 \
  --use_s3_dual_static_volume --s3_voxel_resolution 64 \
  --lambda_s3_G 1.0 --lambda_s3_V 0.2 --lambda_s3_distill 1.0 --lambda_s3_VTV 1e-4 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_s3_v7_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full s3 + v7 example for dir_4d_case2
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_s3_v7 \
  --use_s3_dual_static_volume --s3_voxel_resolution 64 \
  --lambda_s3_G 1.0 --lambda_s3_V 0.2 --lambda_s3_distill 1.0 --lambda_s3_VTV 1e-4 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_s3_v7_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#-----------------------------------------------------------------------------
# s1 + s2 + s3 Combined (all static improvements together)
#-----------------------------------------------------------------------------
# s1 residual + s2 + s3: combines EMA-based reweighting, phase gating, and voxel co-training
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s1_s2_s3 \
  --use_static_reweighting --static_reweight_method residual \
  --static_reweight_burnin_steps 1500 --static_reweight_tau 0.3 \
  --use_phase_gated_static --static_phase_burnin_steps 1500 \
  --static_phase_sigma_target 0.6 --lambda_static_phase_window 0.01 \
  --use_s3_dual_static_volume --s3_voxel_resolution 64 \
  --lambda_s3_G 1.0 --lambda_s3_V 0.2 --lambda_s3_distill 1.0 --lambda_s3_VTV 1e-4 \
  > train_s1_s2_s3_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# s1 + s2 + s3 + v7: full combination of all static and dynamic improvements
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s1_s2_s3_v7 \
  --use_static_reweighting --static_reweight_method residual \
  --static_reweight_burnin_steps 1500 --static_reweight_tau 0.3 \
  --use_phase_gated_static --static_phase_burnin_steps 1500 \
  --static_phase_sigma_target 0.6 --lambda_static_phase_window 0.01 \
  --use_s3_dual_static_volume --s3_voxel_resolution 64 \
  --lambda_s3_G 1.0 --lambda_s3_V 0.2 --lambda_s3_distill 1.0 --lambda_s3_VTV 1e-4 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_s1_s2_s3_v7_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &


#############################################################################
# S4 TEMPORAL-ENSEMBLE GUIDED STATIC WARM-UP
#############################################################################
# s4 is an orthogonal improvement that ONLY affects the static warm-up stage (coarse).
# It can be combined with ANY dynamic model (baseline, v7, v8, v9, v10) and with s1/s2/s3.
#
# Core idea:
# - During static warm-up, s4 uses externally provided pseudo-supervision:
#   - Average CT volume (V_avg): time-averaged 3D volume from traditional reconstruction
#   - Average projections (optional): time-averaged 2D projections per view
# - These serve as "teachers" to guide canonical Gaussians toward a more robust representation
# - After static warm-up, V_avg is no longer used; dynamic stage is unchanged
#
# s4 Loss:
#   L_s4 = λ_G * L_G + λ_vol * L_vol + λ_proj_avg * L_proj_avg
#   where:
#   - L_G = projection reconstruction loss (existing)
#   - L_vol = |σ_G(x) - V_avg(x)| (3D volume distillation)
#   - L_proj_avg = projection distillation with average projections (optional)
#
# s4 Parameters:
#   --use_s4_temporal_ensemble_static   Enable s4 temporal-ensemble static warm-up
#   --s4_avg_ct_path PATH               Path to average CT volume (.npy, .npz, .pt)
#   --s4_avg_proj_path PATH             Path to average projections (optional)
#   --lambda_s4_G λ                     Weight for Gaussian projection loss (default: 1.0)
#   --lambda_s4_vol λ                   Weight for volume distillation loss (default: 0.3)
#   --lambda_s4_proj_avg λ              Weight for projection distillation (default: 0.0)
#   --s4_num_vol_samples N              Number of 3D samples for volume distillation (default: 10000)
#
# Preparing Average CT Volume (V_avg):
#   Option 1: Traditional FDK/FBP reconstruction treating all projections as static
#   Option 2: Time-average of baseline 4D reconstruction: V_avg = mean_t(σ(x, t))
#   Save as .npy file with shape [D, H, W] (e.g., 256x256x256)

#-----------------------------------------------------------------------------
# s4 Basic Usage
#-----------------------------------------------------------------------------
# Basic s4 with only average CT volume
python train.py -s XXX/*.pickle --use_s4_temporal_ensemble_static \
  --s4_avg_ct_path path/to/avg_ct.npy \
  --lambda_s4_G 1.0 --lambda_s4_vol 0.3

# s4 + v7 main model
python train.py -s XXX/*.pickle --use_s4_temporal_ensemble_static \
  --s4_avg_ct_path path/to/avg_ct.npy \
  --lambda_s4_G 1.0 --lambda_s4_vol 0.3 \
  --use_v7_bidirectional_displacement

#-----------------------------------------------------------------------------
# s4 Full Examples
#-----------------------------------------------------------------------------
# Full s4 + baseline example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s4 \
  --use_s4_temporal_ensemble_static \
  --s4_avg_ct_path /path/to/avg_ct_case1.npy \
  --lambda_s4_G 1.0 --lambda_s4_vol 0.3 --s4_num_vol_samples 10000 \
  > train_s4_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Full s4 + v7 example for dir_4d_case1
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s4_v7 \
  --use_s4_temporal_ensemble_static \
  --s4_avg_ct_path /path/to/avg_ct_case1.npy \
  --lambda_s4_G 1.0 --lambda_s4_vol 0.3 --s4_num_vol_samples 10000 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_s4_v7_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#-----------------------------------------------------------------------------
# s4_1: s4 with average projections (CT + Projection Distillation)
#-----------------------------------------------------------------------------
# s4_1 uses both average CT volume and average projections for distillation
# Average projections are generated by forward projecting V_avg at each training angle
#
# Generate avg projections first:
#   python generate_avg_projections.py \
#     --pickle data/dir_4d_case1.pickle \
#     --avg_ct data/avg_ct_case1.npy \
#     --output data/avg_projections_case1.npy

# s4_1 case1: Baseline + avg_ct + avg_projections
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s4_1_baseline \
  --use_s4_temporal_ensemble_static \
  --s4_avg_ct_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/avg_ct_case1.npy \
  --s4_avg_proj_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/avg_projections_case1.npy \
  --lambda_s4_G 1.0 --lambda_s4_vol 0.3 --lambda_s4_proj_avg 0.1 \
  --s4_num_vol_samples 10000 \
  > train_s4_1_baseline_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# s4_1 case2: Baseline + avg_ct + avg_projections
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_s4_1_baseline \
  --use_s4_temporal_ensemble_static \
  --s4_avg_ct_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/avg_ct_case2.npy \
  --s4_avg_proj_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/avg_projections_case2.npy \
  --lambda_s4_G 1.0 --lambda_s4_vol 0.3 --lambda_s4_proj_avg 0.1 \
  --s4_num_vol_samples 10000 \
  > train_s4_1_baseline_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#-----------------------------------------------------------------------------
# s4_2: s4 with avg CT initialization (Full s4 Feature Set)
#-----------------------------------------------------------------------------
# s4_2 combines all s4 features:
# - avg_ct volume distillation
# - avg_projections distillation (optional)
# - Gaussians initialization from avg_ct (instead of random/FDK)
#
# s4_2 Parameters:
#   --use_s4_avg_ct_init                Enable initialization from avg CT
#   --s4_avg_ct_init_thresh THRESH      Threshold for high-density voxels (default: 0.1)
#   --s4_avg_ct_init_num_gaussians N    Number of Gaussians to initialize (default: 50000)
#   --s4_avg_ct_init_method METHOD      Sampling method: "fps" or "random" (default: random)
#   --s4_avg_ct_init_density_rescale R  Density rescaling factor (default: 0.15)

# s4_2 case1: Baseline + full s4 features (avg_ct init + distillation)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s4_2_baseline \
  --use_s4_temporal_ensemble_static \
  --s4_avg_ct_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/avg_ct_case1.npy \
  --s4_avg_proj_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/avg_projections_case1.npy \
  --lambda_s4_G 1.0 --lambda_s4_vol 0.3 --lambda_s4_proj_avg 0.1 \
  --s4_num_vol_samples 10000 \
  --use_s4_avg_ct_init --s4_avg_ct_init_thresh 0.1 --s4_avg_ct_init_num_gaussians 50000 \
  --s4_avg_ct_init_method random --s4_avg_ct_init_density_rescale 0.15 \
  > train_s4_2_baseline_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# s4_2 case2: Baseline + full s4 features (avg_ct init + distillation)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_s4_2_baseline \
  --use_s4_temporal_ensemble_static \
  --s4_avg_ct_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/avg_ct_case2.npy \
  --s4_avg_proj_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/avg_projections_case2.npy \
  --lambda_s4_G 1.0 --lambda_s4_vol 0.3 --lambda_s4_proj_avg 0.1 \
  --s4_num_vol_samples 10000 \
  --use_s4_avg_ct_init --s4_avg_ct_init_thresh 0.1 --s4_avg_ct_init_num_gaussians 50000 \
  --s4_avg_ct_init_method random --s4_avg_ct_init_density_rescale 0.15 \
  > train_s4_2_baseline_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#-----------------------------------------------------------------------------
# s5: 4D Dynamic-Aware Multi-Phase FDK Point Cloud Initialization
#-----------------------------------------------------------------------------
# s5 is a new initialization strategy that uses multi-phase FDK to generate
# 4D-aware Gaussian initialization:
# - Divides projections into P phases based on acquisition time
# - Computes V_ref (reference phase), V_avg (temporal mean), V_var (temporal variance)
# - Uses V_var to identify dynamic vs static regions
# - Samples more Gaussians in dynamic regions with smaller initial scales
# - Samples fewer Gaussians in static regions with larger initial scales
#
# s5 only affects initialization, not training. It can be combined with
# any dynamic model (v7, v9, etc.) and static warm-up methods (s1-s4).
#
# Step 1: Generate s5 init file (offline, run once per case)
python tools/build_init_s5_4d.py \
  --input /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --output /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/init_s5_4d_case1.npy \
  --s5_num_phases 3 \
  --s5_num_points 50000 \
  --s5_static_weight 0.7 \
  --s5_dynamic_weight 0.3 \
  --s5_density_exponent 1.5 \
  --s5_var_exponent 1.0

python tools/build_init_s5_4d.py \
  --input /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --output /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/init_s5_4d_case2.npy \
  --s5_num_phases 3 \
  --s5_num_points 50000 \
  --s5_static_weight 0.7 \
  --s5_dynamic_weight 0.3

# Step 2: Train with s5 initialization
# s5 case1: Baseline + s5 4D-aware initialization
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s5_baseline \
  --use_s5_4d_init \
  --s5_init_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/init_s5_4d_case1.npy \
  > train_s5_baseline_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# s5 case2: Baseline + s5 4D-aware initialization
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case2_s5_baseline \
  --use_s5_4d_init \
  --s5_init_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/init_s5_4d_case2.npy \
  > train_s5_baseline_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# s5 can be combined with other improvements (e.g., s4 distillation, v7 dynamics)
# Example: s5 init + s4 distillation + v7 dynamics
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --dirname dir_4d_case1_s5_s4_v7 \
  --use_s5_4d_init \
  --s5_init_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/init_s5_4d_case1.npy \
  --use_s4_temporal_ensemble_static \
  --s4_avg_ct_path /root/autodl-tmp/4dctgs/x2-gaussian-main/x2-gaussian-main-origin/data/avg_ct_case1.npy \
  --lambda_s4_G 1.0 --lambda_s4_vol 0.3 \
  --use_v7_bidirectional_displacement --lambda_inv 0.1 --lambda_cycle 0.01 \
  > train_s5_s4_v7_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &


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
  --use_low_rank_motion_modes --num_motion_modes 2 \
  --use_mode_regularization --lambda_mode 1e-4 \
  --lambda_inv 0.1 --lambda_cycle 0.1 \
  > train_dir_4d_case2_v9_lowrank_motion2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

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

