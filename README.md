# [ICCV 2025] X2-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction

## [Project Page](https://x2-gaussian.github.io/) | [Paper](https://arxiv.org/abs/2503.21779)

| Animation 1 | Animation 2 | Animation 3 |
|---|---|---|
| ![Animation 1](./media/gif1.gif) | ![Animation 2](./media/gif2.gif) | ![Animation 3](./media/gif3.gif) |

![Tidal volume curve](./media/tidal.jpg)

We achieve genuine continuous-time CT reconstruction without phase-binning. The figure illustrates temporal variations of lung volume in 4D CT reconstructed by our X2-Gaussian.

![Teaser](./media/teaser.jpg)

X2-Gaussian demonstrates state-of-the-art reconstruction performance.

## News

* 2025.10.27: Datasets have been released on [HuggingFace (X2GS)](https://huggingface.co/datasets/vortex778/X2GS). Welcome to have a try!
* 2025.10.17: Training code has been released.
* 2025.06.26: Our work has been accepted to ICCV 2025.
* 2025.03.27: Our paper is available on [arXiv (2503.21779)](https://arxiv.org/abs/2503.21779).

## TODO

* [ ] Release more detailed instructions.
* [ ] Release data generation code.
* [ ] Release evaluation code.
* [ ] Release visualizaton code.

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

You can download datasets used in our paper from [HuggingFace (X2GS)](https://huggingface.co/datasets/vortex778/X2GS). We use [NAF](https://github.com/Ruyi-Zha/naf_cbct) format data (`*.pickle`) used in [SAX-NeRF](https://github.com/caiyuanhao1998/SAX-NeRF).

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

```sh
nohup /root/miniconda3/envs/x2_gaussian/bin/python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --save_checkpoint \
  --dirname dir_4d_case2_baseline \
  > log/train_baseline_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 初始版本的XCAT和S01_004_256_60 baseline 把lp和ltv设置为0了，下面是正确版本

nohup python train.py -s data/XCAT.pickle --save_iterations 30000 50000 --save_checkpoint --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 > log/train_baseline_XCAT_$(date +%Y%m%d_%H%M%S).log 2>&1 &
nohup python train.py -s data/S01_004_256_60.pickle --save_iterations 30000 50000 --save_checkpoint --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 > log/train_baseline_S01_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 \
  --lambda_prior 0.0 --lambda_tv 0.0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_v5_alpha0.99 \
  > log/train_physx_boosted_v5_alpha0.99_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0.01 \
  --lambda_balance 0.0 \
  --lambda_prior 0.0 --lambda_tv 0.0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_v5_alpha0.99_lr0.01 \
  > log/train_physx_boosted_v5_alpha0.99_lr0.01_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# no mask

nohup python train.py -s data/dir_4d_case1_sparse20.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --iterations 50000 \
  --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname DEBUG \
  > log/train_sparse20_physx_nomask_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  >
# v10

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 \
  --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.25 \
  --use_decoupled_mask \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_v10 \
  > log/train_physx_boosted_v10_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
  >
# v16

nohup python train.py -s data/dir_4d_case1_sparse20.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --use_spatiotemporal_mask --lambda_lagbert 0.1 \
  --st_window_size 1 --st_time_delta 0.1 --st_mask_ratio 0.5 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_sparse20_physx_boosted_v16 \
  > log/train_sparse20_physx_boosted_v16_w1_mask50_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V16.1 和 V16.2: 带两个修复

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --use_spatiotemporal_mask --lambda_lagbert 0.1 \
  --st_window_size 1 --st_time_delta 0.1 --st_mask_ratio 0.5 \
  --st_coupled_render --st_mask_embed_scale 0.1\
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_v16_3 \
  > log/train_physx_boosted_v16_3_w1_mask50_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ========================================================

# M1: Uncertainty-Gated Adaptive Fusion

# ========================================================

# M1 replaces fixed α with adaptive β(x,t) based on Eulerian uncertainty

# Formula: Φ(x,t) = (1-β) · Φ_L(x,t) + β(x,t) · Φ_E(x,t)

#

# Key features

# - Lagrangian (Anchor) is the backbone (always contributes)

# - Eulerian (HexPlane) is the residual corrector (contributes when confident)

# - β is learned based on HexPlane's predicted uncertainty s_E = log(σ²_E)

# M1-Bayes v3: β = σ_L² / (σ_L² + σ_E²), σ_L²=0.01 → β≈0.01 (matches V5 α=0.99)

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --fusion_mode uncertainty_gated \
  --gate_mode bayes --sigma_L2 0.01 \
  --m1_lambda_gate 0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_m1_bayes_v3 \
  > log/train_physx_boosted_m1_bayes_v3_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# M1-Sigmoid v3: β = sigmoid((τ - s_E) / λ), τ=-4.6 → β≈0.01 (matches V5 α=0.99)

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --fusion_mode uncertainty_gated \
  --gate_mode sigmoid --gate_tau -4.6 --gate_lambda 1.0 \
  --m1_lambda_gate 0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_m1_sigmoid_v3 \
  > log/train_physx_boosted_m1_sigmoid_v3_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# M1 Visualization: Generate β(x,t) contribution maps

python scripts/visualize_beta.py \
  --checkpoint output/dir_4d_case1_physx_boosted_m1_bayes/ckpt/chkpnt50000.pth \
  --time 0.5 --output output/m1_viz

# ========================================================

# M2: Bounded Learnable Perturbation (ICML formulation)

# ========================================================

# M2 reformulates the fusion as "Base + Bounded Perturbation"

# Φ(x,t) = Φ_L(x,t) + ε · tanh(Φ_E(x,t))

#

# Key advantages over V5's fixed α=0.99

# - Lagrangian (Anchor) is the FULL structural base (100%)

# - Eulerian (HexPlane) is a learned bounded perturbation

# - ε is learnable but bounded: ε = ε_max · sigmoid(ρ)

# - tanh bounds perturbation magnitude, preventing shortcuts

#

# Initialization matches V5's α=0.99 behavior

# ε_init = 0.01 means ~1% scale perturbation from Eulerian

#

# Parameters

# --fusion_mode bounded_perturb  # Enable M2 mode

# --eps_max 0.02                 # Maximum ε bound (default 2%)

# --eps_init 0.01                # Initial ε (default 1%, matches V5)

# --use_tanh                     # Use tanh to bound perturbation (default True)

# ========================================================

# M1.3: Optimized Fixed Alpha (based on M1.2 findings)

# ========================================================

# M1.2 g005 discovered that HexPlane weight 1.5% > 1%

# M1.3 directly uses α=0.985 (hex=1.5%) as the new baseline

# M1.3a: Fixed α=0.985

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.985 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --fusion_mode fixed_alpha \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_m1_3a_alpha0.985 \
  > log/train_physx_boosted_m1_3a_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# M1.3b: Learnable α starting from 0.985

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.985 --balance_lr 0.0001 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --fusion_mode fixed_alpha \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_m1_3b_alpha0.985_lr \
  > log/train_physx_boosted_m1_3b_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ========================================================

# M2.05: Learnable Weighted Average (Fixed from M2)

# ========================================================

# M2 FAILED (-5.5 dB) because

# 1. Formula dx=dx_anchor+ε*tanh(dx_hex) differs from V5's weighted avg

# 2. tanh compressed the HexPlane signal

# 3. Anchor got 100% weight instead of 99%

#

# M2.05 FIX: Return to weighted average structure

# dx = (1-ε)*dx_anchor + ε*dx_hex

# NO tanh, ε_init=0.015 (based on M1.2 findings)

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --fusion_mode bounded_perturb \
  --eps_max 0.03 --eps_init 0.015 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_m2_05 \
  > log/train_physx_boosted_m2_05_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# ========================================================

# M2.1: Trust-Region Schedule (freeze_rho / warmup_cap)

# ========================================================

# Prevents optimizer shortcuts by freezing ρ early or warming up ε

# schedule_mode: none / freeze_rho / warmup_cap

# M2.1-a: freeze_rho (recommended)

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --fusion_mode bounded_perturb \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --eps_max 0.02 --eps_init 0.01 \
  --dirname dir_4d_case1_m2_1a_freeze \
  > log/train_m2_1a_freeze.log 2>&1 &

# ========================================================

# M2.2: Residual Normalization (tanh / rmsnorm / unitnorm)

# ========================================================

# "Residual normalization makes ε a true trust-region radius by

# preventing magnitude leakage from the Eulerian stream."

# M2.2-a: rmsnorm (recommended)

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --fusion_mode bounded_perturb \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --eps_max 0.02 --eps_init 0.01 \
  --residual_mode rmsnorm --norm_eps 1e-6 \
  --dirname dir_4d_case1_m2_2a_rmsnorm \
  > log/train_m2_2a_rmsnorm.log 2>&1 &

# M2.2-b: unitnorm

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --fusion_mode bounded_perturb \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --eps_max 0.02 --eps_init 0.01 \
  --residual_mode unitnorm --norm_eps 1e-6 \
  --dirname dir_4d_case1_m2_2b_unitnorm \
  > log/train_m2_2b_unitnorm.log 2>&1 &

# ========================================================

# M3: Low-Frequency Leakage Penalty

# ========================================================

# "Low-frequency leakage regularization prevents the Eulerian stream

# from explaining global motion, reserving it for high-frequency

# corrective details around the Lagrangian manifold."

# M3: kNN mean LP regularization (基于 M2.1 最佳配置)

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --eps_max 0.03 --eps_init 0.015 \
  --lp_enable --lambda_lp 0.01 --lp_mode knn_mean \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --dirname dir_4d_case1_m3_lp_knn \
  > log/train_m3_lp_knn.log 2>&1 &

# ========================================================

# M4: Subspace Decoupling Regularization

# ========================================================

# "Subspace decoupling regularization discourages the Eulerian residual

# from aligning with the Lagrangian deformation responses, forcing it

# to model complementary details rather than shortcuts."

# M4: velocity correlation decoupling (基于 M2.1 最佳配置)

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --eps_max 0.03 --eps_init 0.015 \
  --decouple_enable --lambda_decouple 0.01 --decouple_mode velocity_corr \
  --decouple_dt 0.02 --decouple_subsample 2048 --decouple_stopgrad_L \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --dirname dir_4d_case1_m4_decouple_vel \
  > log/train_m4_decouple_vel.log 2>&1 &

# M4: stochastic jacobian correlation decoupling (基于 M2.1 最佳配置)

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --eps_max 0.03 --eps_init 0.015 \
  --decouple_enable --lambda_decouple 0.01 --decouple_mode stochastic_jacobian_corr \
  --decouple_num_dirs 1 --decouple_subsample 2048 --decouple_stopgrad_L \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --dirname dir_4d_case1_m4_decouple_jac \
  > log/train_m4_decouple_jac.log 2>&1 &

# 1. 周期扰动数据集 - Baseline

nohup python train.py -s data/dir_4d_case1_noise0.15.pickle \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 30000 50000 --save_checkpoint \
  > log/train_noise0.15_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 2. 周期扰动数据集 - PhysX-Boosted

nohup python train.py -s data/dir_4d_case1_noise0.15.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_prior 0.0 --lambda_tv 0.0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 30000 50000 --save_checkpoint \
  > log/train_noise0.15_physx_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 3. 稀疏视角数据集 - Baseline

nohup python train.py -s data/dir_4d_case1_sparse50.pickle \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 30000 50000 --save_checkpoint \
  > log/train_sparse50_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 4. 稀疏视角数据集 - PhysX-Boosted

nohup python train.py -s data/dir_4d_case1_sparse50.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_prior 0.0 --lambda_tv 0.0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 30000 50000 --save_checkpoint \
  > log/train_sparse50_physx_$(date +%Y%m%d_%H%M%S).log 2>&1 &

```

### PhysX-Gaussian: Anchor-based Spacetime Transformer

PhysX-Gaussian is a new variant that replaces the HexPlane + MLP deformation field with an **Anchor-based Spacetime Transformer**. It learns physical traction relationships between anatomical structures via masked modeling (BERT-style), enabling generalization to irregular breathing patterns.

#### Key Innovation

* Original X²-Gaussian: relies on implicit periodic fitting, poor generalization to irregular breathing
* PhysX-Gaussian: uses physical anchors + attention to infer deformation even with irregular breathing

#### Architecture

1. **FPS Sampling**: Select `num_anchors` points as physical anchors from initial point cloud
2. **KNN Binding**: Each Gaussian binds to `anchor_k` nearest anchors (skinning weights)
3. **Spacetime Transformer**: Anchors attend to each other with time encoding
4. **Masked Modeling**: Randomly mask `mask_ratio` of anchors during training (BERT-style)
5. **Interpolation**: Gaussian displacement = weighted sum of bound anchor displacements

#### Training PhysX-Gaussian

```sh
# PhysX-Gaussian Training (Anchor-based Transformer)
nohup /root/miniconda3/envs/x2_gaussian/bin/python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --save_checkpoint \
  --use_anchor_deformation \
  --num_anchors 1024 \
  --anchor_k 10 \
  --mask_ratio 0.25 \
  --transformer_dim 64 \
  --transformer_heads 4 \
  --transformer_layers 2 \
  --lambda_phys 0.1 \
  --lambda_anchor_smooth 0.01 \
  --dirname dir_4d_case2_physx_gaussian \
  > log/train_physx_gaussian_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### PhysX-Gaussian Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--use_anchor_deformation` | False | Master switch to enable PhysX-Gaussian |
| `--num_anchors` | 1024 | Number of FPS-sampled physical anchors |
| `--anchor_k` | 10 | Number of nearest anchors each Gaussian binds to |
| `--mask_ratio` | 0.25 | Ratio of anchors to mask during training (BERT-style) |
| `--transformer_dim` | 64 | Hidden dimension of spacetime transformer |
| `--transformer_heads` | 4 | Number of attention heads |
| `--transformer_layers` | 2 | Number of transformer encoder layers |
| `--lambda_phys` | 0.1 | Weight for physics completion loss L_phys |
| `--lambda_anchor_smooth` | 0.01 | Weight for anchor motion smoothness |
| `--phys_warmup_steps` | 2000 | Steps before applying L_phys |

#### Fallback to Original Method

When `--use_anchor_deformation` is not specified or set to False, the model falls back to the original X²-Gaussian behavior (HexPlane + MLP deformation).

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

##### 方式一：从 Checkpoint 继续（推荐，完美恢复）

```sh
# 从 checkpoint 继续训练到 50000 iterations
# checkpoint 包含：模型权重 + 优化器状态 + 当前 iteration
python train.py -s XXX/*.pickle --iterations 50000 \
  --save_iterations 50000 \
  --save_checkpoint \
  --dirname my_experiment_50k \
  --start_checkpoint output/xxx/ckpt/chkpnt30000.pth
```

##### 方式二：从保存的模型继续（仅恢复模型权重）

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
