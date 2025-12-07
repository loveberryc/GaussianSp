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
nohup /root/miniconda3/envs/x2_gaussian/bin/python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main-origin/data/dir_4d_case2.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 30000 --save_checkpoint \
  --dirname dir_4d_case2_baseline \
  > log/train_baseline_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &


#初始版本的XCAT和S01_004_256_60 baseline 把lp和ltv设置为0了，下面是正确版本
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

### PhysX-Gaussian: Anchor-based Spacetime Transformer

PhysX-Gaussian is a new variant that replaces the HexPlane + MLP deformation field with an **Anchor-based Spacetime Transformer**. It learns physical traction relationships between anatomical structures via masked modeling (BERT-style), enabling generalization to irregular breathing patterns.

**Key Innovation:**
- Original X²-Gaussian: relies on implicit periodic fitting, poor generalization to irregular breathing
- PhysX-Gaussian: uses physical anchors + attention to infer deformation even with irregular breathing

**Architecture:**
1. **FPS Sampling**: Select `num_anchors` points as physical anchors from initial point cloud
2. **KNN Binding**: Each Gaussian binds to `anchor_k` nearest anchors (skinning weights)
3. **Spacetime Transformer**: Anchors attend to each other with time encoding
4. **Masked Modeling**: Randomly mask `mask_ratio` of anchors during training (BERT-style)
5. **Interpolation**: Gaussian displacement = weighted sum of bound anchor displacements

**Training PhysX-Gaussian:**

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

**PhysX-Gaussian Parameters:**

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

**Fallback to Original Method:**
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