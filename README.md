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
  > train_baseline_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

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