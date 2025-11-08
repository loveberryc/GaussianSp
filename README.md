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

# Activate environment before launching (required when using nohup)
conda activate x2_gaussian
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH"

# Launch the default full training schedule (coarse 5k + fine 30k iters)
python train.py -s <path to data> \
  --coarse_iter 5000 \
  --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 10000 20000 30000

# Example
python train.py -s XXX/*.pickle  
python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 10000 20000 30000 --dirname dir_4d_case1_default

# Detached run (remember to activate env & export LD_LIBRARY_PATH first)
nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 10000 20000 30000 --dirname dir_4d_case1_default \
  > train_dir_4d_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/data/dir_4d_case1.pickle \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 10000 20000 30000  \
  --dirname four_vol_max112_d4case1 \
  > train_four_vol_max112_d4case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

提醒：从现在起，X2-Gaussian 的动态建模默认采用 STNF4D 引入的“四正交体”表示（静态 xyz + xyt/xzt/yzt 体），上述命令会自动使用该实现及默认正则化超参。

- 默认情况下训练不会再落盘 `point_cloud/iteration_*` 结果，以避免磁盘快速写满；若需要导出，请额外添加 `--save_point_cloud`。若还想保留全部历史迭代，可同时添加 `--keep_all_point_cloud`（否则仅保留最近一次）。
- 四正交体的最高空间分辨率默认限制为 `112`（`ModelHiddenParams.kplanes_config.max_spatial_resolution`）；若显存不足，可进一步调低该值（如 `96`）。当前实现采用 float32 网格配合分块正则，避免了半精度在细阶段可能出现的 NaN/Inf。

### 使用原始 HexPlane（未启用四正交体）

若需切换回论文初版的 HexPlane 表达，只需在命令中加入 `--no_grid`，即可禁用四正交体特征场，恢复全 MLP 变形（训练流程保持不变）。

```
python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/data/dir_4d_case1.pickle \
  --no_grid \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 10000 20000 30000 --dirname dir_4d_case1_hexplane

nohup python train.py -s /root/autodl-tmp/4dctgs/x2-gaussian-main/data/dir_4d_case1.pickle \
  --no_grid \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --save_iterations 10000 20000 30000 --dirname dir_4d_case1_hexplane \
  > train_dir_4d_case1_hexplane_$(date +%Y%m%d_%H%M%S).log 2>&1 &

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

