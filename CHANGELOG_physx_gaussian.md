# PhysX-Gaussian 修改日志
<!-- markdownlint-disable MD012 MD024 MD031 MD032 MD040 -->

**生成时间**: 2025-12-16 17:44 (更新)  
**基于提交**: PhysX-Boosted s0/s1/s2/s3 实现

---

## [2025-12-16] s0系列: M1.2 Gate Function Variants

### 背景

M1.2 使用 `γ = γ_max * tanh((τ - s_E) / λ)` 计算融合权重的小扰动。
从实验日志观察到：**tanh 容易饱和**（大量时间 γ ≈ γ_max），削弱了"根据不确定性自适应调节"的意义。

s0 系列探索更优的 gate 函数设计：

### s0.1a: Sigmoid Gate (Positive Only)

**公式**:

```text
γ = γ_max * sigmoid((τ - s_E) / λ)
```

**特点**: γ ∈ (0, γ_max)，无负权，过渡更平滑

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode uncertainty_gated --gamma_max 0.005 \
  --gate_tau 0.0 --gate_lambda 1.0 \
  --s0_gate_type sigmoid \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s0_1a_sigmoid_$(date +%Y%m%d_%H%M%S) \
  > log/s0_1a_sigmoid_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s0.1b: Sigmoid Bipolar Gate

**公式**:

```text
γ = γ_max * (2 * sigmoid((τ - s_E) / λ) - 1)
```

**特点**: γ ∈ (-γ_max, γ_max)，比 tanh 更"软"，不易极端饱和

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode uncertainty_gated --gamma_max 0.005 \
  --gate_tau 0.0 --gate_lambda 1.0 \
  --s0_gate_type sigmoid_bipolar \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s0_1b_sigmoid_bipolar_$(date +%Y%m%d_%H%M%S) \
  > log/s0_1b_sigmoid_bipolar_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s0.2: Normalized s_E with EMA

**公式**:

```text
ŝ_E = (s_E - μ) / (σ + ε)   # μ, σ 用 EMA 统计
γ = γ_max * tanh((τ - ŝ_E) / λ)
```

**特点**: s_E 标准化后，τ 更稳定，减少跨 run 漂移

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode uncertainty_gated --gamma_max 0.005 \
  --gate_tau 0.0 --gate_lambda 1.0 \
  --s0_normalize_se \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s0_2_normalize_se_$(date +%Y%m%d_%H%M%S) \
  > log/s0_2_normalize_se_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s0.3: Residual Mode (Base + Residual)

**公式**:

```text
β = β_min + (β_max - β_min) * sigmoid((τ - s_E) / λ)
Φ = Φ_L + β · Φ_E   # Anchor 为 base，HexPlane 为 residual
```

**特点**: 更清晰的"物理骨架 + 欧拉残差"结构

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode uncertainty_gated \
  --gate_tau 0.0 --gate_lambda 1.0 \
  --s0_residual_mode \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s0_3_residual_mode_$(date +%Y%m%d_%H%M%S) \
  > log/s0_3_residual_mode_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s0 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--s0_gate_type` | `tanh` | Gate function: `tanh`, `sigmoid`, `sigmoid_bipolar` |
| `--s0_normalize_se` | False | s0.2: Normalize s_E with EMA statistics |
| `--s0_se_ema_decay` | 0.99 | s0.2: EMA decay rate for mean/std |
| `--s0_residual_mode` | False | s0.3: Use Φ = Φ_L + β·Φ_E formulation |
| `--s0_beta_min` | 0.005 | s0.3: Minimum residual weight |
| `--s0_beta_max` | 0.015 | s0.3: Maximum residual weight |

---

## [2026-01-02] Anchor Temporal Minimal Action (Mass-weighted)

### 新增：Mass-weighted Minimal Action 时间二阶平滑

目标：抑制 sparse-view 时间插值的 transport jitter，提升 PSNR/SSIM 的稳定性。

核心定义：对 anchor 位移 \(\delta_j(t)\) 的二阶差分（加速度）做质量加权惩罚。

其中质量 \(m_j\) 使用已有 Gaussian→Anchor KNN 绑定权重聚合：

```text
m_j = Σ_i ω_ij
```

时间二阶项（离散三点）：

```text
acc_j(t) = (δ_j(t+Δt) - 2δ_j(t) + δ_j(t-Δt))
L_time = Σ_j m_j ||acc_j(t)||² / (Σ_j m_j + eps)
```

注意：实践中发现 `/(Δt²)` 会导致该项尺度过大（对训练主损失产生过强约束），因此这里采用“不除以 Δt²”的版本，并通过 `lambda_anchor_time` 控制整体权重。

### 代码改动

| 文件 | 修改 |
|------|------|
| `x2_gaussian/gaussian/anchor_module.py` | + `_anchor_mass` 缓存（由 `knn_weights` scatter_add 聚合）；+ `compute_anchor_time_smooth_loss()` |
| `train.py` | fine-stage 新增 `anchor_time` loss，并按 `lambda_anchor_time` 加入 total |
| `x2_gaussian/arguments/__init__.py` | 暴露超参：`lambda_anchor_time`, `anchor_time_delta`, `anchor_time_eps`, `anchor_time_stopgrad_neighbors` |
| `x2_gaussian/utils/argument_utils.py` | 修复 bool 参数解析：支持 `--flag` / `--flag True` / `--flag False`，避免 `unrecognized arguments: True` |

### 新增参数（默认关闭）

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--lambda_anchor_time` | 0.0 | time loss 权重（修复尺度后建议从 1e-5~3e-5 起；更激进可到 1e-4） |
| `--anchor_time_delta` | 0.05 | 有限差分 Δt（归一化时间） |
| `--anchor_time_eps` | 1e-8 | 质量归一化 eps |
| `--anchor_time_stopgrad_neighbors` | True | 是否对 t±Δt 的 forward 停梯度（更稳/省显存） |

### 训练命令（对比实验：关闭 dist）

```bash
nohup python train.py \
  --config output/20260102_1620_case1_np0ns0_A0_cfg_args.yml \
  --start_checkpoint output/20260102_1620_case1_np0ns0_A0/chkpntXXXX.pth \
  --lambda_anchor_distortion 0.0 \
  --lambda_anchor_time 3e-4 \
  --anchor_time_delta 0.05 \
  --anchor_time_stopgrad_neighbors True \
  > log/20260102_1620_case1_np0ns0_A0_time_0to100k.log 2>&1 &
```

### 修复：Time loss 尺度问题（去除 /Δt²）

现象：time loss 版本出现 PSNR/SSIM 显著下降，且训练 log 中该项对 total loss 的贡献异常偏大。

原因：实现中使用了 `/(dt*dt)` 导致加速度项尺度过大。

修复：在 `x2_gaussian/gaussian/anchor_module.py` 的 `compute_anchor_time_smooth_loss()` 中去除 `/(dt*dt)`。

建议：修复后推荐将 `--lambda_anchor_time` 从 `3e-4` 下调至 `3e-5`（或更保守 `1e-5`）再进行 ablation。

---

## [2026-01-02] Anchor Bounded Distortion / Quasi-Isometry Loss

### 新增：Bounded Distortion（抑制折叠/拉伸）

目标：对 anchor 图上的边长比例做 hinge 约束，鼓励局部 quasi-isometry，减少 fold / tear。

定义：在 canonical anchor 上构建 KNN 图（k = `anchor_distortion_k`），对每条边 (i,j) 计算

```text
r_ij = ||(x_i + δ_i) - (x_j + δ_j)|| / (||x_i - x_j|| + eps)
L_dist = mean( max(0, r_min - r_ij)^2 + max(0, r_ij - r_max)^2 )
```

其中可选 `anchor_distortion_sigma` 对边进行距离加权（默认 0.0 表示不加权）。

### 代码改动

| 文件 | 修改 |
|------|------|
| `x2_gaussian/gaussian/anchor_module.py` | + 预计算 anchor KNN 图与 canonical 边长缓存；+ `compute_anchor_distortion_loss()` |
| `train.py` | fine-stage 新增 `anchor_distortion` loss，并按 `lambda_anchor_distortion` 加入 total |
| `x2_gaussian/arguments/__init__.py` | 暴露超参：`lambda_anchor_distortion`, `anchor_distortion_k`, `anchor_distortion_r_min`, `anchor_distortion_r_max`, `anchor_distortion_eps`, `anchor_distortion_sigma` |

### 训练命令（case1 baseline + dist loss）

```bash
nohup python train.py \
  --config output/20260102_1620_case1_np0ns0_A0_cfg_args.yml \
  --dirname 20260102_1620_case1_np0ns0_A0_dist \
  --lambda_anchor_distortion 5e-4 \
  --anchor_distortion_k 8 \
  --anchor_distortion_r_min 0.6 \
  --anchor_distortion_r_max 1.6 \
  --anchor_distortion_eps 1e-6 \
  --anchor_distortion_sigma 0.0 \
  > log/20260102_1620_case1_np0ns0_A0_dist_0to100k.log 2>&1 &
```

### 训练命令（case2 baseline + dist loss）

```bash
nohup python train.py \
  --config output/20251230_032322_np0ns0_A0_cfg_args.yml \
  --dirname 20260103_case2_np0ns0_A0_dist \
  --lambda_anchor_distortion 5e-4 \
  --anchor_distortion_k 8 \
  --anchor_distortion_r_min 0.6 \
  --anchor_distortion_r_max 1.6 \
  --anchor_distortion_eps 1e-6 \
  --anchor_distortion_sigma 0.0 \
  > log/20260103_case2_np0ns0_A0_dist_0to100k.log 2>&1 &
```

---

## [2026-01-03] 下载 DIR case3/case4/case5 数据（hf-mirror） + case3 直接开跑命令

### 下载（建议大陆镜像 hf-mirror）

先下载 case3：

```bash
mkdir -p data && \
wget -O data/dir_4d_case3.pickle "https://hf-mirror.com/datasets/vortex778/X2GS/resolve/main/DIR/dir_4d_case3.pickle?download=true" && \
wget -O data/init_dir_4d_case3.npy "https://hf-mirror.com/datasets/vortex778/X2GS/resolve/main/DIR/init_dir_4d_case3.npy?download=true"
```

再下载 case4/case5：

```bash
wget -O data/dir_4d_case4.pickle "https://hf-mirror.com/datasets/vortex778/X2GS/resolve/main/DIR/dir_4d_case4.pickle?download=true" && \
wget -O data/init_dir_4d_case4.npy "https://hf-mirror.com/datasets/vortex778/X2GS/resolve/main/DIR/init_dir_4d_case4.npy?download=true" && \
wget -O data/dir_4d_case5.pickle "https://hf-mirror.com/datasets/vortex778/X2GS/resolve/main/DIR/dir_4d_case5.pickle?download=true" && \
wget -O data/init_dir_4d_case5.npy "https://hf-mirror.com/datasets/vortex778/X2GS/resolve/main/DIR/init_dir_4d_case5.npy?download=true"
```

### 说明：init 文件无需额外参数

当 `--source_path data/dir_4d_case3.pickle` 时，初始化逻辑会自动寻找同目录的 `data/init_dir_4d_case3.npy`（见 `x2_gaussian/gaussian/initialize.py`）。


### 实测：aria2c 更快更稳（推荐）

case3（pickle）：

```bash
aria2c -c -x 16 -s 16 -k 1M --max-tries=0 --retry-wait=5 --timeout=60   -o dir_4d_case3.pickle   -d /root/autodl-tmp/4dctgs/x2-gaussian-main-origin/data   "https://hf-mirror.com/datasets/vortex778/X2GS/resolve/main/DIR/dir_4d_case3.pickle?download=true"
```

case4（npy，建议低并发避免 TLS handshake 抖动）：

```bash
aria2c -c -x 4 -s 4 -k 1M --max-tries=0 --retry-wait=5 --timeout=60   -o init_dir_4d_case4.npy   -d /root/autodl-tmp/4dctgs/x2-gaussian-main-origin/data   "https://hf-mirror.com/datasets/vortex778/X2GS/resolve/main/DIR/init_dir_4d_case4.npy?download=true"
```

case5（pickle）：

```bash
aria2c -c -x 16 -s 16 -k 1M --max-tries=0 --retry-wait=5 --timeout=60   -o dir_4d_case5.pickle   -d /root/autodl-tmp/4dctgs/x2-gaussian-main-origin/data   "https://hf-mirror.com/datasets/vortex778/X2GS/resolve/main/DIR/dir_4d_case5.pickle?download=true"
```

### case3 nohup（基于 case1 A0 baseline 的 config，直接覆盖 source_path）

```bash
nohup python train.py \
  --config output/20260102_1620_case1_np0ns0_A0_cfg_args.yml \
  --source_path data/dir_4d_case3.pickle \
  --dirname 20260103_1620_case3_np0ns0_A0 \
  > log/20260103_1620_case3_np0ns0_A0_0to100k.log 2>&1 &
```

---

## [2025-12-16] s3系列: Release Scale/Rotation from (1-α)

### 背景

V5 baseline 中 scale 和 rotation 都乘以 (1-α) ≈ 0.01，相当于"压制"了 HexPlane 对 scale/rotation 的贡献：

```text
dx = (1-α)*dx_hex + α*dx_anchor
ds = (1-α)*ds_hex   # ≈ 0.01 * ds_hex
dr = (1-α)*dr_hex   # ≈ 0.01 * dr_hex
```

s3 系列"放开"这个约束，让 HexPlane 的 scale/rotation 保持原始强度。

### s3.1: Release Scale

**公式**:

```text
dx = (1-α)*dx_hex + α*dx_anchor
ds = ds_hex   # ← RELEASED (full HexPlane scale)
dr = (1-α)*dr_hex
```

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s3_1_release_scale_$(date +%Y%m%d_%H%M%S) \
  > log/s3_1_release_scale_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s3.2: Release Rotation

**公式**:

```text
dx = (1-α)*dx_hex + α*dx_anchor
ds = (1-α)*ds_hex
dr = dr_hex   # ← RELEASED (full HexPlane rotation)
```

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_rotation \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s3_2_release_rotation_$(date +%Y%m%d_%H%M%S) \
  > log/s3_2_release_rotation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s3.3: Release Scale + Rotation

**公式**:

```text
dx = (1-α)*dx_hex + α*dx_anchor
ds = ds_hex   # ← RELEASED
dr = dr_hex   # ← RELEASED
```

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale --s3_release_rotation \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s3_3_release_scale_rotation_$(date +%Y%m%d_%H%M%S) \
  > log/s3_3_release_scale_rotation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s3.4: Release Scale + Zero Rotation

**公式**:

```text
dx = (1-α)*dx_hex + α*dx_anchor  (α=0.99)
ds = ds_hex   # ← RELEASED (full HexPlane scale)
dr = 0        # ← ZEROED (HexPlane rotation completely disabled)
```

**动机**: s3.1 (release_scale) 效果最好，s3.2 (release_rotation) 效果最差。
s3.4 进一步测试：完全禁用 HexPlane rotation 会不会更好？

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale --s3_zero_rotation \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s3_4_zero_rotation_$(date +%Y%m%d_%H%M%S) \
  > log/s3_4_zero_rotation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s3.5: s3.4 with α=0.95

**公式**:

```text
dx = (1-α)*dx_hex + α*dx_anchor  (α=0.95, 更多 HexPlane 位移权重)
ds = ds_hex
dr = 0
```

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.95 --balance_lr 0 \
  --s3_release_scale --s3_zero_rotation \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s3_5_alpha095_$(date +%Y%m%d_%H%M%S) \
  > log/s3_5_alpha095_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s3.6: s3.4 with α=0.90

**公式**:

```text
dx = (1-α)*dx_hex + α*dx_anchor  (α=0.90, 10% HexPlane + 90% Anchor)
ds = ds_hex
dr = 0
```

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.90 --balance_lr 0 \
  --s3_release_scale --s3_zero_rotation \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s3_6_alpha090_$(date +%Y%m%d_%H%M%S) \
  > log/s3_6_alpha090_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s3 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--s3_release_scale` | False | ds = ds_hex (not multiplied by 1-α) |
| `--s3_release_rotation` | False | dr = dr_hex (not multiplied by 1-α) |
| `--s3_zero_rotation` | False | dr = 0 (completely disable HexPlane rotation) |

---

## [2025-12-16] s2系列: Anchor Fusion for Scale/Rotation

### 背景

V5 baseline 的融合公式只对位移 dx 应用了 anchor 融合：

```text
dx = (1-α)*dx_hex + α*dx_anchor
ds = (1-α)*ds_hex
dr = (1-α)*dr_hex
```

s2 系列尝试将 anchor 融合扩展到 scale 和 rotation，验证是否能进一步提升性能。

### s2.1: Anchor Fusion to Scale

**公式**:

```text
dx = (1-α)*dx_hex + α*dx_anchor
ds = (1-α)*ds_hex + α*dx_anchor   # ← NEW
dr = (1-α)*dr_hex
```

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s2_anchor_to_scale \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s2_1_anchor_to_scale_$(date +%Y%m%d_%H%M%S) \
  > log/s2_1_anchor_to_scale_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s2.2: Anchor Fusion to Rotation

**公式**:

```text
dx = (1-α)*dx_hex + α*dx_anchor
ds = (1-α)*ds_hex
dr = (1-α)*dr_hex + α*dx_anchor_4d   # ← NEW (padded to 4D)
```

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s2_anchor_to_rotation \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s2_2_anchor_to_rotation_$(date +%Y%m%d_%H%M%S) \
  > log/s2_2_anchor_to_rotation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s2.3: Anchor Fusion to Scale + Rotation

**公式**:

```text
dx = (1-α)*dx_hex + α*dx_anchor
ds = (1-α)*ds_hex + α*dx_anchor   # ← NEW
dr = (1-α)*dr_hex + α*dx_anchor_4d   # ← NEW (padded to 4D)
```

**Bug Fix (2025-12-16 04:45)**:

- `dr_hex` 是四元数 `[N, 4]`，`dx_anchor` 是 `[N, 3]`
- 修复：`dx_anchor_4d = cat([dx_anchor, zeros], dim=1)` 补零到 4D

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s2_anchor_to_scale --s2_anchor_to_rotation \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s2_3_anchor_to_scale_rotation_$(date +%Y%m%d_%H%M%S) \
  > log/s2_3_anchor_to_scale_rotation_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## [2025-12-16] s1系列: Per-Anchor Small-Perturbation

### s1: Per-Anchor γᵢ (无额外正则)

**目标**: 验证"把全局γ变成 per-anchor γᵢ"本身是否带来稳定增益。

**核心改动**:

- 每个锚点 i 有独立的 bounded perturbation: γᵢ(t) ∈ [-γ_max, γ_max]
- 通过 KNN 权重传播到高斯点: γ(x,t) = Σ wᵢ(x)·γᵢ(t)
- Fusion 严格保持系数和: Φ = (0.99-γ(x,t))·Φ_L + (0.01+γ(x,t))·Φ_E
- s_E(i,t) 通过从点到锚点的聚合得到（不新增网络）

**公式**:

```text
s_E(i,t) = Σ_x wᵢ(x)·s_E(x,t) / (Σ_x wᵢ(x) + ε)  # 聚合到锚点
γᵢ = γ_max * tanh((τ - s_E(i,t)) / λ)             # 每锚点 γ
γ(x,t) = Σ wᵢ(x)·γᵢ                               # 插值到高斯
```

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode uncertainty_gated --gamma_max 0.005 \
  --gate_tau 0.0 --gate_lambda 1.0 \
  --per_anchor_gamma \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s1_per_anchor_gamma_$(date +%Y%m%d_%H%M%S) \
  > log/s1_per_anchor_gamma_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s1.1: s1 + Anchor Graph 空间平滑

**目标**: 防止相邻锚点 γᵢ 巨不一致（棋盘格 gate）。

**公式**: L_graph = λ · Σ_{(i,j)∈E} (γᵢ - γⱼ)²

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode uncertainty_gated --gamma_max 0.005 \
  --gate_tau 0.0 --gate_lambda 1.0 \
  --per_anchor_gamma --lambda_gamma_graph 0.01 \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s1_1_graph_smooth_$(date +%Y%m%d_%H%M%S) \
  > log/s1_1_graph_smooth_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s1.2: s1 + 时间平滑

**目标**: 直接压 flicker，防止 γᵢ(t) 时间抖动。

**公式**: L_temp = λ · Σᵢ |γᵢ(t) - γᵢ(t-Δt)|²

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode uncertainty_gated --gamma_max 0.005 \
  --gate_tau 0.0 --gate_lambda 1.0 \
  --per_anchor_gamma --lambda_gamma_temp 0.01 \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s1_2_temp_smooth_$(date +%Y%m%d_%H%M%S) \
  > log/s1_2_temp_smooth_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**备注**:

- README 与本日志中的 s1/s1.1/s1.2/s1.3 训练命令已统一显式加入 `--gate_tau 0.0 --gate_lambda 1.0`，避免默认值漂移导致实验不可比。

### s1.4: s0.1b + s1.1（sigmoid_bipolar gate + Anchor Graph 空间平滑）

**目标**: 将 s0.1b 的更平滑双极性 gate 与 s1.1 的空间平滑正则组合，抑制棋盘格 gate，同时允许 γ 取负值（减弱残差）。

**配置**:

- `s0.1b`: `--s0_gate_type sigmoid_bipolar`
- `s1.1`: `--per_anchor_gamma --lambda_gamma_graph 0.01`

其中 `--lambda_gamma_graph 0.01` 与 s1.1 的默认实验一致。

**训练命令**:

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --per_anchor_gamma --s0_gate_type sigmoid_bipolar --lambda_gamma_graph 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s1_4_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s1_4_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

**通过标准**:

- flicker 指标明显下降
- PSNR/LPIPS 不降（最好略升）
- γᵢ 的空间分布更连贯、时间曲线更平滑

---

## [2025-12-15] M7/M8: 结构性融合改进

### M7: High-Pass Structural Decomposition on M1.2

**核心思想**: 将 M6 的高通频率分解应用到 M1.2 (uncertainty_gated) 融合模式上。

**公式**:

```text
r = Φ_E - Φ_L                    # Eulerian residual
r_low = LP(r), r_high = r - r_low  # 频率分解
Φ = Φ_L + hex_weight · (r_high + tied_factor · r_low)
```

**tied 模式**: `tied_factor = 1.0` 时，数学上退化回原始 M1.2。

**训练命令**:

```bash
# M7-tied (基于 M1.2 g005 + hpass tied)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode uncertainty_gated --gamma_max 0.005 --gate_tau 0.0 --gate_lambda 1.0 \
  --hpass_enable --hpass_eps_low_mode tied \
  --mask_ratio 0.0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 7000 10000 20000 30000 40000 50000 \
  --save_iterations 10000 30000 50000 --save_checkpoint \
  --dirname m7_tied_$(date +%Y%m%d_%H%M%S) \
  > log/m7_tied_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### M8: Transport-Correction Decomposition (Predictor-Corrector)

**核心创新**: 将并行融合 `Φ = α·Φ_L + (1-α)·Φ_E` 升级为串行 Predictor-Corrector 分解。

**论文叙事**: *"Instead of parallel blending, we decompose into serial predictor-corrector: Lagrangian transport followed by Eulerian closure in the comoving frame."*

**公式 (Operator Splitting / ALE style)**:

```text
1. Predictor (Lagrangian transport):  x' = x + Φ_L(x,t)
2. Corrector (Eulerian at x'):        Δ = Φ_E(x',t)  [comoving frame]
3. Update (budgeted residual):        x(t) = x' + ε·Δ
```

**关键洞察**: 残差在 comoving frame (x') 上评估，天然无法学习 Φ_L 已捕获的大尺度运输。

**消融实验设计**:

1. **comoving=True** (M8 主实验): Eulerian 在 x' 上查询
2. **comoving=False** (消融): Eulerian 在 x 上查询 (应该更差)
3. **learnable_beta**: 学习空间变化的 β(x',t) + budget 约束

**训练命令**:

```bash
# M8 comoving (主实验 - Eulerian 在 transported position 查询)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --transport_correct_enable --transport_correct_comoving \
  --transport_correct_eps 0.01 \
  --mask_ratio 0.0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 7000 10000 20000 30000 40000 50000 \
  --save_iterations 10000 30000 50000 --save_checkpoint \
  --dirname m8_comoving_$(date +%Y%m%d_%H%M%S) \
  > log/m8_comoving_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# M8 learnable beta (学习 β(x',t) + budget 约束)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --transport_correct_enable --transport_correct_comoving \
  --transport_correct_learnable_beta \
  --transport_correct_beta_max 0.03 --transport_correct_beta_init 0.01 \
  --transport_correct_beta_budget 0.01 --transport_correct_lambda_budget 0.1 \
  --mask_ratio 0.0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 7000 10000 20000 30000 40000 50000 \
  --save_iterations 10000 30000 50000 --save_checkpoint \
  --dirname m8_learnable_beta_$(date +%Y%m%d_%H%M%S) \
  > log/m8_learnable_beta_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 修改文件

| 文件 | 修改 |
|------|------|
| `x2_gaussian/arguments/__init__.py` | 添加 M8 参数: `transport_correct_*` |
| `x2_gaussian/gaussian/anchor_module.py` | 实现 M7/M8 融合逻辑, 添加 `get_m8_statistics()` |
| `train.py` | 添加 M8 日志记录 |

---

## [2025-12-15] M5/M6 Baseline 修复

### 问题描述

M5/M6 实验的 baseline 结果（psnr3d ~39）远低于历史最佳（psnr3d ~45）。

### 根本原因

1. **缺失 V5 LEARNABLE BALANCE**: 历史最佳配置使用了 `--use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0`
2. **residual_mode 默认值错误**: 原默认值 `tanh` 会破坏位移幅度，改为 `none`
3. **mask_ratio 默认值错误**: 需要显式设置 `--mask_ratio 0.0`

### 修复内容

| 文件 | 修改 |
|------|------|
| `x2_gaussian/gaussian/anchor_module.py` | `residual_mode` 默认值改为 `none` |
| `x2_gaussian/arguments/__init__.py` | 更新 `residual_mode` 文档和默认值 |
| `README.md` | 更新 M5/M6 训练命令，添加完整参数 |
| `CHANGELOG_physx_gaussian.md` | 更新训练命令文档 |

### 正确的 baseline 参数

```bash
--use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0
--mask_ratio 0.0
--lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0
--eps_max 0.03 --eps_init 0.015
--schedule_mode freeze_rho --freeze_steps 2000
```

### sparse50 数据集复现实验（dir_4d_case1_sparse50.pickle）

```bash
# M1.2 g005 (对应 train_physx_boosted_m1_2_g005_case1_... 配置迁移到 sparse50)
nohup python train.py -s data/dir_4d_case1_sparse50.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --fusion_mode uncertainty_gated \
  --gate_tau 0.0 --gate_lambda 1.0 \
  --gamma_max 0.005 \
  --m1_lambda_gate 0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 7000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname m1_2_g005_sparse50_$(date +%Y%m%d_%H%M%S) \
  > log/m1_2_g005_sparse50_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# M2.1a freeze_rho (对应 train_physx_boosted_m2_1a_freeze_case1_... 配置迁移到 sparse50)
nohup python train.py -s data/dir_4d_case1_sparse50.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode bounded_perturb \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --eps_max 0.03 --eps_init 0.015 \
  --mask_ratio 0.0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 7000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname m2_1a_sparse50_$(date +%Y%m%d_%H%M%S) \
  > log/m2_1a_sparse50_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## [2025-12-14] M6: High-Pass Structural Decomposition

### 核心创新（M6）

**论文叙事**: *"Unlike penalty-based regularization, we enforce a structural frequency split of the Eulerian residual in the forward pass, allocating a bounded correction budget to the high-frequency component to prevent shortcut learning."*

关键洞察：M3/M4 的惩罚式正则无法提升（多半压掉必要的中频修正），M6 在 forward 中强制分解残差：

```text
r(x,t) = Φ_E(x,t) - Φ_L(x,t)    # Eulerian residual
r_low = LP(r)                    # 低频（邻域平均）
r_high = r - r_low               # 高频
Φ = Φ_L + ε_high·r_high + ε_low·r_low
```

### 三种 eps_low 模式

| 模式 | 描述 | 用途 |
|------|------|------|
| `zero` | ε_low = 0（硬高通） | 主实验，结构创新 |
| `tied` | ε_low = ε_high | Sanity check（应等于 baseline） |
| `bounded_small` | ε_low 可学习但 max << ε_high | 消融实验 |

### 新增配置参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--hpass_enable` | False | M6 主开关 |
| `--hpass_lp_mode` | knn_cached | LP 模式: graph / knn_cached |
| `--hpass_k` | 8 | 邻域大小 |
| `--hpass_eps_low_mode` | zero | 低频预算模式 |
| `--hpass_eps_low_max` | 0.005 | bounded_small 模式的 eps_low 上限 |

### 关键特性

1. **继承 M2.1a freeze 策略**: 前 `freeze_steps` 步冻结 ρ_high/ρ_low
2. **诊断日志**: E_low、E_high、ratio = E_low/(E_high+1e-8)
3. **无新 loss**: 纯 forward 结构改动，避免 confound

### [2025-12-14] M6 v2 修复（实现细节）

为满足“forward 内 LP 必须高效 + tied 必须严格退化回 baseline”的要求，做了两处关键修复：

1. **LP 复杂度修复**: `knn_cached` 不再对 5w Gaussians 做 `torch.cdist` (O(N^2))。
   - 复用项目已有的 **Gaussian→Anchor KNN binding**（`knn_indices/knn_weights`），
   - 通过 `scatter -> (anchor-space smooth) -> gather` 实现 LP，复杂度 **O(N·K + M·k)**。

2. **Sanity 严格退化修复**: residual 定义改为 `r = Φ_E - Φ_L`（位移上 `r = dx_H - dx_anchor`）。
   - 当 `eps_low_mode="tied"` 时，公式严格退化为 M2.1a：`Φ = Φ_L + ε_eff*(Φ_E - Φ_L)`。

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `x2_gaussian/arguments/__init__.py` | +30 行 M6 hpass 配置 |
| `x2_gaussian/gaussian/anchor_module.py` | +200 行 LowPassOperator + 双预算融合 |
| `train.py` | +40 行 M6 日志 + freeze 逻辑 |

### 训练命令

**重要**: 必须包含 `--use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0` 以匹配历史最佳 baseline (psnr3d ~45)。

```bash
# Baseline (M2.1a + V5 balance) - 标准 baseline
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode bounded_perturb \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --eps_max 0.03 --eps_init 0.015 \
  --mask_ratio 0.0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 7000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname m5_baseline_$(date +%Y%m%d_%H%M%S) \
  > log/m5_baseline_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# M6-tied (sanity check - 应与 baseline 结果一致)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode bounded_perturb \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --mask_ratio 0.0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --hpass_enable --hpass_eps_low_mode tied \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 7000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname m6_tied_$(date +%Y%m%d_%H%M%S) \
  > log/m6_tied_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# M6-hard (主实验 - 硬高通，ε_low=0)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode bounded_perturb \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --mask_ratio 0.0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --hpass_enable --hpass_eps_low_mode zero \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 7000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname m6_hard_$(date +%Y%m%d_%H%M%S) \
  > log/m6_hard_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## [2025-12-13] M5: Phase-Aware Trust-Region ε(t)

### 核心创新（M5）

**论文叙事**: *"Phase-aware trust-region allocates a bounded residual budget across respiratory phases, preserving Lagrangian dominance while enabling demand-driven corrections."*

将 M2.1a 的全局标量 ε 升级为时间相位条件化的 ε(t)：

```text
ε(t) = ε_max * sigmoid(g(t))
```

其中 g(t) 是低容量函数，允许不同呼吸相位有不同的残差预算。

### 两种模式

| 模式 | 描述 | 适用场景 |
|------|------|----------|
| `per_frame` | g_k 为可学习向量 [T]，每帧一个标量 | 最稳定，离散相位 |
| `tiny_mlp` | g(t) 是小型 MLP + Fourier 编码 | 连续时间，论文味 |

### 新增配置参数

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--phase_eps_enable` | False | M5 主开关 |
| `--phase_eps_mode` | per_frame | 模式: per_frame / tiny_mlp |
| `--phase_eps_num_frames` | 10 | 离散相位数 (per_frame) |
| `--phase_eps_mlp_hidden` | 32 | MLP 隐藏层维度 (tiny_mlp) |
| `--phase_eps_mlp_layers` | 2 | MLP 层数 (tiny_mlp) |
| `--phase_eps_smooth_lambda` | 1e-4 | L_smooth 平滑先验权重 |

### 关键特性

1. **继承 M2.1a freeze 策略**: 前 `freeze_steps` 步冻结 g 参数，ε(t) 固定为 init_eps
2. **时间平滑先验**: L_smooth = mean_k (ε_{k+1} - ε_k)² 防止过拟合
3. **日志/可视化**: 每 1000 步输出 ε(t) 统计，支持曲线可视化

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `x2_gaussian/arguments/__init__.py` | +33 行 M5 配置参数 |
| `x2_gaussian/gaussian/anchor_module.py` | +220 行 PhaseEpsilon 模块 |
| `train.py` | +50 行 L_smooth 损失 + freeze 逻辑 |
| `scripts/visualize_phase_eps.py` | 新建，ε(t) 曲线可视化 |

### 训练命令

**重要**: 必须包含 `--use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0` 以匹配历史最佳 baseline。

```bash
# M5-per_frame (离散相位)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode bounded_perturb \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --mask_ratio 0.0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --phase_eps_enable --phase_eps_mode per_frame \
  --phase_eps_num_frames 10 --phase_eps_smooth_lambda 1e-4 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 7000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname m5_perframe_$(date +%Y%m%d_%H%M%S) \
  > log/m5_perframe_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# M5-tiny_mlp (连续时间 MLP)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --fusion_mode bounded_perturb \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --mask_ratio 0.0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --phase_eps_enable --phase_eps_mode tiny_mlp \
  --phase_eps_mlp_hidden 32 --phase_eps_mlp_layers 2 \
  --phase_eps_smooth_lambda 1e-4 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 7000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname m5_tinymlp_$(date +%Y%m%d_%H%M%S) \
  > log/m5_tinymlp_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 可视化命令

```bash
python scripts/visualize_phase_eps.py --model_path output/xxx/ --output_dir plots/
```

---

## 概述

本项目实现了 **PhysX-Gaussian 系列**形变场变体，包括：

1. **PhysX-Gaussian**: 纯 Anchor-based Spacetime Transformer（替代 HexPlane+MLP）
2. **PhysX-Boosted**: HexPlane + Anchor 双分支融合（站在巨人肩膀上）
3. **PhysX-Boosted V5-V9**: 多种融合策略的消融实验版本

### 核心创新（总览）

- **原始 X²-Gaussian**: 依赖隐式周期拟合，对不规则呼吸泛化能力差
- **PhysX-Gaussian**: 使用物理锚点 + 注意力机制，即使呼吸不规则也能推断形变

### 架构设计

1. **FPS 采样**: 从初始点云中选择 `num_anchors` 个点作为物理锚点
2. **KNN 绑定**: 每个高斯绑定到 `anchor_k` 个最近锚点（蒙皮权重）
3. **时空 Transformer**: 锚点之间通过时间编码进行相互注意力
4. **掩码建模**: 训练时随机掩码 `mask_ratio` 比例的锚点（BERT 风格）

5. **插值**: 高斯位移 = 绑定锚点位移的加权和

---

## 修改的文件列表

| 文件 | 修改类型 | 修改行数 |
|------|----------|----------|
| `README.md` | 修改 | +56/-2 |
| `train.py` | 修改 | +76 |
| `x2_gaussian/arguments/__init__.py` | 修改 | +33 |
| `x2_gaussian/gaussian/gaussian_model.py` | 修改 | +206 |
| `x2_gaussian/gaussian/render_query.py` | 修改 | +37/-25 |
| `x2_gaussian/gaussian/anchor_module.py` | **新建** | +711 |

---

## 详细修改内容

### 1. README.md

新增 PhysX-Gaussian 使用文档和训练命令：

```markdown
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
```

#### PhysX-Gaussian 训练命令

```sh
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

#### 参数表

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--use_anchor_deformation` | False | 启用 PhysX-Gaussian 的主开关 |
| `--num_anchors` | 1024 | FPS 采样的物理锚点数量 |
| `--anchor_k` | 10 | 每个高斯绑定的最近锚点数量 |
| `--mask_ratio` | 0.25 | 训练时掩码的锚点比例 |
| `--transformer_dim` | 64 | 时空 Transformer 隐藏维度 |
| `--transformer_heads` | 4 | 注意力头数量 |
| `--transformer_layers` | 2 | Transformer 编码器层数 |
| `--lambda_phys` | 0.1 | 物理补全损失 L_phys 权重 |
| `--lambda_anchor_smooth` | 0.01 | 锚点运动平滑损失权重 |
| `--phys_warmup_steps` | 2000 | 应用 L_phys 前的预热步数 |

---

### 2. train.py

#### 2.1 新增 `apply_physx_preset()` 函数

```python
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
    # ... 打印配置信息
```

#### 2.2 在 `training()` 函数中调用预设

```python
# Apply PhysX-Gaussian preset if enabled
apply_physx_preset(opt, hyper)
```

#### 2.3 在 `scene_reconstruction()` 中添加损失计算

```python
# PhysX-Gaussian: Anchor-based deformation losses
use_anchor = getattr(hyper, 'use_anchor_deformation', False)
if stage == 'fine' and use_anchor and gaussians.use_anchor_deformation:
    lambda_phys = getattr(hyper, 'lambda_phys', 0.1)
    lambda_anchor_smooth = getattr(hyper, 'lambda_anchor_smooth', 0.01)
    phys_warmup_steps = getattr(hyper, 'phys_warmup_steps', 2000)
    
    # Only apply physics completion loss after warmup
    if iteration >= phys_warmup_steps and lambda_phys > 0:
        time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device)
        L_phys = gaussians.compute_physics_completion_loss(time_tensor)
        loss["phys_completion"] = L_phys
        loss["total"] = loss["total"] + lambda_phys * L_phys
    
    # Anchor smoothness regularization (always active)
    if lambda_anchor_smooth > 0:
        L_anchor_smooth = gaussians.compute_anchor_smoothness_loss()
        loss["anchor_smooth"] = L_anchor_smooth
        loss["total"] = loss["total"] + lambda_anchor_smooth * L_anchor_smooth
```

#### 2.4 添加 TensorBoard 日志记录

```python
# PhysX-Gaussian: Log anchor-based deformation statistics
if stage == 'fine' and getattr(hyper, 'use_anchor_deformation', False) and gaussians.use_anchor_deformation:
    if "phys_completion" in loss:
        metrics['physx_L_phys'] = loss["phys_completion"].item()
    if "anchor_smooth" in loss:
        metrics['physx_L_smooth'] = loss["anchor_smooth"].item()
```

---

### 3. x2_gaussian/arguments/**init**.py

新增 PhysX-Gaussian 参数定义：

```python
# PhysX-Gaussian: Anchor-based Spacetime Transformer Deformation
self.use_anchor_deformation = False  # Master switch for PhysX-Gaussian
self.num_anchors = 1024  # Number of FPS-sampled physical anchors
self.anchor_k = 10  # Number of nearest anchors each Gaussian binds to (KNN)
self.mask_ratio = 0.25  # Ratio of anchors to mask during training (BERT-style)
self.transformer_dim = 64  # Hidden dimension of spacetime transformer
self.transformer_heads = 4  # Number of attention heads
self.transformer_layers = 2  # Number of transformer encoder layers
self.anchor_time_embed_dim = 16  # Time embedding dimension for anchors
self.anchor_pos_embed_dim = 32  # Position embedding dimension for anchors

# PhysX-Gaussian loss parameters
self.lambda_phys = 0.1  # Weight for physics completion loss L_phys
self.lambda_anchor_smooth = 0.01  # Weight for anchor motion smoothness regularization
self.phys_warmup_steps = 2000  # Steps before applying L_phys
```

---

### 4. x2_gaussian/gaussian/gaussian_model.py

#### 4.1 新增 import

```python
from x2_gaussian.gaussian.anchor_module import AnchorDeformationNet
```

#### 4.2 `__init__` 中初始化锚点形变网络

```python
# PhysX-Gaussian: Anchor-based Spacetime Transformer parameters
self.use_anchor_deformation = getattr(args, 'use_anchor_deformation', False)
self.num_anchors = getattr(args, 'num_anchors', 1024)
self.anchor_k = getattr(args, 'anchor_k', 10)
self.mask_ratio = getattr(args, 'mask_ratio', 0.25)
self._deformation_anchor = None

# Create anchor deformation network if enabled
if self.use_anchor_deformation:
    self._deformation_anchor = AnchorDeformationNet(args)
    print(f"[PhysX-Gaussian] Anchor-based deformation ENABLED")
```

#### 4.3 `create_from_pcd()` 中初始化锚点和 KNN 绑定

```python
# PhysX-Gaussian: Initialize anchors and KNN binding
if self.use_anchor_deformation and self._deformation_anchor is not None:
    self._deformation_anchor = self._deformation_anchor.to("cuda")
    self._deformation_anchor.initialize_anchors(fused_point_cloud)
    self._deformation_anchor.update_knn_binding(fused_point_cloud)
    print(f"[PhysX-Gaussian] Anchors initialized and KNN binding computed")
```

#### 4.4 `training_setup()` 中添加优化器参数

```python
# PhysX-Gaussian: Add anchor deformation parameters to optimizer
if self.use_anchor_deformation and self._deformation_anchor is not None:
    l.append({
        "params": list(self._deformation_anchor.get_mlp_parameters()),
        "lr": training_args.deformation_lr_init * self.spatial_lr_scale,
        "name": "anchor_deformation",
    })
    l.append({
        "params": list(self._deformation_anchor.get_grid_parameters()),
        "lr": training_args.grid_lr_init * self.spatial_lr_scale,
        "name": "anchor_transformer",
    })
```

#### 4.5 `prune_points()` 和 `densification_postfix()` 中更新 KNN 绑定

```python
# PhysX-Gaussian: Update KNN binding after pruning/densification
if self.use_anchor_deformation and self._deformation_anchor is not None:
    self._deformation_anchor.update_knn_binding(self._xyz)
```

#### 4.6 新增 PhysX-Gaussian 专用方法

```python
def get_active_deformation_network(self):
    """Get the active deformation network (anchor-based or original HexPlane)."""

def compute_anchor_deformation(self, time, is_training=True):
    """Compute deformation using anchor-based spacetime transformer."""

def compute_physics_completion_loss(self, time):
    """Compute PhysX-Gaussian physics completion loss L_phys."""

def compute_anchor_smoothness_loss(self):
    """Compute PhysX-Gaussian anchor motion smoothness loss."""

def update_anchor_knn_binding(self):
    """Update KNN binding between Gaussians and anchors."""

def save_anchor_deformation(self, path):
    """Save anchor deformation network state."""

def load_anchor_deformation(self, path):
    """Load anchor deformation network state."""
```

#### 4.7 修改 `get_deformed_centers()` 方法

新增 `is_training` 参数，支持 PhysX-Gaussian 锚点形变：

```python
def get_deformed_centers(self, time, use_v7_1_correction=False, correction_alpha=0.0, is_training=True):
    # ...
    # PhysX-Gaussian: Use anchor-based transformer instead of HexPlane+MLP
    if self.use_anchor_deformation and self._deformation_anchor is not None:
        means3D_deformed, scales_deformed, rotations_deformed = self._deformation_anchor(
            means3D, scales, rotations, density, time, is_training=is_training
        )
    else:
        # Original X²-Gaussian: HexPlane+MLP deformation
        means3D_deformed, scales_deformed, rotations_deformed = self._deformation(
            means3D, scales, rotations, density, time
        )
    
    if self.use_anchor_deformation:
        # PhysX-Gaussian doesn't use V7.2 correction - skip entirely
        return means3D_deformed, scales_deformed, rotations_deformed
    # ...
```

---

### 5. x2_gaussian/gaussian/render_query.py

修改三个渲染函数以统一使用 `get_deformed_centers()`，支持 PhysX-Gaussian：

#### 5.1 `query()` 函数

```python
# 旧代码:
if use_v7_1_correction and correction_alpha != 0.0:
    means3D_final, scales_final, rotations_final = pc.get_deformed_centers(...)
else:
    means3D_final, scales_final, rotations_final = pc._deformation(...)

# 新代码:
means3D_final, scales_final, rotations_final = pc.get_deformed_centers(
    time, 
    use_v7_1_correction=use_v7_1_correction, 
    correction_alpha=correction_alpha,
    is_training=False  # Query is typically for evaluation
)
```

#### 5.2 `render()` 函数

```python
means3D_final, scales_final, rotations_final = pc.get_deformed_centers(
    time, 
    use_v7_1_correction=use_v7_1_correction, 
    correction_alpha=correction_alpha,
    is_training=True  # Render is called during training
)
```

#### 5.3 `render_prior_oneT()` 函数

```python
means3D_final, scales_final, rotations_final = pc.get_deformed_centers(
    time, is_training=False  # Prior rendering doesn't need masking
)
```

---

### 6. x2_gaussian/gaussian/anchor_module.py (新建文件，711行)

这是 PhysX-Gaussian 的核心模块，完整实现了 Anchor-based Spacetime Transformer。

#### 6.1 工具函数

```python
def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """FPS 采样选择代表性锚点"""

def compute_knn_weights(query_points, anchor_points, k, temperature=1.0):
    """计算 KNN 索引和基于距离的蒙皮权重"""
```

#### 6.2 编码模块

```python
class PositionalEncoding(nn.Module):
    """3D 位置的正弦位置编码"""

class TimeEncoding(nn.Module):
    """时间信息的傅里叶时间编码"""

class AnchorEmbedding(nn.Module):
    """锚点位置嵌入到特征空间"""
```

#### 6.3 Transformer 编码器

```python
class SpacetimeTransformerEncoder(nn.Module):
    """
    时空锚点交互的 Transformer 编码器。
    
    学习锚点之间如何基于空间关系和时间上下文（呼吸相位）相互影响运动。
    
    参数:
        d_model: 隐藏维度 (default: 64)
        nhead: 注意力头数量 (default: 4)
        num_layers: 编码器层数 (default: 2)
        dim_feedforward: FFN 维度 (default: 256)
        dropout: Dropout 比例 (default: 0.1)
    """
```

#### 6.4 核心类 `AnchorDeformationNet`

```python
class AnchorDeformationNet(nn.Module):
    """
    PhysX-Gaussian: Anchor-based Spacetime Transformer for Deformation.
    
    替代 HexPlane + MLP 的方式:
    1. 使用 FPS 采样的锚点作为物理控制点
    2. 通过自注意力学习锚点交互
    3. 训练时掩码锚点以实现鲁棒形变推断
    4. 通过蒙皮插值锚点位移到高斯位置
    
    关键见解: 呼吸运动受物理约束（肋骨、膈肌、肺组织）控制，
    学习这些关系可以泛化到不规则呼吸模式。
    """
```

##### 主要方法

| 方法 | 功能 |
|------|------|
| `initialize_anchors(points)` | 从点云用 FPS 初始化锚点 |
| `update_knn_binding(positions)` | 更新高斯与锚点的 KNN 绑定 |
| `forward_anchors(time, is_training)` | 计算锚点位移（可选掩码） |
| `forward_anchors_unmasked(time)` | 计算锚点位移（无掩码，用于教师强制） |
| `interpolate_displacements(dx, positions)` | 用蒙皮权重插值位移到高斯 |
| `forward(positions, scales, rotations, density, time)` | 完整前向传播（兼容原始接口） |
| `forward_backward_position(deformed_pts, time)` | 反向形变（用于逆一致性） |
| `compute_physics_completion_loss()` | 计算 L_phys 物理补全损失 |
| `compute_anchor_smoothness_loss()` | 计算锚点运动平滑损失 |
| `get_mlp_parameters()` | 返回 MLP 参数（兼容优化器） |
| `get_grid_parameters()` | 返回 Transformer 参数 |

##### 网络结构

- `anchor_embed`: 锚点位置嵌入 MLP
- `time_encode`: 傅里叶时间编码
- `input_proj`: 输入投影层
- `mask_token`: 可学习的 [MASK] token
- `transformer`: Spacetime Transformer 编码器
- `displacement_head`: 位移预测头
- `displacement_head_backward`: 反向位移预测头
- `scale_head`: 尺度预测头
- `rotation_head`: 旋转预测头

---

## 兼容性

- `use_anchor_deformation=False`（默认）: 行为与原始 X²-Gaussian 完全相同
- `use_anchor_deformation=True`: 使用 PhysX-Gaussian 锚点形变，禁用 V7.2 一致性校正（两者是替代方案）

---

## 当前修改汇总 (基于 git diff HEAD)

> 以下内容基于 `git diff HEAD` 核实，确保准确无遗漏。

### 1. 删除文件

- `idea.md` - 旧的 idea 文档已删除 (156 行)

### 2. train.py 修改

#### 新增注释

```python
# torch.autograd.set_detect_anomaly(True)  # DEBUG: Disabled - may cause issues
```

**新增 `apply_physx_preset()` 函数** (第 27-48 行):

- 打印 PhysX-Gaussian 配置信息
- 显示锚点数量、KNN、mask_ratio、transformer 参数
- 显示损失权重 λ_phys, λ_anchor_smooth

#### 调用 preset

```python
apply_physx_preset(opt, hyper)  # 在 apply_v7_preset 之后
```

**跳过 HexPlane 相关损失** (当 `use_anchor_deformation=True` 时):

| 损失 | 跳过原因 |
|------|----------|
| Prior loss | `render_prior_oneT` 会产生第二次前向传播 |
| 3D TV loss | `query()` 会产生第二次前向传播 |
| 4D TV loss | HexPlane 正则化，PhysX-Gaussian 不使用 HexPlane |
| L_inv (逆一致性) | 使用 HexPlane 内部计算 |
| Cycle motion | 使用 HexPlane 内部计算 |
| Jacobian reg | 使用 HexPlane 内部计算 |
| Trajectory smoothing | 使用 HexPlane 内部计算 |

#### PhysX-Gaussian 损失计算

```python
if stage == 'fine' and use_anchor and gaussians.use_anchor_deformation:
    # L_phys (只在 warmup 后)
    if iteration >= phys_warmup_steps and lambda_phys > 0:
        L_phys = gaussians.compute_physics_completion_loss(time_tensor)
        loss["total"] = loss["total"] + lambda_phys * L_phys
    
    # L_anchor_smooth
    if lambda_anchor_smooth > 0:
        L_anchor_smooth = gaussians.compute_anchor_smoothness_loss()
        loss["total"] = loss["total"] + lambda_anchor_smooth * L_anchor_smooth
```

#### PhysX-Gaussian 统计日志

```python
if "phys_completion" in loss:
    metrics['physx_L_phys'] = loss["phys_completion"].item()
if "anchor_smooth" in loss:
    metrics['physx_L_smooth'] = loss["anchor_smooth"].item()
```

### 3. gaussian_model.py 修改

#### 导入

```python
from x2_gaussian.gaussian.anchor_module import AnchorDeformationNet
```

#### 新增属性初始化

```python
self.use_anchor_deformation = getattr(args, 'use_anchor_deformation', False)
self.num_anchors = getattr(args, 'num_anchors', 1024)
self.anchor_k = getattr(args, 'anchor_k', 10)
self.mask_ratio = getattr(args, 'mask_ratio', 0.25)
self._deformation_anchor = None
if self.use_anchor_deformation:
    self._deformation_anchor = AnchorDeformationNet(args)
```

#### `create_from_pcd()` 中初始化锚点

```python
if self.use_anchor_deformation and self._deformation_anchor is not None:
    self._deformation_anchor = self._deformation_anchor.to("cuda")
    self._deformation_anchor.initialize_anchors(fused_point_cloud)
    self._deformation_anchor.update_knn_binding(fused_point_cloud)
```

#### `training_setup()` 添加优化器参数

```python
if self.use_anchor_deformation and self._deformation_anchor is not None:
    l.append({"params": list(self._deformation_anchor.get_mlp_parameters()), ...})
    l.append({"params": list(self._deformation_anchor.get_grid_parameters()), ...})
```

#### 剪枝/密集化后更新 KNN

```python
# prune_points() 和 densification_postfix() 中
if self.use_anchor_deformation and self._deformation_anchor is not None:
    self._deformation_anchor.update_knn_binding(self._xyz)
```

#### 新增 PhysX-Gaussian 方法

- `get_active_deformation_network()`
- `compute_anchor_deformation(time, is_training)`
- `compute_physics_completion_loss(time)`
- `compute_anchor_smoothness_loss()`
- `update_anchor_knn_binding()`
- `save_anchor_deformation(path)`
- `load_anchor_deformation(path)`

#### `get_deformed_centers()` 修改

1. 添加 `is_training` 参数
2. 使用 `.clone()` 避免 in-place 修改:

   ```python
   means3D = self.get_xyz.clone()
   density = self.get_density.clone()
   scales = self._scaling.clone()
   rotations = self._rotation.clone()
   ```

3. PhysX-Gaussian 分支使用 `.contiguous()`:

   ```python
   if self.use_anchor_deformation and self._deformation_anchor is not None:
       means3D_deformed, scales_deformed, rotations_deformed = self._deformation_anchor(...)
       means3D_deformed = means3D_deformed.contiguous()
       scales_deformed = scales_deformed.contiguous()
       rotations_deformed = rotations_deformed.contiguous()
   ```

4. PhysX-Gaussian 跳过 V7.2 校正:

   ```python
   if self.use_anchor_deformation:
       return means3D_deformed, scales_deformed, rotations_deformed
   ```

### 4. render_query.py 修改

#### 所有渲染函数添加 `.clone()`

- `query()`: `means3D = pc.get_xyz.clone()`, `density = pc.get_density.clone()`, `scales = pc._scaling.clone()`, `rotations = pc._rotation.clone()`
- `render()`: 同上
- `render_prior_oneT()`: 同上

#### 统一使用 `get_deformed_centers()`

- `query()`: `pc.get_deformed_centers(time, ..., is_training=False)`
- `render()`: `pc.get_deformed_centers(time, ..., is_training=True)`
- `render_prior_oneT()`: `pc.get_deformed_centers(time, is_training=False)`

#### 清理

- 移除 `render_prior_oneT()` 中的 `# breakpoint()` 注释

### 5. arguments/**init**.py 修改

**新增 PhysX-Gaussian 参数** (ModelHiddenParams 类):

```python
# 架构参数
self.use_anchor_deformation = False
self.num_anchors = 1024
self.anchor_k = 10
self.mask_ratio = 0.25
self.transformer_dim = 64
self.transformer_heads = 4
self.transformer_layers = 2
self.anchor_time_embed_dim = 16
self.anchor_pos_embed_dim = 32

# 损失参数
self.lambda_phys = 0.1
self.lambda_anchor_smooth = 0.01
self.phys_warmup_steps = 2000
```

### 6. README.md 修改

新增 **PhysX-Gaussian** 章节:

- 架构说明（FPS、KNN、Transformer、Masking）
- 训练命令示例
- 参数表格说明

---

## 当前状态

✅ **PhysX-Gaussian 完全可用**：

- 使用 `.contiguous()` 确保 CUDA 兼容性
- 梯度正常流经 anchor transformer 网络
- L_phys 和 L_anchor_smooth 损失已启用
- 训练正常运行 (GPU 利用率 ~82%)

---

## 2025-12-02 ~ 2025-12-04 更新

## PhysX-Boosted: 双分支融合架构

### 设计思路

**策略**: "站在巨人肩膀上，触及更高处"

```text
Δμ_total = Δμ_hexplane(t) + Δμ_anchor(t)
```

- 保留 100% X²-Gaussian Baseline（HexPlane、所有损失、渲染）
- 添加 Anchor Transformer 作为"物理校正力"
- HexPlane: "画皮肤"（高频纹理、微形变）
- Anchor: "画骨架"（解剖结构、物理一致性）

### Boosted 架构 新增参数

```python
# x2_gaussian/arguments/__init__.py
self.use_boosted = False  # 启用 PhysX-Boosted 模式
self.disable_4d_tv = False  # 消融研究：禁用 L_4d_tv
```

---

## PhysX-Boosted V5: 可学习权重融合

### V5 公式

```text
Δx_total = (1 - α) · Δx_hexplane + α · Δx_anchor
α = sigmoid(τ), τ 是可学习参数
```

### V5 新增参数

```python
self.use_learnable_balance = False  # 启用 V5
self.balance_alpha_init = 0.5       # 初始 α 值
self.balance_lr = 0.001             # α 的学习率
self.lambda_balance = 0.0           # L_balance = (α - 0.5)² 正则化权重
```

### 特殊处理

- `α = 0.0`: 纯 HexPlane 模式（禁用 Anchor）
- `α = 1.0`: 纯 Anchor 模式（禁用 HexPlane）
- `balance_lr = 0`: 固定 α，不学习

---

## PhysX-Boosted V6: 正交梯度投影

### 核心思想

HexPlane (A) 是"基底"，Anchor (B) 学习残差

- **Forward**: `Δx_total = Δx_hex + Δx_anchor`（直接相加）
- **Backward**: 投影掉 Anchor 梯度沿 HexPlane 梯度方向的分量

  ```text
  grad_B_orth = grad_B - proj_{grad_A}(grad_B)
  ```

### V6 新增参数

```python
self.use_orthogonal_projection = False  # 启用 V6
self.ortho_projection_strength = 1.0   # 投影强度
```

### V6 实现

```python
# anchor_module.py
class OrthogonalGradientProjection(torch.autograd.Function):
    @staticmethod
    def backward(ctx, grad_output):
        dx_hex, = ctx.saved_tensors
        unit_hex = dx_hex / (torch.norm(dx_hex, dim=-1, keepdim=True) + 1e-8)
        dot_product = torch.sum(grad_output * unit_hex, dim=-1, keepdim=True)
        projection = dot_product * unit_hex
        grad_anchor_orthogonal = grad_output - strength * projection
        return grad_anchor_orthogonal, None, None
```

---

## PhysX-Boosted V7: 不确定性感知融合

### V7 公式

HexPlane 和 Anchor 都输出位移 + 不确定性 (log σ²)

```text
w_A = 1/(σ_A² + ε), w_B = 1/(σ_B² + ε)
Δx_final = (w_A·Δx_hex + w_B·Δx_anchor) / (w_A + w_B)
```

### Kendall 损失 (CVPR 2017)

```text
L_total = L_render/(2Σ) + 0.5·log(Σ)  where Σ = σ_A² + σ_B²
```

### V7 新增参数

```python
self.use_uncertainty_fusion = False  # 启用 V7
self.uncertainty_eps = 1e-6
self.lambda_uncertainty = 0.5
self.uncertainty_init = 0.0  # 初始 log(σ²)
```

---

## PhysX-Boosted V8: 反向正交梯度投影

### 与 V6 对调

- Anchor (A) 是"基底"，学习容易捕捉的模式
- HexPlane (B) 被约束只学习残差（正交方向）

### V8 新增参数

```python
self.use_reverse_orthogonal_projection = False  # 启用 V8
```

---

## PhysX-Boosted V9: V5 + 极端情况支持

### 特性

结合 V5 可学习权重，并支持 α=0 和 α=1 极端情况：

```python
# anchor_module.py __init__
if balance_alpha_init == 0.0:
    self._is_pure_hexplane = True
    self.balance_logit = None
elif balance_alpha_init == 1.0:
    self._is_pure_anchor = True
    self.balance_logit = None
else:
    tau_init = math.log(alpha_clamped / (1 - alpha_clamped))
    self.balance_logit = nn.Parameter(torch.tensor(tau_init))
```

### 融合逻辑

```python
# forward 中
if self._is_pure_hexplane:
    dx_combined = dx_hex
elif self._is_pure_anchor:
    dx_combined = dx_anchor
else:
    alpha = torch.sigmoid(self.balance_logit)
    dx_combined = (1 - alpha) * dx_hex + alpha * dx_anchor
```

---

## Bug 修复: backward through graph a second time

### 根本原因

`anchor_positions` 从 Gaussian 参数 (`self._xyz`) 初始化时保留了计算图，导致跨迭代图冲突。

### 修复方案

1. **initialize_anchors()**: 存储前 detach

   ```python
   indices = farthest_point_sampling(points.detach(), actual_num_anchors)
   self.anchor_positions = points[indices].detach().clone()
   ```

2. **forward_anchors()**: 嵌入前 detach

   ```python
   anchor_pos = self.anchor_positions.detach()
   ```

3. **update_knn_binding()**: 输入输出都 detach

   ```python
   knn_indices, knn_weights = compute_knn_weights(gaussian_positions.detach(), ...)
   self.knn_indices = knn_indices.detach()
   self.knn_weights = knn_weights.detach()
   ```

4. **get_deformed_centers()**: 添加 `.contiguous()`

   ```python
   means3D_deformed = means3D_deformed.contiguous()
   ```

### 关键洞察

Rasterizer **兼容** anchor deformation。问题是**计算图生命周期管理**，不是 rasterizer 不兼容。任何来自 `requires_grad=True` 参数并存储为类属性的张量必须显式 `.detach()`。

---

## 新增工具

### 1. STNF4D 数据集转换 (`tools/convert_stnf4d_to_x2gaussian.py`)

将 STNF4D 项目的 `.pickle` 数据集转换为 X2-Gaussian 兼容格式：

- 调整 phase 索引（1-based → 0-based）
- 添加 time 字段
- 保持原始 train/val 划分
- 复制 scanner 参数和 GT volumes

```bash
python tools/convert_stnf4d_to_x2gaussian.py \
  --input_dir /path/to/STNF4D_code/data \
  --output_dir data/
```

### 2. 鲁棒性测试数据集生成 (`tools/create_robustness_datasets.py`)

创建两种鲁棒性测试数据集：

#### 方向1: 周期扰动（模拟不均匀呼吸）

```bash
python tools/create_robustness_datasets.py \
  --input data/dir_4d_case1.pickle \
  --phase_noise 0.15  # 15% 相位扰动
```

#### 方向2: 稀疏视角

```bash
python tools/create_robustness_datasets.py \
  --input data/dir_4d_case1.pickle \
  --view_ratio 0.5  # 保留 50% 视角
```

### 3. PSNR/SSIM 计算方法对比 (`tools/compare_metrics.py`)

对比 X2-Gaussian 和 STNF4D 的评价指标计算差异：

| 方法 | 归一化 | 公式 |
|------|--------|------|
| X2-Gaussian | 无 | `PSNR = 10 * log10(MAX² / MSE)` |
| STNF4D | 分别归一化到 [0,1] | `PSNR = 20 * log10(1.0 / sqrt(MSE))` |

```bash
python tools/compare_metrics.py \
  --model_path output/xxx/point_cloud/iteration_5000 \
  --data_path data/XCAT.pickle
```

---

## 已生成的数据集

| 数据集 | 描述 | 训练视角 |
|--------|------|----------|
| `dir_4d_case1_noise0.15.pickle` | 15% 周期扰动 | 300 |
| `dir_4d_case1_sparse50.pickle` | 50% 稀疏视角 | 150 |
| `XCAT.pickle` | STNF4D 转换 | 100 |
| `S01_004_256_60.pickle` | STNF4D 转换 | 240 |
| `100_HM.pickle` | STNF4D 转换 | 100 |

---

## 训练命令汇总

### Baseline

```bash
nohup python train.py -s data/XCAT.pickle \
  --save_iterations 30000 50000 --save_checkpoint \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  > log/train_baseline_XCAT.log 2>&1 &
```

### PhysX-Boosted V9 (α=0.99)

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_prior 0.0 --lambda_tv 0.0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  > log/train_physx_boosted_v9_alpha0.99.log 2>&1 &
```

### 鲁棒性测试

```bash
# 周期扰动 - PhysX-Boosted
nohup python train.py -s data/dir_4d_case1_noise0.15.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --iterations 50000 > log/train_noise0.15_physx.log 2>&1 &

# 稀疏视角 - PhysX-Boosted
nohup python train.py -s data/dir_4d_case1_sparse50.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --iterations 50000 > log/train_sparse50_physx.log 2>&1 &
```

---

## 文件修改汇总 (2025-12-02 ~ 2025-12-04)

| 文件 | 修改类型 | 描述 |
|------|----------|------|
| `x2_gaussian/arguments/__init__.py` | 修改 | V5-V9 参数定义 |
| `x2_gaussian/gaussian/anchor_module.py` | 修改 | Boosted 融合、正交投影、不确定性融合 |
| `x2_gaussian/gaussian/gaussian_model.py` | 修改 | load_from_model_path 修复、V5 优化器参数 |
| `train.py` | 修改 | V5-V9 损失计算、日志记录 |
| `tools/convert_stnf4d_to_x2gaussian.py` | 新建 | STNF4D 数据转换 |
| `tools/create_robustness_datasets.py` | 新建 | 鲁棒性测试数据生成 |
| `tools/compare_metrics.py` | 新建 | PSNR/SSIM 计算方法对比 |
| `README.md` | 修改 | 训练命令更新 |

---

## 当前实验状态

✅ **正在运行的实验**:

- XCAT Baseline
- S01 Baseline
- XCAT PhysX-Boosted V9 (α=0.99)
- S01 PhysX-Boosted V9 (α=0.99)

⏳ **待运行的实验**:

- 周期扰动 Baseline vs PhysX-Boosted
- 稀疏视角 Baseline vs PhysX-Boosted
- 不同 α 值消融 (0.0, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0)

---

### 2025-12-05 ~ 2025-12-06 更新

## PhysX-Boosted V10-V16: 掩码建模策略演进

本次更新实现了一系列掩码建模策略的变体（V10-V16），旨在通过自监督学习增强锚点形变网络的鲁棒性。

### 设计目标

**核心问题**: 如何利用 BERT 风格的掩码建模让锚点 Transformer 学习到更鲁棒的形变表示？

**理想效果**: 即使部分锚点信息缺失，网络也能通过学习到的空间-时间关系推断正确的形变。

---

## V10: 解耦掩码 (Decoupled Mask)

### V10 设计思路

将掩码建模与渲染**解耦**：

- 渲染路径：使用完整的 `forward_anchors()` 输出
- L_phys 路径：使用独立的 `forward_anchors_masked()` 输出

```text
渲染: forward_anchors(t, mask=False) → dx_full → render() → L_render
                    ↓ detach
L_phys: forward_anchors_masked(t) → dx_masked → L1(dx_masked[mask], dx_full[mask])
```

### V10 参数

```python
# x2_gaussian/arguments/__init__.py
self.use_decoupled_mask = False  # 启用 V10
```

### V10 实现位置

| 文件 | 方法 | 说明 |
|------|------|------|
| `anchor_module.py` | `forward_anchors()` | 当 `use_decoupled_mask=True` 时跳过掩码 |
| `anchor_module.py` | `forward_anchors_masked()` | 专用于 L_phys 的掩码前向传播 |
| `anchor_module.py` | `compute_physics_completion_loss()` | 计算 L_phys，只在被掩码的锚点上 |

### V10 问题分析

**失败原因**: L_phys 是"自我预测"任务——教师和学生都来自同一个网络。网络可能学会作弊（记忆），而不是学习真正的物理关系。

---

## V11: 预训练-微调 (Pretrain-Finetune)

### V11 设计思路

分两阶段训练：

1. **预训练阶段** (前 N 步): 高掩码比例 (70%)，只用 L_phys
2. **微调阶段** (N 步后): 低掩码比例或无掩码，加入 L_render

```text
Stage 1 (Pretrain): mask_ratio=0.7, L = L_phys only
Stage 2 (Finetune): mask_ratio=0.25, L = L_render + L_phys
```

### V11 参数

```python
self.use_pretrain_finetune = False  # 启用 V11
self.pretrain_steps = 5000          # 预训练步数
self.pretrain_mask_ratio = 0.7     # 预训练阶段掩码比例
```

### V11 实现位置

| 文件 | 位置 | 说明 |
|------|------|------|
| `anchor_module.py` | `__init__` | 添加 `_in_pretrain_stage` 状态变量 |
| `anchor_module.py` | `forward_anchors()` | 根据阶段选择 mask_ratio |
| `train.py` | `scene_reconstruction()` | 预训练阶段跳过 densification |

### V11 问题分析

**失败原因**: 预训练阶段没有外部监督（L_render），L_phys 仍然是自我预测。网络无法学到有意义的表示。

---

## V12: 时间掩码 (Temporal Mask)

### V12 设计思路

掩码整个时间步，而不是空间锚点：

```text
时间 t1: [a1, a2, a3, ..., aM] → 正常处理
时间 t2: [MASK, MASK, MASK, ..., MASK] → 被掩码
时间 t3: [a1, a2, a3, ..., aM] → 正常处理
```

模型需要从其他时间步的信息推断被掩码时间步的形变。

### V12 参数

```python
self.use_temporal_mask = False  # 启用 V12
self.temporal_mask_ratio = 0.2  # 时间步被掩码的概率
```

### 实现位置

| 文件 | 位置 | 说明 |
|------|------|------|
| `anchor_module.py` | `forward_anchors()` | 基于 `time_bin` 决定是否掩码所有锚点 |
| `anchor_module.py` | `forward_anchors_masked()` | 同上 |

### 问题分析

**失败原因**: 单时间步处理时，掩码整个时间步 = 丢失所有空间信息。这太难了——模型没有任何线索来推断形变。

---

## V13: 一致性正则化 (Consistency Regularization)

### V13 设计思路

将掩码作为**数据增强**，而不是预测目标：

```text
Teacher (无掩码): forward_anchors_unmasked(t) → dx_full (detach)
Student (有掩码): forward_with_mask(t) → dx_masked
Loss: L_consist = ||dx_masked - dx_full.detach()||

# 渲染使用 dx_full（无掩码），L_consist 是辅助损失
```

**关键区别**: 损失在**所有锚点**上计算，不仅仅是被掩码的锚点。

### V13 参数

```python
self.use_consistency_mask = False  # 启用 V13
self.lambda_consist = 0.1         # L_consist 权重
```

### V13 实现

```python
# anchor_module.py: compute_consistency_loss()
def compute_consistency_loss(self, time_emb: torch.Tensor) -> torch.Tensor:
    # 1. 获取教师输出（无掩码，detach）
    unmasked_out = self.forward_anchors_unmasked(time_emb).detach()
    
    # 2. 学生分支：重新嵌入 + 掩码 + transformer
    anchor_features = self.input_proj(anchor_input).unsqueeze(0)
    num_mask = int(M * self.mask_ratio)
    perm = torch.randperm(M, device=device)
    masked_indices = perm[:num_mask]
    mask_tokens = self.mask_token.expand(1, num_mask, -1)
    anchor_features[0, masked_indices] = mask_tokens.squeeze(0)
    
    anchor_features = self.transformer(anchor_features)
    masked_out = self.displacement_head(anchor_features).squeeze(0)
    
    # 3. L1 损失（所有锚点）
    loss = F.l1_loss(masked_out, unmasked_out)
    return loss
```

### 物理意义

教导网络：**即使部分输入被扰动，输出应该保持稳定**。这增强了对输入噪声的鲁棒性。

---

## V14/V15: 时间平滑 (Temporal Smoothness)

### V14/V15 设计思路

惩罚锚点运动的"加速度"：

```text
dx(t-ε), dx(t), dx(t+ε)
acceleration = dx(t+ε) - 2*dx(t) + dx(t-ε)  # 二阶差分
L_temporal = ||acceleration||²
```

**物理意义**: 自然运动应该是平滑的（加速度接近零）。惩罚高加速度 = 鼓励线性运动。

### V14/V15 参数

```python
self.use_temporal_interp = False  # 启用 V14
self.lambda_interp = 0.1         # L_temporal 权重
self.interp_context_range = 0.2  # 时间范围 ε
```

### 实现

```python
# anchor_module.py: compute_temporal_interp_loss()
def compute_temporal_interp_loss(self, time_emb: torch.Tensor) -> torch.Tensor:
    t_val = t.item()
    epsilon = self.interp_context_range / 2
    t_prev_val = max(0.0, t_val - epsilon)
    t_next_val = min(1.0, t_val + epsilon)
    
    # 当前时间步（有梯度）
    dx_t = self._last_anchor_displacements
    
    # 邻近时间步（无梯度）
    with torch.no_grad():
        dx_prev = self.forward_anchors_unmasked(t_prev)
        dx_next = self.forward_anchors_unmasked(t_next)
    
    # 二阶差分（加速度）
    acceleration = dx_next - 2 * dx_t + dx_prev
    loss = (acceleration ** 2).mean()
    return loss
```

### V15 = V13 + V14

同时启用一致性正则化和时间平滑：

```bash
python train.py ... --use_consistency_mask --lambda_consist 0.1 \
                    --use_temporal_interp --lambda_interp 0.1
```

---

## V16: 拉格朗日时空掩码建模 (Lagrangian Spatio-Temporal Masked Modeling)

### 核心创新 (V16)

#### V10-V15 的问题

1. 单时间步处理，无法建模时间关系
2. [MASK] token **替换**原始 token，丢失位置信息
3. 掩码建模是辅助损失 (λ=0.1)，不是主要目标

#### V16 解决方案

1. Token 是 (锚点, 时间) 对，Transformer 同时建模空间和时间
2. mask_flag_embed 是**加性**嵌入，保留位置/时间信息
3. L_lagbert 是主要目标 (λ=0.5)

### 架构

```text
输入: anchor_pos [M, 3], t_center (e.g., 0.5)

1. 采样时间窗口: t_vec = [t-Δ, t, t+Δ] = [0.4, 0.5, 0.6]  (K=3)

2. 构建 K*M 个时空 token:
   token_{k,j} = pos_embed(anchor_j) + time_embed(t_k)

3. 加 mask_flag_embed (不是替换！):
   token_{k,j} += mask_flag_embed(flag_{k,j})  # flag ∈ {0, 1}

4. Transformer 跨所有 (锚点, 时间) token attention:
   features = transformer([1, K*M, d_model])

5. 预测位移: dx = displacement_head(features) → [K, M, 3]
```

### 损失计算

```python
# Full pass (无 mask)
mask_full = zeros(K, M)
dx_full = forward_anchors_st(anchor_pos, t_vec, mask_full)

# Masked pass (有 mask)
mask_flags = sample_st_mask(K, M)  # 随机选择 30% token
dx_masked = forward_anchors_st(anchor_pos, t_vec, mask_flags)

# L_lagbert: 只在被 mask 的 token 上计算
L_lagbert = L1(dx_masked[mask==1], dx_full[mask==1].detach())

# 渲染用 center 时间步的 full pass 输出
dx_center = dx_full[center_idx]  # [M, 3]
```

### V16 参数

```python
# 核心参数
self.use_spatiotemporal_mask = False  # 启用 V16
self.lambda_lagbert = 0.5            # L_lagbert 权重（主要目标！）
self.st_window_size = 3              # 时间窗口大小 K
self.st_time_delta = 0.1             # 时间步长 Δ
self.st_mask_ratio = 0.3             # (锚点, 时间) token 掩码比例

# Fix 1: mask_embed 缩放因子
self.st_mask_embed_scale = 1.0       # 默认 1.0 = 原始行为
                                      # 设为 0.1 可减少 mask_embed 干扰

# Fix 2: 渲染与 L_lagbert 耦合
self.st_coupled_render = False        # 默认 False = 分离的前向传播
                                      # 设为 True = 共享前向传播
```

### 关键实现

#### 1. mask_flag_embed (不是替换，是相加)

```python
# anchor_module.py: __init__
if self.use_spatiotemporal_mask:
    # Mask flag embedding: {0: unmasked, 1: masked} -> d_model
    self.mask_flag_embed = nn.Embedding(2, self.d_model)
    nn.init.normal_(self.mask_flag_embed.weight, std=0.02)

# anchor_module.py: forward_anchors_st()
if mask_flags is not None and self.use_spatiotemporal_mask:
    mask_flags_flat = mask_flags.reshape(K * M).long()
    mask_embed = self.mask_flag_embed(mask_flags_flat)
    # Fix 1: 应用缩放因子
    features_flat = features_flat + self.st_mask_embed_scale * mask_embed
```

#### 与 BERT 的区别

- BERT: `token[mask] = [MASK]` (替换，丢失位置信息)
- V16: `token += mask_flag_embed(flag)` (加性，保留位置信息)

#### 2. 时空前向传播

```python
# anchor_module.py: forward_anchors_st()
def forward_anchors_st(self, anchor_pos, t_vec, mask_flags=None):
    K = t_vec.shape[0]  # 时间步数
    M = anchor_pos.shape[0]  # 锚点数
    
    # 1. 位置嵌入（所有锚点共享）
    pos_embed = self.anchor_embed(anchor_pos.detach())  # [M, pos_dim]
    
    # 2. 时间嵌入
    time_embeds = [self.time_encode(t_vec[k].unsqueeze(0)) for k in range(K)]
    time_embeds = torch.cat(time_embeds, dim=0)  # [K, time_dim]
    
    # 3. 构建时空 token
    tokens = []
    for k in range(K):
        time_k = time_embeds[k:k+1].expand(M, -1)
        token_k = torch.cat([pos_embed, time_k], dim=-1)
        tokens.append(token_k)
    tokens = torch.stack(tokens, dim=0)  # [K, M, pos_dim + time_dim]
    
    # 4. 投影到 d_model
    features_flat = self.input_proj(tokens.reshape(K*M, -1))
    
    # 5. 添加 mask_flag_embed
    if mask_flags is not None:
        mask_embed = self.mask_flag_embed(mask_flags.reshape(K*M).long())
        features_flat = features_flat + self.st_mask_embed_scale * mask_embed
    
    # 6. Transformer (跨所有 K*M tokens)
    features = self.transformer(features_flat.unsqueeze(0))
    
    # 7. 预测位移
    displacements = self.displacement_head(features.squeeze(0))
    return displacements.reshape(K, M, 3)
```

#### 3. Fix 2: 渲染与 L_lagbert 耦合

**问题**: 原实现中，渲染用 `forward_anchors()`，L_lagbert 用 `compute_lagbert_loss()`，是两次独立的前向传播。

**解决方案**: 当 `st_coupled_render=True` 时：

1. 渲染前先调用 `compute_lagbert_loss()`
2. 缓存 `dx_center` 和 `L_lagbert`
3. `forward_anchors()` 检测到缓存后直接返回

```python
# anchor_module.py: compute_lagbert_loss()
if self.st_coupled_render:
    self._st_coupled_dx_center = dx_center  # 缓存

# anchor_module.py: forward_anchors()
if self.st_coupled_render and self.use_spatiotemporal_mask:
    if hasattr(self, '_st_coupled_dx_center') and self._st_coupled_dx_center is not None:
        dx_center = self._st_coupled_dx_center
        self._st_coupled_dx_center = None  # 清除缓存
        return dx_center

# train.py: 渲染前调用
_v16_lagbert_cached = None
if stage == 'fine' and gaussians.is_st_coupled_render():
    time_tensor = torch.tensor(viewpoint_cam.time).to(gaussians.get_xyz.device)
    _, _v16_lagbert_cached = gaussians.compute_lagbert_loss(time_tensor, is_training=True)

# 渲染（会使用缓存的 dx_center）
render_pkg = render(viewpoint_cam, gaussians, ...)

# 后面使用缓存的 L_lagbert
if _v16_lagbert_cached is not None:
    L_lagbert = _v16_lagbert_cached
```

### V16 训练命令

```bash
# V16 基础版
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --use_spatiotemporal_mask --lambda_lagbert 0.1 \
  --st_window_size 1 --st_time_delta 0.1 --st_mask_ratio 0.5 \
  --dirname dir_4d_case1_physx_boosted_v16 \
  > log/train_v16.log 2>&1 &

# V16 + Fix 1 (降低 mask_embed 干扰)
nohup python train.py ... \
  --st_mask_embed_scale 0.1 \
  > log/train_v16_fix1.log 2>&1 &

# V16 + Fix 2 (耦合渲染)
nohup python train.py ... \
  --st_coupled_render \
  > log/train_v16_fix2.log 2>&1 &

# V16 + 两个 Fix
nohup python train.py ... \
  --st_mask_embed_scale 0.1 --st_coupled_render \
  > log/train_v16_both_fixes.log 2>&1 &
```

---

## 文件修改汇总 (2025-12-05 ~ 2025-12-06)

| 文件 | 修改类型 | 新增/修改行数 | 说明 |
|------|----------|---------------|------|
| `x2_gaussian/arguments/__init__.py` | 修改 | +50 | V10-V16 参数定义 |
| `x2_gaussian/gaussian/anchor_module.py` | 修改 | +300 | V13-V16 核心实现 |
| `x2_gaussian/gaussian/gaussian_model.py` | 修改 | +40 | V13-V16 包装方法 |
| `train.py` | 修改 | +30 | V13-V16 损失计算 |

### 详细修改内容 (V10-V16)

#### 1. arguments/**init**.py

```python
# V10: 解耦掩码
self.use_decoupled_mask = False

# V11: 预训练-微调
self.use_pretrain_finetune = False
self.pretrain_steps = 5000
self.pretrain_mask_ratio = 0.7

# V12: 时间掩码
self.use_temporal_mask = False
self.temporal_mask_ratio = 0.2

# V13: 一致性正则化
self.use_consistency_mask = False
self.lambda_consist = 0.1

# V14: 时间平滑
self.use_temporal_interp = False
self.lambda_interp = 0.1
self.interp_context_range = 0.2

# V16: 时空掩码建模
self.use_spatiotemporal_mask = False
self.lambda_lagbert = 0.5
self.st_window_size = 3
self.st_time_delta = 0.1
self.st_mask_ratio = 0.3
self.st_mask_embed_scale = 1.0   # Fix 1
self.st_coupled_render = False    # Fix 2
```

#### 2. anchor_module.py 新增方法

| 方法 | 版本 | 说明 |
|------|------|------|
| `forward_anchors_unmasked()` | V13 | 无掩码前向传播（教师） |
| `compute_consistency_loss()` | V13 | 一致性正则化损失 |
| `compute_temporal_interp_loss()` | V14 | 时间平滑损失 |
| `forward_anchors_st()` | V16 | 时空前向传播 |
| `sample_time_window()` | V16 | 采样时间窗口 |
| `sample_st_mask()` | V16 | 采样时空掩码 |
| `compute_lagbert_loss()` | V16 | 拉格朗日-BERT 损失 |

#### 3. gaussian_model.py 新增方法

| 方法 | 说明 |
|------|------|
| `compute_consistency_loss(time)` | V13 包装器 |
| `compute_temporal_smoothness_loss(time)` | V14 包装器 |
| `compute_lagbert_loss(time, is_training)` | V16 包装器 |
| `is_st_coupled_render()` | 检查是否启用 Fix 2 |
| `get_st_cached_dx_center()` | 获取缓存的 dx_center |

#### 4. train.py 损失计算

```python
# V13: 一致性正则化
if use_consistency_mask and lambda_consist > 0:
    L_consist = gaussians.compute_consistency_loss(time_tensor)
    loss["consist"] = L_consist
    loss["total"] = loss["total"] + lambda_consist * L_consist

# V14: 时间平滑
if use_temporal_interp and lambda_interp > 0:
    L_temporal = gaussians.compute_temporal_smoothness_loss(time_tensor)
    loss["temporal_smooth"] = L_temporal
    loss["total"] = loss["total"] + lambda_interp * L_temporal

# V16: 时空掩码建模 (Fix 2: 使用缓存)
if use_spatiotemporal_mask and lambda_lagbert > 0:
    if _v16_lagbert_cached is not None:
        L_lagbert = _v16_lagbert_cached
    else:
        _, L_lagbert = gaussians.compute_lagbert_loss(time_tensor, is_training=True)
    loss["lagbert"] = L_lagbert
    loss["total"] = loss["total"] + lambda_lagbert * L_lagbert
```

---

## 版本对比总结

| 版本 | 核心思想 | Token 定义 | 掩码方式 | 损失目标 | 问题 |
|------|----------|------------|----------|----------|------|
| V10 | 解耦掩码 | 单时间步锚点 | 替换为 [MASK] | 被掩码锚点 | 自我预测 |
| V11 | 预训练-微调 | 单时间步锚点 | 替换为 [MASK] | 被掩码锚点 | 无外部监督 |
| V12 | 时间掩码 | 单时间步锚点 | 掩码整个时间步 | 被掩码锚点 | 丢失所有信息 |
| V13 | 一致性正则 | 单时间步锚点 | 替换为 [MASK] | 所有锚点 | 仍是弱正则 |
| V14 | 时间平滑 | 单时间步锚点 | 无掩码 | 加速度 | 不涉及掩码 |
| **V16** | **时空建模** | **(锚点,时间)对** | **加性嵌入** | **被掩码token** | 主要目标 |

---

## 当前实验状态 (V16)

✅ **V16 实验运行中**:

- `dir_4d_case1_physx_boosted_v16` (λ_lagbert=0.1, window=1, mask=0.5)

⏳ **待运行的实验**:

- V16 + Fix 1 (st_mask_embed_scale=0.1)
- V16 + Fix 2 (st_coupled_render)
- V16 + 两个 Fix
- 不同 λ_lagbert 消融 (0.1, 0.2, 0.5)
- 不同 st_window_size 消融 (1, 3, 5)

---

## PhysX-Boosted M1: Uncertainty-Gated Residual Fusion

**日期**: 2025-12-11

### 设计思想

M1 是一个重大的模型结构升级，将原来固定标量 α≈0.99 的线性融合改成**基于不确定性的自适应门控融合**。

#### 论文记号

- **Φ_L(x,t)**: 拉格朗日场（Anchor-based Transformer）- 捕获骨架运动
- **Φ_E(x,t)**: 欧拉场（HexPlane）- 捕获高频残差细节
- **s_E(x,t)**: 欧拉场的对数方差输出 = log(σ_E²)
- **β(x,t)**: 自适应门控系数，取决于欧拉场的不确定性

#### 融合公式

##### V5 (固定 α)

```text
Φ(x,t) = (1 - α) · Φ_E(x,t) + α · Φ_L(x,t)
```

##### M1 (不确定性门控残差)

```text
Φ(x,t) = Φ_L(x,t) + β(x,t) · Φ_E(x,t)
```

设计哲学：

- 拉格朗日是"骨架"（始终贡献）
- 欧拉是"残差校正器"（只有在有信心时才贡献）
- 高 σ_E（不确定）→ 低 β → 更信任拉格朗日
- 低 σ_E（有信心）→ 高 β → 欧拉贡献更多

#### β(x,t) 计算方式

**Bayes 模式**（基于逆方差加权）:

```text
β = σ_L² / (σ_L² + σ_E²(x,t))
σ_E² = exp(s_E)
```

其中 σ_L² 是常数超参数（如 1e-4）

##### Sigmoid 模式

```text
β = sigmoid((τ - s_E(x,t)) / λ)
```

其中 τ 是阈值，λ 是温度

#### 稀疏正则 L_gate

为了鼓励"能用拉格朗日解释的尽量用拉格朗日"：

```text
L_gate = E_{x,t}[|β(x,t)|_1]
```

### M1 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--fusion_mode` | `fixed_alpha` | 融合模式：`fixed_alpha` 或 `uncertainty_gated` |
| `--gate_mode` | `bayes` | 门控模式：`bayes` 或 `sigmoid` |
| `--sigma_L2` | `1e-4` | Bayes 模式下的拉格朗日方差常数 |
| `--gate_tau` | `0.0` | Sigmoid 模式下的阈值 τ |
| `--gate_lambda` | `1.0` | Sigmoid 模式下的温度 λ |
| `--beta_min` | `0.0` | β 最小值 |
| `--beta_max` | `1.0` | β 最大值 |
| `--m1_lambda_gate` | `0.0` | L_gate 稀疏正则权重 |
| `--eulerian_uncertainty_hidden_dim` | `32` | 不确定性头隐藏层维度 |
| `--eulerian_s_E_init` | `0.0` | s_E 输出的初始值 |

### M1 修改的文件

| 文件 | 修改内容 |
|------|----------|
| `x2_gaussian/arguments/__init__.py` | 新增 M1 参数 |
| `x2_gaussian/gaussian/deformation.py` | 新增 uncertainty_head，输出 s_E |
| `x2_gaussian/gaussian/anchor_module.py` | 新增 M1 融合逻辑，β 计算，L_gate 计算 |
| `train.py` | 新增 L_gate 损失，M1 统计日志 |
| `scripts/visualize_beta.py` | **新建** 可视化脚本 |
| `README.md` | 新增 M1 训练命令 |
| `CHANGELOG_physx_gaussian.md` | 新增 M1 变更日志 |

### 核心代码变更

#### 1. deformation.py - Eulerian 不确定性输出

```python
# 在 create_net() 中新增
self.uncertainty_head = nn.Sequential(
    nn.ReLU(),
    nn.Linear(self.W, eulerian_uncertainty_hidden),
    nn.ReLU(),
    nn.Linear(eulerian_uncertainty_hidden, 1)  # Output: s_E = log(σ²)
)

# 在 forward_dynamic() 中计算 s_E
if self.fusion_mode == 'uncertainty_gated':
    self._last_s_E = self.uncertainty_head(hidden)  # [N, 1]
```

#### 2. anchor_module.py - M1 融合

```python
# 在 forward() 中的 fusion 分支
elif self.fusion_mode == 'uncertainty_gated':
    # 获取 s_E
    s_E = self.original_deformation.get_last_s_E()  # [N, 1]
    
    # 计算 β
    if self.gate_mode == 'bayes':
        sigma2_E = torch.exp(s_E)
        beta = self.sigma_L2 / (self.sigma_L2 + sigma2_E + 1e-8)
    else:  # sigmoid
        beta = torch.sigmoid((self.gate_tau - s_E) / (self.gate_lambda + 1e-8))
    
    beta = beta.clamp(min=self.beta_min, max=self.beta_max)
    
    # M1 融合公式: Φ = Φ_L + β · Φ_E
    dx_combined = dx_anchor + beta * dx_hex
```

#### 3. train.py - L_gate 损失

```python
# M1: Uncertainty-Gated Residual Fusion
fusion_mode = getattr(hyper, 'fusion_mode', 'fixed_alpha')
lambda_gate = getattr(hyper, 'lambda_gate', 0.0)
if fusion_mode == 'uncertainty_gated' and gaussians._deformation_anchor is not None:
    if lambda_gate > 0:
        L_gate = gaussians._deformation_anchor.compute_gate_sparsity_loss()
        loss["gate_sparsity"] = L_gate
        loss["total"] = loss["total"] + lambda_gate * L_gate
    
    # Log M1 statistics
    m1_stats = gaussians._deformation_anchor.get_m1_statistics()
    if m1_stats.get('beta_mean') is not None:
        loss["m1_beta_mean"] = m1_stats['beta_mean']
```

### 可视化工具

新增 `scripts/visualize_beta.py`：

```bash
# 生成 β(x,t) 贡献图
python scripts/visualize_beta.py \
    --checkpoint path/to/ckpt \
    --time 0.5 \
    --output output/m1_viz

# 输出:
#   - beta_slice_t0.50.png: β 的2D切片可视化
#   - beta_stats_t0.50.png: β 和 s_E 的统计分布
#   - beta_volume_t0.50.npz: 体素化的 β 数据
```

### 训练命令示例

```bash
# M1-Bayes
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --mask_ratio 0.0 \
  --fusion_mode uncertainty_gated \
  --gate_mode bayes --sigma_L2 1e-4 \
  --m1_lambda_gate 1e-4 \
  --iterations 50000 \
  > log/train_m1_bayes_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# M1-Sigmoid
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --mask_ratio 0.0 \
  --fusion_mode uncertainty_gated \
  --gate_mode sigmoid --gate_tau 0.0 --gate_lambda 1.0 \
  --m1_lambda_gate 1e-4 \
  --iterations 50000 \
  > log/train_m1_sigmoid_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 向后兼容性

- 当 `fusion_mode="fixed_alpha"` 时，行为与 V5 完全一致
- 所有原有参数和实验仍然有效

### 待验证实验

- [ ] M1-Bayes vs V5 baseline
- [ ] M1-Sigmoid vs V5 baseline  
- [ ] 不同 σ_L² 消融 (0.1, 1.0, 10.0)
- [ ] 不同 m1_lambda_gate 消融 (0, 1e-4, 1e-3)
- [ ] β 均值是否在合理范围 (0.3 ~ 0.7)

---

## M1 Bug 修复 (2025-12-11)

### Bug 1: 融合公式错误

**问题**: 原公式将两个预测完整位移的分支直接相加，导致过冲。

```python
# 错误 (导致位移过冲):
dx_combined = dx_anchor + beta * dx_hex

# 正确 (加权平均):
dx_combined = (1 - beta) * dx_anchor + beta * dx_hex
```

**影响**: M1-Sigmoid PSNR 下降 0.4 dB，M1-Bayes PSNR 下降 4.6 dB

### Bug 2: σ_L² 默认值过小

**问题**: σ_L² = 1e-4 导致 β ≈ 0.0001，Eulerian 贡献被完全压制。

```python
# 错误:
self.sigma_L2 = 1e-4  # β = 1e-4 / (1e-4 + 1) ≈ 0.0001

# 正确:
self.sigma_L2 = 1.0   # β = 1.0 / (1.0 + 1) = 0.5
```

#### 修复后行为

- β 初始值 ≈ 0.5（当 s_E = 0）
- 网络可以学习调整 s_E 来控制 β
- σ_L² 越小 → β 越小 → 越信任 Lagrangian

### Bug 3: ds_anchor/dr_anchor 未定义

**问题**: 尝试对 Anchor 不存在的 scale/rotation 输出进行融合。

```python
# 错误 (ds_anchor 不存在):
ds_combined = (1 - beta) * ds_anchor + beta * ds_hex

# 正确 (Anchor 只预测位置):
ds_combined = beta * ds_hex  # Scale 只来自 HexPlane
dr_combined = beta * dr_hex  # Rotation 只来自 HexPlane
```

### Bug 4: 参数未从 V5 学习

**问题**: σ_L²=1.0 导致 β=0.5，与 V5 最优 α=0.99 差异太大。

```python
# 从 V5 学习: α=0.99 最优 → HexPlane 权重 = 0.01
# 在 M1 中: β = HexPlane 权重
# 目标: β ≈ 0.01 当 s_E=0

# Bayes: β = σ_L² / (σ_L² + 1) = 0.01 → σ_L² ≈ 0.01
self.sigma_L2 = 0.01  # 而不是 1.0

# Sigmoid: sigmoid(τ/λ) = 0.01 → τ ≈ -4.6 (λ=1)
self.gate_tau = -4.6  # 而不是 0.0
```

### 修正后的公式解释

```text
Φ(x,t) = (1 - β) · Φ_L + β · Φ_E   [位置]
ds = β · ds_hex                     [Scale - 只来自 HexPlane]
dr = β · dr_hex                     [Rotation - 只来自 HexPlane]

其中:
- β = σ_L² / (σ_L² + exp(s_E))  [Bayes, σ_L²=0.01 → β≈0.01]
- β = sigmoid((τ - s_E) / λ)    [Sigmoid, τ=-4.6 → β≈0.01]

设计理念 (从 V5 α=0.99 学习):
- 初始时 β ≈ 0.01，行为类似 V5
- 网络可以学习 s_E，在 HexPlane 有信心的区域增加 β
- 实现"自适应"的融合而不是固定的 α=0.99
```

## 结果对比

| Model | PSNR3D | vs V5 |
|-------|--------|-------|
| **V5 baseline (α=0.99)** | **45.117** | — |
| M1-Sigmoid v3 | 45.119 | +0.002 |
| M1.1-Sigmoid | 44.316 | **-0.801** ❌ |

## M1.1 失败原因分析

### 梯度解耦破坏了 V5 的训练动态

- V5: HexPlane 贡献 1%，获得 1% 梯度 → **平衡**
- M1.1: HexPlane 贡献 1-3%，获得 100% 梯度 → **失衡**

这种失衡导致 HexPlane 过度训练，学习了过大的位移预测。

## M1.2 设计：V5 基础上的微小扰动

**核心思想**：保持 V5 的 99:1 梯度流，只在此基础上做**微小**的不确定性调整。

```text
# V5 公式 (保持不变)
dx_v5 = 0.99 * dx_anchor + 0.01 * dx_hex

# M1.2: 添加微小扰动 γ
dx_m1.2 = (0.99 - γ) * dx_anchor + (0.01 + γ) * dx_hex

其中 |γ| ≤ 0.005 (最多 ±0.5% 调整)
```

### γ 基于不确定性计算

- s_E 高 (不确定) → γ < 0 → 减少 HexPlane (从1%降到0.5%)
- s_E 低 (确定) → γ > 0 → 增加 HexPlane (从1%升到1.5%)

添加 `gamma_max` 参数：

更新打印语句：

修复 M1.2 代码，移除旧的 beta 计算逻辑：

更新 train.py 中的日志代码：

运行快速测试验证 M1.2：

M1.2 测试结果：

- **hex_weight=0.015 (γ=+0.005)** - 达到最大值，说明网络想要更多 HexPlane
- **s_E_mean=-6.97** - 非常确定，γ 被推到最大

γ 达到上限，说明可能需要更大的 `gamma_max`。让我调整参数并提供训练命令：

---

## M1.2 训练命令

基于测试结果，γ 达到上限说明网络想要更多 HexPlane 贡献。提供两个版本测试：

```bash
# M1.2 v1: γ_max=0.005 (HexPlane 权重范围: 0.5% ~ 1.5%)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --fusion_mode uncertainty_gated \
  --gate_tau 0.0 --gate_lambda 1.0 \
  --gamma_max 0.005 \
  --m1_lambda_gate 0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_m1_2_g005 \
  > log/train_physx_boosted_m1_2_g005_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# M1.2 v2: γ_max=0.01 (HexPlane 权重范围: 0% ~ 2%)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --fusion_mode uncertainty_gated \
  --gate_tau 0.0 --gate_lambda 1.0 \
  --gamma_max 0.01 \
  --m1_lambda_gate 0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_m1_2_g01 \
  > log/train_physx_boosted_m1_2_g01_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

## M1.2 设计总结

| 版本 | 设计 | 问题 |
|------|------|------|
| M1.0/v3 | β从s_E计算，范围0-1 | β固定在0.01，无自适应 |
| M1.1 | 梯度解耦 | 破坏V5训练动态，-0.8dB |
| **M1.2** | V5基础+微小γ扰动 | **保持99:1梯度流** |

**M1.2 核心公式**：

```text
dx = (0.99 - γ) * dx_anchor + (0.01 + γ) * dx_hex
γ = γ_max * tanh((τ - s_E) / λ)

γ_max=0.005时: HexPlane 权重 ∈ [0.5%, 1.5%]
γ_max=0.01时:  HexPlane 权重 ∈ [0%, 2%]
```

**预期**：M1.2 保持 V5 的训练动态，同时允许基于不确定性的微调，应该能够匹配或略微超越 V5。

---

## M2: Bounded Learnable Perturbation (ICML formulation)

**日期**: 2025-12-12

### M2 设计动机

从 M1 实验中学到:

- M1.0/v3: β 固定在 0.01，无自适应（本质上就是 V5）
- M1.1: 梯度解耦破坏了 V5 的训练动态，性能下降 0.8 dB
- M1.2: 在 V5 基础上做微小扰动，但仍受限于加权平均公式

M2 采用更优雅的 ICML 风格公式：**Base + Bounded Perturbation**

### M2 核心公式

```text
Φ(x,t) = Φ_L(x,t) + ε · tanh(Φ_E(x,t))

其中:
- Φ_L: Lagrangian (Anchor) - 完整结构基底 (100%)
- Φ_E: Eulerian (HexPlane) - 有界可学习微扰
- ε = ε_max · sigmoid(ρ), ρ 是可学习标量
- tanh 约束微扰幅度，防止 shortcut learning
```

### 与 V5 的关系

```text
V5:  dx = 0.01·dx_hex + 0.99·dx_anchor  [固定加权平均]
M2:  dx = dx_anchor + ε·tanh(dx_hex)    [基底 + 有界微扰]

M2 更优雅因为:
1. 结构-微扰分离明确（Base + Perturbation）
2. Lagrangian 是完整基底，不是 99%
3. ε 有界（sigmoid）防止 shortcut
4. tanh 约束微扰幅度，保证数值稳定
```

### 初始化匹配 V5

```python
# ε_init = 0.01 复现 V5 α=0.99 的经验优势
# ρ_init = logit(ε_init / ε_max)
eps_ratio = min(max(self.eps_init / self.eps_max, 1e-6), 1 - 1e-6)
rho_init = math.log(eps_ratio / (1 - eps_ratio))  # logit
self.rho = nn.Parameter(torch.tensor(rho_init, dtype=torch.float32))
```

### M2 新增参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `fusion_mode` | `"fixed_alpha"` | 设为 `"bounded_perturb"` 启用 M2 |
| `eps_max` | `0.02` | ε 的上界 (2%) |
| `eps_init` | `0.01` | ε 的初始值 (1%, 匹配 V5) |
| `use_tanh` | `True` | 是否使用 tanh 约束微扰 |

### 修改的文件

| 文件 | 修改内容 |
|------|---------|
| `anchor_module.py` | M2 参数初始化、bounded_perturb fusion 模式、getter 方法 |
| `arguments/__init__.py` | M2 config 参数 |
| `train.py` | M2 日志记录 |
| `README.md` | M2 文档和训练命令 |

### M2 核心代码

```python
elif self.fusion_mode == 'bounded_perturb':
    # Compute ε = ε_max * sigmoid(ρ)
    eps = self.eps_max * torch.sigmoid(self.rho)
    self._last_eps = eps.item()  # Cache for logging
    
    # Apply H(·) = tanh(·) to bound perturbation magnitude
    if self.use_tanh:
        dx_perturb = torch.tanh(dx_hex)
        ds_perturb = torch.tanh(ds_hex)
        dr_perturb = torch.tanh(dr_hex)
    else:
        dx_perturb = dx_hex
        ds_perturb = ds_hex
        dr_perturb = dr_hex
    
    # M2 Fusion: Base (Lagrangian) + Bounded Perturbation (Eulerian)
    dx_combined = dx_anchor + eps * dx_perturb
    ds_combined = eps * ds_perturb
    dr_combined = eps * dr_perturb
```

### M2 训练命令

```bash
# M2 (bounded_perturb)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --fusion_mode bounded_perturb \
  --eps_max 0.02 --eps_init 0.01 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_m2 \
  > log/train_physx_boosted_m2_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### ICML 叙事优势

> M2 将 V5 的经验性发现（α=0.99 最优）提升为更优雅的数学表述：
> Lagrangian 场作为**完整结构基底**，Eulerian 场作为**有界可学习微扰**。
> 初始化 ε≈0.01 复现 V5 的经验优势，同时 ε 的端到端学习允许模型
> 自动发现最优的 Eulerian 贡献比例。这是一种"先验引导的自适应"。

---

## M1.3 & M2.05: 基于实验分析的改进

**日期**: 2025-12-12

### 实验结果回顾

| 模型 | 50K PSNR3D | Δ vs V5 NoMask |
|------|------------|----------------|
| V5 NoMask (α=0.99) | 45.001 | baseline |
| **M1.2 g005** | **45.298** | **+0.297 dB ✓** |
| M1.2 g01 | 45.001 | 0 |
| **M2** | **39.486** | **-5.515 dB ✗** |

### M1.2 g005 成功原因

```text
观察:
- hex_weight = 0.0150 (γ = +0.005, 达到最大值)
- s_E_mean: -5.7 → -3.5 (不确定性降低)

关键洞察:
1. 保持 V5 的加权平均公式结构
2. HexPlane 权重从 1% 增加到 1.5% 提升了性能
3. V5 的 α=0.99 不是最优，α=0.985 更好
```

### M2 失败原因

```text
观察:
- ε = 0.010000 (恒定), ρ = 0.0000 (从未学习)

致命问题:
1. 公式 dx = dx_anchor + ε·tanh(dx_hex) 与 V5 结构不同
2. Anchor 得到 100% 权重而非 99%
3. tanh 压缩了 HexPlane 信号
4. ε 没有学习（可能未加入优化器）
```

### M1.3: 基于 M1.2 发现的优化

**M1.3a**: 固定 α=0.985 (hex=1.5%)

```bash
--balance_alpha_init 0.985 --balance_lr 0
```

**M1.3b**: 可学习 α，从 0.985 开始

```bash
--balance_alpha_init 0.985 --balance_lr 0.0001
```

### M2.05: 修复公式结构

#### 问题修复

1. 恢复加权平均结构: `dx = (1-ε)·dx_anchor + ε·dx_hex`
2. 移除 tanh（它压缩了信号）
3. ε_init = 0.015（基于 M1.2 发现）

#### M2.05 核心代码

```python
# M2.05: Weighted average (same structure as V5!)
eps = self.eps_max * torch.sigmoid(self.rho)
alpha = 1.0 - eps

dx_combined = alpha * dx_anchor + eps * dx_hex
ds_combined = eps * ds_hex
dr_combined = eps * dr_hex
```

### 训练命令

```bash
# M1.3a: Fixed α=0.985
python train.py ... --balance_alpha_init 0.985 --balance_lr 0

# M1.3b: Learnable α from 0.985
python train.py ... --balance_alpha_init 0.985 --balance_lr 0.0001

# M2.05: Learnable weighted average
python train.py ... --fusion_mode bounded_perturb --eps_max 0.03 --eps_init 0.015
```

---

## M2.1: Trust-Region Schedule

**日期**: 2025-12-12

### M2.1 设计动机

M2 的 ρ 从第一步就开始学习，可能导致优化器"走捷径"。M2.1 引入 trust-region schedule 强制模型先在 Lagrangian manifold 收敛。

### 两种模式

**freeze_rho** (硬冻结):

- 前 N 步完全冻结 ρ，不更新梯度
- ε 维持在 eps_init 附近

**warmup_cap** (软约束):

- ε_eff = min(ε_raw, ε_max * step/warmup_steps)
- 逐步放开 residual 容量

### M2.1 新增参数

```python
schedule_mode = "freeze_rho"  # ["none", "freeze_rho", "warmup_cap"]
freeze_steps = 2000           # For freeze_rho
warmup_steps = 5000           # For warmup_cap
```

### M2.1 训练命令

```bash
# M2.1-a: freeze_rho
--fusion_mode bounded_perturb --schedule_mode freeze_rho --freeze_steps 2000

# M2.1-b: warmup_cap  
--fusion_mode bounded_perturb --schedule_mode warmup_cap --warmup_steps 5000
```

---

## M2.2: Residual Normalization

**日期**: 2025-12-13

### M2.2 设计动机

> "Residual normalization makes ε a true trust-region radius by preventing magnitude leakage from the Eulerian stream."

M2/M2.1 中 tanh 可能无法完全控制 residual 幅值，导致 ε_eff 不能真正代表"微扰半径"。M2.2 引入更强的归一化方式。

### 三种 H(Δ) 模式

| 模式 | 公式 | 特点 |
|------|------|------|
| **tanh** | H(Δ) = tanh(Δ) | M2/M2.1 baseline，[-1,1] 约束 |
| **rmsnorm** | H(Δ) = Δ / rms(Δ) | RMS 归一化，幅值 O(1) |
| **unitnorm** | H(Δ) = Δ / ‖Δ‖ | L2 单位化，ε 精确控制半径 |

### M2.2 新增参数

```python
residual_mode = "tanh"  # ["tanh", "rmsnorm", "unitnorm"]
norm_eps = 1e-6         # Numerical stability
```

### M2.2 Logging 新增

- `mean_norm_E`: ‖Δ‖ 均值（归一化前）
- `mean_norm_H`: ‖H(Δ)‖ 均值（归一化后）

### M2.2 训练命令

```bash
# M2.2-a: rmsnorm (推荐)
--fusion_mode bounded_perturb --schedule_mode freeze_rho \
--residual_mode rmsnorm --norm_eps 1e-6

# M2.2-b: unitnorm
--fusion_mode bounded_perturb --schedule_mode freeze_rho \
--residual_mode unitnorm --norm_eps 1e-6
```

#### M2.2a: rmsnorm + freeze_rho (FIXED eps)

nohup /root/miniconda3/envs/x2_gaussian/bin/python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --fusion_mode bounded_perturb \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --eps_max 0.03 --eps_init 0.015 \
  --residual_mode rmsnorm --norm_eps 1e-6 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_m2_2a_rmsnorm_v2 \
  > log/train_physx_boosted_m2_2a_rmsnorm_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &

#### M2.2b: unitnorm + freeze_rho (FIXED eps)

nohup /root/miniconda3/envs/x2_gaussian/bin/python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --fusion_mode bounded_perturb \
  --schedule_mode freeze_rho --freeze_steps 2000 \
  --eps_max 0.03 --eps_init 0.015 \
  --residual_mode unitnorm --norm_eps 1e-6 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname dir_4d_case1_physx_boosted_m2_2b_unitnorm_v2 \
  > log/train_physx_boosted_m2_2b_unitnorm_v2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
---

## M3: Low-Frequency Leakage Penalty

**日期**: 2025-12-13

### M3 设计动机

> "Low-frequency leakage regularization prevents the Eulerian stream from explaining global motion, reserving it for high-frequency corrective details around the Lagrangian manifold."

问题：Eulerian stream 可能"偷学"低频/大尺度运动，绕过 ε 的约束。
解决：直接惩罚 residual 的低频分量，逼迫它只补高频细节。

### M3 核心公式

```text
L_LP = mean_i || LP(Δ_i) ||^2
```

其中 LP(·) 是低通算子：

- 若 Δ 在邻域内变化缓慢（低频）→ LP(Δ) 大 → 被惩罚
- 若 Δ 是局部高频修正 → LP(Δ) ≈ 0 → 不惩罚

### 两种 LP 模式

| 模式 | 公式 | 特点 |
|------|------|------|
| **knn_mean** | LP(Δ_i) = mean_{j∈N_k(i)} Δ_j | 惩罚局部均值，推荐 |
| **graph_laplacian** | LP(Δ_i) = Δ_i - mean_{j∈N(i)} Δ_j | 图拉普拉斯，更理论 |

### M3 新增参数

```python
lp_enable = False       # Master switch
lambda_lp = 0.01        # L_LP weight
lp_mode = "knn_mean"    # ["knn_mean", "graph_laplacian"]
lp_k = 8                # Number of neighbors
lp_subsample = 2048     # Subsample for efficiency
```

### M3 Logging 新增

- `m3_lp_loss`: L_LP 值
- `m3_lp_mean`: mean ||LP(Δ)||
- `m3_lp_ratio`: ||LP(Δ)|| / ||Δ|| (越小说明高频占比越高)

### M3 训练命令

```bash
# M3: LP regularization with kNN mean
--fusion_mode bounded_perturb --schedule_mode freeze_rho \
--lp_enable --lambda_lp 0.01 --lp_mode knn_mean --lp_k 8

# M3: LP regularization with graph Laplacian
--fusion_mode bounded_perturb --schedule_mode freeze_rho \
--lp_enable --lambda_lp 0.01 --lp_mode graph_laplacian --lp_k 8
```

---

## M4: Subspace Decoupling Regularization

**日期**: 2025-12-13

### M4 设计动机

> "Subspace decoupling regularization discourages the Eulerian residual from aligning with the Lagrangian deformation responses, forcing it to model complementary details rather than shortcuts."

问题：Eulerian 可能学到与 Lagrangian 相同方向的变形（shortcut），导致两个分支冗余而非互补。
解决：惩罚两个分支的"导数信息"（速度或 Jacobian）之间的余弦相似度，强制它们解耦。

### M4 核心公式

```text
L_decouple = mean_i(cos²(v_L, v_E))  # velocity_corr
L_decouple = mean_i(cos²(g_L, g_E))  # stochastic_jacobian_corr
```

### 两种 decouple 模式

| 模式 | 公式 | 特点 |
|------|------|------|
| **velocity_corr** | v = deform(x, t+dt) - deform(x, t) | 比较时间导数，便宜稳定 |
| **stochastic_jacobian_corr** | g = grad(dot(deform, w), x) | 比较空间 Jacobian，更理论 |

### M4 新增参数

```python
decouple_enable = False           # Master switch
lambda_decouple = 0.01            # L_decouple weight
decouple_mode = "velocity_corr"   # ["velocity_corr", "stochastic_jacobian_corr"]
decouple_subsample = 2048         # Subsample for efficiency
decouple_stopgrad_L = True        # Detach Lagrangian (only train Eulerian)

# velocity_corr specific
decouple_dt = 0.02                # Time step for velocity

# stochastic_jacobian_corr specific
decouple_num_dirs = 1             # Number of random directions
```

### Logging 新增

- `m4_decouple_loss`: L_decouple 值
- `m4_corr_mean`: mean |cos(v_L, v_E)| 或 |cos(g_L, g_E)|
- `m4_grad_L_norm`, `m4_grad_E_norm`: Jacobian 模式下的梯度范数

### M4 训练命令

```bash
# M4: velocity correlation decoupling
--fusion_mode bounded_perturb --schedule_mode freeze_rho \
--decouple_enable --lambda_decouple 0.01 --decouple_mode velocity_corr

# M4: stochastic Jacobian decoupling
--fusion_mode bounded_perturb --schedule_mode freeze_rho \
--decouple_enable --lambda_decouple 0.01 --decouple_mode stochastic_jacobian_corr
```

### Bug 修复记录 (2025-12-13)

1. **Bug #1**: `NameError: name 'means3D' is not defined`
   - **位置**: `anchor_module.py` forward() 中的 M3/M4 缓存代码
   - **原因**: 使用了错误的变量名 `means3D` 和 `times`
   - **修复**: 改为正确的 `gaussian_positions` 和 `time_emb`

2. **Bug #2**: `AttributeError: 'AnchorDeformationNet' object has no attribute 'anchors_initialized'`
   - **位置**: `anchor_module.py` `_get_anchor_deformation()` 方法
   - **原因**: 使用了不存在的属性名 `anchors_initialized`
   - **修复**: 改为正确的 `initialized`

3. **Bug #3**: `ValueError: too many values to unpack (expected 3)`
   - **位置**: `anchor_module.py` `_get_anchor_deformation()` 方法
   - **原因**: `forward_anchors()` 返回单个 tensor，不是 tuple
   - **修复**: `anchor_dx, _, _ = self.forward_anchors(times)` → `anchor_dx = self.forward_anchors(times, is_training=False)`

4. **Bug #4**: `AttributeError: 'NoneType' object has no attribute 'unsqueeze'`
   - **位置**: `anchor_module.py` `_get_eulerian_deformation()` 方法
   - **原因**: HexPlane deformation 需要 scales, rotations, density 参数
   - **修复**: 创建 dummy tensors 传递给 HexPlane forward

5. **Bug #5**: `RuntimeError: derivative for aten::grid_sampler_2d_backward is not implemented`
   - **位置**: `anchor_module.py` `_compute_jacobian_corr_loss()` 方法
   - **原因**: Jacobian 模式使用 autograd 计算二阶导数，但 grid_sampler_2d 不支持
   - **修复**: 改用空间有限差分代替 autograd 计算 Jacobian 方向导数

### 配置修复 (2025-12-13)

**问题**: M2 best_baseline 和 M3/M4 实验使用了错误的 `bounded_perturb` + `residual_mode tanh` 配置，导致 ||Δ|| 爆炸 (0.37→2.12) 和 psnr3d 下降 (40.5→38.6)。

**修复**: 所有 M3/M4 实验现在继承 M2.1 最佳配置 (m2_1a_freeze_v2: psnr3d 45.325, ssim3d 0.980):

- 使用 `schedule_mode freeze_rho` (不使用 `fusion_mode bounded_perturb`)
- 移除 `residual_mode tanh`
- 保持 `eps_max 0.03`, `eps_init 0.015`, `freeze_steps 2000`

#### 影响的实验

- M2 best_baseline → 重新运行
- M3 LP knn → 重新运行  
- M4 velocity_corr v2 → 新增
- M4 jacobian_corr v2 → 新增

---

## M 系列实验完整结果汇总 (2025-12-13)

### M1.x 系列 - α Balance 优化

| 版本 | 核心公式 | 关键参数 | psnr3d | ssim3d |
|------|----------|----------|--------|--------|
| M1 bayes (早期) | Δx = (1-α)·Δx_hex + α·Δx_anchor | α=0.99 可学习, balance_lr=0.001 | 40.51 | 0.967 |
| M1 sigmoid (早期) | 同上 | α=0.99 可学习 | 44.73 | 0.977 |
| M1.1 bayes | 同上 | 改进初始化 | 44.27 | 0.975 |
| M1.1 sigmoid | 同上 | 改进初始化 | 44.32 | 0.976 |
| **M1.2 g=0.05** | Φ = (0.99-γ)·Φ_L + (0.01+γ)·Φ_E | γ=0.05, hex 权重 6% | **45.30** | **0.981** |
| M1.2 g=0.1 | 同上 | γ=0.1, hex 权重 11% | 45.00 | 0.979 |
| M1.3a α=0.985 | Δx = (1-α)·Δx_hex + α·Δx_anchor | α=0.985 固定, balance_lr=0 | 45.23 | 0.980 |
| M1.3b α=0.985 lr | 同上 | α=0.985 可学习, balance_lr=0.0001 | 45.11 | 0.980 |

**M1 结论**: M1.2 g=0.05 最优，HexPlane 权重 ~6% 是最佳平衡点

### M2.x 系列 - Trust-Region + Residual Normalization

| 版本 | 核心公式 | 关键参数 | psnr3d | ssim3d |
|------|----------|----------|--------|--------|
| M2 (早期) | dx = dx_anchor + ε·tanh(dx_hex) | eps=0.01 恒定 | 39.49 | 0.953 |
| M2.05 | Φ = (1-ε)·Φ_L + ε·Φ_E | eps_max=0.03, eps_init=0.015 | 45.13 | 0.980 |
| **M2.1a freeze_rho** | Φ = (1-ε_eff)·Φ_L + ε_eff·Φ_E | schedule=freeze_rho, freeze_steps=2000 | **45.33** | **0.980** |
| M2.1b warmup | 同上 | schedule=warmup_cap, warmup_steps=5000 | 45.18 | 0.980 |
| M2 none | 同上 | schedule=none | 45.30 | 0.980 |
| M2.2 tanh | Φ = (1-ε_eff)·Φ_L + ε_eff·H(Φ_E) | H=tanh, bounded_perturb | 39.50 | 0.953 |
| M2.2a rmsnorm | 同上 | H=rmsnorm | 37.69 | 0.919 |
| M2.2b unitnorm | 同上 | H=unitnorm | 38.96 | 0.940 |

**M2 结论**: M2.1a freeze_rho 最优。M2.2 残差归一化严重损害性能（-6~7 dB）

### M3 系列 - Low-Frequency Leakage Penalty

| 版本 | 核心公式 | 关键参数 | psnr3d | ssim3d |
|------|----------|----------|--------|--------|
| M3 LP knn (旧配置) | L_LP = λ·‖LP(Δ)‖² | bounded_perturb + lp_mode=knn_mean | 41.92 | 0.970 |
| **M3 LP knn v2** | 同上 | 继承 M2.1 + lp_enable, λ=0.01 | **45.09** | 0.979 |

**M3 结论**: 正确配置后 LP 正则化略微降低性能 (-0.24 dB vs M2.1 baseline)

### M4 系列 - Subspace Decoupling Regularization

| 版本 | 核心公式 | 关键参数 | psnr3d | ssim3d |
|------|----------|----------|--------|--------|
| M4 vel (旧配置) | L_dec = λ·\|cos(v_L, v_E)\| | bounded_perturb + velocity_corr | 41.77 | 0.970 |
| M4 jac (旧配置) | L_dec = λ·\|cos(J_L·w, J_E·w)\| | bounded_perturb + jacobian_corr | 41.98 | 0.972 |
| **M4 vel v2** | 同上 | 继承 M2.1 + velocity_corr, λ=0.01 | **45.07** | 0.979 |
| **M4 jac v2** | 同上 | 继承 M2.1 + jacobian_corr, λ=0.01 | **45.09** | 0.980 |

**M4 结论**: 正确配置后 decoupling 正则化无明显提升 (基本持平 M2.1 baseline)

### 总排名 (psnr3d 降序)

| 排名 | 实验 | psnr3d | ssim3d | Δ vs M2.1 |
|------|------|--------|--------|-----------|
| 1 | **M2.1a freeze_rho** | **45.33** | 0.980 | **基准** |
| 2 | M2 none | 45.30 | 0.980 | -0.03 |
| 3 | M1.2 g=0.05 | 45.30 | 0.981 | -0.03 |
| 4 | M1.3a α=0.985 | 45.23 | 0.980 | -0.10 |
| 5 | M2 best_baseline | 45.19 | 0.980 | -0.14 |
| 6 | M2.1b warmup | 45.18 | 0.980 | -0.15 |
| 7 | M2.05 | 45.13 | 0.980 | -0.20 |
| 8 | M3 LP knn v2 | 45.09 | 0.979 | -0.24 |
| 9 | M4 jac v2 | 45.09 | 0.980 | -0.24 |
| 10 | M4 vel v2 | 45.07 | 0.979 | -0.26 |
| ... | ... | ... | ... | ... |
| 27 | M2.2a rmsnorm | 37.69 | 0.919 | **-7.64** |

### 关键发现

1. **最优配置**: M2.1a freeze_rho (无残差归一化，前 2000 步冻结 ε)
2. **M3/M4 正则化无效**: 所有正则化尝试都没有超越简单的 M2.1 baseline
3. **M2.2 残差归一化有害**: rmsnorm/unitnorm 导致 6-7 dB 严重下降
4. **HexPlane 权重最优点**: ~1.5%-6% (对应 α=0.985~0.94)
5. **旧配置 (bounded_perturb) 有缺陷**: 导致 ||Δ|| 爆炸，性能下降 3-4 dB

---

## [2025-12-19] 多数据集对比实验

### 背景

为验证 V5 nomask 和 s3.1 在不同数据集上的泛化能力，启动以下对比实验：

### 实验设计

对比三个版本：
1. **x2-gaussian baseline**: 原始论文方法（无 anchor deformation）
2. **V5 nomask**: PhysX-Boosted V5 配置（α=0.99, mask_ratio=0.0）
3. **s3.1**: 在 V5 基础上 release scale（ds = ds_hex）

### 启动的实验

#### dir_4d_case2 数据集

```bash
# V5 nomask
nohup python train.py -s data/dir_4d_case2.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname v5_nomask_case2_$(date +%Y%m%d_%H%M%S) \
  > log/v5_nomask_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### 4dlung_case4 数据集

```bash
# x2-gaussian baseline
nohup python train.py -s data/4dlung_case4.pickle \
  --save_iterations 30000 50000 --save_checkpoint \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --dirname baseline_4dlung_case4_$(date +%Y%m%d_%H%M%S) \
  > log/baseline_4dlung_case4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V5 nomask
nohup python train.py -s data/4dlung_case4.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname v5_nomask_4dlung_case4_$(date +%Y%m%d_%H%M%S) \
  > log/v5_nomask_4dlung_case4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# s3.1
nohup python train.py -s data/4dlung_case4.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s3_1_4dlung_case4_$(date +%Y%m%d_%H%M%S) \
  > log/s3_1_4dlung_case4_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### spare_mc_4d_case1 数据集

```bash
# x2-gaussian baseline
nohup python train.py -s data/spare_mc_4d_case1.pickle \
  --save_iterations 30000 50000 --save_checkpoint \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --dirname baseline_spare_mc_case1_$(date +%Y%m%d_%H%M%S) \
  > log/baseline_spare_mc_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# V5 nomask
nohup python train.py -s data/spare_mc_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --iterations 50000 --test_iterations 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname v5_nomask_spare_mc_case1_$(date +%Y%m%d_%H%M%S) \
  > log/v5_nomask_spare_mc_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# s3.1
nohup python train.py -s data/spare_mc_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --mask_ratio 0.0 --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s3_1_spare_mc_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s3_1_spare_mc_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### 实验日志

| 数据集 | 版本 | Log 文件 |
|--------|------|----------|
| dir_4d_case2 | V5 nomask | `log/v5_nomask_case2_20251219_022341.log` |
| 4dlung_case4 | baseline | `log/baseline_4dlung_case4_20251219_022341.log` |
| 4dlung_case4 | V5 nomask | `log/v5_nomask_4dlung_case4_20251219_022341.log` |
| 4dlung_case4 | s3.1 | `log/s3_1_4dlung_case4_20251219_022341.log` |
| spare_mc_4d_case1 | baseline | `log/baseline_spare_mc_case1_20251219_022341.log` |
| spare_mc_4d_case1 | V5 nomask | `log/v5_nomask_spare_mc_case1_20251219_022341.log` |
| spare_mc_4d_case1 | s3.1 | `log/s3_1_spare_mc_case1_20251219_022341.log` |

---

## [2025-12-19] 鲁棒性数据集生成工具扩展

### 概述

扩展了数据集生成工具，支持更多真实 CT 采集挑战的模拟：

1. **组合扰动** (phase_noise + sparse views)
2. **投影测量噪声** (Poisson / Gaussian)
3. **有限角度** (Limited-angle)
4. **金属伪影/条纹** (Metal artifacts / Stripes)
5. **运动模糊** (Motion blur)

### 1. 组合扰动数据集 (`tools/create_robustness_datasets.py --combine`)

现在支持在单个 pickle 中同时应用相位扰动和稀疏视角：

```bash
# 生成 phase_noise=0.5 + sparse=50% 组合数据集
python tools/create_robustness_datasets.py \
  --input data/dir_4d_case1.pickle \
  --phase_noise 0.5 \
  --view_ratio 0.5 \
  --combine

# 输出: data/dir_4d_case1_noise0.5_sparse50.pickle
```

### 2. 投影测量噪声 (`tools/add_projection_noise.py`)

模拟真实 CT 采集中的量子噪声和电子噪声：

| 噪声类型 | 描述 | 关键参数 |
|----------|------|----------|
| `poisson` | 光子计数统计噪声 | `--photon_scale` (1e4=中等, 1e3=重度) |
| `gaussian` | 电子/读出噪声 | `--gaussian_std` (0.05=中等) |
| `mixed` | Poisson + Gaussian | 两者组合 |

```bash
# Poisson 噪声 (光子计数 1e4)
python tools/add_projection_noise.py \
  --input data/dir_4d_case1.pickle \
  --noise_type poisson \
  --photon_scale 1e4

# Gaussian 噪声 (std=5%)
python tools/add_projection_noise.py \
  --input data/dir_4d_case1.pickle \
  --noise_type gaussian \
  --gaussian_std 0.05

# 混合噪声
python tools/add_projection_noise.py \
  --input data/dir_4d_case1.pickle \
  --noise_type mixed \
  --photon_scale 1e4 \
  --gaussian_std 0.02
```

### 3. 有限角度 CT (`tools/create_limited_angle.py`)

模拟角度覆盖不完整的情况（比稀疏视角更难）：

| 模式 | 描述 |
|------|------|
| `single` | 单一连续扇区 (如仅 0°-120°) |
| `dual` | 两个对立扇区 (如 0°-90° 和 180°-270°) |

```bash
# 仅保留 120° 范围
python tools/create_limited_angle.py \
  --input data/dir_4d_case1.pickle \
  --angle_range 120

# 保留两个 90° 对立扇区
python tools/create_limited_angle.py \
  --input data/dir_4d_case1.pickle \
  --angle_range 90 \
  --mode dual
```

### 4. 金属伪影/条纹 (`tools/add_metal_artifacts.py`)

模拟探测器故障和金属物体引起的伪影：

| 伪影类型 | 描述 | 关键参数 |
|----------|------|----------|
| `stripe` | 条纹伪影（探测器行故障） | `--stripe_ratio`, `--stripe_intensity` |
| `dead` | 坏死像素 | `--dead_ratio` |
| `metal` | 金属高衰减 | `--metal_intensity`, `--metal_width` |
| `ring` | 环形伪影 | `--ring_count`, `--ring_intensity` |

```bash
# 条纹伪影 (5% 探测器行受影响)
python tools/add_metal_artifacts.py \
  --input data/dir_4d_case1.pickle \
  --artifact_type stripe \
  --stripe_ratio 0.05

# 金属伪影
python tools/add_metal_artifacts.py \
  --input data/dir_4d_case1.pickle \
  --artifact_type metal \
  --metal_intensity 2.0

# 环形伪影
python tools/add_metal_artifacts.py \
  --input data/dir_4d_case1.pickle \
  --artifact_type ring \
  --ring_count 3
```

### 5. 运动模糊 (`tools/add_motion_blur.py`)

模拟曝光时间跨越多个呼吸相位导致的模糊：

| 模糊类型 | 描述 | 关键参数 |
|----------|------|----------|
| `phase_mix` | 相邻相位混合 | `--mix_ratio` (0-0.5) |
| `temporal_avg` | 时间方向平均 | `--window_size` |
| `exposure` | 长曝光模拟 | `--exposure_phases` |

```bash
# 相位混合模糊 (20% 混合)
python tools/add_motion_blur.py \
  --input data/dir_4d_case1.pickle \
  --blur_type phase_mix \
  --mix_ratio 0.2

# 时间平均模糊
python tools/add_motion_blur.py \
  --input data/dir_4d_case1.pickle \
  --blur_type temporal_avg \
  --window_size 3

# 长曝光模糊 (覆盖 30% 呼吸周期)
python tools/add_motion_blur.py \
  --input data/dir_4d_case1.pickle \
  --blur_type exposure \
  --exposure_phases 0.3
```

### 已生成的新数据集

| 数据集 | 描述 | 训练视角 |
|--------|------|----------|
| `dir_4d_case1_noise0.7.pickle` | 70% 相位扰动 | 300 |
| `dir_4d_case1_noise1.0.pickle` | 100% 相位扰动（极限） | 300 |
| `dir_4d_case1_noise0.5_sparse50.pickle` | 50% 相位扰动 + 50% 稀疏 | 150 |

### 难度等级参考

| 难度 | 相位扰动 | 稀疏视角 | 投影噪声 | 有限角度 |
|------|----------|----------|----------|----------|
| 简单 | 0.15 | 80% | 1e5 | 300° |
| 中等 | 0.3-0.5 | 50% | 1e4 | 180° |
| 困难 | 0.7-1.0 | 25-33% | 1e3 | 120° |
| 极限 | 1.0 + sparse | 20% | 5e2 | 90° |

### 工具文件列表

| 文件 | 功能 |
|------|------|
| `tools/create_robustness_datasets.py` | 相位扰动 + 稀疏视角 (支持 --combine) |
| `tools/add_projection_noise.py` | Poisson/Gaussian 投影噪声 |
| `tools/create_limited_angle.py` | 有限角度 CT |
| `tools/add_metal_artifacts.py` | 金属伪影/条纹/环形伪影 |
| `tools/add_motion_blur.py` | 运动模糊 |

---

## [2025-12-19] s4.1：Anchor-only 位置变化场（基于 s3.1）

### 定义

在 s3.1（`--s3_release_scale`）基础上，引入 s4.1 位置变化场：

1. **位置变化场**：

```
Δx(n,t) = α · Δx_anchor(n,t)
```

1. **尺度变化场**（沿用 s3.1）：

```
Δs(n,t) = Δs_hex(n,t)
```

1. **旋转变化场**（保持 V5 默认）：

```
Δr(n,t) = (1-α) · Δr_hex(n,t)
```

其中 `α` 仍由 V5 的 learnable balance 系数给出（通常用 `--balance_alpha_init 0.99 --balance_lr 0` 固定在 0.99）。

### 新增参数

- `--s4_1_anchor_only_position`

### case1 启动命令

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s4_1_anchor_only_position \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_1_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_1_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s4.2/s4.3/s4.4：可控系数的 dx/dr 覆盖（基于 s3.1/s4.1）

为精确复现固定系数（例如 1.0/0.95/0.05），新增两个参数用于覆盖 V5 分支中的融合权重：

- `--s4_dx_anchor_weight wA`
  - 作用：覆盖位置融合为 `Δx = (1-wA)·Δx_hex + wA·Δx_anchor`
  - 默认 `-1` 表示关闭覆盖（继续使用 V5 的 `α` 或 s4.1 的 `α·Δx_anchor`）
- `--s4_dr_hex_weight k`
  - 作用：覆盖旋转融合为 `Δr = k·Δr_hex`
  - 默认 `-1` 表示关闭覆盖（继续使用 V5 的 `Δr=(1-α)·Δr_hex` 或 s3.* 的 rotation 选项）

#### s4.2: 在 s4.1 基础上 Δx = Δx_anchor（k=0.01）

```
Δx = 1.0·Δx_anchor
Δs = Δs_hex
Δr = 0.01·Δr_hex
```

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_2_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_2_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### s4.3: 在 s4.2 基础上 Δr = 0.05·Δr_hex

```
Δx = 1.0·Δx_anchor
Δs = Δs_hex
Δr = 0.05·Δr_hex
```

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.05 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_3_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_3_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### s4.4: 在 s3.1 基础上 Δx = 0.05·Δx_hex + 0.95·Δx_anchor（k=0.01）

```
Δx = 0.05·Δx_hex + 0.95·Δx_anchor
Δs = Δs_hex
Δr = 0.01·Δr_hex
```

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s4_dx_anchor_weight 0.95 \
  --s4_dr_hex_weight 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_4_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_4_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### s4.5: wA=1.0（纯 Anchor position），大 rotation HexPlane（k=0.1/0.5/1.0）

```
Δx = 0·Δx_hex + 1.0·Δx_anchor
Δs = Δs_hex
Δr = k·Δr_hex
```

##### s4.5-k0.1

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.1 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_5_k01_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_5_k01_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

##### s4.5-k0.5

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.5 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_5_k05_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_5_k05_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

##### s4.5-k1.0

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 1.0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_5_k10_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_5_k10_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### s4.6: wA=1.0（纯 Anchor position），k=0（完全禁用 rotation HexPlane）

```
Δx = 0·Δx_hex + 1.0·Δx_anchor
Δs = Δs_hex
Δr = 0
```

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.0 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_6_k00_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_6_k00_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

---

## [2025-12-21] 近期补充记录（参数核对/覆盖顺序/参数量统计）

### 1) s4.5-k0.0 的 dirname 命名错误（但 cfg_args 真实参数为 k=0）

- `log/s4_5_k0_case1_20251221_051305.log`
  - Output: `./output/eb2dd3e3-e`
  - `cfg_args.yml` 关键字段：
    - `s4_dx_anchor_weight: 1.0`
    - `s4_dr_hex_weight: 0.0`
    - `s5_rot_nlerp: false`
  - 但 `cfg_args.yml` 中 `dirname` 被写成：`s4_5_k10_case1_20251221_051305`（应理解为命名错误，不代表真实参数）

### 2) s5.5-k1.0 并不等价于 clean s5.1：k 覆盖发生在 nlerp 之后

结论：在 `use_learnable_balance` 分支中，rotation 的覆盖顺序是：

1. 若 `--s5_rot_nlerp`，先得到 `dr_combined = q_fused - rotations`；
2. 若 `--s4_dr_hex_weight k >= 0` 且满足条件（未开启 `s3_zero_rotation/s3_release_rotation/s2_anchor_to_rotation`），最后会强制：

```
dr_combined = k · dr_hex
```

因此：

- `k=0` 会把 rotation 完全冻结（`dr=0`），即使开启 `--s5_rot_nlerp` 也会被覆盖掉。
- `k=1` 会使最终 rotation 等同于 HexPlane rotation（`dr=dr_hex`），同样覆盖掉 `--s5_rot_nlerp`。

对照：

- `s5_5_k10_case1_20251221_052826`
  - Output: `./output/6ad53cb6-4`
  - `cfg_args.yml`：`s4_dx_anchor_weight=1.0, s4_dr_hex_weight=1.0, s5_rot_nlerp=true`

### 3) 参数量统计与 learnable/buffer 严格区分（基于 checkpoint）

以 `output/eb2dd3e3-e/ckpt/chkpnt50000.pth`（iter=50000）为例，按 checkpoint 中的张量元素数（numel）统计：

#### 3.1 参数量粗分（不区分 learnable/buffer）

- Gaussians（可学习参数）：`578,919`
  - xyz: 157,887
  - scaling: 157,887
  - rotation: 210,516
  - density: 52,629
- HexPlane deform_network state_dict（总）：`47,282,248`
  - grid/K-Planes: 47,247,366
  - trunk_feature_out: 8,256
  - heads（pos/scales/rot + backward + uncertainty 等）：其余
- AnchorDeformationNet state_dict（总）：`48,456,660`
  - 其中 `original_deformation`（embedded HexPlane）: 47,282,248
  - 其余 anchor 网络 + KNN 缓存：1,174,412

#### 3.2 learnable parameters vs buffers（严格区分）

HexPlane deform_network（pack[10] / `self._deformation.state_dict()`）：

- total: 47,282,248
- learnable_params: 47,282,226
- buffers（register_buffer: `time_poc/pos_poc/rotation_scaling_poc`）: 16
- nonlearnable_tensors（例如 `deformation_net.grid.aabb` 为 requires_grad=False）: 6

AnchorDeformationNet（pack[14] / `self._deformation_anchor.state_dict()`）：

- total: 48,456,660
- learnable_params: 47,399,982
- buffers: 1,056,678

其中（剔除 embedded HexPlane baseline `original_deformation.*` 后）：

- learnable_params: 117,734
- buffers: 1,056,678

并且以下关键缓存均为 buffer（非 learnable）：

- `knn_indices`: 526,290（shape = [N, K]）
- `knn_weights`: 526,290（shape = [N, K]）
- `anchor_positions`: 3,072（shape = [M, 3]）
- `anchor_indices`: 1,024（shape = [M]）
- `initialized/knn_valid`: 1/1（bool 标量）

---

## [2025-12-22] s4：k 与 ds_weight 消融结果汇总 + 下一轮组合实验设计

### 0) 磁盘清理：删除所有实验的点云输出

为节省磁盘空间，决定删除所有实验目录下的：

- `output/*/point_cloud/`

保留以下关键产物用于对比与复现：

- `output/*/cfg_args.yml`
- `output/*/eval/*`
- `output/*/ckpt/*`

### 1) 本轮已完成实验（iter=50000 指标）

#### 1.1 k sweep（固定 wA=1.0, ds_weight=1.0）

注：ds_weight=1.0 通过 `--s3_release_scale` 实现（`Δs=Δs_hex`）。

| setting | log | output | PSNR2D | SSIM2D | PSNR3D_mean | SSIM3D_mean | 备注 |
|---|---|---|---:|---:|---:|---:|---|
| wA=1, k=0.02 | `log/s4_5_k002_case1_20251221_210537.log` | `output/27c660a3-a` | 44.1292 | 0.989998 | 45.1445 | 0.980067 | 正常 |
| wA=1, k=0.08 | `log/s4_5_k008_case1_20251221_210540.log` | `output/5d5ac6c3-d` | 43.9678 | 0.989986 | 45.4020 | 0.980673 | 3D 最优 |
| wA=1, k=0.9 | `log/s4_5_k09_case1_20251221_210544.log` | `output/cdf927b7-f` | NaN | NaN | 24.7051 | 0.339739 | 训练出现 NaN（loss=nan，log 报 "NaN or Inf"） |

结论：

- k=0.9 明确不稳定（NaN），不作为候选。
- 在可用范围内，k=0.02 更偏 2D 指标，k=0.08 更偏 3D 指标。

#### 1.2 ds_weight sweep（固定 wA=1.0, k=0.01）

注：为使 `ds_weight` 可控，本组实验不使用 `--s3_release_scale`，并通过固定 `α` 实现：

```
Δs = (1-α)·Δs_hex
ds_weight = 1-α
```

| setting | log | output | PSNR2D | SSIM2D | PSNR3D_mean | SSIM3D_mean | 备注 |
|---|---|---|---:|---:|---:|---:|---|
| wA=1, k=0.01, ds=0.99 | `log/s4_5_k001_ds099_case1_20251221_210852.log` | `output/096401a3-3` | 43.9916 | 0.990007 | 45.0711 | 0.980120 | 正常 |
| wA=1, k=0.01, ds=0.90 | `log/s4_5_k001_ds090_case1_20251221_210856.log` | `output/dea48e27-6` | 44.0813 | 0.990098 | 45.3725 | 0.980672 | 本轮综合最优点之一 |
| wA=1, k=0.01, ds=0.50 | `log/s4_5_k001_ds050_case1_20251221_210900.log` | `output/7d9673b9-2` | NaN | NaN | 24.7051 | 0.339739 | eval2d 为 NaN（疑似训练不稳定） |
| wA=1, k=0.01, ds=0.10 | `log/s4_5_k001_ds010_case1_20251221_210903.log` | `output/c6d55162-1` | 43.7947 | 0.989728 | 45.1387 | 0.980105 | 正常但变差 |
| wA=1, k=0.01, ds=0.01 | `log/s4_5_k001_ds001_case1_20251221_210907.log` | `output/2d954d33-f` | 43.7814 | 0.989819 | 45.1971 | 0.980275 | 正常但不占优 |

结论：

- ds_weight≈0.9 是明显的甜点（2D/3D 同时占优）。
- ds_weight 太小（0.1/0.01）会退化。
- ds_weight=0.5 出现 NaN，视为不稳定点，不作为候选。

#### 1.3 wA=0 对照（dx 纯 HexPlane）

对照实验：

- `log/s4_wA0_k001_ds100_case1_20251221_232931.log`

结论：效果不佳（符合预期：wA=0 意味 dx 完全回退到 HexPlane，Anchor 对 motion skeleton 的贡献被移除）。

### 2) 下一轮最有价值的组合实验（围绕当前最优附近）

当前观察到的最优邻域：

- 候选中心 A：`k≈0.02, ds≈0.9`（偏 2D）
- 候选中心 B：`k≈0.08, ds≈0.9`（偏 3D）

统一设定：

- 固定 `wA=1.0`（`--s4_dx_anchor_weight 1.0`）
- 固定 `ds_weight` 通过 `α` 实现（不使用 `--s3_release_scale`）
- `ds_weight = 1 - α`

#### 2.1 围绕中心 A（k≈0.02, ds≈0.9）

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.015 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_combo_wA1_k0015_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_combo_wA1_k0015_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.025 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_combo_wA1_k0025_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_combo_wA1_k0025_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.15 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.02 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_combo_wA1_k0020_ds085_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_combo_wA1_k0020_ds085_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.05 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.02 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_combo_wA1_k0020_ds095_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_combo_wA1_k0020_ds095_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### 2.2 围绕中心 B（k≈0.08, ds≈0.9）

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.06 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_combo_wA1_k0060_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_combo_wA1_k0060_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.10 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_combo_wA1_k0100_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_combo_wA1_k0100_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.15 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.08 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_combo_wA1_k0080_ds085_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_combo_wA1_k0080_ds085_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.05 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.08 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_combo_wA1_k0080_ds095_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_combo_wA1_k0080_ds095_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### 2.3 [2025-12-22] s4_combo(wA=1) 本轮 8 个组合实验结果（优先 3D）

说明：所有实验均为 `case1`，`mask_ratio=0.0`，`wA=1.0`（`--s4_dx_anchor_weight 1.0`）。其中 `ds_weight = 1 - balance_alpha_init`。

| dirname | output | k (`s4_dr_hex_weight`) | ds_weight | PSNR3D_mean | SSIM3D_mean | 备注 |
|---|---|---:|---:|---:|---:|---|
| s4_combo_wA1_k0015_ds090_case1_20251222_025636 | `output/4e526ec0-e` | 0.015 | 0.90 | 45.353639 | 0.980733 | 稳定 |
| s4_combo_wA1_k0020_ds085_case1_20251222_025644 | `output/cf65f2d1-a` | 0.020 | 0.85 | 45.380846 | 0.980884 | 本轮最好 |
| s4_combo_wA1_k0020_ds095_case1_20251222_025648 | `output/f1a1f57c-0` | 0.020 | 0.95 | 45.244195 | 0.980585 | 稳定 |
| s4_combo_wA1_k0025_ds090_case1_20251222_025640 | `output/429d08e5-d` | 0.025 | 0.90 | 36.759800 | 0.903665 | 严重退化（未 NaN） |
| s4_combo_wA1_k0060_ds090_case1_20251222_025652 | `output/407f63bf-0` | 0.060 | 0.90 | 24.705123 | 0.339739 | NaN 崩溃 |
| s4_combo_wA1_k0080_ds085_case1_20251222_025659 | `output/9e0c0257-2` | 0.080 | 0.85 | 44.981205 | 0.979276 | 稳定但下降 |
| s4_combo_wA1_k0080_ds095_case1_20251222_025703 | `output/fb8f4148-a` | 0.080 | 0.95 | 43.975829 | 0.974935 | 稳定但明显下降 |
| s4_combo_wA1_k0100_ds090_case1_20251222_025655 | `output/8fcc286b-7` | 0.100 | 0.90 | 45.145577 | 0.979993 | 稳定 |

结论（用于下一轮设计）：

- **k 的稳定性呈现非单调**：`k=0.06` 直接崩溃；`k≈0.015~0.02` 表现最好且稳定。
- **ds_weight 并非越大越好**：在 `k=0.02` 与 `k=0.08` 附近，`ds=0.85` 相比 `ds=0.95` 更稳定/更高。
- `k=0.025, ds=0.90` 出现严重退化，建议做重复跑来判断是否“偶发坏解/随机种子问题”。

#### 2.4 下一轮：更合理的 8 个新实验（含额外 k001 ds09）

设计原则：集中精修 `k≈0.0125~0.02` 与 `ds≈0.85~0.90`，并增加一个 `k=0.01, ds=0.90` 作为你要求的对照；同时保留 `k=0.025, ds=0.90` 做复现验证。

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.15 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.0125 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next_wA1_k00125_ds085_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next_wA1_k00125_ds085_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.0125 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next_wA1_k00125_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next_wA1_k00125_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.15 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.015 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next_wA1_k0015_ds085_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next_wA1_k0015_ds085_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next_wA1_k00175_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next_wA1_k00175_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.15 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.02 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next_wA1_k0020_ds085_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next_wA1_k0020_ds085_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.02 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next_wA1_k0020_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next_wA1_k0020_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.025 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next_wA1_k0025_ds090_case1_repeat_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next_wA1_k0025_ds090_case1_repeat_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next_wA1_k0010_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next_wA1_k0010_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### 2.5 [2025-12-23] s4_next(wA=1) 本轮 8 个实验结果（优先 3D）

说明：本轮为 `2.4` 中的 `s4_next_*` 实际跑出的结果。`ds_weight = 1 - balance_alpha_init`。

| dirname | output | k (`s4_dr_hex_weight`) | ds_weight | PSNR3D_mean | SSIM3D_mean | 备注 |
|---|---|---:|---:|---:|---:|---|
| s4_next_wA1_k00125_ds085_case1_20251222_225310 | `output/1a11ecda-d` | 0.0125 | 0.85 | 24.705123 | 0.339739 | NaN 崩溃 |
| s4_next_wA1_k00125_ds090_case1_20251222_225310 | `output/8ee24fc3-2` | 0.0125 | 0.90 | 45.376411 | 0.980655 | 稳定 |
| s4_next_wA1_k0015_ds085_case1_20251222_225310 | `output/9e69cfe6-8` | 0.0150 | 0.85 | 45.209491 | 0.980570 | 稳定 |
| s4_next_wA1_k00175_ds090_case1_20251222_225310 | `output/5934d912-b` | 0.0175 | 0.90 | 45.447739 | 0.980683 | 本轮最好 |
| s4_next_wA1_k0020_ds085_case1_20251222_225310 | `output/291226de-5` | 0.0200 | 0.85 | 45.162651 | 0.980145 | 稳定 |
| s4_next_wA1_k0020_ds090_case1_20251222_225310 | `output/dafd60eb-f` | 0.0200 | 0.90 | 24.705123 | 0.339739 | NaN 崩溃 |
| s4_next_wA1_k0025_ds090_case1_repeat_20251222_225310 | `output/9d50f1d6-5` | 0.0250 | 0.90 | 45.379347 | 0.980707 | 稳定（复现后正常） |
| s4_next_wA1_k0010_ds090_case1_20251222_225311 | `output/2bcd5b64-8` | 0.0100 | 0.90 | 45.150013 | 0.979998 | 稳定 |

结论（用于下一轮设计）：

- 本轮最优点在 `k=0.0175, ds=0.90`。
- 出现明显的 **k 与 ds 的耦合不稳定带**：`k=0.0125, ds=0.85` 与 `k=0.0200, ds=0.90` 均 NaN 崩溃。
- `k=0.025, ds=0.90` 本轮复现后正常且接近最优，可作为第二分支继续探索。

#### 2.6 下一轮：围绕最优点与 k=0.025 分支的 8 个新实验（wA=1）

设计原则：以 `k=0.0175, ds=0.90` 为中心做小步搜索，并对 `k=0.025` 分支做 ds±0.02 探索；避免已知不稳定组合（例如 `k=0.0200, ds=0.90`）。

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.016 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next2_wA1_k0016_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next2_wA1_k0016_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.08 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next2_wA1_k00175_ds092_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next2_wA1_k00175_ds092_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.12 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next2_wA1_k00175_ds088_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next2_wA1_k00175_ds088_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.0185 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next2_wA1_k00185_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next2_wA1_k00185_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.019 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next2_wA1_k0019_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next2_wA1_k0019_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.10 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.015 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next2_wA1_k0015_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next2_wA1_k0015_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.08 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.025 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next2_wA1_k0025_ds092_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next2_wA1_k0025_ds092_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.12 --balance_lr 0 \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.025 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4_next2_wA1_k0025_ds088_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4_next2_wA1_k0025_ds088_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### 2.7 [2025-12-23] s4_next2(wA=1) 本轮 8 个实验结果（优先 3D）

说明：本轮为 `2.6` 中的 `s4_next2_*` 实际跑出的结果。`ds_weight = 1 - balance_alpha_init`。

| dirname | output | k (`s4_dr_hex_weight`) | ds_weight | PSNR3D_mean | SSIM3D_mean | 备注 |
|---|---|---:|---:|---:|---:|---|
| s4_next2_wA1_k0016_ds090_case1_20251223_041753 | `output/fba41c91-7` | 0.0160 | 0.90 | 45.408836 | 0.980750 | 稳定 |
| s4_next2_wA1_k00175_ds092_case1_20251223_041753 | `output/bea98ca4-c` | 0.0175 | 0.92 | 45.439277 | 0.980953 | 本轮最好 |
| s4_next2_wA1_k00175_ds088_case1_20251223_041753 | `output/bb3c9617-e` | 0.0175 | 0.88 | 45.275594 | 0.980504 | 稳定 |
| s4_next2_wA1_k00185_ds090_case1_20251223_041753 | `output/af171d3c-d` | 0.0185 | 0.90 | 45.364124 | 0.980564 | 稳定 |
| s4_next2_wA1_k0019_ds090_case1_20251223_041753 | `output/6f7be48a-d` | 0.0190 | 0.90 | 45.216299 | 0.980348 | 稳定 |
| s4_next2_wA1_k0015_ds090_case1_20251223_041753 | `output/0991f8e5-4` | 0.0150 | 0.90 | 45.080593 | 0.980101 | 稳定 |
| s4_next2_wA1_k0025_ds092_case1_20251223_041753 | `output/0ab22030-b` | 0.0250 | 0.92 | 45.053810 | 0.979267 | 稳定但退化 |
| s4_next2_wA1_k0025_ds088_case1_20251223_041755 | `output/2edd6d7a-c` | 0.0250 | 0.88 | 45.270483 | 0.980818 | 稳定 |

结论（用于下一轮设计）：

- 在已验证稳定的区域里，`k` 最优集中在 `0.016~0.0185`，其中 `k=0.0175` 表现最强。
- 在 `k=0.0175` 下，`ds` 从 `0.88 -> 0.92` 有明显增益（0.92 最好）。
- 在 `k=0.025` 下，`ds=0.88` 明显优于 `ds=0.92`，提示 `k` 与 `ds` 依然存在耦合。
- 结合上一轮 `s4_next` 的 NaN 结论（`k=0.020, ds=0.90` 与 `k=0.0125, ds=0.85`），后续应尽量避免跨越式扫参，优先做局部小步精修。

下一步建议（不强制）：

- 以 `k=0.0175` 为中心，继续细化 `ds`：`0.91/0.92/0.93`，并对 `k` 做 ±0.001 微调（`0.0165/0.0175/0.0185`）。
- 以 `k=0.025` 分支，只在 `ds=0.86~0.90` 附近探索（避免 `ds=0.92`）。

---

#### 2.8 [2025-12-24] S4 固定融合：显式控制 (wA, ds_weight, k) 并移除对 α 的依赖

背景：跨数据集测试中发现，S4 系列在后期可能出现 NaN 或者 3D 指标突然掉崖。排查后发现核心风险是 ds 的计算仍被 `(1-α)` 隐式 gate：即使命令行设置了 `--balance_lr 0`，只要进入 `use_learnable_balance` 分支，`ds_combined` 的幅度仍然由 α 决定；并且在实际运行中还观察到日志可能仍打印 `Added learnable balance parameter to optimizer (lr=0.001)`，导致 α 实际仍可能在被优化。

为确保 S4 实验中的 `ds_weight` 完全由实验参数显式控制，新增：

- `--s4_ds_hex_weight`：直接指定 `ds_weight`，按 `ds = ds_weight * ds_hex` 计算。

并在 `anchor_module.py` 中实现：当 `s4_dx_anchor_weight / s4_ds_hex_weight / s4_dr_hex_weight` 任一被设置（>=0）时，启用 **S4 (independent fixed fusion)**，不再依赖 `use_learnable_balance`/α。

##### 2.8.1 重新跑 3 个跨数据集实验（k=0.0175, ds_weight=0.92）

```bash
# 4dlung_case4
nohup python train.py -s data/4dlung_case4.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.92 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k00175_ds092_4dlung_case4_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k00175_ds092_4dlung_case4_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# spare_mc_case1
nohup python train.py -s data/spare_mc_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.92 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k00175_ds092_spare_mc_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k00175_ds092_spare_mc_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# dir_4d_case2
nohup python train.py -s data/dir_4d_case2.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.92 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k00175_ds092_dir_4d_case2_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k00175_ds092_dir_4d_case2_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

##### 2.8.2 [2025-12-24] dir_4d_case1：s4fix 固定融合 8 组合 rerun（pending）

说明：本轮用于替代旧的 `s4_combo_*`（learnable balance / α 依赖）组合实验。通过显式设置 `--s4_dx_anchor_weight/--s4_ds_hex_weight/--s4_dr_hex_weight` 启用 fixed fusion，并确保 `ds_weight` 不再受 `(1-α)` gate 影响。

```bash
# 1) k=0.0150, ds=0.90
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.90 \
  --s4_dr_hex_weight 0.0150 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k00150_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k00150_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 2) k=0.0160, ds=0.90
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.90 \
  --s4_dr_hex_weight 0.0160 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k00160_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k00160_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 3) k=0.0175, ds=0.88
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.88 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k00175_ds088_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k00175_ds088_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 4) k=0.0175, ds=0.92
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.92 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k00175_ds092_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k00175_ds092_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 5) k=0.0185, ds=0.90
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.90 \
  --s4_dr_hex_weight 0.0185 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k00185_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k00185_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 6) k=0.0190, ds=0.90
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.90 \
  --s4_dr_hex_weight 0.0190 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k00190_ds090_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k00190_ds090_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 7) k=0.0250, ds=0.88
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.88 \
  --s4_dr_hex_weight 0.0250 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k00250_ds088_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k00250_ds088_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# 8) k=0.0250, ds=0.92
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.92 \
  --s4_dr_hex_weight 0.0250 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k00250_ds092_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k00250_ds092_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

本次启动 log（2025-12-24）：

| k | ds_weight | log |
|---:|---:|---|
| 0.0150 | 0.90 | `log/s4fix_wA1_k00150_ds090_case1_20251224_123534.log` |
| 0.0160 | 0.90 | `log/s4fix_wA1_k00160_ds090_case1_20251224_123533.log` |
| 0.0175 | 0.88 | `log/s4fix_wA1_k00175_ds088_case1_20251224_123534.log` |
| 0.0175 | 0.92 | `log/s4fix_wA1_k00175_ds092_case1_20251224_123533.log` |
| 0.0185 | 0.90 | `log/s4fix_wA1_k00185_ds090_case1_20251224_123534.log` |
| 0.0190 | 0.90 | `log/s4fix_wA1_k00190_ds090_case1_20251224_123534.log` |
| 0.0250 | 0.88 | `log/s4fix_wA1_k00250_ds088_case1_20251224_123534.log` |
| 0.0250 | 0.92 | `log/s4fix_wA1_k00250_ds092_case1_20251224_123534.log` |

本轮结果（[ITER 50000]，按 PSNR3D_mean 排序）：

| k (`s4_dr_hex_weight`) | ds_weight (`s4_ds_hex_weight`) | output | PSNR3D_mean | SSIM3D_mean | PSNR2D_mean | SSIM2D_mean |
|---:|---:|---|---:|---:|---:|---:|
| 0.0175 | 0.88 | `output/5dac34f6-4` | 45.402 | 0.981 | 44.045 | 0.990 |
| 0.0150 | 0.90 | `output/7dd76797-b` | 45.354 | 0.981 | 44.212 | 0.990 |
| 0.0250 | 0.92 | `output/7735813c-2` | 45.218 | 0.981 | 44.225 | 0.990 |
| 0.0190 | 0.90 | `output/a30734dc-f` | 45.135 | 0.980 | 43.893 | 0.990 |
| 0.0160 | 0.90 | `output/40ec6c31-e` | 44.920 | 0.979 | 43.663 | 0.990 |
| 0.0250 | 0.88 | `output/64a04b6e-b` | 44.920 | 0.979 | 43.663 | 0.990 |
| 0.0185 | 0.90 | `output/99cbb0bf-a` | 44.856 | 0.979 | 43.571 | 0.990 |
| 0.0175 | 0.92 | `output/53b0ffb4-0` | 44.644 | 0.978 | 43.182 | 0.989 |

对比结论（vs 未修复版本 `s4_next2_*` 8 组合）：

- 本轮（s4fix）最优：`k=0.0175, ds=0.88`，PSNR3D_mean=45.402。
- 旧版（s4_next2）最优：`k=0.0175, ds=0.92`，PSNR3D_mean=45.439。
- 最优点从 `ds=0.92` 转移到 `ds=0.88`；峰值差异约 0.037（很小）。
- 重要：本轮 8 个均稳定跑满 50k，未出现旧版所观测到的后期 NaN/掉崖风险。

下一步（建议 8 个组合，s4fix 固定融合，局部精修）：

说明：将 8 个点分成两条线，各 4 个。

- 线 A（围绕本轮最优 `k=0.0175, ds=0.88` 做局部精修）：
  - `k=0.0165, ds=0.88`
  - `k=0.0185, ds=0.88`
  - `k=0.0175, ds=0.87`
  - `k=0.0175, ds=0.89`
- 线 B（沿 `k=0.025, ds≈0.92` 的高分支做局部精修）：
  - `k=0.0240, ds=0.92`
  - `k=0.0260, ds=0.92`
  - `k=0.0250, ds=0.91`
  - `k=0.0250, ds=0.93`

下一步命令模板（case1，wA=1.0）：

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight <DS_WEIGHT> \
  --s4_dr_hex_weight <K> \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s4fix_wA1_k${K_STR}_ds${DS_STR}_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_wA1_k${K_STR}_ds${DS_STR}_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### [2025-12-25] dir_4d_case1：s4fix 围绕 k=0.0175, ds=0.88 的 coarse sweep（debug 全开）

说明：目标是“颗粒度更大”的下一轮扫点，在保证不重复既有 s4fix 8 组合以及已启动的 `k=0.0175, ds=0.87` 前提下，扩大 k/ds 扰动范围以获得更明显差异。所有命令均开启 `--debug` 用于 NaN/Inf 首发定位。

已启动（debug 复现，观察稳定性）：

- `k=0.0175, ds=0.87`：`log/s4fix_wA1_k00175_ds087_case1_20251225_142640.log`

coarse sweep（8 组，debug 全开）：

- 线 A（固定 ds=0.88，扫 k）：
  - `k=0.0125, ds=0.88`
  - `k=0.0150, ds=0.88`
  - `k=0.0200, ds=0.88`
  - `k=0.0225, ds=0.88`
- 线 B（固定 k=0.0175，扫 ds）：
  - `k=0.0175, ds=0.82`
  - `k=0.0175, ds=0.84`
  - `k=0.0175, ds=0.94`
  - `k=0.0175, ds=0.96`

```bash
# A1) k=0.0125, ds=0.88 (debug)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.88 \
  --s4_dr_hex_weight 0.0125 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --debug \
  --dirname s4fix_dbg_wA1_k00125_ds088_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_dbg_wA1_k00125_ds088_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# A2) k=0.0150, ds=0.88 (debug)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.88 \
  --s4_dr_hex_weight 0.0150 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --debug \
  --dirname s4fix_dbg_wA1_k00150_ds088_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_dbg_wA1_k00150_ds088_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# A3) k=0.0200, ds=0.88 (debug)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.88 \
  --s4_dr_hex_weight 0.0200 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --debug \
  --dirname s4fix_dbg_wA1_k00200_ds088_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_dbg_wA1_k00200_ds088_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# A4) k=0.0225, ds=0.88 (debug)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.88 \
  --s4_dr_hex_weight 0.0225 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --debug \
  --dirname s4fix_dbg_wA1_k00225_ds088_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_dbg_wA1_k00225_ds088_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# B1) k=0.0175, ds=0.82 (debug)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.82 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --debug \
  --dirname s4fix_dbg_wA1_k00175_ds082_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_dbg_wA1_k00175_ds082_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# B2) k=0.0175, ds=0.84 (debug)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.84 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --debug \
  --dirname s4fix_dbg_wA1_k00175_ds084_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_dbg_wA1_k00175_ds084_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# B3) k=0.0175, ds=0.94 (debug)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.94 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --debug \
  --dirname s4fix_dbg_wA1_k00175_ds094_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_dbg_wA1_k00175_ds094_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# B4) k=0.0175, ds=0.96 (debug)
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 1.0 \
  --s4_ds_hex_weight 0.96 \
  --s4_dr_hex_weight 0.0175 \
  --lambda_dssim 0.25 --lambda_phys 0.1 --lambda_anchor_smooth 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --debug \
  --dirname s4fix_dbg_wA1_k00175_ds096_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s4fix_dbg_wA1_k00175_ds096_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### [2025-12-26/27] dir_4d_case1：s6(TRGF) 诊断与修复（性能损失定位 + ds 权重局部扫点）

说明：围绕此前不稳定的 `k=0.0200, ds≈0.90`，引入 s6(TRGF) 的可控限幅（pos/scale/rot），并用 6 组对照定位“早期性能损失”与“后期掉崖/NaN”的权衡。

#### 6 组对照（k=0.0200, ds=0.90, wA=1.0）

- A0 (baseline/s4fix)：`log/s6diag_A0_base_s4fix_dbg_wA1_k00200_ds090_case1_20251226_155642.log`
  - 20k: psnr3d 44.206, psnr2d 43.049
  - 50k: psnr3d 45.339, psnr2d 44.379
- A1 (TRGF-tight)：`log/s6diag_A1_trgf_fixed_tight_dbg_ds090_20251226_155645.log`
  - 20k: psnr3d 43.696, psnr2d 41.183
  - 50k: psnr3d 44.874, psnr2d 42.153
- A2 (TRGF-relaxSR)：`log/s6diag_A2_trgf_fixed_relaxSR_dbg_ds090_20251226_155648.log`
  - 20k: psnr3d 44.113, psnr2d 42.917
  - 50k: psnr3d 45.230, psnr2d 44.123
- B1 (late-on + schedule, full)：`log/s6diag_B1_trgf_lateOn_sched_dbg_ds090_20251226_155653.log`
  - 20k: psnr3d 44.227, psnr2d 42.988
  - 50k: psnr3d 43.277, psnr2d 34.238
- B2 (late-on + schedule, SR-only)：`log/s6diag_B2_trgf_sched_SRonly_dbg_ds090_20251226_155657.log`
  - 20k: psnr3d 44.248, psnr2d 43.171
  - 50k: psnr3d 42.535, psnr2d 33.073
- B3 (late-on + schedule, DX-only)：`log/s6diag_B3_trgf_sched_DXonly_dbg_ds090_20251226_155702.log`
  - 20k: psnr3d 44.113, psnr2d 42.917
  - 50k: psnr3d 45.230, psnr2d 44.123

结论（本轮）：

- A1 的性能损失主要体现在 2D 指标（20k/50k 均明显偏低），说明 TRGF 过紧会带来欠拟合。
- A2 将 tau 放松后，性能基本回到 baseline 水平（50k: psnr2d 44.123 vs 44.379），且全程稳定。
- B1/B2 在 30k 后出现 2D 指标显著退化（psnr2d 39.540/39.995 → 36.759/36.103 → 34.238/33.073），表现为“开启/调度 TRGF 的副作用”。
- B3（只对 dx schedule）未出现上述 2D 退化，最终指标与 A2 接近。

#### A2 版本：ds 权重局部扫点（k=0.0200，TRGF-relaxSR）

说明：固定 TRGF-relaxSR 配置，扫 `ds_hex_weight∈{0.88,0.89,0.91,0.92}`（额外跑的点用于局部最优确认）。

- ds=0.88：`log/s6diag_A2_trgf_fixed_relaxSR_dbg_ds088_20251226_213116.log`
  - 20k: psnr3d 44.213, psnr2d 43.070
  - 50k: psnr3d 45.357, psnr2d 44.071
- ds=0.89：`log/s6diag_A2_trgf_fixed_relaxSR_dbg_ds089_20251226_213219.log`
  - 20k: psnr3d 44.139, psnr2d 42.904
  - 50k: psnr3d 45.315, psnr2d 44.038
- ds=0.91：`log/s6diag_A2_trgf_fixed_relaxSR_dbg_ds091_20251226_213243.log`
  - 20k: psnr3d 44.090, psnr2d 42.924
  - 50k: psnr3d 45.266, psnr2d 44.169
- ds=0.92：`log/s6diag_A2_trgf_fixed_relaxSR_dbg_ds092_20251226_213150.log`
  - 20k: psnr3d 44.170, psnr2d 42.909
  - 50k: psnr3d 45.337, psnr2d 44.125

结论（局部扫点）：

- 在 A2(TRGF-relaxSR) 下，`ds=0.91` 给出本轮最高的 50k 2D 指标（psnr2d 44.169）。
- `ds=0.88~0.92` 整体差异很小（44.038~44.169），说明 TRGF-relaxSR 能把训练稳定性与性能拉到较鲁棒的区间。

#### [2025-12-27] dir_4d_case1：wA=0.99, k=0.0100, ds=1.0（A0 baseline vs A2 TRGF-relaxSR）

- A0 (baseline/s4fix)：`log/s6diag_A0_base_s4fix_dbg_wA099_k00100_ds100_case1_20251227_115654.log`
  - 5k:  psnr3d 39.495, psnr2d 38.791
  - 15k: psnr3d 44.000, psnr2d 43.156
  - 20k: psnr3d 44.310, psnr2d 43.029
  - 30k: psnr3d 45.249, psnr2d 44.085
  - 40k: psnr3d 45.265, psnr2d 44.302
  - 50k: psnr3d 45.418, psnr2d 44.506
- A2 (TRGF-relaxSR)：`log/s6diag_A2_trgf_fixed_relaxSR_dbg_wA099_k00100_ds100_case1_20251227_115654.log`
  - 5k:  psnr3d 39.495, psnr2d 38.792
  - 15k: psnr3d 43.679, psnr2d 42.920
  - 20k: psnr3d 44.164, psnr2d 42.989
  - 30k: psnr3d 45.078, psnr2d 43.854
  - 40k: psnr3d 45.322, psnr2d 44.061
  - 50k: psnr3d 45.496, psnr2d 44.401

结论：

- A2 相比 A0：50k 的 psnr3d 略高（45.496 vs 45.418），psnr2d 略低（44.401 vs 44.506），整体非常接近。

#### [2025-12-27] dir_4d_case2：复刻 ds=0.90 的 A2(TRGF-relaxSR) 配置（case2）

- A2 (TRGF-relaxSR, wA=1.0, k=0.0200, ds=0.90)：`log/s6diag_A2_trgf_fixed_relaxSR_dbg_wA1_k00200_ds090_case2_20251227_115654.log`
  - 5k:  psnr3d 32.205, psnr2d 37.106
  - 15k: psnr3d 34.276, psnr2d 42.332
  - 20k: psnr3d 35.115, psnr2d 42.538
  - 30k: psnr3d 35.295, psnr2d 42.901
  - 40k: psnr3d 35.486, psnr2d 43.125
  - 50k: psnr3d 35.670, psnr2d 43.158

#### [2025-12-27] 新启动 nohup（待跑完补结果）

- case2 A0 (baseline/s4fix, wA=0.99, k=0.0100, ds=1.0)：`log/s6diag_A0_base_s4fix_dbg_wA099_k00100_ds100_case2_20251227_201237.log`
- case2 A2 (TRGF-relaxSR, wA=0.99, k=0.0100, ds=1.0)：`log/s6diag_A2_trgf_fixed_relaxSR_dbg_wA099_k00100_ds100_case2_20251227_201237.log`
- case1 A2 (TRGF-relaxSR, HexPlane=0 / pure anchor via V5 fixed balance α=1.0)：`log/s6diag_A2_trgf_fixed_relaxSR_dbg_hex0_alpha1_case1_20251227_201237.log`

#### [2025-12-28] dir_4d_case2：从 50k checkpoint 续训到 100k（每 10k 测一次）

目标：对 case2 的 A0/A2（wA=0.99, k=0.0100, ds=1.0）做 50k → 100k 的续训，并在 60/70/80/90/100k 评测。

续训参数要点：

- 使用 `--start_checkpoint output/<id>/ckpt/chkpnt50000.pth`
- 设 `--iterations 100000`
- 设 `--test_iterations 60000 70000 80000 90000 100000`
- 设 `--save_iterations 100000 --save_checkpoint`
- 设 `--model_path` 指向原来的 output 目录（继续写入同一目录）

工程修复（resume 必需）：

- 修复从 checkpoint 恢复时 `AnchorDeformationNet` 的 `knn_indices/knn_weights` 因高斯数变化导致的 shape mismatch。
  - 实现：在 `GaussianModel.restore()` 中过滤掉 shape 不匹配的 KNN buffer，并在加载后对当前 `xyz` 重新 `update_knn_binding()`。

已启动：
bash -lc 'TS=$(date +%Y%m%d_%H%M%S)
nohup python train.py -s data/dir_4d_case2.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 0.99 --s4_ds_hex_weight 1.0 --s4_dr_hex_weight 0.01 \
  --s6_trust_region --s6_tau_pos 0.02 --s6_tau_scale 0.25 --s6_tau_rot 0.25 --s6_trust_region_log --s6_trust_region_log_interval 200 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 100000 \
  --test_iterations 60000 70000 80000 90000 100000 \
  --save_iterations 100000 --save_checkpoint \
  --start_checkpoint output/2ded576b-2/ckpt/chkpnt50000.pth \
  --model_path ./output/2ded576b-2 \
  --dirname s6diag_A2_trgf_fixed_relaxSR_dbg_wA099_k00100_ds100_case2_resume100k_${TS} \
  > log/s6diag_A2_trgf_fixed_relaxSR_dbg_wA099_k00100_ds100_case2_resume100k_${TS}.log 2>&1 &


TS=$(date +%Y%m%d_%H%M%S)
LOG=log/s6diag_A0_base_s4fix_dbg_wA099_k00100_ds100_case2_resume100k_${TS}.log
nohup python train.py -s data/dir_4d_case2.pickle \
  --use_anchor_deformation --use_boosted \
  --s4_dx_anchor_weight 0.99 --s4_ds_hex_weight 1.0 --s4_dr_hex_weight 0.01 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 100000 \
  --test_iterations 60000 70000 80000 90000 100000 \
  --save_iterations 100000 --save_checkpoint \
  --start_checkpoint output/b6fdb07b-5/ckpt/chkpnt50000.pth \
  --model_path ./output/b6fdb07b-5 \
  --dirname s6diag_A0_base_s4fix_dbg_wA099_k00100_ds100_case2_resume100k_${TS} \
  > "$LOG" 2>&1 &

  
- case2 A2 (TRGF-relaxSR) resume100k：`log/s6diag_A2_trgf_fixed_relaxSR_dbg_wA099_k00100_ds100_case2_resume100k_20251228_161150.log`
- case2 A0 (baseline/s4fix) resume100k：`log/s6diag_A0_base_s4fix_dbg_wA099_k00100_ds100_case2_resume100k_20251228_161516.log`
  - 注：首次启动的 A0 resume 日志文件被删除（进程 stdout/stderr 仍指向 deleted inode），因此停止旧进程并重新启动以保证可追踪。

续训评测结果（从日志提取；注意：两条日志中未找到 60k 的 `Evaluating` 行，只有 70k 起）：

- case2 A0 resume100k：
  - 70k：psnr3d=36.017, ssim3d=0.943, psnr2d=43.316, ssim2d=0.990
  - 80k：psnr3d=36.536, ssim3d=0.946, psnr2d=43.951, ssim2d=0.991
  - 90k：psnr3d=36.142, ssim3d=0.945, psnr2d=43.681, ssim2d=0.991
  - 100k：psnr3d=36.384, ssim3d=0.947, psnr2d=43.989, ssim2d=0.991
- case2 A2 (TRGF-relaxSR) resume100k：
  - 70k：psnr3d=36.023, ssim3d=0.942, psnr2d=43.603, ssim2d=0.991
  - 80k：psnr3d=36.105, ssim3d=0.943, psnr2d=43.790, ssim2d=0.991
  - 90k：psnr3d=36.130, ssim3d=0.943, psnr2d=43.764, ssim2d=0.991
  - 100k：psnr3d=36.185, ssim3d=0.943, psnr2d=44.026, ssim2d=0.991

对比结论（case2, 100k）：A0 的 3D 指标略优（36.384 vs 36.185），A2 的 2D PSNR 略优（44.026 vs 43.989）。

另：A2(case2) TRGF τpos 扫描（从 0 开始跑 50k，用于观察点数/性能变化）：

- A2-TRGF τpos=0.03：output=`output/14ef713c-t`，pid=19086
  - cfg：`output/14ef713c-t_cfg_args.yml`（基于 `output/2ded576b-2/cfg_args.yml`，仅覆盖 `model_path/dirname/iterations/test_iterations/save_iterations/checkpoint_iterations/start_checkpoint/s6_tau_pos`）
  - log：`log/s6diag_A2_trgf_relaxSR_taupos003_case2_20251228_164359.log`
- A2-TRGF τpos=0.04：output=`output/16169395-t`，pid=19087
  - cfg：`output/16169395-t_cfg_args.yml`
  - log：`log/s6diag_A2_trgf_relaxSR_taupos004_case2_20251228_164359.log`

τpos 扫描评测结果（从日志提取；与 baseline τpos=0.02 的历史日志对齐比较）：

- baseline A2 (τpos=0.02)：`log/s6diag_A2_trgf_fixed_relaxSR_dbg_wA099_k00100_ds100_case2_20251227_201237.log`
  - 5k：psnr3d=32.202, ssim3d=0.895, psnr2d=37.100, ssim2d=0.983
  - 20k：psnr3d=35.015, ssim3d=0.928, psnr2d=42.519, ssim2d=0.989
  - 30k：psnr3d=35.299, ssim3d=0.934, psnr2d=43.001, ssim2d=0.989
  - 40k：psnr3d=35.617, ssim3d=0.937, psnr2d=43.193, ssim2d=0.990
  - 50k：psnr3d=35.796, ssim3d=0.939, psnr2d=43.290, ssim2d=0.990
- τpos=0.03：`log/s6diag_A2_trgf_relaxSR_taupos003_case2_20251228_164359.log`
  - 5k：psnr3d=32.206, ssim3d=0.895, psnr2d=37.107, ssim2d=0.983
  - 20k：psnr3d=35.000, ssim3d=0.929, psnr2d=42.524, ssim2d=0.989
  - 30k：psnr3d=35.296, ssim3d=0.933, psnr2d=42.912, ssim2d=0.989
  - 40k：psnr3d=35.578, ssim3d=0.936, psnr2d=43.193, ssim2d=0.990
  - 50k：psnr3d=35.765, ssim3d=0.938, psnr2d=43.416, ssim2d=0.990
- τpos=0.04：`log/s6diag_A2_trgf_relaxSR_taupos004_case2_20251228_164359.log`
  - 5k：psnr3d=32.202, ssim3d=0.895, psnr2d=37.100, ssim2d=0.983
  - 20k：psnr3d=34.774, ssim3d=0.927, psnr2d=42.602, ssim2d=0.989
  - 30k：psnr3d=35.088, ssim3d=0.932, psnr2d=43.110, ssim2d=0.989
  - 40k：psnr3d=35.226, ssim3d=0.935, psnr2d=43.249, ssim2d=0.990
  - 50k：psnr3d=35.464, ssim3d=0.937, psnr2d=43.297, ssim2d=0.990

#### [2025-12-28] s7：case2 A0 baseline 上做 per-anchor wA（per-region transport gain）

思路：将全局 `wA`（S4 的 dx 融合权重）扩展为 per-anchor 的 `wA_i(t)`，由 Anchor Transformer 的特征预测，并通过 Gaussian→Anchor 的 KNN skinning 权重插值成 per-Gaussian 的 `wA(x,t)`：

- 预测：`wA_i = clamp(wA_base + ΔwA_i, 0, 1)`，`ΔwA_i = Δ_max * tanh(MLP(f_i))`
- 插值：`wA(x,t) = Σ_k w_k(x) * wA_{a_k}(t)`
- 正则：可选的 anchor-graph spatial smoothness 与 temporal smoothness（对应 loss 会加到 total）

case2 基准：保持 `wA_base=0.99, k=0.01, ds=1.0`，只开启 s7 以允许局部偏离。

已启动（从 0 开始跑 50k；test @ 5/15/20/30/40/50k）：

- E1_free：Δ_max=0.01, λ_graph=0, λ_temp=0
  - output=`output/73cefa60-s7`, pid=8663
  - cfg=`output/73cefa60-s7_cfg_args.yml`
  - log=`log/s7_A0_case2_E1_free_wA099_k001_ds100_20251228_221652.log`
- E2_graph：Δ_max=0.01, λ_graph=0.01, λ_temp=0
  - output=`output/927b694e-s7`, pid=8664
  - cfg=`output/927b694e-s7_cfg_args.yml`
  - log=`log/s7_A0_case2_E2_graph_wA099_k001_ds100_20251228_221652.log`
- E3_graph_temp：Δ_max=0.01, λ_graph=0.01, λ_temp=0.01
  - output=`output/92d44e73-s7`, pid=8666
  - cfg=`output/92d44e73-s7_cfg_args.yml`
  - log=`log/s7_A0_case2_E3_graph_temp_wA099_k001_ds100_20251228_221652.log`
- E4_strong：Δ_max=0.02, λ_graph=0.01, λ_temp=0.01
  - output=`output/526aa0b9-s7`, pid=8668
  - cfg=`output/526aa0b9-s7_cfg_args.yml`
  - log=`log/s7_A0_case2_E4_strong_wA099_k001_ds100_20251228_221652.log`

后验检查（重要）：

- 四个 s7 run 的日志中 `[s7] wA_std=0`，`wA_min=max=0.99000`，说明 per-anchor `wA_i` 没有发生学习。
- 定位原因：`s7_wA_head` 未被加入 optimizer 参数列表（`AnchorDeformationNet.get_mlp_parameters()` 漏掉了 `self.s7_wA_head.parameters()`），导致 head 权重始终为 0，`ΔwA_i ≡ 0`。
- 修复：在 `x2_gaussian/gaussian/anchor_module.py` 的 `get_mlp_parameters()` 中加入 `if self.s7_wA_head is not None: params.extend(self.s7_wA_head.parameters())`。
- 因此，本批 4 个 run 的指标主要反映“重复训练的随机波动”，不能用于验证 s7 的有效性，需要在修复后重跑。

本批 4-run 指标（供参考，视为 baseline 噪声）：

- E1_free：
  - 5k：psnr3d=32.203, ssim3d=0.895, psnr2d=37.097, ssim2d=0.983
  - 15k：psnr3d=34.582, ssim3d=0.926, psnr2d=42.149, ssim2d=0.988
  - 20k：psnr3d=35.570, ssim3d=0.934, psnr2d=42.811, ssim2d=0.989
  - 30k：psnr3d=35.811, ssim3d=0.938, psnr2d=43.411, ssim2d=0.990
  - 40k：psnr3d=35.973, ssim3d=0.941, psnr2d=43.539, ssim2d=0.990
  - 50k：psnr3d=36.140, ssim3d=0.942, psnr2d=43.525, ssim2d=0.991
- E2_graph：
  - 5k：psnr3d=32.205, ssim3d=0.895, psnr2d=37.103, ssim2d=0.983
  - 15k：psnr3d=34.637, ssim3d=0.927, psnr2d=42.157, ssim2d=0.988
  - 20k：psnr3d=35.540, ssim3d=0.934, psnr2d=42.764, ssim2d=0.989
  - 30k：psnr3d=35.785, ssim3d=0.939, psnr2d=43.330, ssim2d=0.990
  - 40k：psnr3d=35.988, ssim3d=0.941, psnr2d=43.522, ssim2d=0.990
  - 50k：psnr3d=36.171, ssim3d=0.943, psnr2d=43.688, ssim2d=0.991
- E3_graph_temp：
  - 5k：psnr3d=32.204, ssim3d=0.895, psnr2d=37.103, ssim2d=0.983
  - 15k：psnr3d=34.702, ssim3d=0.927, psnr2d=42.124, ssim2d=0.988
  - 20k：psnr3d=35.517, ssim3d=0.934, psnr2d=42.770, ssim2d=0.989
  - 30k：psnr3d=35.777, ssim3d=0.938, psnr2d=43.292, ssim2d=0.990
  - 40k：psnr3d=35.953, ssim3d=0.941, psnr2d=43.545, ssim2d=0.990
  - 50k：psnr3d=36.169, ssim3d=0.943, psnr2d=43.600, ssim2d=0.991
- E4_strong：
  - 5k：psnr3d=32.205, ssim3d=0.895, psnr2d=37.098, ssim2d=0.983
  - 15k：psnr3d=34.606, ssim3d=0.926, psnr2d=42.318, ssim2d=0.988
  - 20k：psnr3d=35.349, ssim3d=0.932, psnr2d=42.631, ssim2d=0.989
  - 30k：psnr3d=35.726, ssim3d=0.938, psnr2d=43.445, ssim2d=0.990
  - 40k：psnr3d=35.981, ssim3d=0.941, psnr2d=43.654, ssim2d=0.990
  - 50k：psnr3d=36.082, ssim3d=0.942, psnr2d=43.698, ssim2d=0.991

阶段性结论（τpos 扫描）：

- τpos=0.03：整体与 baseline(0.02) 接近，50k 的 2D PSNR 略高（43.416 vs 43.290），3D 基本持平。
- τpos=0.04：20k~50k 的 3D 指标明显偏低（如 50k: 35.464 vs 35.796），不推荐继续放大。

下一步建议：

- 优先保留 τpos=0.02/0.03 两档，后续如果要继续“促点数/促细节”，建议只在 τpos=0.03 基础上做更小幅的增量（例如 0.025/0.03/0.035），避免直接到 0.04。
- 若要验证“点数是否真的增多”，建议在日志中同时统计 `pts=` 的演化（densify/prune 期间的点数跳变），再结合 20k~50k 指标判断是否是容量变化而非纯训练随机性。

2025-12-29

s7v2（修复 optimizer 注册后重跑）：case2 A0，per-anchor wA（从 0 开始 50k）

- 代码修复：`AnchorDeformationNet.get_mlp_parameters()` 已加入 `s7_wA_head.parameters()`，保证 per-anchor `wA_i` 可学习。

cfg（从上一批 s7 E1 模板生成，仅修改 model_path/dirname/s7 超参）：

- E1_free：Δ_max=0.01, λ_graph=0, λ_temp=0
  - output_id=`3b9e5c72-s7v2`
  - cfg=`output/3b9e5c72-s7v2_cfg_args.yml`
  - log=`log/s7v2_A0_case2_E1_free_wA099_k001_ds100_20251229_034232.log`
- E2_graph：Δ_max=0.01, λ_graph=0.01, λ_temp=0
  - output_id=`79724d62-s7v2`
  - cfg=`output/79724d62-s7v2_cfg_args.yml`
  - log=`log/s7v2_A0_case2_E2_graph_wA099_k001_ds100_20251229_034232.log`
- E3_graph_temp：Δ_max=0.01, λ_graph=0.01, λ_temp=0.01
  - output_id=`c1450625-s7v2`
  - cfg=`output/c1450625-s7v2_cfg_args.yml`
  - log=`log/s7v2_A0_case2_E3_graph_temp_wA099_k001_ds100_20251229_034232.log`
- E4_strong：Δ_max=0.02, λ_graph=0.01, λ_temp=0.01
  - output_id=`1528cf9c-s7v2`
  - cfg=`output/1528cf9c-s7v2_cfg_args.yml`
  - log=`log/s7v2_A0_case2_E4_strong_wA099_k001_ds100_20251229_034232.log`

nohup（待运行）：

- E1_free：`nohup python train.py --config output/3b9e5c72-s7v2_cfg_args.yml > log/s7v2_A0_case2_E1_free_wA099_k001_ds100_20251229_034232.log 2>&1 &`
- E2_graph：`nohup python train.py --config output/79724d62-s7v2_cfg_args.yml > log/s7v2_A0_case2_E2_graph_wA099_k001_ds100_20251229_034232.log 2>&1 &`
- E3_graph_temp：`nohup python train.py --config output/c1450625-s7v2_cfg_args.yml > log/s7v2_A0_case2_E3_graph_temp_wA099_k001_ds100_20251229_034232.log 2>&1 &`
- E4_strong：`nohup python train.py --config output/1528cf9c-s7v2_cfg_args.yml > log/s7v2_A0_case2_E4_strong_wA099_k001_ds100_20251229_034232.log 2>&1 &`

A0 case2：L_phys / L_anchor_smooth ablation（从 0 开始 50k；基于 `output/b6fdb07b-5/cfg_args.yml`）

- no_smooth：仅关闭 smooth（`lambda_anchor_smooth=0`）
  - cfg=`output/20251229_094639_ps0_A0_cfg_args.yml`
  - log=`log/ablate_A0_case2_no_smooth_wA099_k00100_ds100_20251229_094639.log`
  - nohup：`nohup python train.py --config output/20251229_094639_ps0_A0_cfg_args.yml > log/ablate_A0_case2_no_smooth_wA099_k00100_ds100_20251229_094639.log 2>&1 &`
- no_phys：仅关闭 L_phys（`lambda_phys=0`）
  - cfg=`output/20251229_094639_pp0_A0_cfg_args.yml`
  - log=`log/ablate_A0_case2_no_phys_wA099_k00100_ds100_20251229_094639.log`
  - nohup：`nohup python train.py --config output/20251229_094639_pp0_A0_cfg_args.yml > log/ablate_A0_case2_no_phys_wA099_k00100_ds100_20251229_094639.log 2>&1 &`
- both：两者都带（baseline 重跑，控制随机性）
  - cfg=`output/20251229_094639_base_A0_cfg_args.yml`
  - log=`log/ablate_A0_case2_both_wA099_k00100_ds100_20251229_094639.log`
  - nohup：`nohup python train.py --config output/20251229_094639_base_A0_cfg_args.yml > log/ablate_A0_case2_both_wA099_k00100_ds100_20251229_094639.log 2>&1 &`

A0 case2：no_phys + no_smooth（从 0 直接跑到 100k；每 10k eval；基于 `output/20251229_094639_base_A0_cfg_args.yml`）

- cfg=`output/20251230_032322_np0ns0_A0_cfg_args.yml`
- log=`log/ablate100k_from0_A0_case2_no_phys_no_smooth_wA099_k00100_ds100_20251230_032322.log`
- nohup：`nohup python train.py --config output/20251230_032322_np0ns0_A0_cfg_args.yml > log/ablate100k_from0_A0_case2_no_phys_no_smooth_wA099_k00100_ds100_20251230_032322.log 2>&1 &`

s7 / s7v2（case2 A0）阶段性复盘：

- s7（20251228_221652 这批）：`wA_std` 全程为 0（`wA` 恒等于 `wA_base=0.99`），说明 per-anchor `wA` 没有学习到（当时 `s7_wA_head` 未加入 optimizer）。
- s7v2（20251229_034232 这批）：`wA_std` 在 7k~20k 明显 >0（约 1e-4~5e-4），说明 per-anchor `wA` 分支已生效；但随着训练推进 `wA` 很快塌到接近常数（均值大约落到 0.98/0.97 一带，std 逐步趋近 0），正则项（graph/temp）在当前配置下基本为 0。
- 50k 指标（仅用于粗对比）：E1/E2/E3/E4 的 psnr3d/psnr2d 差异很小，尚不足以证明 s7v2 带来稳定提升。

s7v3（case2 A0）下一批（目标：避免 wA 塌缩为常数；让 graph/temp 正则项非零）

- E1_dmax005（只放开 delta 上限，观察 wA 是否仍塌缩）：
  - cfg=`output/20251230_071406_E1_dmax005_A0_cfg_args.yml`
  - dmax=0.05, λ_graph=0.0, λ_temp=0.0, k=8
- E2_dmax005_g01（增加 graph 正则，观察 L_graph 是否非零且 wA_std 是否维持）：
  - cfg=`output/20251230_071406_E2_dmax005_g01_A0_cfg_args.yml`
  - dmax=0.05, λ_graph=0.1, λ_temp=0.0, k=8
- E3_dmax005_g01_t01（graph+temp 同时加，防止过拟合/局部振荡）：
  - cfg=`output/20251230_071406_E3_dmax005_g01_t01_A0_cfg_args.yml`
  - dmax=0.05, λ_graph=0.1, λ_temp=0.1, k=8
- E4_dmax010_g05_t05_k16（强正则 + 更大邻域，尝试维持结构差异）：
  - cfg=`output/20251230_071406_E4_dmax010_g05_t05_k16_A0_cfg_args.yml`
  - dmax=0.10, λ_graph=0.5, λ_temp=0.5, k=16

s7v3 跑完结论（失败）：

- 现象：wA 训练过程中显著下滑（例如 E4: 0.99 → ~0.89），最终 wA_std 也趋近 0；指标变差。
- 推断：wA 在当前参数化下可向下调（tanh 允许负 delta），优化器会把“降低 wA”当作快速降主损的捷径，破坏 0.99 的甜点设定。

s7v4（case2 A0）改进方向：固定 0.99 甜点（wA 只允许上调）

- 代码开关：`s7_wA_only_up=true`，令 wA ∈ [wA_base, wA_base + delta_max]，避免 wA 被训练压低。
- E1_onlyup_d001：
  - cfg=`output/20251230_114833_E1_onlyup_d001_A0_cfg_args.yml`
  - dmax=0.01, λ_graph=0.0, λ_temp=0.0, k=8
- E2_onlyup_d002：
  - cfg=`output/20251230_114833_E2_onlyup_d002_A0_cfg_args.yml`
  - dmax=0.02, λ_graph=0.0, λ_temp=0.0, k=8
- E3_onlyup_d002_g005：
  - cfg=`output/20251230_114833_E3_onlyup_d002_g005_A0_cfg_args.yml`
  - dmax=0.02, λ_graph=0.05, λ_temp=0.0, k=8
- E4_onlyup_d002_g005_t005：
  - cfg=`output/20251230_114833_E4_onlyup_d002_g005_t005_A0_cfg_args.yml`
  - dmax=0.02, λ_graph=0.05, λ_temp=0.05, k=8

- nohup（log 不冲突）：
  - `nohup python train.py --config output/20251230_114833_E1_onlyup_d001_A0_cfg_args.yml > log/s7v4_A0_case2_E1_onlyup_d001_wA099_k001_ds100_20251230_114833.log 2>&1 &`
  - `nohup python train.py --config output/20251230_114833_E2_onlyup_d002_A0_cfg_args.yml > log/s7v4_A0_case2_E2_onlyup_d002_wA099_k001_ds100_20251230_114833.log 2>&1 &`
  - `nohup python train.py --config output/20251230_114833_E3_onlyup_d002_g005_A0_cfg_args.yml > log/s7v4_A0_case2_E3_onlyup_d002_g005_wA099_k001_ds100_20251230_114833.log 2>&1 &`
  - `nohup python train.py --config output/20251230_114833_E4_onlyup_d002_g005_t005_A0_cfg_args.yml > log/s7v4_A0_case2_E4_onlyup_d002_g005_t005_wA099_k001_ds100_20251230_114833.log 2>&1 &`

2025-12-30

a1：可观测性加权的统一连续体正则（替代/统一 L_phys + L_smooth 的先验角色；anchor graph surrogate）

- 训练设置：case2 A0，从 0 开始 50k，每 10k eval
- baseline：`lambda_phys=0, lambda_anchor_smooth=0, mask_ratio=0`
- a1 超参：`a1_reg_enable=true, a1_reg_lambda=0.01, a1_reg_k=8, a1_reg_mask_ratio=0`

4-run：

- A_g1：只开一阶（邻域位移差）
  - cfg=`output/20251230_1441_a1A_g1_cfg_args.yml`
  - log=`log/a1_A0_case2_A_g1_only_lambda0010_k8_mask0_20251230_1441.log`
- B_g2：只开二阶（离散 laplacian 差，薄板倾向）
  - cfg=`output/20251230_1441_a1B_g2_cfg_args.yml`
  - log=`log/a1_A0_case2_B_g2_only_lambda0010_k8_mask0_20251230_1441.log`
- C_g1g2：一阶 + 二阶
  - cfg=`output/20251230_1441_a1C_g1g2_cfg_args.yml`
  - log=`log/a1_A0_case2_C_g1g2_lambda0010_k8_mask0_20251230_1441.log`
- D_g1g2_p2：一阶 + 二阶，且权重更硬（power=2）
  - cfg=`output/20251230_1441_a1D_g1g2_p2_cfg_args.yml`
  - log=`log/a1_A0_case2_D_g1g2_p2_lambda0010_k8_mask0_20251230_1441.log`

nohup（待运行）：

- `nohup python train.py --config output/20251230_1441_a1A_g1_cfg_args.yml > log/a1_A0_case2_A_g1_only_lambda0010_k8_mask0_20251230_1441.log 2>&1 &`
- `nohup python train.py --config output/20251230_1441_a1B_g2_cfg_args.yml > log/a1_A0_case2_B_g2_only_lambda0010_k8_mask0_20251230_1441.log 2>&1 &`
- `nohup python train.py --config output/20251230_1441_a1C_g1g2_cfg_args.yml > log/a1_A0_case2_C_g1g2_lambda0010_k8_mask0_20251230_1441.log 2>&1 &`
- `nohup python train.py --config output/20251230_1441_a1D_g1g2_p2_cfg_args.yml > log/a1_A0_case2_D_g1g2_p2_lambda0010_k8_mask0_20251230_1441.log 2>&1 &`

跑完结果（50k）：

- A_g1（g1 only）：失败（OOM）
  - log=`log/a1_A0_case2_A_g1_only_lambda0010_k8_mask0_20251230_1441.log`
  - 报错：`torch.cuda.OutOfMemoryError`（在 `compute_regulation -> _plane_regulation -> compute_plane_smoothness`）
- B_g2（g2 only）：
  - 50k：psnr3d=36.293, ssim3d=0.943, psnr2d=43.576, ssim2d=0.991
- C_g1g2（g1+g2）：
  - 50k：psnr3d=36.256, ssim3d=0.943, psnr2d=43.570, ssim2d=0.990
- D_g1g2_p2（g1+g2, power=2）：
  - 50k：psnr3d=36.235, ssim3d=0.943, psnr2d=43.519, ssim2d=0.991

阶段性结论（a1）：

- 在当前设置（`mask_ratio=0`, `a1_reg_mask_ratio=0`）下，B/C/D 的 50k 指标差异非常小（均在噪声级别）。
- A_g1 的 OOM 与 a1 本身关联性不强，更像是当时 GPU 上并行进程过多导致的系统性 OOM（日志中列出了多个进程占用显存）。
- 下一步如果要验证 “可观测性加权” 的有效性，应当在 a1 中引入非零 `a1_reg_mask_ratio`（或复用 `mask_ratio`），使 c(x) 在训练中形成非平凡场。

2025-12-31

a1（补）A_g1 重跑（避免上次 OOM；cfg/log 重新命名避免覆盖旧结果）

- cfg=`output/20251231_0352_a1A_g1_rerun_cfg_args.yml`
- log=`log/a1_rerun_A0_case2_A_g1_only_lambda0010_k8_mask0_20251231_0352.log`
- `nohup python train.py --config output/20251231_0352_a1A_g1_rerun_cfg_args.yml > log/a1_rerun_A0_case2_A_g1_only_lambda0010_k8_mask0_20251231_0352.log 2>&1 &`

a2：a1 的 mask 实验（`a1_reg_mask_ratio=0.25`，其余保持一致；验证 observability-weighted 的实际作用）

- 训练设置：case2 A0，从 0 开始 50k，每 10k eval
- baseline：同 a1（`lambda_phys=0, lambda_anchor_smooth=0, mask_ratio=0`）
- a2 超参：`a1_reg_enable=true, a1_reg_lambda=0.01, a1_reg_k=8, a1_reg_mask_ratio=0.25`

4-run：

- A_g1：只开一阶（邻域位移差）
  - cfg=`output/20251231_0352_a2A_g1_mask025_cfg_args.yml`
  - log=`log/a2_A0_case2_A_g1_only_mask025_lambda0010_k8_20251231_0352.log`
- B_g2：只开二阶（离散 laplacian 差，薄板倾向）
  - cfg=`output/20251231_0352_a2B_g2_mask025_cfg_args.yml`
  - log=`log/a2_A0_case2_B_g2_only_mask025_lambda0010_k8_20251231_0352.log`
- C_g1g2：一阶 + 二阶
  - cfg=`output/20251231_0352_a2C_g1g2_mask025_cfg_args.yml`
  - log=`log/a2_A0_case2_C_g1g2_mask025_lambda0010_k8_20251231_0352.log`
- D_g1g2_p2：一阶 + 二阶，且权重更硬（power=2）
  - cfg=`output/20251231_0352_a2D_g1g2_p2_mask025_cfg_args.yml`
  - log=`log/a2_A0_case2_D_g1g2_p2_mask025_lambda0010_k8_20251231_0352.log`

nohup：

- `nohup python train.py --config output/20251231_0352_a2A_g1_mask025_cfg_args.yml > log/a2_A0_case2_A_g1_only_mask025_lambda0010_k8_20251231_0352.log 2>&1 &`
- `nohup python train.py --config output/20251231_0352_a2B_g2_mask025_cfg_args.yml > log/a2_A0_case2_B_g2_only_mask025_lambda0010_k8_20251231_0352.log 2>&1 &`
- `nohup python train.py --config output/20251231_0352_a2C_g1g2_mask025_cfg_args.yml > log/a2_A0_case2_C_g1g2_mask025_lambda0010_k8_20251231_0352.log 2>&1 &`
- `nohup python train.py --config output/20251231_0352_a2D_g1g2_p2_mask025_cfg_args.yml > log/a2_A0_case2_D_g1g2_p2_mask025_lambda0010_k8_20251231_0352.log 2>&1 &`

2025-12-30

s7v4（case2 A0）only-up：固定 0.99 甜点（wA 只允许上调），从 0 开始 50k

- E1_onlyup_d001：
  - log=`log/s7v4_A0_case2_E1_onlyup_d001_wA099_k001_ds100_20251230_114833.log`
  - 50k：psnr3d=36.186, ssim3d=0.943, psnr2d=43.790, ssim2d=0.991
- E2_onlyup_d002：
  - log=`log/s7v4_A0_case2_E2_onlyup_d002_wA099_k001_ds100_20251230_114833.log`
  - 50k：psnr3d=36.093, ssim3d=0.943, psnr2d=43.692, ssim2d=0.991
- E3_onlyup_d002_g005：
  - log=`log/s7v4_A0_case2_E3_onlyup_d002_g005_wA099_k001_ds100_20251230_114833.log`
  - 50k：psnr3d=36.069, ssim3d=0.942, psnr2d=43.609, ssim2d=0.990
  - 统计：`L_graph=0.000000`
- E4_onlyup_d002_g005_t005：
  - log=`log/s7v4_A0_case2_E4_onlyup_d002_g005_t005_wA099_k001_ds100_20251230_114833.log`
  - 50k：psnr3d=36.166, ssim3d=0.943, psnr2d=43.665, ssim2d=0.991
  - 统计：`L_graph=0.000000, L_temp=0.000000`

阶段性结论（s7v4）：

- only-up 成功避免了 wA 被训练压低（wA 始终 >= 0.99），但 `wA_std` 在约 25k 后快速塌到 0，最终 wA 基本退化为常数。
- 在 E3/E4 中，即便设置了 graph/temp 正则，日志中 `L_graph/L_temp` 仍长期为 0，说明当前实现/权重下正则项基本没有生效或数值过小。
- 50k 指标：E1 最好（2D PSNR=43.790），但四个实验差异总体仍偏小；目前更像是 “稳定性/不掉点” 而非 “明确提升”。

## [2025-12-20] s5：将 dx 的融合思想拓展到 ds/dr（逐步实验）

### 背景

V5 在 position 上的核心融合是：

```
Δx = (1-α)·Δx_hex + α·Δx_anchor
```

在 s5 系列中，我们尝试将同样的“结构（Anchor）+ 残差（HexPlane）”融合思想，拓展到 scale 与 rotation。

### 新增参数

- `--s5_rot_nlerp`：rotation 使用单位四元数 nlerp（绝对旋转融合）
- `--s5_scale_log_fusion`：scale 使用 log-space 融合（乘性更新）
- `--s5_jacobian_sr`：从 Anchor 位移场 u(x) 估计 Jacobian 并做 polar 分解，得到 Anchor 的 (scale,rotation) reference
- `--s5_jacobian_k`：Jacobian 估计的邻域 K（默认 8）
- `--s5_eps`：数值稳定 epsilon（默认 1e-8）

### s5.0（baseline）：只做 dx 融合（ds/dr 沿用旧逻辑）

对应当前默认 V5/s3.* 的 ds/dr 逻辑。

### s5.1：Rotation nlerp 融合（reference=原始 rotations）

```
q_hex = rotations_hex
q_ref = rotations
q_new = normalize((1-wA)·q_hex + wA·q_ref)
```

开启：`--s5_rot_nlerp`

### s5.2：Scale log-space 融合（reference=原始 scales）

```
s_hex = scales_hex
s_ref = scales
log(s_new) = (1-wA)·log(s_hex) + wA·log(s_ref)
```

开启：`--s5_scale_log_fusion`

### s5.3：Jacobian→polar 分解得到 Anchor SR reference

从 Anchor 位移场 u(x) 估计局部 Jacobian：

```
F = I + ∂u/∂x
polar(F) ≈ R · S
```

使用 R 给旋转 reference，S 的奇异值给 scale reference（目前实现为 isotropic log-stretch）。

开启：`--s5_jacobian_sr --s5_jacobian_k 8`

### s5.4：组合（推荐对照）：log-scale + rot-nlerp + jacobian SR

开启：`--s5_scale_log_fusion --s5_rot_nlerp --s5_jacobian_sr --s5_jacobian_k 8`

### case1 启动命令模板（wA 与 dx 一致）

说明：wA 取自 dx 的融合权重（若设置了 `--s4_dx_anchor_weight` 则使用该值，否则使用 learnable balance 的 α）。

#### s5.1(case1)

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s5_rot_nlerp \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s5_1_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s5_1_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### s5.2(case1)

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s5_scale_log_fusion \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s5_2_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s5_2_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### s5.4(case1)（log-scale + rot-nlerp + jacobian SR reference）

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s5_scale_log_fusion --s5_rot_nlerp \
  --s5_jacobian_sr --s5_jacobian_k 8 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s5_4_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s5_4_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s5.5：在 wA=1.0（纯 Anchor position）基础上加入 s5.1（rot nlerp），并对照 k=0/0.01/1.0

```
Δx = 0·Δx_hex + 1.0·Δx_anchor
Δs = Δs_hex
Δr: 先做 q_new = nlerp(q_hex, q_ref)，再用 k·Δr_hex 覆盖（对照实验）
```

#### s5.5-k00(case1)

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.0 \
  --s5_rot_nlerp \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s5_5_k00_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s5_5_k00_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### s5.5-k001(case1)

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 0.01 \
  --s5_rot_nlerp \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s5_5_k001_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s5_5_k001_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

#### s5.5-k10(case1)

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s4_dx_anchor_weight 1.0 \
  --s4_dr_hex_weight 1.0 \
  --s5_rot_nlerp \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s5_5_k10_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s5_5_k10_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### s5.6：使用 s5.1 + s5.3，不使用 s5.2（避免 scale log 融合 NaN）

```
Δx: 沿用 V5（或 s4 的 wA 覆盖）
Δs: 沿用 s3.*（此处为 s3_release_scale -> Δs=Δs_hex）
Δr: rot nlerp，reference 来自 Jacobian→polar（Anchor 位移场）
```

```bash
nohup python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation --use_boosted \
  --use_learnable_balance --balance_alpha_init 0.99 --balance_lr 0 \
  --s3_release_scale \
  --s5_rot_nlerp \
  --s5_jacobian_sr --s5_jacobian_k 8 \
  --lambda_balance 0.0 --lambda_prior 0.0 --lambda_tv 0.0 \
  --mask_ratio 0.0 \
  --coarse_iter 5000 --iterations 50000 \
  --test_iterations 5000 10000 20000 30000 40000 50000 \
  --save_iterations 50000 --save_checkpoint \
  --dirname s5_6_case1_$(date +%Y%m%d_%H%M%S) \
  > log/s5_6_case1_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```
