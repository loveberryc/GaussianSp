# Time-Aware Adaptive Residual Sparsification (TARS)

## 🎯 核心创新

**TARS**是一种时间感知的自适应残差稀疏化方法，专为动态4D CT重建设计。该方法结合了：
1. **可学习时间权重** - 让网络自动学习心动周期中哪些时刻需要更多动态表达
2. **自适应稀疏正则化** - 鼓励不重要时刻的残差接近零
3. **时间平滑约束** - 保持相邻时刻的连续性

## 📊 方法对比

| 方法 | 参数效率 | 时间适应性 | 医学合理性 | 实现复杂度 |
|------|---------|-----------|-----------|-----------|
| 原始四体积 | ⭐⭐ | ❌ | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| 静态+残差 | ⭐⭐⭐⭐ | ❌ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **静态+残差+TARS** | ⭐⭐⭐⭐⭐ | ✅ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## 🔬 核心原理

### 数学表达

```
输出特征 = 静态特征 + α(t) · w_global · 残差特征(t)

其中：
- α(t) ∈ [0, 1]: 时间步 t 的自适应权重
- w_global: 全局残差权重（原有参数）
- 残差特征(t): 时间相关的残差部分
```

### 损失函数

```
L_total = L_recon + L_4D_TV + L_TARS

L_TARS = λ₁ · L1(α) + λ₂ · Smooth(α)

其中：
- L1(α) = mean(|α|): 稀疏性正则化，鼓励部分α→0
- Smooth(α) = mean((α[t+1] - α[t])²): 时间平滑性
- λ₁: 稀疏性权重（默认0.001）
- λ₂: 平滑性权重（默认0.01）
```

### 为什么有效？

#### 1. 医学先验知识
- **心动周期收缩期**（变化大）→ α(t) ≈ 1.0（高残差权重）
- **舒张期**（变化小）→ α(t) ≈ 0.0（低残差权重）
- 网络自动学习这种模式，无需人工标注

#### 2. 参数效率
- 只增加 T 个可学习参数（T = 时间步数，通常150）
- 远小于残差体积参数（百万级）
- 通过稀疏化减少有效参数量

#### 3. 动态资源分配
- 自动将计算资源分配到关键时刻
- 减少冗余计算
- 提升收敛速度和最终质量

## 🚀 使用方法

### 基础用法

```bash
# 启用TARS（在静态+残差模式基础上）
python train.py -s data/dir_4d_case1.pickle \
  --grid_mode static_residual_four_volume \
  --use_static_prior \
  --use_time_aware_residual \
  --coarse_iter 5000 --iterations 30000 \
  --dirname dir_4d_case1_with_tars
```

### 完整参数配置

```bash
# 推荐配置（带静态先验 + TARS）
python train.py -s data/dir_4d_case1.pickle \
  --grid_mode static_residual_four_volume \
  --use_static_prior \
  --static_prior_resolution 64 \
  --use_time_aware_residual \
  --time_weights_sparsity_weight 0.001 \
  --time_weights_smoothness_weight 0.01 \
  --max_spatial_resolution 80 \
  --max_time_resolution 150 \
  --coarse_iter 5000 --iterations 30000 \
  --test_iterations 5000 7000 10000 20000 30000 \
  --dirname dir_4d_case1_tars_full
```

### 参数说明

```bash
# TARS核心参数
--use_time_aware_residual                 # 启用TARS（布尔标志）
--time_weights_sparsity_weight 0.001      # 时间权重L1稀疏正则化权重
--time_weights_smoothness_weight 0.01     # 时间权重平滑正则化权重

# 配合使用的参数
--use_static_prior                        # 启用静态先验初始化（强烈推荐）
--static_prior_resolution 64              # 先验计算分辨率
--static_resolution_multiplier 1.0        # 静态体积分辨率倍数
--residual_resolution_multiplier 0.5      # 残差体积分辨率倍数
--max_time_resolution 150                 # 最大时间分辨率（TARS时间权重数量）
```

### 超参数调节指南

#### 稀疏性权重 (`time_weights_sparsity_weight`)

```bash
# 弱稀疏化（接近不使用TARS）
--time_weights_sparsity_weight 0.0001

# 适中稀疏化（推荐，默认）
--time_weights_sparsity_weight 0.001

# 强稀疏化（更多时间步被抑制）
--time_weights_sparsity_weight 0.01
```

**调节原则**：
- 如果观察到所有α(t)都接近1.0 → 增大此权重
- 如果观察到过多α(t)接近0.0 → 减小此权重
- 目标：约30-50%的时间步α < 0.5

#### 平滑性权重 (`time_weights_smoothness_weight`)

```bash
# 弱平滑（允许更多突变）
--time_weights_smoothness_weight 0.001

# 适中平滑（推荐，默认）
--time_weights_smoothness_weight 0.01

# 强平滑（强制连续）
--time_weights_smoothness_weight 0.1
```

**调节原则**：
- 如果α(t)曲线抖动严重 → 增大此权重
- 如果α(t)曲线过于平坦 → 减小此权重
- 目标：平滑过渡 + 保留关键变化点

## 📈 预期效果

### 参数效率提升

| 配置 | 总参数量 | 有效参数量 | 压缩比 |
|------|---------|-----------|--------|
| 四体积（原始） | ~30M | ~30M | 1.0x |
| 静态+残差 | ~15M | ~15M | 2.0x |
| **静态+残差+TARS** | ~15M + 150 | ~10M (估计) | **3.0x** |

### 训练效率

- **收敛速度**：预计提升 20-30%（相比无TARS）
- **最终质量**：PSNR +0.5~1.0 dB（估计）
- **显存占用**：与静态+残差基本相同（+150个参数可忽略）

### 可解释性

- **时间权重可视化**：可导出α(t)曲线
- **医学验证**：α(t)峰值应对应心动周期收缩期
- **自动发现**：无需人工标注心动相位

## 🔬 技术细节

### 实现架构

```
StaticPlusResidualVolumeField
├── static_grids: nn.ParameterList        # 静态3D体积
├── residual_grids: nn.ModuleList         # 残差4D体积 (xyt, xzt, yzt)
└── time_weights: nn.Parameter [T]        # TARS时间权重 (NEW!)
```

### Forward Pass

```python
# 1. 采样静态特征
static_feat = sample_static_3d(static_vol, pts)

# 2. 采样残差特征
residual_feat = sample_residual_4d(residual_vols, pts, t)

# 3. 应用TARS自适应权重
if use_time_aware_residual:
    alpha_t = sigmoid(time_weights[t])      # [0, 1]
    residual_feat = residual_feat * alpha_t

# 4. 应用全局残差权重
residual_feat = residual_feat * residual_weight

# 5. 组合
output = static_feat + residual_feat
```

### 正则化计算

```python
# L1稀疏性损失
L_sparsity = mean(|time_weights|)  # 鼓励权重接近0

# 时间平滑性损失
L_smoothness = mean((time_weights[t+1] - time_weights[t])^2)

# 总TARS损失
L_TARS = λ₁ · L_sparsity + λ₂ · L_smoothness
```

## 🧪 实验建议

### 对比实验

```bash
# A组：无TARS（基线）
python train.py -s data/dir_4d_case1.pickle \
  --grid_mode static_residual_four_volume \
  --use_static_prior \
  --dirname exp_no_tars

# B组：有TARS（新方法）
python train.py -s data/dir_4d_case1.pickle \
  --grid_mode static_residual_four_volume \
  --use_static_prior \
  --use_time_aware_residual \
  --dirname exp_with_tars

# 对比：收敛曲线、PSNR、参数有效性、时间权重模式
```

### 消融实验

```bash
# 1. 只稀疏化（无平滑）
--use_time_aware_residual \
--time_weights_sparsity_weight 0.001 \
--time_weights_smoothness_weight 0.0

# 2. 只平滑（无稀疏化）
--use_time_aware_residual \
--time_weights_sparsity_weight 0.0 \
--time_weights_smoothness_weight 0.01

# 3. 完整TARS
--use_time_aware_residual \
--time_weights_sparsity_weight 0.001 \
--time_weights_smoothness_weight 0.01
```

## 📊 可视化时间权重

训练完成后，可通过以下方式获取时间权重：

```python
import torch
from x2_gaussian.gaussian import GaussianModel

# 加载模型
gaussians = GaussianModel.load_checkpoint("output/dir_4d_case1_tars_full/point_cloud/iteration_30000/")

# 获取时间权重（已应用sigmoid）
if hasattr(gaussians._deformation.deformation_net.grid, 'get_time_weights_visualization'):
    time_weights = gaussians._deformation.deformation_net.grid.get_time_weights_visualization()
    
    # 可视化
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 4))
    plt.plot(time_weights.numpy())
    plt.xlabel('Time Step')
    plt.ylabel('Adaptive Weight α(t)')
    plt.title('TARS Time-Aware Weights')
    plt.grid(True)
    plt.savefig('tars_weights.png')
    print(f"Time weights range: [{time_weights.min():.3f}, {time_weights.max():.3f}]")
    print(f"Sparse time steps (α<0.5): {(time_weights < 0.5).sum()}/{len(time_weights)}")
```

## 🔧 故障排查

### 问题1：所有时间权重都接近1.0

**原因**：稀疏性正则化权重太小

**解决**：
```bash
--time_weights_sparsity_weight 0.01  # 增大到0.01
```

### 问题2：时间权重曲线抖动严重

**原因**：平滑性正则化权重太小

**解决**：
```bash
--time_weights_smoothness_weight 0.1  # 增大到0.1
```

### 问题3：重建质量下降

**原因**：过度稀疏化，丢失重要动态信息

**解决**：
```bash
# 方案1：减小稀疏性权重
--time_weights_sparsity_weight 0.0001

# 方案2：增大残差分辨率
--residual_resolution_multiplier 0.7  # 从0.5增到0.7

# 方案3：增大全局残差权重
--residual_weight 1.5  # 从1.0增到1.5
```

### 问题4：训练不稳定

**原因**：TARS正则化权重太大

**解决**：
```bash
# 降低两个权重
--time_weights_sparsity_weight 0.0001
--time_weights_smoothness_weight 0.001
```

## 🎓 引用

如果您使用TARS方法，请引用：

```bibtex
@article{x2gaussian_tars2024,
  title={Time-Aware Adaptive Residual Sparsification for Dynamic 4D CT Reconstruction},
  author={[Your Name]},
  journal={X2-Gaussian with TARS},
  year={2024}
}
```

## 📚 相关资源

- **静态先验文档**: [STATIC_PRIOR_USAGE.md](STATIC_PRIOR_USAGE.md)
- **静态+残差基础**: [STATIC_RESIDUAL_IMPROVEMENT.md](STATIC_RESIDUAL_IMPROVEMENT.md)
- **主项目README**: [README.md](README.md)

## 🔮 未来改进

1. **自适应正则化权重**：根据训练阶段自动调整λ₁和λ₂
2. **多尺度时间权重**：不同分辨率级别使用不同时间权重
3. **频域TARS**：在频率域应用自适应权重（捕捉周期性）
4. **心动相位对齐**：结合ECG信号自动对齐α(t)峰值
5. **分层稀疏化**：空间+时间联合自适应权重

---

**版本**: 1.0.0  
**日期**: 2024-11-14  
**作者**: X2-Gaussian TARS Extension

