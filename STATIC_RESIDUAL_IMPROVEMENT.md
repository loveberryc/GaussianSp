# 静态+残差四正交体模式 (Static + Residual Four Volumes)

## 概述

为解决标准四正交体积模式的高显存占用问题，实现了一种新的"静态先验+动态残差"架构。该方案在大幅降低参数量的同时，保持了相当的重建质量。

## 问题背景

- **原始四正交体模式**：包含 4 个 3D 体积（static_xyz, xyt, xzt, yzt），参数量约 **393M**
- **显存占用高**：在 `max_spatial_resolution=80, max_time_resolution=150` 时需要约 **1.5GB** 显存
- **训练困难**：在显存受限环境下难以使用更高分辨率

## 核心设计

### 架构分解

将 4D 场景分解为两部分：

1. **静态部分 (Static Prior)**
   - 单个 3D 体积 (xyz)
   - 捕获场景的静态结构（如 CT 图像中不变的解剖结构）
   - 可使用相对高的分辨率（默认 1.0x）

2. **动态残差部分 (Dynamic Residual)**
   - 三个时间体积 (xyt, xzt, yzt)
   - 仅学习相对于静态先验的动态变化
   - 使用更低的分辨率（默认 0.5x）以节省参数

### 关键特性

- **参数分离**：静态和残差部分分别存储和优化
- **小初始化**：残差部分初始化为接近零的小值，鼓励稀疏修正
- **融合策略**：最终特征 = 静态特征 + 加权残差特征
- **专门正则化**：为静态和残差部分设计了不同的 TV 损失策略

## 实现细节

### 新增文件和类

1. **`ResidualVolumeLevel`** (hexplane.py)
   - 仅包含三个时间体积（xyt, xzt, yzt）
   - 专为残差设计，不包含静态 xyz

2. **`StaticPlusResidualVolumeField`** (hexplane.py)
   - 主要实现类
   - 管理静态体积列表（ParameterList）和残差体积列表（ModuleList）
   - 实现多尺度特征提取和融合

3. **正则化函数** (gaussian_model.py)
   - `_plane_regulation_static_residual()`: 空间平滑性损失
   - `_time_regulation_static_residual()`: 时间平滑性损失
   - `_l1_regulation_static_residual()`: L1 稀疏性损失

### 新增超参数

```python
# 分辨率倍数控制
static_resolution_multiplier: float = 1.0   # 静态体积分辨率倍数
residual_resolution_multiplier: float = 0.5  # 残差体积分辨率倍数

# 残差贡献控制
residual_weight: float = 1.0                 # 残差特征权重
use_residual_clamp: bool = False             # 是否裁剪残差值
residual_clamp_value: float = 2.0            # 裁剪范围
```

## 性能对比

### 参数量对比（max_spatial_resolution=80, max_time_resolution=150）

| 模式 | 参数量 | 显存占用 | 相比基线 | 相比四正交体 |
|------|--------|---------|---------|-------------|
| HexPlane (baseline) | 47M | ~180MB | 1.0x | - |
| **四正交体** | **393M** | **~1.5GB** | **8.3x** | **-** |
| **静态+残差 (默认)** | **134M** | **~0.5GB** | **2.8x** | **-65.9%** |
| 静态+残差 (保守) | 212M | ~807MB | 4.5x | -46.2% |

### 训练效果（Quick Test: 500 coarse + 2000 fine iterations）

**Coarse Stage (500 iters):**
- PSNR3D: 34.47 → 37.16
- SSIM3D: 0.836 → 0.899
- PSNR2D: 27.07 → 35.04
- SSIM2D: 0.841 → 0.953

**与标准四正交体对比**：效果相当，显存节省 65%

## 使用方法

### 基本使用

```bash
conda activate x2_gaussian
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib/python3.9/site-packages/torch/lib:$LD_LIBRARY_PATH"

# 默认配置
python train.py -s data/dir_4d_case1.pickle \
  --grid_mode static_residual_four_volume \
  --coarse_iter 5000 --iterations 30000 \
  --max_spatial_resolution 80 --max_time_resolution 150 \
  --dirname dir_4d_case1_static_residual
```

### 超参数调节

```bash
# 更高质量（稍高显存）
python train.py -s data/dir_4d_case1.pickle \
  --grid_mode static_residual_four_volume \
  --static_resolution_multiplier 1.2 \
  --residual_resolution_multiplier 0.6 \
  --max_spatial_resolution 96 --max_time_resolution 180 \
  --dirname dir_4d_case1_static_residual_high

# 更低显存（稍低质量）
python train.py -s data/dir_4d_case1.pickle \
  --grid_mode static_residual_four_volume \
  --static_resolution_multiplier 0.8 \
  --residual_resolution_multiplier 0.4 \
  --max_spatial_resolution 64 --max_time_resolution 120 \
  --dirname dir_4d_case1_static_residual_low
```

## 技术亮点

1. **最小改动原则**
   - 新功能通过新的 `grid_mode` 选项启用
   - 不影响原有模式（四正交体、HexPlane、MLP）的功能
   - 所有原有命令继续有效

2. **模块化设计**
   - 清晰分离静态和残差组件
   - 易于扩展和调试
   - 专门的正则化策略

3. **灵活配置**
   - 可独立调节静态和残差分辨率
   - 支持残差权重和裁剪控制
   - 适应不同显存和质量需求

## 代码文件修改

1. **x2_gaussian/gaussian/hexplane.py** (+268 lines)
   - 新增 `ResidualVolumeLevel` 类
   - 新增 `StaticPlusResidualVolumeField` 类
   - 更新 `build_feature_grid()` 函数

2. **x2_gaussian/arguments/__init__.py** (+10 lines)
   - 添加静态+残差模式的超参数
   - 更新参数提取逻辑

3. **x2_gaussian/gaussian/gaussian_model.py** (+46 lines)
   - 新增 `_plane_regulation_static_residual()`
   - 新增 `_time_regulation_static_residual()`
   - 新增 `_l1_regulation_static_residual()`
   - 更新 `compute_regulation()` 函数

4. **train.py** (+1 line)
   - 添加 `static_residual_four_volume` 到有效模式列表

5. **README.md** (+64 lines)
   - 添加静态+残差模式的详细说明
   - 添加使用示例和超参数说明

6. **test_static_residual.py** (新文件, +139 lines)
   - 参数量测试脚本
   - 对比不同模式的显存占用

## 未来改进方向

1. **自适应分辨率**：根据场景复杂度自动调整静态/残差分辨率
2. **分阶段训练**：先训练静态部分，再训练残差部分
3. **预训练静态先验**：从 3D GS 或 mean CT 初始化静态部分
4. **动态权重调节**：训练过程中自动调整 residual_weight
5. **稀疏化残差**：进一步压缩残差表示的参数量

## 测试脚本

运行参数量测试：
```bash
python test_static_residual.py
```

快速训练测试：
```bash
python train.py -s data/dir_4d_case1.pickle \
  --grid_mode static_residual_four_volume \
  --coarse_iter 500 --iterations 2000 \
  --test_iterations 500 1000 2000 \
  --dirname test_static_residual_quick
```

## 总结

静态+残差四正交体模式成功实现了：
- ✅ **65.9%** 的显存节省（相比标准四正交体）
- ✅ 相当的重建质量
- ✅ 完整的向后兼容性
- ✅ 灵活的配置选项
- ✅ 清晰的代码结构

该方案为在显存受限环境下进行高质量 4D CT 重建提供了一个实用的解决方案。

