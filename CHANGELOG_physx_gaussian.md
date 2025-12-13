# PhysX-Gaussian 修改日志

**生成时间**: 2025-12-04 18:50 (更新)  
**基于提交**: PhysX-Boosted V9 + Robustness Testing

---

## 概述

本项目实现了 **PhysX-Gaussian 系列**形变场变体，包括：

1. **PhysX-Gaussian**: 纯 Anchor-based Spacetime Transformer（替代 HexPlane+MLP）
2. **PhysX-Boosted**: HexPlane + Anchor 双分支融合（站在巨人肩膀上）
3. **PhysX-Boosted V5-V9**: 多种融合策略的消融实验版本

### 核心创新

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
