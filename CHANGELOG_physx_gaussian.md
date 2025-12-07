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

**训练命令**:
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

**参数表**:

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

### 3. x2_gaussian/arguments/__init__.py

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

**主要方法**:

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

**网络结构**:
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

**新增注释**:
```python
# torch.autograd.set_detect_anomaly(True)  # DEBUG: Disabled - may cause issues
```

**新增 `apply_physx_preset()` 函数** (第 27-48 行):
- 打印 PhysX-Gaussian 配置信息
- 显示锚点数量、KNN、mask_ratio、transformer 参数
- 显示损失权重 λ_phys, λ_anchor_smooth

**调用 preset**:
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

**PhysX-Gaussian 损失计算**:
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

**PhysX-Gaussian 统计日志**:
```python
if "phys_completion" in loss:
    metrics['physx_L_phys'] = loss["phys_completion"].item()
if "anchor_smooth" in loss:
    metrics['physx_L_smooth'] = loss["anchor_smooth"].item()
```

### 3. gaussian_model.py 修改

**导入**:
```python
from x2_gaussian.gaussian.anchor_module import AnchorDeformationNet
```

**新增属性初始化**:
```python
self.use_anchor_deformation = getattr(args, 'use_anchor_deformation', False)
self.num_anchors = getattr(args, 'num_anchors', 1024)
self.anchor_k = getattr(args, 'anchor_k', 10)
self.mask_ratio = getattr(args, 'mask_ratio', 0.25)
self._deformation_anchor = None
if self.use_anchor_deformation:
    self._deformation_anchor = AnchorDeformationNet(args)
```

**`create_from_pcd()` 中初始化锚点**:
```python
if self.use_anchor_deformation and self._deformation_anchor is not None:
    self._deformation_anchor = self._deformation_anchor.to("cuda")
    self._deformation_anchor.initialize_anchors(fused_point_cloud)
    self._deformation_anchor.update_knn_binding(fused_point_cloud)
```

**`training_setup()` 添加优化器参数**:
```python
if self.use_anchor_deformation and self._deformation_anchor is not None:
    l.append({"params": list(self._deformation_anchor.get_mlp_parameters()), ...})
    l.append({"params": list(self._deformation_anchor.get_grid_parameters()), ...})
```

**剪枝/密集化后更新 KNN**:
```python
# prune_points() 和 densification_postfix() 中
if self.use_anchor_deformation and self._deformation_anchor is not None:
    self._deformation_anchor.update_knn_binding(self._xyz)
```

**新增 PhysX-Gaussian 方法**:
- `get_active_deformation_network()`
- `compute_anchor_deformation(time, is_training)`
- `compute_physics_completion_loss(time)`
- `compute_anchor_smoothness_loss()`
- `update_anchor_knn_binding()`
- `save_anchor_deformation(path)`
- `load_anchor_deformation(path)`

**`get_deformed_centers()` 修改**:
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

**所有渲染函数添加 `.clone()`**:
- `query()`: `means3D = pc.get_xyz.clone()`, `density = pc.get_density.clone()`, `scales = pc._scaling.clone()`, `rotations = pc._rotation.clone()`
- `render()`: 同上
- `render_prior_oneT()`: 同上

**统一使用 `get_deformed_centers()`**:
- `query()`: `pc.get_deformed_centers(time, ..., is_training=False)`
- `render()`: `pc.get_deformed_centers(time, ..., is_training=True)`
- `render_prior_oneT()`: `pc.get_deformed_centers(time, is_training=False)`

**清理**:
- 移除 `render_prior_oneT()` 中的 `# breakpoint()` 注释

### 5. arguments/__init__.py 修改

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

# 2025-12-02 ~ 2025-12-04 更新

## PhysX-Boosted: 双分支融合架构

### 设计思路

**策略**: "站在巨人肩膀上，触及更高处"

```
Δμ_total = Δμ_hexplane(t) + Δμ_anchor(t)
```

- 保留 100% X²-Gaussian Baseline（HexPlane、所有损失、渲染）
- 添加 Anchor Transformer 作为"物理校正力"
- HexPlane: "画皮肤"（高频纹理、微形变）
- Anchor: "画骨架"（解剖结构、物理一致性）

### 新增参数

```python
# x2_gaussian/arguments/__init__.py
self.use_boosted = False  # 启用 PhysX-Boosted 模式
self.disable_4d_tv = False  # 消融研究：禁用 L_4d_tv
```

---

## PhysX-Boosted V5: 可学习权重融合

### 公式

```
Δx_total = (1 - α) · Δx_hexplane + α · Δx_anchor
α = sigmoid(τ), τ 是可学习参数
```

### 新增参数

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
  ```
  grad_B_orth = grad_B - proj_{grad_A}(grad_B)
  ```

### 新增参数

```python
self.use_orthogonal_projection = False  # 启用 V6
self.ortho_projection_strength = 1.0   # 投影强度
```

### 实现

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

### 公式

HexPlane 和 Anchor 都输出位移 + 不确定性 (log σ²)

```
w_A = 1/(σ_A² + ε), w_B = 1/(σ_B² + ε)
Δx_final = (w_A·Δx_hex + w_B·Δx_anchor) / (w_A + w_B)
```

### Kendall 损失 (CVPR 2017)

```
L_total = L_render/(2Σ) + 0.5·log(Σ)  where Σ = σ_A² + σ_B²
```

### 新增参数

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

### 新增参数

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

**方向1: 周期扰动（模拟不均匀呼吸）**
```bash
python tools/create_robustness_datasets.py \
  --input data/dir_4d_case1.pickle \
  --phase_noise 0.15  # 15% 相位扰动
```

**方向2: 稀疏视角**
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

# 2025-12-05 ~ 2025-12-06 更新

## PhysX-Boosted V10-V16: 掩码建模策略演进

本次更新实现了一系列掩码建模策略的变体（V10-V16），旨在通过自监督学习增强锚点形变网络的鲁棒性。

### 设计目标

**核心问题**: 如何利用 BERT 风格的掩码建模让锚点 Transformer 学习到更鲁棒的形变表示？

**理想效果**: 即使部分锚点信息缺失，网络也能通过学习到的空间-时间关系推断正确的形变。

---

## V10: 解耦掩码 (Decoupled Mask)

### 设计思路

将掩码建模与渲染**解耦**：
- 渲染路径：使用完整的 `forward_anchors()` 输出
- L_phys 路径：使用独立的 `forward_anchors_masked()` 输出

```
渲染: forward_anchors(t, mask=False) → dx_full → render() → L_render
                    ↓ detach
L_phys: forward_anchors_masked(t) → dx_masked → L1(dx_masked[mask], dx_full[mask])
```

### 参数

```python
# x2_gaussian/arguments/__init__.py
self.use_decoupled_mask = False  # 启用 V10
```

### 实现位置

| 文件 | 方法 | 说明 |
|------|------|------|
| `anchor_module.py` | `forward_anchors()` | 当 `use_decoupled_mask=True` 时跳过掩码 |
| `anchor_module.py` | `forward_anchors_masked()` | 专用于 L_phys 的掩码前向传播 |
| `anchor_module.py` | `compute_physics_completion_loss()` | 计算 L_phys，只在被掩码的锚点上 |

### 问题分析

**失败原因**: L_phys 是"自我预测"任务——教师和学生都来自同一个网络。网络可能学会作弊（记忆），而不是学习真正的物理关系。

---

## V11: 预训练-微调 (Pretrain-Finetune)

### 设计思路

分两阶段训练：
1. **预训练阶段** (前 N 步): 高掩码比例 (70%)，只用 L_phys
2. **微调阶段** (N 步后): 低掩码比例或无掩码，加入 L_render

```
Stage 1 (Pretrain): mask_ratio=0.7, L = L_phys only
Stage 2 (Finetune): mask_ratio=0.25, L = L_render + L_phys
```

### 参数

```python
self.use_pretrain_finetune = False  # 启用 V11
self.pretrain_steps = 5000          # 预训练步数
self.pretrain_mask_ratio = 0.7     # 预训练阶段掩码比例
```

### 实现位置

| 文件 | 位置 | 说明 |
|------|------|------|
| `anchor_module.py` | `__init__` | 添加 `_in_pretrain_stage` 状态变量 |
| `anchor_module.py` | `forward_anchors()` | 根据阶段选择 mask_ratio |
| `train.py` | `scene_reconstruction()` | 预训练阶段跳过 densification |

### 问题分析

**失败原因**: 预训练阶段没有外部监督（L_render），L_phys 仍然是自我预测。网络无法学到有意义的表示。

---

## V12: 时间掩码 (Temporal Mask)

### 设计思路

掩码整个时间步，而不是空间锚点：

```
时间 t1: [a1, a2, a3, ..., aM] → 正常处理
时间 t2: [MASK, MASK, MASK, ..., MASK] → 被掩码
时间 t3: [a1, a2, a3, ..., aM] → 正常处理
```

模型需要从其他时间步的信息推断被掩码时间步的形变。

### 参数

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

### 设计思路

将掩码作为**数据增强**，而不是预测目标：

```
Teacher (无掩码): forward_anchors_unmasked(t) → dx_full (detach)
Student (有掩码): forward_with_mask(t) → dx_masked
Loss: L_consist = ||dx_masked - dx_full.detach()||

# 渲染使用 dx_full（无掩码），L_consist 是辅助损失
```

**关键区别**: 损失在**所有锚点**上计算，不仅仅是被掩码的锚点。

### 参数

```python
self.use_consistency_mask = False  # 启用 V13
self.lambda_consist = 0.1         # L_consist 权重
```

### 实现

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

### 设计思路

惩罚锚点运动的"加速度"：

```
dx(t-ε), dx(t), dx(t+ε)
acceleration = dx(t+ε) - 2*dx(t) + dx(t-ε)  # 二阶差分
L_temporal = ||acceleration||²
```

**物理意义**: 自然运动应该是平滑的（加速度接近零）。惩罚高加速度 = 鼓励线性运动。

### 参数

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

### 核心创新

**V10-V15 的问题**:
1. 单时间步处理，无法建模时间关系
2. [MASK] token **替换**原始 token，丢失位置信息
3. 掩码建模是辅助损失 (λ=0.1)，不是主要目标

**V16 解决方案**:
1. Token 是 (锚点, 时间) 对，Transformer 同时建模空间和时间
2. mask_flag_embed 是**加性**嵌入，保留位置/时间信息
3. L_lagbert 是主要目标 (λ=0.5)

### 架构

```
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

### 参数

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

**与 BERT 的区别**:
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

### 训练命令

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

### 详细修改内容

#### 1. arguments/__init__.py

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

## 当前实验状态

✅ **V16 实验运行中**:
- `dir_4d_case1_physx_boosted_v16` (λ_lagbert=0.1, window=1, mask=0.5)

⏳ **待运行的实验**:
- V16 + Fix 1 (st_mask_embed_scale=0.1)
- V16 + Fix 2 (st_coupled_render)
- V16 + 两个 Fix
- 不同 λ_lagbert 消融 (0.1, 0.2, 0.5)
- 不同 st_window_size 消融 (1, 3, 5)
