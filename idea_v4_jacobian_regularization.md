# X2-Gaussian 扩展方案 v4：基于 Jacobian 的局部防折叠正则（Jacobian Regularization for Non-Folding Motion）

本文件是在已完成以下模块基础上的第四阶段扩展设计：

- v1：前向形变场 D_f、反向形变场 D_b、位置逆一致性 L_inv_pos
- v2：弱对称正则 L_sym（D_b ≈ -D_f）
- v3：基于 SSRML 学到的周期 T_hat 的整周期运动闭环约束 L_cycle_motion

假设当前代码已经：
- 使用 D_f / D_b 对 Gaussian 位置进行时空形变；
- 在损失中加入了 L_inv_pos、L_sym、L_cycle_motion，并具有对应开关和权重。

本阶段的目标是：**在不改变现有网络结构的前提下，引入一个对前向形变场的局部 Jacobian 正则，显式惩罚局部折叠（det(J) < 0）和极端体积变形。**

---

## 1. 现有形变场设定回顾（针对位置）

我们只关注 Gaussian 中心的位置形变（不涉及旋转/尺度）：

- canonical 高斯中心：mu_i ∈ R^3
- 前向形变场 D_f(x, t)：从 canonical 到时间 t 的位移
- 对于给定时间 t，前向映射：
  - phi_f(x, t) = x + D_f(x, t)
- 在实现中，时间 t 通常通过 K-Planes + MLP 编码器/解码器输入 D_f

已有的逆一致性和周期约束都在位置层面使用了 phi_f 和 D_f。本次扩展**只在 phi_f 上做局部 Jacobian 正则，不修改任何现有结构。**

---

## 2. Jacobian 矩阵与防折叠直觉

对于固定时间 t，phi_f(x, t) 是 R^3 → R^3 的映射。其在局部的线性近似由 Jacobian 给出：

- J_f(x, t) = ∂phi_f(x, t) / ∂x    （3×3 矩阵）

直觉：

- 如果 det(J_f) < 0，局部发生翻转/折叠（orientation 反转）；
- 如果 det(J_f) 非常小或非常大，局部体积变化极端（强压缩/膨胀）。

我们希望：

1. 优先惩罚 det(J_f) < 0 的情况（防止折叠 / 自交）；
2. 适度惩罚 det(J_f) 在数值上偏离 1 太多（volume 过度不守恒）。

为保持实现简单，这里使用一个**soft penalty**：

- 对负的 det(J_f) 做 ReLU 惩罚；
- 对 log(det(J_f)) 的偏离做二次惩罚（可选，MVP 可以只实现第一项）。

---

## 3. MVP：Jacobian 基本防折叠正则 L_jac（只惩罚 det < 0）

### 3.1 采样点定义

在训练过程中，我们只在少量样本点上估计 Jacobian，以控制计算成本。  
建议的采样策略：

1. 在当前 batch 中的若干时间戳 t（可以复用已有训练时选择的 t_j）；
2. 对每个 t，从当前参与渲染的 Gaussian 中心 mu_i 里随机采样一小部分（例如 N_jac 个）作为 x 点；
3. 在这些 (x, t) 上估计 J_f(x, t) 的行列式 det_J。

### 3.2 使用有限差分近似 Jacobian（避免二阶梯度）

为避免使用二阶自动微分（会比较重），这里采用**有限差分**来近似 Jacobian：

记：

- x = mu_i  ∈ R^3
- e_x, e_y, e_z 为标准基向量
- h 为一个小的步长（常数，例如 1e-3 或与场尺度相适配）

定义：

- y0 = phi_f(x, t)
- yx = phi_f(x + h·e_x, t)
- yy = phi_f(x + h·e_y, t)
- yz = phi_f(x + h·e_z, t)

则 Jacobian 的三列近似为：

- J_col_x ≈ (yx - y0) / h
- J_col_y ≈ (yy - y0) / h
- J_col_z ≈ (yz - y0) / h

将三列拼成 3×3 矩阵：

- J ≈ [ J_col_x, J_col_y, J_col_z ]  （列拼接）

然后计算：

- det_J = det(J)

上述过程只需要前向调用 D_f/phi_f，**不需要**对参数求二阶导数。

### 3.3 防折叠损失定义（MVP 版本）

最小实现版，我们只惩罚 det_J < 0 的情况：

- 对每个采样点 (x, t) 计算 det_J
- 定义：
  - penalty_neg = ReLU(-det_J)    # 如果 det_J < 0，则给正罚，否则为 0
- 最终：
  - L_jac = mean_{(x,t)} penalty_neg

备注：

- 这是一个非常保守的正则：只有发生局部翻转时才罚；
- 不限制 det_J 大小，只防止“反转”，不会抑制正常的适度压缩/膨胀。

如果之后希望更强，可以在新一轮 idea 中加入对 |log(det_J)| 的惩罚。

---

## 4. 总损失中的接入方式

当前总损失结构类似：

- L_total =
  - L_render
  + alpha * L_pc
  + beta * L_TV_3D
  + gamma * L_TV_4D
  + lambda_inv * L_inv_pos
  + lambda_sym * L_sym
  + lambda_cycle * L_cycle_motion

本次扩展新增一项：

- L_total_new =
  - L_total
  + lambda_jac * L_jac

其中：

- lambda_jac 是新的可配置超参数（float）
- 建议：
  - 默认值为一个较小的数，比如 0.01 或 0.05
  - 支持通过配置文件或命令行修改
- 当 lambda_jac = 0 时，L_jac 关闭，模型完全退化为上一版本（无 Jacobian 正则）

---

## 5. 接口与配置要求（最小改动原则）

1. 不破坏现有 README 中的所有命令：
   - 如果用户不显式开启新特性或不设置 lambda_jac，现有命令必须行为不变

2. 新增配置项（命名可按项目风格调整）：
   - use_jacobian_reg: bool, optional
   - lambda_jac: float, optional
   - jacobian_num_samples: int, optional（每个 batch 用于估计 Jacobian 的点数，可以有默认值，例如 64 或 128）
   - jacobian_step_size: float, optional（有限差分步长 h，可以有默认值，例如 1e-3）

3. 建议逻辑：

   - 当 use_jacobian_reg 为 true 且 lambda_jac > 0：
     - 在训练中对每个 batch 采样少量 (x, t)，使用上述有限差分方法估计 det_J，并计算 L_jac。
   - 否则：
     - 不计算 L_jac 或计算后不加入总损失。

4. 可以在 README 中新增一个示例命令，例如：

   - `python train.py --use_inverse_consistency --lambda_inv 0.05 --use_symmetry_reg --lambda_sym 0.01 --use_cycle_motion --lambda_cycle 0.01 --use_jacobian_reg --lambda_jac 0.01`

   但不得要求用户修改原有命令。

---

## 6. 本阶段暂不实现的进一步扩展（仅备注）

以下内容属于未来可能进一步增强的方向，本阶段请不要实现：

1. 对 det_J 进行更强的 volume regularization，例如对 log(det_J) 做二次惩罚；
2. 计算 Jacobian 的旋转/尺度分解，并分别对局部的 shear / anisotropy 做约束；
3. 在反向场 D_b 上也加入对 Jacobian 的正则，形成完整的“双向 diffeomorphic regularization”。

这些可以在未来新建 idea_v5 / v6 再做详细设计。
