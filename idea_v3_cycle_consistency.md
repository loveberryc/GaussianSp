# X2-Gaussian 扩展方案 v3：基于呼吸周期 T 的整周期运动闭环约束（motion cycle-consistency）

本文件是在已经完成以下实现的基础上的第三阶段扩展：

- 第一阶段（idea.md）：前向形变场 D_f、反向形变场 D_b、位置逆一致性损失 L_inv_pos
- 第二阶段（idea_v2_inverse_consistency.md）：弱对称正则 L_sym（D_b ≈ -D_f）

假设当前代码已经实现并可以正常训练：
- 使用 SSRML 学习呼吸周期 T_hat（通常是通过一个 tau_hat 参数，T_hat = exp(tau_hat)）
- 使用 D_f / D_b、L_inv_pos、L_sym，并有相应的开关和超参数

本阶段的目标是：**在形变空间显式利用 T_hat，添加一个“整周期运动闭环约束” L_cycle_motion**。

---

## 1. 已有时间与形变的设置（简要回顾）

当前实现中：

- canonical 高斯中心：mu_i
- 对任意时间 t，前向变形得到该时间的高斯位置：
  - mu_i_t = mu_i + D_f(mu_i, t)

- SSRML 中有一个可学习的周期参数（记为 T_hat）：
  - T_hat = exp(tau_hat)  （具体名字由实现决定）
  - SSRML 的周期一致性已经用 T_hat 在图像空间做约束（I(t) ≈ I(t ± T_hat)）

我们现在要做的是：**在形变空间，用同一个 T_hat 做“整周期闭环”的运动一致性约束**。

---

## 2. 新增损失：L_cycle_motion（一个周期的运动一致性）

### 2.1 直觉

呼吸运动是近似周期性的：

- 经过一个完整周期 T_hat 后，同一解剖点在空间中的位置应该基本回到原处；
- 在 X2-Gaussian 中，mu_i + D_f(mu_i, t) 是时间 t 对应的位置；
- 那么 mu_i + D_f(mu_i, t + T_hat) 应该和 mu_i + D_f(mu_i, t) 接近。

因此我们定义一个“整周期运动闭环”损失：

- 对任意 mu_i 和时间 t，要求：
  - mu_i_t = mu_i + D_f(mu_i, t)
  - mu_i_t_plus = mu_i + D_f(mu_i, t + T_hat)
  - mu_i_t_plus ≈ mu_i_t

### 2.2 形式定义

在实现上，对若干高斯索引 i 和时间 t（可复用当前 batch 的 t_j），定义：

- x1 = mu_i + D_f(mu_i, t)
- x2 = mu_i + D_f(mu_i, t + T_hat)

整周期运动闭环损失：

- L_cycle_motion = mean_{i,t} || x2 - x1 ||_1

说明：

- 使用 L1 范数即可（和其他位置相关 loss 风格一致）；
- t + T_hat 不一定对应真实数据采样点，但 D_f(x, t) 是连续函数，可以在任意时间上查询；
- SSRML 原本为了周期一致性也会在 t + T_hat 上渲染投影，因此代码中通常已有类似时间偏移的逻辑，可以复用。

### 2.3 采样与效率

出于效率考虑，可以采用以下简单策略之一：

1. 对当前 batch 中使用到的时间戳 t_j：
   - 从中随机采样一部分 t_j，用于 L_cycle_motion；
2. 对每个 sampled t_j：
   - 在其对应的高斯子集上计算 x1, x2 和损失；
3. 高斯子集可以复用 L_inv_pos / L_sym 中的采样策略（例如随机采样部分 Gaussian 或只选密度较大的 Gaussian）。

---

## 3. 总损失更新

在已有总损失的基础上增加一个新项 L_cycle_motion。

当前（已有）结构类似：

- L_total =
  - L_render
  + alpha * L_pc
  + beta * L_TV_3D
  + gamma * L_TV_4D
  + lambda_inv * L_inv_pos
  + lambda_sym * L_sym

加入 L_cycle_motion 后：

- L_total_new =
  - L_total
  + lambda_cycle * L_cycle_motion

其中：

- lambda_cycle 是新的可配置超参数（float）
- 建议：
  - 默认值为一个较小数，比如 0.01 或 0.05（可以根据实验调整）
  - 支持通过配置文件或命令行修改
- 当 lambda_cycle = 0 时，L_cycle_motion 关闭，模型退化为上一版本（不影响 D_f / D_b / L_inv_pos / L_sym 的行为）

---

## 4. 接口与配置要求

1. 不破坏现有 README 中的所有命令：
   - 如果用户不显式开启新特性或不设定 lambda_cycle，原有命令必须保持行为不变

2. 建议新增配置项（命名可按项目风格调整）：
   - use_cycle_motion (bool, optional)
   - lambda_cycle (float, optional)

3. 实现建议：

   - 在训练时，如果 use_cycle_motion 为 true 且 lambda_cycle > 0：
     - 计算 L_cycle_motion，并加到总损失中；
   - 否则：
     - 不计算或计算后不加到总损失中。

4. 可以在 README 中新增一个示例命令，例如：

   - `python train.py --use_inverse_consistency --lambda_inv 0.05 --use_symmetry_reg --lambda_sym 0.01 --use_cycle_motion --lambda_cycle 0.01`

   但不得要求修改原有命令才能正常运行。

---

## 5. 本次暂不实现的更重扩展（仅备注）

本阶段只实现 L_cycle_motion，其它更重的扩展暂不实现，包括但不限于：

- 使用 velocity field + 时间积分（diffeomorphic 形变）
- 对旋转 / 尺度的周期一致性或整周期闭环约束
- 多阶段 / 多步的时序 cycle consistency（比如在 T_hat 的多个分段时间点上施加额外约束）

这些可在未来的 idea_v4 / v5 中再做计划。
