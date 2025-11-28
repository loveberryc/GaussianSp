# X2-Gaussian Inverse-Consistency 扩展方案 v2（在已有 D_f / D_b 基础上新增弱对称约束）

本文件是在已有 `idea.md`（第一阶段：前向 + 反向形变 + 逆一致性）基础上的第二阶段扩展设计。  
假设当前代码已经实现了：

- 共享 K-Planes 编码器
- 前向形变场 D_f(x,t)
- 反向形变场 D_b(x,t)
- 位置级别的逆一致性损失 L_inv_pos（phi_b(phi_f(mu_i,t), t) ≈ mu_i）
- 可配置的 lambda_inv 以及开关

本阶段只新增一个轻量但有价值的扩展：**弱对称约束 D_b ≈ -D_f**。

---

## 1. 当前已有结构回顾（简要）

在现有实现中：

- canonical 高斯中心：mu_i
- 时间 t 的前向位置：
  - mu_i_t = mu_i + D_f(mu_i, t)
- 反向形变把 mu_i_t 拉回 canonical：
  - mu_hat_i = mu_i_t + D_b(mu_i_t, t)

已有的逆一致性损失：

- L_inv_pos = mean_{i,t} || mu_hat_i - mu_i ||_1

---

## 2. 本次扩展的唯一目标：增加 D_b ≈ -D_f 的弱对称正则（L_sym）

### 2.1 设计动机

直觉上，如果形变是“小量”的、可逆的局部平移，那么：

- 从 canonical 到时间 t 的位移大致是 v
- 从时间 t 回到 canonical 的位移大致是 -v

也就是说，在同一对 (mu_i, mu_i_t) 上：

- D_f(mu_i, t) ≈ - D_b(mu_i_t, t)

这并不会强制 D_f / D_b 完全相等，而是作为一个**软正则**，帮助：

- 限制 D_b 的搜索空间，不至于和 D_f 完全“唱反调”；
- 提高 D_f / D_b 的稳定性和数值对齐程度；
- 和已有的 L_inv_pos 形成互补（L_inv_pos 是要求整体闭环，L_sym 是对局部位移对称的偏好）。

### 2.2 具体形式定义

在训练过程中，对每个高斯 mu_i 和时间 t，已有：

- 前向位置：
  - x = mu_i
  - y = mu_i_t = mu_i + D_f(mu_i, t)
- 反向位移：
  - D_b(mu_i_t, t)

我们定义弱对称损失 L_sym 为：

- L_sym = mean_{i,t} || D_b(mu_i_t, t) + D_f(mu_i, t) ||_1

说明：

- 这里使用 L1 范数即可（实现简单，和 L_inv_pos 一致风格）；
- 也可以用 L2，但这次实现优先用 L1。

注意：

- L_sym 和 L_inv_pos 是两个不同的约束：
  - L_inv_pos：要求 phi_b(phi_f(mu_i,t),t) ≈ mu_i（路径闭合）
  - L_sym：要求“来回位移在数值上互为相反”（局部线性近似的物理合理性）

### 2.3 对计算成本的考虑

L_sym 只引入一次简单的向量差异计算：

- D_f(mu_i, t) 和 D_b(mu_i_t, t) 在已有代码中应该已经被计算（用于 mu_i_t 和 mu_hat_i）
- 因此计算 L_sym 的额外开销很小（只是多一个减法和范数）

如果有必要：

- 可以沿用 L_inv_pos 相同的子采样策略（只在子集高斯上算）
- 但第一版实现也可以直接对所有高斯计算 L_sym，视 profiling 决定

---

## 3. 总损失更新

在已有的总损失基础上新增 L_sym：

- 当前（已有）结构类似：
  - L_total = L_render + alpha * L_pc + beta * L_TV_3D + gamma * L_TV_4D + lambda_inv * L_inv_pos

- 新结构：
  - L_total_new = L_total + lambda_sym * L_sym

其中：

- lambda_sym 是新的可配置超参数（float）
- 建议：
  - 默认值为一个较小的数，比如 0.01 或 0.05
  - 必须支持通过配置 / 命令行进行修改
- 当 lambda_sym = 0 时，L_sym 关闭，整个模型退化为上一版本（仅 L_inv_pos 的情况）

---

## 4. 接口与配置要求（保持最小改动原则）

1. 不允许破坏现有 README 中的任何命令：
   - 如果用户不显式打开新功能或配置 lambda_sym，现有命令必须保持原行为

2. 新增配置项（命名仅为建议，可以按照项目风格调整）：
   - use_symmetry_reg (bool, optional)
   - lambda_sym (float, optional)

3. 推荐实现方式：
   - 当 use_symmetry_reg 为 true 且 lambda_sym > 0 时启用 L_sym
   - 否则不计算 L_sym，或者计算后直接不加入总损失

4. 可以在 README 中新增一个示例命令演示如何启用 L_sym，但**不得修改或废弃已有命令**。

---

## 5. 本次暂不实现的后续扩展（仅做备注）

以下属于后续阶段的可能扩展，本轮实现请忽略：

1. 对高斯旋转 R_i、尺度 S_i 的逆一致性约束
2. 基于 SSRML 学到的周期 T，加入完整呼吸周期的 forward-cycle 或 multi-step 一致性
3. 用 velocity field + 时间积分替代 displacement field（diffeomorphic 形变）

这些可以在未来新建 idea_v3 / idea_v4 再逐步引入。
