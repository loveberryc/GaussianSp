# Inverse-Consistent Motion Field for X2-Gaussian (MVP 版本规格)

本文件说明在现有 X2-Gaussian 项目中，引入「前向 + 反向形变场 + 逆一致性正则」的 **最小实现版本**，用于第一次基础实验。  
目标是：**在尽量少改动原代码的前提下，实现一个可跑通、有明显方法创新点的版本**。

---

## 0. 背景（简要）

现有 X2-Gaussian 使用一个时空形变场 D(x, t)，从 canonical 空间（例如静态 Gaussian 中心 mu_i）预测在时间 t 的位移：

- mu_i(t) = mu_i + D(mu_i, t)

我们希望在此基础上，引入：

- 前向形变场 D_f(x, t)：从 canonical 到时间 t
- 反向形变场 D_b(x, t)：从时间 t 回到 canonical

并在 Gaussian 中心上添加一个「逆一致性」正则，鼓励：

- 从 canonical 走到时间 t，再走回 canonical，位置尽量一致
- 形式上：phi_f(x,t) = x + D_f(x,t), phi_b(y,t) = y + D_b(y,t)  
  要满足 phi_b(phi_f(x,t), t) ≈ x

---

## 1. MVP 必须实现的内容（这部分是这次实现的唯一目标）

本次实现 **只做以下内容**，其它进阶想法一律暂不实现。

### 1.1 双形变场结构（共享编码器，双 MLP 头）

在现有 K-Planes + MLP 形变框架基础上：

1. 保持 **K-Planes 编码器不变**：  
   - 输入仍是 (x, t)，输出时空特征 f_h(x, t)

2. 在解码器部分，从原来的「单个形变 head」改为：

   - 前向形变 head：
     - D_f(x, t) = F_f(f_h(x, t))
   - 反向形变 head：
     - D_b(x, t) = F_b(f_h(x, t))

3. 对于 MVP，可以采用任意一种简单实现：
   - 推荐：共享一部分 MLP，再用两个线性层分别输出 D_f / D_b
   - 也可以直接两个独立 MLP head，只要实现简单可靠即可

> 限制：本次只在 Gaussian **位置**（mu_i）上使用前后形变，**不对旋转和尺度做逆一致性**（即 R_i / S_i 不参与 inverse consistency）。

### 1.2 前向形变：保持原有功能

前向形变 D_f 必须兼容并尽量复用现有逻辑：

- 原逻辑为：mu_i(t) = mu_i + D(mu_i, t)
- 现在改为：mu_i(t) = mu_i + D_f(mu_i, t)

要求：

- 在不开启新功能开关时，行为应尽量等价于原来的 D（例如可以初始化 D_b 为 0，不影响原训练）

### 1.3 反向形变 + 逆一致性损失（只在位置上）

在训练过程中，为每个时间 t 和高斯中心 mu_i：

1. 使用前向形变得到时间 t 的位置：

   - x = mu_i（canonical 坐标）
   - y = mu_i_t = mu_i + D_f(mu_i, t)

2. 用反向形变把 y 映射回 canonical：

   - x_hat = y + D_b(y, t)

3. 定义位置逆一致性损失（L1 范数）：

   - L_inv_pos = mean over (i, t) of || x_hat - x ||_1

实现细节要求：

- 出于性能考虑，可以只在一个子集高斯上计算 L_inv_pos：
  - 比如随机采样一部分高斯
  - 或只选密度较大的高斯
- 但逻辑上要支持对所有高斯进行计算（方便之后改实验）

### 1.4 总损失中加入 L_inv_pos

在现有总损失基础上新增一项：

- 原有结构类似：
  - L_total = L_render + alpha * L_pc + beta * L_TV_3D + gamma * L_TV_4D

- 新结构：
  - L_total_new = L_total + lambda_inv * L_inv_pos

要求：

- 新增一个可配置超参数 `lambda_inv`（例如通过 config / 命令行传入）
- 建议默认值可以是一个较小数，例如 0.01 或 0.05（具体数值可以先硬编码，然后再暴露到配置中）
- 当 `lambda_inv = 0` 时，模型应退化回“没有逆一致性约束”的原始行为（除了结构上多了 D_b，但不会产生额外影响）

---

## 2. 接口与最小改动原则

### 2.1 原有 README 命令必须不被破坏

- 所有 README 中已有的命令（训练/测试/可视化等）必须继续保持原行为和默认效果
- 默认情况下（无额外参数时），应尽量保持与原论文结果兼容

### 2.2 新功能通过新开关启用

要求：

1. 通过 **新的配置项或命令行选项** 来开启 inverse-consistency：

   - 示例（仅供参考，可按项目风格命名）：
     - `--use_inverse_consistency`
     - 配置文件中 `use_inverse_consistency: true/false`
     - `lambda_inv: 0.01`

2. 当此开关为 `false` 或 `lambda_inv = 0` 时：
   - D_b 虽然可以被构造，但不应影响任何原有训练逻辑或输出结果
   - 程序不会报错或改变原有 pipeline

3. 允许在后续 README 中补充新命令，例如：
   - `python train.py --use_inverse_consistency --lambda_inv 0.01`
   - 但必须保证原命令不需要修改即可照常运行

---

## 3. 暂不实现的扩展内容（本次请忽略）

以下内容属于方法的潜在进一步扩展，本次实验 **请不要实现**，以免增加复杂度、影响首次跑通：

1. 对高斯旋转 R_i、尺度 S_i 的逆一致性（如 L_inv_rot）
2. D_b ≈ -D_f 这样的对称性正则（L_sym）
3. 使用 velocity field + 积分来替代位移（diffeomorphic 变形）
4. 基于完整呼吸周期 T 的多步 / 周期闭环一致性
5. 修改 SSRML 部分的任何实现逻辑（本次只在形变场上改动）

如果后续基础版本稳定跑通，可以在新的 idea_xxx.md 中再逐步扩展这些想法。

---

## 4. 验证与预期效果（简单版本）

本次实现后，优先关注：

1. 模型可以在开启 `use_inverse_consistency` 后正常训练、收敛（不 crash）
2. 对同样的训练设置：
   - 对比关闭 / 开启 inverse-consistency 时的
     - 重建 PSNR/SSIM（特别是未见时间点的插值）
     - 形变场是否更加平滑、无明显异常漂移
3. 保证原始 README 中的实验命令依然可用，结果不明显退化

