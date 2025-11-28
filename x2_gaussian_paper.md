# X²-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction — 方法设计说明

> 目标：从一段时间内采集到的 X-ray 投影序列，重建**任意时间点的 3D CT 体数据**，实现真正的连续时间 4D CT（而不是传统的 10 个离散相位）。

核心思想：

1. 用 **Radiative Gaussian Splatting** 表达 3D CT 体（高斯体元集合）。
2. 在此基础上，引入一个 **时间相关的变形场**，让每个高斯随时间发生平移 / 旋转 / 缩放，从而实现连续时间 4D 建模（Dynamic Gaussian Motion Modeling, DGMM）。
3. 不用额外呼吸带等硬件，而是通过 **呼吸周期的自监督学习**（Self-Supervised Respiratory Motion Learning, SSRML），在训练中自动拟合病人的呼吸周期，并通过“周期一致性损失”约束时间维度。

---

## 1. 问题设定与输入输出

- 输入：
  - 一组投影图像：\(\{ I_j \}_{j=1}^N\)
  - 对应的时间戳：\(\{ t_j \}_{j=1}^N\)，扫描期间均匀分布
  - 对应的几何参数（view / projection matrices）：\(\{ M_j \}_{j=1}^N\)

- 目标：
  - 学习一个**连续时间的密度场** \(\sigma(x, t)\)，可以在任意时间 \(t\) 输出 3D CT 体（或渲染出对应时间的 X-ray 投影）。
  - **不使用**外部呼吸信号 / gating 设备，也**不做**传统的 phase-binning 到 10 个离散相位。

---

## 2. Radiative Gaussian Splatting 预备知识

### 2.1 静态高斯场表示

CT 体被表示为一组 3D 高斯核：
\[
\mathcal{G} = \{ G_i \}_{i=1}^K
\]
每个高斯 \(G_i\) 包含：

- 中心：\(\mu_i \in \mathbb{R}^3\)
- 协方差矩阵：\(\Sigma_i \in \mathbb{R}^{3\times3}\)
- 密度（幅度）：\(\rho_i\)

高斯核：
\[
G_i(x|\rho_i,\mu_i,\Sigma_i)
=
\rho_i \cdot \exp\left(
-\frac12 (x-\mu_i)^T \Sigma_i^{-1}(x-\mu_i)
\right)
\]

协方差分解为旋转 + 缩放：
\[
\Sigma_i = R_i S_i S_i^T R_i^T
\]
其中 \(R_i\) 是 \(3\times3\) 旋转矩阵，\(S_i\) 是对角尺度矩阵。

整体密度场：
\[
\sigma(x) = \sum_{i=1}^K G_i(x|\rho_i,\mu_i,\Sigma_i)
\]

### 2.2 X-ray 前向模型（Radiative GS）

对一条射线 \(r(t) = o + t d\)，根据 Beer–Lambert 定律：

- \(I_0\)：入射强度
- \(I'(r)\)：出射强度

\[
I(r) = \log I_0 - \log I'(r) = \int \sigma(r(t)) \, dt
\]

得到投影像素值（简写为高斯积分和）：
\[
I_r(r) = \sum_{i=1}^K \int G_i(r(t)|\rho_i,\mu_i,\Sigma_i) \, dt
\]

Radiative Gaussian Splatting 的工作就是**高效地近似上式积分**，并在 GPU 上用 3D 高斯“splat” 到 2D 图像上完成渲染。

---

## 3. 整体框架概览

X²-Gaussian 的整体结构（对应原文 Fig. 2，文字复述）：

1. **静态 Radiative GS 预训练**
   - 忽略时间，只用所有投影对同一个“静态”体进行重建。
   - 得到初始的高斯集合 \(\{ G_i \}\) 作为**参考状态（canonical pose）**。

2. **Dynamic Gaussian Motion Modeling (DGMM)**
   - 为每个高斯引入随时间变化的偏移：
     \[
     G_i'(t) = (\mu_i + \Delta\mu_i(t),\; R_i + \Delta R_i(t),\; S_i + \Delta S_i(t),\; \rho_i)
     \]
   - \(\Delta\mu_i(t), \Delta R_i(t), \Delta S_i(t)\) 由一个**时空编码器 + 解码器**预测：
     - Encoder：基于 4D K-Planes，输入 \((x,y,z,t)\)
     - Decoder：多头 MLP，分别输出位置 / 旋转 / 缩放的偏移

3. **Self-Supervised Respiratory Motion Learning (SSRML)**
   - 把呼吸周期 \(T\) 作为**可学习参数** \( \hat T = \exp(\hat\tau) \)
   - 根据“**呼吸周期内同结构会回到近似相同位置**”的生理先验：
     - 强制网络在时间 \(t\) 和 \(t + n \hat T\)（\(n = \pm 1\)）渲染出的图像保持一致。
   - 通过这个“周期一致性损失”在训练中自适应拟合病人专属的呼吸周期。

4. **端到端优化**
   - 渲染损失（与真实投影比对） + 周期一致性损失 + 3D/4D TV 正则共同训练：
     - 高斯参数（\(\mu_i, R_i, S_i, \rho_i\)）
     - K-Plane 网格参数
     - 解码器参数
     - 呼吸周期参数 \(\hat \tau\)

---

## 4. Dynamic Gaussian Motion Modeling（DGMM）

### 4.1 高斯运动参数化

在参考时间（例如 \(t=0\)）下，每个高斯有固定参数：
\[
G_i = (\mu_i, R_i, S_i, \rho_i)
\]

为了在任意时间 \(t\) 得到其状态，引入时间依赖的偏移（由网络预测）：
\[
G_i'(t)
=
(\mu_i + \Delta\mu_i(t),\; R_i + \Delta R_i(t),\; S_i + \Delta S_i(t),\; \rho_i)
\]

这里的设计要点：

- **平移 \(\Delta\mu_i(t)\)**：建模器官的整体移动（如膈肌上下）。
- **旋转 \(\Delta R_i(t)\)**：建模局部结构的姿态变化。
- **缩放 \(\Delta S_i(t)\)**：建模肺叶鼓胀 / 收缩等体积变化。
- 整体上是“现有高斯参数 + 小偏移”的形式，便于稳定训练。

### 4.2 4D K-Planes 时空编码器 \(E\)

输入：

- 高斯中心位置 \(\mu = (x, y, z)\)
- 时间戳 \(t\)

组成 4D 坐标：
\[
v = (x, y, z, t)
\]

将 \(v\) 投影到 6 个二维平面（K-Planes）上：

- 3 个纯空间平面：\(P_{xy}, P_{xz}, P_{yz}\)
- 3 个时空平面：\(P_{xt}, P_{yt}, P_{zt}\)

每个平面是多分辨率的 feature grid：
- 第 \(l\) 个分辨率：\(P_{ab}^{(l)} \in \mathbb{R}^{d \times lM \times lM}\)
- \(M\)：基本分辨率，\(l\) 控制多尺度

对每个平面，使用双线性插值获得特征：
\[
f_{ab}^{(l)} = \psi\big(P_{ab}^{(l)}, \text{proj}_{ab}(v)\big)
\]

把所有平面、所有分辨率的特征通过**特征拼接 + Hadamard 乘积**组合：
\[
f_e = \bigoplus_l\ \bigotimes_{(a,b)\in\{(x,y),(x,z),(y,z),(x,t),(y,t),(z,t)\}}
f_{ab}^{(l)}
\]

然后通过一个很小的 MLP \(\phi_h\) 融合：
\[
f_h = \phi_h(f_e)
\]

注意点：

- 采用 K-Planes 的原因：
  - 相比 4D dense grid 大幅节省内存。
  - 训练速度快，足够表达复杂的呼吸运动。

### 4.3 解码器 \(F\)：从特征到形变参数

使用三个解码头（都是轻量 MLP），分别预测不同形变分量：

\[
\Delta\mu = F_\mu(f_h), \quad
\Delta R = F_R(f_h), \quad
\Delta S = F_S(f_h)
\]

然后按前述公式构造时间 \(t\) 下的高斯：

\[
G_i'(t) = (\mu_i + \Delta\mu_i(t),\; R_i + \Delta R_i(t),\; S_i + \Delta S_i(t),\; \rho_i)
\]

> 实现建议：\(\Delta R\) 实际上可以用轴角 / 四元数等更稳定的旋转参数化，论文形式上写成 \(R + \Delta R\)，你在工程实现里可以选择更适合的方式。

### 4.4 渲染过程（结合几何）

对某个投影 \(I_j\)：

1. 已知时间 \(t_j\) 和投影几何 \(M_j\)。
2. 对每个高斯，计算时间 \(t_j\) 下的参数 \(G_i'(t_j)\)。
3. 用 Radiative GS 的 rasterizer：
   - 把高斯投影到 2D 平面；
   - 在图像平面进行 splatting + 按 Beer–Lambert 模型累积。
4. 得到渲染图像 \(\hat I_j = I_{\text{render}}(t_j; M_j)\)。

这些渲染结果将用于：

- 与真实投影 \(I_j\) 做**渲染损失**；
- 与 \(t_j \pm \hat T\) 时刻的渲染一起，做**周期一致性损失**。

---

## 5. Self-Supervised Respiratory Motion Learning（SSRML）

SSRML 的目标：**不依赖外部呼吸带**，在训练过程中从投影数据中自动学出病人的呼吸周期 \(T\)。

### 5.1 生理先验：周期一致性

呼吸是**近似周期性的**，同一解剖结构在间隔一个呼吸周期后会回到差不多的位置。  

在图像级别表现为：  
对于任意时间 \(t\)，在理想情况下有：
\[
I(t) \approx I(t + nT),\quad n \in \mathbb{Z}
\]

论文中只使用相邻周期：
- \(n \in \{-1, 1\}\)

以避免收敛到错误的“倍数 / 分数周期”。

于是定义**周期一致性损失**：
\[
\mathcal{L}_{pc}
=
\| I(t) - I(t + nT) \|_1
+ \lambda_1\ \text{D-SSIM}(I(t), I(t + nT))
\]

这里 D-SSIM 是基于结构相似度（SSIM）的度量，常用于图像质量评估。

在实际实现中：

- \(I(t)\) 和 \(I(t+nT)\) 都是**模型渲染出的图像**（即 synthetic views），不需要真实投影在 \(t+nT\) 时刻的数据。
- 对每个 mini-batch 中的若干时间点 \(t\)，随机取 \(n\in\{-1,1\}\)，计算相应损失。

### 5.2 可学习的周期 \(\hat T\) 与 log-space 参数化

真实周期 \(T\) 在临床中未知，因此作为**可学习参数**：
\[
\hat T = \exp(\hat\tau),\quad \hat\tau \in \mathbb{R}
\]

优点：

- 保证 \(\hat T > 0\)，避免出现负周期。
- log-space 有更好的数值稳定性，梯度更新更平滑。

训练中实际使用的周期一致性损失：
\[
\mathcal{L}_{pc}
=
\| I(t) - I(t + n\exp(\hat\tau)) \|_1
+ \lambda_1\ \text{D-SSIM}\big(I(t), I(t + n\exp(\hat\tau))\big)
,\quad n\in\{-1,1\}
\]

优化目标：
\[
\hat\tau^* = \arg\min_{\hat\tau} \mathcal{L}_{pc},\quad
T^* = \exp(\hat\tau^*)
\]

训练时通过对 \(\hat\tau\) 反向传播梯度即可。

### 5.3 Bounded Cycle Shifts：为什么只用相邻周期

如果允许 \(n\) 很大，例如 \(n=6\)：

- 真实周期 \(T=3\) s；
- 模型错误收敛到 \(\hat T = 4\) s；
- \(n\hat T = 24\) s，刚好是真实周期的整数倍 \(8T\)，  
  周期一致性约束仍然能被“错误周期”满足。

这会导致网络收敛到“子谐波 / 倍频周期”，而不是正确的基本周期。  

因此论文采取：

- **只用 \(n = \pm 1\)**（相邻周期）
- 配合 log-space 参数化，使得 \(\hat T\) 在训练中更平稳、准确地收敛。

---

## 6. 损失函数与正则化

### 6.1 渲染损失 \(\mathcal{L}_{render}\)

对每个真实投影 \(I_j\) 和渲染结果 \(\hat I_j\)：

\[
\mathcal{L}_{render}
=
\| \hat I_j - I_j \|_1
+ \lambda_2\ \text{D-SSIM}(\hat I_j, I_j)
\]

- L1：保证像素级一致性；
- D-SSIM：强调结构信息（边缘、纹理），能更好对齐 CT 结构。

### 6.2 TV 正则

1. **3D TV 正则 \(\mathcal{L}^{3D}_{TV}\)**（作用于 CT 体）：
   - 对重建的 3D 密度体 \(\sigma(x)\) 做各向 TV；
   - 用来降低噪声、抑制不必要的高频伪影。

2. **4D TV 正则 \(\mathcal{L}^{4D}_{TV}\)**（作用于 K-Planes 网格）：
   - 对 4D K-Planes 的多分辨率网格进行 TV；
   - 平滑时空特征，有利于形成连贯的运动场。

### 6.3 总损失

\[
\mathcal{L}_{total}
=
\mathcal{L}_{render}
+ \alpha\ \mathcal{L}_{pc}
+ \beta\ \mathcal{L}^{3D}_{TV}
+ \gamma\ \mathcal{L}^{4D}_{TV}
\]

论文中使用的典型权重（可作为实现参考）：

- \(\lambda_1 = \lambda_2 = 0.25\)
- \(\alpha = 1.0\)（周期一致性与渲染损失同量级）
- \(\beta = 0.05\)
- \(\gamma = 0.001\)

---

## 7. 训练流程与实现细节

### 7.1 总体训练策略：静态预热 + 4D 联合训练

1. **Phase 1：静态 3D Radiative GS 预训练**
   - 忽略时间，假设所有投影都来自同一个静态体。
   - 使用 R²-GS 风格的 radiative GS 训练约 5000 iter：
     - 优化高斯位置、尺度、密度、旋转；
     - 得到一个高质量的静态 CT 表示。
   - 这个阶段的目的：
     - 先学到“解剖结构 + 投影几何”的稳定对应；
     - 给后续 4D 动态模型一个好的初始化。

2. **Phase 2：开启 4D 动态建模 + 周期学习**
   - 固定初始高斯作为参考状态，添加：
     - K-Planes 作为时空编码器；
     - 解码器 \(F_\mu, F_R, F_S\)；
     - 呼吸周期参数 \(\hat\tau\)。
   - 总训练迭代：约 30k iter（论文实现），包括预热阶段。

**关键点：**

- 周期一致性只依赖模型渲染出的两个时间点，无需真实投影。
- 周期参数 \(\hat{\tau}\) 通过 K-Planes 插值中的 \(t_j + n\hat{T}\) 影响渲染，从而能被梯度更新。
- TV 正则可按一定频率（如每几步）计算，以节约算力。

---

## 8. 推理 / 应用方式

训练完毕后，我们获得：

- 参考状态高斯集合 \(\{ G_i \}\)
- 时空变形网络（K-Planes + 解码器）
- 学到的呼吸周期 \(T^*\)

### 8.1 重建任意时间点的 3D CT

给定任意时间 \(t^*\)：

1. 对每个高斯 \(G_i\)：
   - 用 \(v = (\mu_i, t^*)\) 通过 K-Planes + Decoder 得到  
     \(\Delta\mu_i(t^*), \Delta R_i(t^*), \Delta S_i(t^*)\)；
   - 得到变形后高斯  
     \(G_i'(t^*)\)。

2. 将所有高斯栅格化成 3D 密度体，输出为该时间点的 CT 体。

按多个时间点采样 \(\{ t_k^* \}\)，即可得到时间连续的 4D CT 序列。

### 8.2 提取呼吸相关的临床参数（示例）

论文示例：在 9 s 内密集采样 180 帧 3D CT（0.05 s 间隔），然后对每帧做肺分割并统计体积。得到体积–时间曲线后，可以计算：

- Tidal Volume（潮气量）
- Minute Ventilation（分钟通气量）
- 吸呼比 I:E ratio 等

这些应用都建立在“连续时间 4D 重建 + 正确的呼吸周期估计”之上。

---

## 9. 与传统/现有方法的关键差异

1. **不再做 phase-binning**
   - 传统 4D CT：投影 → 分 10 个相位 → 分别做 10 次 3D 重建。
   - X²-Gaussian：直接学习 \(\sigma(x,t)\) 的连续表示，可在任意时间采样。

2. **不依赖外部 gating 设备**
   - 周期由 SSRML 自动从投影数据中学习，减少硬件依赖和病人不适。

3. **基于 Radiative Gaussian Splatting 的动态图像重建**
   - 相比 NeRF / TensoRF 等，GS 在渲染速度和样本效率方面更优秀。
   - 相比静态 R²-GS，X²-Gaussian 通过 DGMM + SSRML 支持真正的 4D 重建。
