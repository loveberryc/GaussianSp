# X2-Gaussian s4: Temporal-Ensemble Guided Static Warm-Up

本文件定义 s4 版本，仅作用于 X2-Gaussian 的 **静态 3D radiative Gaussians 粗训练阶段**（canonical warm-up，例如前 5000 步），动态阶段（v7/v8/v9/v10 等）完全不改。s4 通过一个训练开关独立启用，可与任意动态版本组合。

核心想法：

> 在静态阶段，不仅用原始动态投影训练 canonical Gaussians，
> 还引入一个“时间集成的 pseudo 3D/2D 老师”（平均 CT / 平均投影）作为额外监督，
> 让 Gaussians 收敛到更鲁棒、更一致的 canonical 表征。

---

## 0. Baseline 静态 warm-up 回顾

当前静态 warm-up 的目标大致为：

[
\mathcal{L}*{\text{static}}^{\mathrm{G}} =
\frac{1}{N} \sum*{j=1}^N
\mathcal{L}_{\text{render}}\big(\hat I_j^{\mathrm{G}}, I_j\big),
]

其中：

* (\hat I_j^{\mathrm{G}})：canonical Gaussians 渲染的第 j 个投影；
* (\mathcal{L}_{\text{render}})：L1 + D-SSIM 等组合。

静态模型用全部动态投影监督，会在强呼吸运动区域形成时间平均的 canonical blur。

---

## 1. s4 的伪监督来源：平均 CT / 平均投影

### 1.1 平均 3D 体（Temporal-Averaged CT Volume）

构造一个离线的 3D 平均体：

* 记作 (V_{\mathrm{avg}}(x))，定义在固定 3D 规则网格上（如 128³）；
* 可通过两种方式生成：

1. 传统静态重建（FDK/FBP）
   将全部投影视为静态输入 → 得到一个 time-averaged volume（模糊但结构稳定）。

2. baseline 4D 重建的时间平均
   若已有 baseline (X2-Gaussian / v7 / v9) 的 4D 重建 (\sigma(x,t))，则做：

[
V_{\mathrm{avg}}(x) =
\frac{1}{T} \int \sigma(x,t),dt
\approx
\frac{1}{N_t}\sum_k \sigma(x,t_k).
]

训练时 (V_{\mathrm{avg}}) 作为 fixed pseudo-GT，仅提供数值监督，不回传梯度。

---

### 1.2 平均 2D 投影（Temporal-Averaged Projections，可选）

如果数据包含同一视角在多个时间相位的投影（如多圈采集），可以对同一视角做平均：

* 记作 ({ I_j^{\mathrm{avg}} })；
* 或由 (V_{\mathrm{avg}}) forward project 得到。

训练时同样作为 fixed pseudo 监督。

---

## 2. s4 静态阶段的多目标训练

静态 warm-up 阶段，同时优化 canonical Gaussians 参数 (\theta_{\mathrm{can}})，目标包含：

### 2.1 原始投影重建损失（保持 baseline）

[
\mathcal{L}*{\mathrm{G}} =
\frac{1}{N} \sum_j
\mathcal{L}*{\text{render}}\big(\hat I_j^{\mathrm{G}}, I_j\big).
]

### 2.2 平均 CT 的 3D distillation（核心新增）

在 3D 空间采样点 ({x_m})，令 Gaussians 静态体值贴近平均体：

[
\mathcal{L}*{\mathrm{vol}} =
\mathbb{E}*{x \sim \mathcal{X}}
\big|\sigma_{\mathrm{G}}(x) - V_{\mathrm{avg}}(x)\big|_1,
]

其中：

* (\sigma_{\mathrm{G}}(x))：radiative Gaussians 在点 x 的静态 attenuation（所有高斯叠加）；
* (V_{\mathrm{avg}}(x))：平均 CT 体插值采样值；
* (\mathcal{X})：在 canonical bbox 内均匀采样或偏向高梯度区域采样。

### 2.3 平均投影 distillation（可选）

若 ({I_j^{\mathrm{avg}}}) 可用，则：

[
\mathcal{L}*{\mathrm{proj_avg}} =
\frac{1}{N*{\mathrm{avg}}} \sum_j
\mathcal{L}_{\text{render}}\big(\hat I_j^{\mathrm{G,avg}}, I_j^{\mathrm{avg}}\big),
]

其中：

* (\hat I_j^{\mathrm{G,avg}})：canonical Gaussians 在对应几何下渲染的投影；
* (I_j^{\mathrm{avg}})：该几何下的平均投影。

---

## 3. s4 静态总损失

[
\begin{split}
\mathcal{L}*{\mathrm{static}}^{\mathrm{s4}} =
&;\lambda*{\mathrm{G}} , \mathcal{L}*{\mathrm{G}}
+ \lambda*{\mathrm{vol}} , \mathcal{L}*{\mathrm{vol}} \
&+ \lambda*{\mathrm{proj_avg}} , \mathcal{L}_{\mathrm{proj_avg}}
\quad (\text{如果提供平均投影}).
\end{split}
]

推荐初始设置：

* (\lambda_{\mathrm{G}} = 1)（主导监督）；
* (\lambda_{\mathrm{vol}} \in [0.1, 0.5])（中等偏小，提供结构牵引但不压死）；
* (\lambda_{\mathrm{proj_avg}} \in [0, 0.1])（可先关掉）。

---

## 4. 用平均 CT 改善 Gaussians 初始化（可选但潜力大）

除 distillation 外，可用 (V_{\mathrm{avg}}) 做初始化：

1. 从 (V_{\mathrm{avg}}) 中选取高密度区域体素（>阈值）；
2. 在这些点上做 FPS/K-Means 选 N 个中心 → 作为 initial μ_i；
3. 用局部邻域协方差/梯度估计初始 scale/orientation；
4. ρ_i 取 (V_{\mathrm{avg}}(\mu_i)) 或邻域均值。

初始化更贴近 anatomy，可显著增强静态收敛稳定性。

---

## 5. 与动态阶段的兼容性

* s4 仅作用于静态 warm-up；
* 静态阶段结束后：

  * 不再计算 (\mathcal{L}*{\mathrm{vol}})、(\mathcal{L}*{\mathrm{proj_avg}})；
  * 动态训练完全沿用现有逻辑（v7/v9/v10 等）；
* 因此 s4 与任意动态版本正交，可自由组合。

---

## 6. 配置项建议

* `use_s4_temporal_ensemble_static: bool`
* `s4_avg_ct_path: str`（平均 CT volume 路径）
* `s4_avg_proj_path: str`（平均投影路径，可选）
* `lambda_s4_G: float`（默认 1.0）
* `lambda_s4_vol: float`
* `lambda_s4_proj_avg: float`（默认 0.0）
* `s4_num_vol_samples: int`（每 step 3D distillation 采样点数量）
* 可选初始化相关：

  * `use_s4_avg_ct_init: bool`
  * `s4_avg_ct_init_thresh: float`
  * `s4_avg_ct_init_method: {"fps","kmeans"}`

---
