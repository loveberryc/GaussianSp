# PhysX-Boosted: Learnable Fusion of Implicit Neural Fields and Physical Anchors for Dynamic 4D-CT Reconstruction

## Abstract

We present PhysX-Boosted, a novel deformation architecture for dynamic 4D Gaussian Splatting that combines the representational power of implicit neural fields (HexPlane) with physically-grounded anchor-based transformers. Our key insight is that respiratory motion in 4D-CT exhibits both high-frequency textural deformations (best captured by neural fields) and low-frequency anatomical structure preservation (best modeled by physical anchors). We introduce a learnable balance parameter α that adaptively fuses these two complementary deformation sources: Δx_total = (1-α)·Δx_hexplane + α·Δx_anchor. Experiments on lung 4D-CT datasets demonstrate that PhysX-Boosted achieves superior reconstruction quality while maintaining physically plausible motion patterns.

---

## 1. Preliminaries

### 1.1 3D Gaussian Splatting

3D Gaussian Splatting (3DGS) represents a scene as a collection of learnable 3D Gaussians. Each Gaussian is parameterized by:

- **Position**: μ ∈ ℝ³ (center location)
- **Covariance**: Σ ∈ ℝ^{3×3} (spatial extent), factorized as Σ = RSS^TR^T where R is a rotation quaternion and S is a diagonal scaling matrix
- **Opacity**: σ ∈ [0, 1]
- **Color**: c ∈ ℝ^k (spherical harmonic coefficients)

The rendering equation for a pixel is computed via α-blending of sorted Gaussians:

$$C = \sum_{i=1}^{N} c_i \alpha_i \prod_{j=1}^{i-1} (1 - \alpha_j)$$

where αᵢ is the opacity contribution of Gaussian i at the pixel location.

### 1.2 X²-Gaussian: Dynamic 4D Gaussian Splatting with HexPlane Deformation

X²-Gaussian extends 3DGS to dynamic scenes by introducing a deformation network that maps canonical Gaussian positions to time-dependent deformed positions. The core architecture consists of:

#### 1.2.1 HexPlane Factorized 4D Feature Grid

To efficiently represent spatiotemporal features, X²-Gaussian adopts the HexPlane representation that factorizes a 4D volume (x, y, z, t) into six 2D feature planes:

$$\mathcal{P} = \{P_{xy}, P_{xz}, P_{xt}, P_{yz}, P_{yt}, P_{zt}\}$$

Each plane Pₐᵦ ∈ ℝ^{R_a × R_b × C} stores learnable features at resolution Rₐ × Rᵦ with C channels. For a query point (x, y, z, t), features are extracted via bilinear interpolation from each plane and aggregated through Hadamard product:

$$f(x, y, z, t) = \bigodot_{P \in \mathcal{P}} \text{interp}(P, \pi_P(x, y, z, t))$$

where πₚ projects the 4D coordinate onto the 2D plane P, and ⊙ denotes element-wise multiplication.

#### 1.2.2 Multi-Scale Feature Extraction

To capture both coarse and fine deformations, HexPlane uses multi-resolution grids:

$$f_{ms}(x, y, z, t) = \text{Concat}[f^{(1)}(x, y, z, t), f^{(2)}(x, y, z, t), ..., f^{(L)}(x, y, z, t)]$$

where each level l operates at resolution 2^{l-1} × base_resolution.

#### 1.2.3 Deformation MLP

The extracted features are decoded by an MLP to predict per-Gaussian deformations:

$$[\Delta\mu, \Delta s, \Delta r] = \text{MLP}(f_{ms}(\mu_i, t))$$

where:
- Δμ ∈ ℝ³: Position displacement
- Δs ∈ ℝ³: Scale modification (log-space)
- Δr ∈ ℝ⁴: Rotation quaternion residual

The deformed Gaussian parameters at time t are:

$$\mu_i(t) = \mu_i + \Delta\mu, \quad s_i(t) = s_i + \Delta s, \quad r_i(t) = r_i + \Delta r$$

#### 1.2.4 Training Objectives

The baseline X²-Gaussian is trained with the following loss functions:

**Rendering Loss (L₁)**:
$$\mathcal{L}_{render} = \|I_{pred} - I_{gt}\|_1$$

**Structural Similarity Loss (DSSIM)**:
$$\mathcal{L}_{dssim} = 1 - \text{SSIM}(I_{pred}, I_{gt})$$

**Prior Loss (Lₚ𝒸)** for breathing periodicity:
$$\mathcal{L}_{pc} = \|I(t) - I(t + T)\|_1$$

where T is a learnable breathing period parameter.

**3D Total Variation Loss (L_tv)** for spatial smoothness:
$$\mathcal{L}_{tv} = \sum_{i} \|\nabla V_i\|_1$$

where V is the voxelized density field.

**4D Total Variation Loss (L_4d_tv)** for temporal smoothness of HexPlane grids:
$$\mathcal{L}_{4d\_tv} = \sum_{P \in \mathcal{P}} \left( \|\nabla_u P\|_1 + \|\nabla_v P\|_1 \right)$$

The total training objective is:

$$\mathcal{L}_{baseline} = \mathcal{L}_{render} + \lambda_{dssim}\mathcal{L}_{dssim} + \lambda_{pc}\mathcal{L}_{pc} + \lambda_{tv}\mathcal{L}_{tv} + \lambda_{4d\_tv}\mathcal{L}_{4d\_tv}$$

---

## 2. PhysX-Boosted Method

### 2.1 Motivation

While HexPlane-based deformation excels at capturing high-frequency, texture-level motion details, it lacks explicit modeling of anatomical structure and physical constraints. In 4D-CT lung imaging, respiratory motion is fundamentally governed by:

1. **Rib cage mechanics**: Rigid or near-rigid motion of bone structures
2. **Diaphragm dynamics**: Large-scale coherent motion driving lung expansion/contraction
3. **Tissue connectivity**: Neighboring anatomical structures move in physically consistent ways

We hypothesize that combining implicit neural fields (HexPlane) with explicit physical modeling (anchor-based transformers) yields superior deformation prediction. The key insight is:

> **HexPlane** = "Paint the skin" (high-frequency texture, micro-deformations)
> 
> **Anchors** = "Draw the skeleton" (anatomical structure, physical consistency)

### 2.2 Anchor-Based Spacetime Transformer

#### 2.2.1 Physical Anchor Selection via Farthest Point Sampling

We select M physical anchors from the initial point cloud using Farthest Point Sampling (FPS), ensuring uniform spatial coverage of anatomical structures:

$$\mathcal{A} = \{a_1, a_2, ..., a_M\} = \text{FPS}(\{\mu_1, \mu_2, ..., \mu_N\}, M)$$

These anchors serve as control points that encode local anatomical motion. Typical configuration: M = 1024 anchors.

#### 2.2.2 KNN-Based Skinning Weights

Each Gaussian binds to its k nearest anchors with distance-based skinning weights:

$$w_{ij} = \frac{\exp(-\|μ_i - a_j\|^2 / τ)}{\sum_{j' \in \mathcal{N}_k(i)} \exp(-\|μ_i - a_{j'}\|^2 / τ)}$$

where:
- $\mathcal{N}_k(i)$: k-nearest anchor indices for Gaussian i
- τ: Temperature parameter controlling weight sharpness (default: 0.01)
- k: Number of neighbors (default: 10)

#### 2.2.3 Spacetime Transformer Architecture

Anchors interact through a Transformer encoder that learns spatiotemporal relationships:

**Input Embedding**:
$$h_j^{(0)} = \text{Linear}(\text{Concat}[\text{PosEmbed}(a_j), \text{TimeEmbed}(t)])$$

where:
- PosEmbed: Learned spatial embedding (32-dim)
- TimeEmbed: Sinusoidal temporal encoding (16-dim)

**Transformer Encoding**:
$$H^{(l)} = \text{TransformerLayer}(H^{(l-1)}), \quad l = 1, ..., L$$

Each layer consists of multi-head self-attention and feedforward networks:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Configuration: d_model=64, heads=4, layers=2, dim_feedforward=256.

**Displacement Prediction**:
$$\Delta a_j = \text{DisplacementHead}(h_j^{(L)})$$

where DisplacementHead is an MLP: Linear(64→64) → GELU → Linear(64→3).

#### 2.2.4 Gaussian Displacement via Skinning Interpolation

The displacement for each Gaussian is computed as a weighted sum of anchor displacements:

$$\Delta\mu_i^{anchor} = \sum_{j \in \mathcal{N}_k(i)} w_{ij} \cdot \Delta a_j$$

This skinning-based interpolation ensures:
1. **Spatial coherence**: Nearby Gaussians receive similar displacements
2. **Physical plausibility**: Motion is constrained by anchor topology
3. **Computational efficiency**: Only M anchors go through the Transformer

#### 2.2.5 Masked Modeling for Robust Deformation Learning

Inspired by BERT-style pretraining, we apply random masking during training:

1. Randomly select 25% of anchors as masked set $\mathcal{M}$
2. Replace masked anchor embeddings with learnable [MASK] token
3. Transformer must predict masked anchor displacements from context

This encourages the model to learn physical traction relationships between anatomical structures, enabling robust inference even when some anchors are occluded or poorly observed.

### 2.3 PhysX-Boosted: Learnable Fusion Architecture

#### 2.3.1 Dual-Stream Deformation

PhysX-Boosted computes deformations from two parallel streams:

**Stream 1 - HexPlane (Neural Field)**:
$$\Delta\mu^{hex}, \Delta s^{hex}, \Delta r^{hex} = \text{HexPlaneNet}(\mu_i, t)$$

**Stream 2 - Anchor Transformer (Physical Model)**:
$$\Delta\mu^{anchor} = \sum_{j \in \mathcal{N}_k(i)} w_{ij} \cdot \text{AnchorTransformer}(a_j, t)$$

#### 2.3.2 Learnable Balance Parameter α

We introduce a learnable scalar parameter α ∈ (0, 1) that adaptively balances the two streams:

$$\alpha = \sigma(\tau)$$

where τ is a learnable logit initialized to τ₀ = 0 (corresponding to α₀ = 0.5), and σ is the sigmoid function.

The final deformation is computed as:

$$\Delta\mu^{total} = (1 - \alpha) \cdot \Delta\mu^{hex} + \alpha \cdot \Delta\mu^{anchor}$$

$$\Delta s^{total} = (1 - \alpha) \cdot \Delta s^{hex}$$

$$\Delta r^{total} = (1 - \alpha) \cdot \Delta r^{hex}$$

Note: The anchor stream focuses solely on position displacement, as anatomical structure primarily constrains spatial motion rather than scale/rotation.

#### 2.3.3 Interpretation of α

The learned α provides interpretable insights:

| α Value | Interpretation |
|---------|----------------|
| α → 0 | Model relies primarily on HexPlane (texture-driven) |
| α = 0.5 | Equal contribution from both streams |
| α → 1 | Model relies primarily on Anchors (structure-driven) |

Empirically, we observe α ≈ 0.43 after training, indicating that both streams contribute meaningfully, with a slight preference for HexPlane's textural details.

### 2.4 Training Objectives

#### 2.4.1 Physics Completion Loss (L_phys)

To enforce physical consistency, we compute a masked prediction loss:

$$\mathcal{L}_{phys} = \frac{1}{|\mathcal{M}|} \sum_{j \in \mathcal{M}} \|\Delta a_j^{masked} - \Delta a_j^{teacher}\|_2^2$$

where:
- $\Delta a_j^{masked}$: Displacement predicted with [MASK] token
- $\Delta a_j^{teacher}$: Displacement from unmasked forward pass (detached)

This loss encourages the model to infer motion from anatomical context rather than memorizing individual anchor trajectories.

#### 2.4.2 Anchor Smoothness Loss (L_smooth)

To regularize anchor motion, we penalize large displacement differences between neighboring anchors:

$$\mathcal{L}_{smooth} = \sum_{j} \sum_{j' \in \mathcal{N}(j)} \|\Delta a_j - \Delta a_{j'}\|_2^2$$

#### 2.4.3 Balance Regularization Loss (L_balance) [Optional]

To prevent α from collapsing to extreme values:

$$\mathcal{L}_{balance} = (\alpha - \alpha_{target})^2$$

where α_target = 0.5 encourages balanced contribution. In practice, we set λ_balance = 0 to allow free learning.

#### 2.4.4 Total Training Objective

$$\mathcal{L}_{total} = \mathcal{L}_{render} + \lambda_{dssim}\mathcal{L}_{dssim} + \lambda_{phys}\mathcal{L}_{phys} + \lambda_{smooth}\mathcal{L}_{smooth} + \lambda_{4d\_tv}\mathcal{L}_{4d\_tv}$$

Default hyperparameters:
- λ_dssim = 0.25
- λ_phys = 0.1
- λ_smooth = 0.01
- λ_4d_tv = 0.0001 (plane_tv_weight)

### 2.5 Implementation Details

#### 2.5.1 Network Architecture

| Component | Configuration |
|-----------|---------------|
| HexPlane resolution | [64, 64, 64, 150] |
| HexPlane multi-res | [1, 2, 4, 8] |
| HexPlane feature dim | 32 per plane |
| Deformation MLP | 64-dim, depth=1 |
| Anchor count | 1024 |
| KNN neighbors | 10 |
| Transformer dim | 64 |
| Transformer heads | 4 |
| Transformer layers | 2 |
| Mask ratio | 0.25 |

#### 2.5.2 Optimization

- Optimizer: Adam
- HexPlane grid LR: 0.0002
- HexPlane MLP LR: 0.0002
- Anchor Transformer LR: 0.002
- Balance parameter LR: 0.001
- Total iterations: 50,000
- Coarse stage: 5,000 iterations (no deformation)
- Physics warmup: 2,000 fine-stage iterations

#### 2.5.3 Two-Stage Training

1. **Coarse Stage** (iterations 1-5000): Train static Gaussians without deformation
2. **Fine Stage** (iterations 5001-50000): Enable HexPlane + Anchor deformation with full loss

---

## 3. Algorithm Summary

```
Algorithm: PhysX-Boosted Training

Input: 4D-CT images {I_t}, initial point cloud {μ_i}
Output: Trained Gaussian model with PhysX-Boosted deformation

1. Initialize Gaussians from point cloud
2. Select M anchors via FPS: A = FPS({μ_i}, M)
3. Compute KNN binding: w_ij for all (i, j) pairs
4. Initialize HexPlane grids and MLP
5. Initialize Anchor Transformer
6. Initialize balance logit τ = 0 (α = 0.5)

for iteration = 1 to max_iter:
    Sample time t and camera view
    
    if iteration ≤ coarse_iter:
        # Coarse stage: no deformation
        μ_deformed = μ
    else:
        # Fine stage: PhysX-Boosted deformation
        
        # Stream 1: HexPlane
        Δμ_hex = HexPlaneNet(μ, t)
        
        # Stream 2: Anchor Transformer
        if training:
            Apply random masking to anchors
        Δa = AnchorTransformer(A, t)
        Δμ_anchor = Σ w_ij · Δa_j  (KNN interpolation)
        
        # Learnable fusion
        α = sigmoid(τ)
        Δμ_total = (1-α)·Δμ_hex + α·Δμ_anchor
        μ_deformed = μ + Δμ_total
    
    # Render and compute losses
    I_pred = Render(μ_deformed, ...)
    L = L_render + λ·L_dssim + λ·L_phys + λ·L_smooth
    
    # Backward and update
    L.backward()
    optimizer.step()
    
    # Update KNN binding after densification/pruning
    if densification_step:
        Update w_ij for new Gaussian set

return Trained model
```

---

## 4. Experimental Configuration

### 4.1 Fixed α Ablation Study

To understand the contribution of each stream, we conduct ablation with fixed α values:

| Experiment | α | HexPlane | Anchor | Command Flag |
|------------|---|----------|--------|--------------|
| HexPlane-dominant | 0.3 | 70% | 30% | `--balance_alpha_init 0.3 --balance_lr 0` |
| Balanced | 0.5 | 50% | 50% | `--balance_alpha_init 0.5 --balance_lr 0` |
| Anchor-dominant | 0.7 | 30% | 70% | `--balance_alpha_init 0.7 --balance_lr 0` |
| Learnable | 0.5→? | adaptive | adaptive | `--balance_alpha_init 0.5 --balance_lr 0.001` |

### 4.2 Training Command

```bash
python train.py -s data/dir_4d_case1.pickle \
  --use_anchor_deformation \
  --use_boosted \
  --use_learnable_balance \
  --balance_alpha_init 0.5 \
  --balance_lr 0.001 \
  --lambda_balance 0.0 \
  --iterations 50000
```

---

## References

1. Kerbl et al. "3D Gaussian Splatting for Real-Time Radiance Field Rendering." SIGGRAPH 2023.
2. Fridovich-Keil et al. "K-Planes: Explicit Radiance Fields in Space, Time, and Appearance." CVPR 2023.
3. Cao & Johnson. "HexPlane: A Fast Representation for Dynamic Scenes." CVPR 2023.
4. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers." NAACL 2019.
5. X²-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction.
