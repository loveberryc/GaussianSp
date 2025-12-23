# PhysX-Boosted (NoMask-S4): Stable Anisotropic Fusion of Anchor Motion and Neural Residuals for Dynamic 4D-CT Gaussian Splatting

## Abstract

We describe a stabilized variant of PhysX-Boosted for dynamic 4D-CT reconstruction with 4D Gaussian Splatting. Unlike the earlier masked-anchor formulation, our best-performing regime on the V5-no-mask baseline uses an *anisotropic fusion schedule across deformation channels*: (i) **positions** are driven almost entirely by the anchor-based spacetime transformer (**wA = 1**), (ii) **scales** are updated with a **large HexPlane gain** (`ds_weight` ≳ 0.8–0.9), and (iii) **rotations** are updated with a **small HexPlane gain** (**k ≲ 0.1–0.2**). We argue that this asymmetric design matches the physical structure of respiratory motion: the anchor field provides a low-frequency, topology-preserving *skeleton motion* in position space, while scale absorbs volumetric expansion/compression (diaphragm-driven) and rotation is a numerically fragile degree of freedom that should be regularized via a small-step update. We provide a mechanistic analysis based on (1) invertibility/stability of local deformation maps, (2) quaternion normalization sensitivity, and (3) a coarse-to-fine separation of anatomical motion and appearance-driven residuals.

---

## 1. Preliminaries

### 1.1 Dynamic 4D Gaussian Splatting

A dynamic 4D Gaussian scene is represented by a set of canonical Gaussians \(\{\mathcal{G}^i\}_{i=1}^N\). Each Gaussian has canonical parameters:

- Mean (position) \(\mu_i \in \mathbb{R}^3\)
- Scale (log-scale or diagonal scale) \(s_i \in \mathbb{R}^3\)
- Rotation quaternion \(q_i \in \mathbb{H}\) (unit quaternion)
- Opacity \(\sigma_i\) and SH color \(c_i\)

At time \(t\), a deformation model predicts increments \(\Delta\mu_i(t),\ \Delta s_i(t),\ \Delta q_i(t)\), and the deformed parameters are applied prior to rendering.

### 1.2 HexPlane Deformation Field

HexPlane factorizes 4D spatiotemporal features into a set of 2D planes
\(\mathcal{P}=\{P_{xy},P_{xz},P_{xt},P_{yz},P_{yt},P_{zt}\}\), with multi-resolution concatenation. For a query \((\mu_i,t)\), HexPlane predicts residual updates:

\[
\Delta\mu_i^{hex},\ \Delta s_i^{hex},\ \Delta q_i^{hex} = f_{hex}(\mu_i,t).
\]

HexPlane excels at representing high-frequency motion residuals but may violate anatomical coherence without additional structure.

---

## 2. NoMask-S4 Method

### 2.1 Motivation: Why Anisotropic Fusion?

Respiratory motion is **not** isotropic across deformation channels:

- **Position** should follow a globally coherent, topology-preserving field (bones, organ boundaries, connectivity).
- **Scale** primarily encodes *local expansion/compression*, which is physically meaningful in lung CT (ventilation/volume change).
- **Rotation** is an auxiliary degree of freedom for Gaussian covariance orientation. It is powerful but also **numerically stiff**: aggressive rotation updates can cause unstable gradients, invalid covariance behavior, and downstream NaNs.

Therefore, we decouple the fusion strategy:

- Use **anchors for position** (structure).
- Use **HexPlane strongly for scale** (volume change).
- Use **HexPlane weakly for rotation** (stability).

This differs from earlier “single-α” fusion which implicitly ties the magnitude of \(\Delta\mu\), \(\Delta s\), and \(\Delta q\) together.

### 2.2 Anchor Spacetime Transformer (Skeleton Motion)

We select \(M\) anchors by FPS from the initial point set \(\{\mu_i\}\):

\[
\mathcal{A} = \{a_j\}_{j=1}^M = \mathrm{FPS}(\{\mu_i\},M).
\]

Each Gaussian binds to its \(K\) nearest anchors \(\mathcal{N}_K(i)\) with temperature-controlled weights:

\[
\omega^{ij}=\frac{\exp(-\|\mu_i-a^j\|^2/\tau)}{\sum_{j'\in\mathcal{N}_K(i)}\exp(-\|\mu_i-a^{j'}\|^2/\tau)}.
\]

A transformer encoder predicts anchor displacements \(\Delta a_j(t)\), and Gaussians receive skinned displacements:

\[
\Delta\mu_i^{anchor}(t)=\sum_{j\in\mathcal{N}_K(i)}\omega^{ij}\,\Delta a^j(t).
\]

### 2.3 S4: Explicit Per-Channel Fusion

Let HexPlane predict residual updates \((\Delta\mu_i^{hex},\Delta s_i^{hex},\Delta q_i^{hex})\), and anchors predict \(\Delta\mu_i^{anchor}\).

We define three scalars controlling the effective step size per channel:

- \(w_A\in[0,1]\): **position** anchor weight
- \(d_s\in[0,1]\): **scale** HexPlane gain (`ds_weight`)
- \(k\ge 0\): **rotation** HexPlane gain

**Position (wA = 1 in the target regime).**

\[
\Delta\mu\_i = (1-w_A)\,\Delta\mu\_i^{hex} + w_A\,\Delta\mu\_i^{anchor}.
\]

**Scale (large gain).**

In the V5-no-mask implementation used here, when `--s3_release_scale` is disabled, scale is updated as:

\[
\Delta s\_i = d_s\,\Delta s\_i^{hex},\qquad d_s = 1-\alpha.
\]

We typically set \(d_s\approx 0.85\)–\(0.95\) (equivalently \(\alpha\approx 0.15\)–\(0.05\)).

**Rotation (small gain).**

\[
\Delta q\_i = k\,\Delta q\_i^{hex}.
\]

We typically set \(k\lesssim 0.1\) (sometimes up to \(0.2\) depending on stability).

Finally, the deformed parameters are applied:

\[
\mu\_i(t)=\mu\_i+\Delta\mu\_i,\qquad s\_i(t)=s\_i+\Delta s\_i,
\]

and rotations are updated with normalization:

\[
q\_i(t)=\mathrm{normalize}(q\_i + \Delta q\_i).
\]

---

## 3. Why Small-k / Large-ds Works: Mechanistic Analysis

### 3.1 Stability of Local Deformation Maps

A deformation field defines a local mapping \(x' = x + u(x)\) with Jacobian \(J = I + \nabla u\). In tomographic reconstruction, large non-invertible folds (det\(J\le 0\)) correspond to physically implausible motion and lead to inconsistent projections.

- The anchor field \(u_A\) is **smooth by construction** (KNN skinning + transformer with limited capacity), making \(\nabla u_A\) bounded.
- The HexPlane field can encode high-frequency residuals, which is valuable but can produce large gradients.

By enforcing **wA=1**, we constrain the primary mapping to a topology-preserving skeleton field:

\[
J \approx I + \nabla u_A,
\]

and allow HexPlane to influence scale (volume change) without injecting high-frequency positional folding.

### 3.2 Scale as a Physically Compatible Degree of Freedom

Lung motion contains strong local expansion/compression. In Gaussian splatting, scale directly modifies covariance extents and can represent ventilation-induced density changes more naturally than forcing positional displacements to explain all intensity variation.

Large \(d_s\) enables HexPlane to model these volumetric effects while the anchor position field maintains structure. Empirically, ds_weight around \(0.9\) is a consistent sweet spot.

### 3.3 Rotation as a Numerically Stiff Channel

Rotation updates are applied through quaternions; the normalization operator is non-linear:

\[
\mathrm{normalize}(q+\Delta q)=\frac{q+\Delta q}{\|q+\Delta q\|}.
\]

When \(\|q+\Delta q\|\) becomes very small or gradients amplify through normalization, training can become unstable. Large rotation residuals can also interact adversely with covariance rendering.

Using **small k** is therefore a *trust-region style* constraint:

\[
\|\Delta q\| \le k\,\|\Delta q^{hex}\|,\quad k\ll 1,
\]

which reduces the chance of NaNs and keeps covariance orientation changes smooth.

### 3.4 Why No Masking?

In earlier variants, masking anchors acted as an auxiliary regularizer. In the V5-no-mask baseline with S4 anisotropic fusion, the anchor branch already behaves as a low-frequency backbone due to:

- FPS anchor sparsity,
- KNN skinning interpolation,
- limited transformer depth,
- and wA=1 eliminating HexPlane position shortcuts.

Thus, removing masking avoids unnecessary stochasticity and keeps optimization simpler while preserving physical coherence.

---

## 4. Algorithm Summary

```text
Algorithm: PhysX-Boosted (NoMask-S4) Training

Input: time-indexed projections {I_t}, initial Gaussians {G_i}
Output: trained Gaussians with stable anisotropic deformation

1. Initialize Gaussians from point cloud
2. Sample M anchors via FPS
3. Precompute KNN weights ω_ij
4. Initialize HexPlane (grids + MLP)
5. Initialize anchor transformer

for iter = 1..T:
  if iter <= coarse_iter:
    use identity deformation
  else:
    # Anchor position field
    Δa = AnchorTransformer(A, t)
    Δμ_anchor = Σ_j ω_ij Δa_j

    # HexPlane residual field
    (Δμ_hex, Δs_hex, Δq_hex) = HexPlane(μ, t)

    # S4 fusion
    Δμ = (1-wA)Δμ_hex + wAΔμ_anchor
    Δs = ds_weight * Δs_hex
    Δq = k * Δq_hex

    apply deformation and render

  optimize with rendering + regularizers

return model
```

---

## 5. Practical Configuration (Recommended Regime)

We recommend the following high-performing and stable range:

- **wA**: 1.0 (anchor-only position)

- **k**: 0.016–0.019 (best observed near 0.0175)
- **ds_weight**: 0.88–0.92 (best observed near 0.92)

In our internal ablations, too-large k (e.g., \(\approx 0.9\)) can trigger NaNs, and we also observed coupling-driven instabilities for specific (k, ds_weight) pairs (e.g., \(k=0.020,\ ds\_weight=0.90\) and \(k=0.0125,\ ds\_weight=0.85\) can collapse). Therefore, we recommend local refinement around the stable sweet spot above rather than coarse sweeps.

---

## References

1. Kerbl et al. “3D Gaussian Splatting for Real-Time Radiance Field Rendering.” SIGGRAPH 2023.
2. Fridovich-Keil et al. “K-Planes: Explicit Radiance Fields in Space, Time, and Appearance.” CVPR 2023.
3. Cao & Johnson. “HexPlane: A Fast Representation for Dynamic Scenes.” CVPR 2023.
4. X²-Gaussian: 4D Radiative Gaussian Splatting for Continuous-time Tomographic Reconstruction.
