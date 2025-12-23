# M1.2 (g005) Small-Perturbation Fusion (ICML-Style Method Note)

<!-- markdownlint-disable MD049 -->

## Problem Setting

We consider *PhysX-Boosted* deformation modeling, which combines:

- A **Lagrangian backbone** \(\Phi_L\) implemented by an anchor-based spacetime transformer (captures globally coherent anatomical motion).
- An **Eulerian corrector** \(\Phi_E\) implemented by a full-capacity HexPlane branch (captures residual, high-frequency details).

The goal is to exploit the complementary strengths of both streams while preventing the Eulerian branch from dominating optimization.

## V5 NoMask Baseline (Learnable Balance)

The V5 baseline blends Lagrangian and Eulerian predictions via a balance coefficient \(\alpha\) (initialized to 0.99):

\[
\Delta \mathbf{x} \;=\; (1-\alpha)\,\Delta \mathbf{x}_{\text{hex}} \; + \; \alpha\,\Delta \mathbf{x}_{\text{anchor}},
\]

where \(\alpha = \sigma(\tau)\) is parameterized by a scalar \(\tau\) (e.g., \(\tau_{\text{init}}\approx 4.595\Rightarrow \alpha_{\text{init}}\approx 0.99\)). In the **NoMask** setting, the anchor transformer is trained without random masking (mask ratio = 0), avoiding an implicit regularization that can harm accuracy.

This fixed ratio (approximately 99:1 in favor of the Lagrangian stream) was empirically found to be a strong operating point.

## M1.2: Uncertainty-Gated *Small Perturbation* Around the V5 Optimum

M1.2 keeps the V5 training dynamics *intact* by preserving the 99:1 ratio as a **base point**, but introduces a *bounded* and *data-dependent* perturbation \(\gamma\) that slightly shifts the Eulerian weight when the Eulerian branch is confident.

### Fusion Rule

Let \(\Phi_L\) denote the Lagrangian field and \(\Phi_E\) the Eulerian field. M1.2 defines:

\[
\Phi \;=\; (0.99-\gamma)\,\Phi_L \; + \; (0.01+\gamma)\,\Phi_E.
\]

Equivalently, the effective Eulerian weight is:

\[
\beta_{\text{eff}} \;=\; 0.01 + \gamma, \qquad \beta_{\text{eff}} \in [0.01-\gamma_{\max},\; 0.01+\gamma_{\max}].
\]

With \(\gamma_{\max}=0.005\) (the **g005** setting), the HexPlane weight is constrained to:

\[
\beta_{\text{eff}} \in [0.005,\; 0.015],
\]

i.e., the Eulerian contribution can only move within a narrow band of \([0.5\%, 1.5\%]\). This is the key design choice: **the Eulerian branch never receives the full-gradient privilege of larger mixing coefficients**, avoiding the failure mode observed in more aggressive gating variants.

### Uncertainty-to-Perturbation Mapping

M1.2 converts an Eulerian uncertainty (or confidence) scalar \(s_E\) into \(\gamma\) using a smooth saturating gate:

\[
\gamma \;=\; \gamma_{\max} \cdot \tanh\Big(\frac{\tau - s_E}{\lambda}\Big),
\]

where \(\tau\) is a threshold and \(\lambda\) is a temperature. In the case1 run:

- \(\gamma_{\max}=0.005\)
- \(\tau=0.0\)
- \(\lambda=1.0\)

Intuitively:

- When the Eulerian stream is **confident** (lower uncertainty / lower \(s_E\)), \(\tanh((\tau-s_E)/\lambda)\) becomes positive, increasing the Eulerian weight toward 1.5%.
- When the Eulerian stream is **uncertain**, \(\gamma\) decreases, reducing the Eulerian weight toward 0.5%.

### What This Experiment “Did” in Practice

The log of `train_physx_boosted_m1_2_g005_case1_20251212_165700.log` confirms that this run:

- Enabled **PhysX-Boosted** (HexPlane baseline + anchor correction).
- Used **NoMask** for anchors (mask ratio = 0.0).
- Activated **M1.2 small-perturbation fusion** with \(\gamma_{\max}=0.005\), \(\tau=0.0\), \(\lambda=1.0\).
- Kept the underlying V5 balance initialization \(\alpha_{\text{init}}\approx 0.99\) while allowing only a tiny, uncertainty-driven deviation via \(\gamma\).

Empirically, the printed diagnostics show the effective Eulerian weight `hex_weight = 0.01 + γ` staying near the upper bound (e.g., 0.015 early on), indicating that the model often prefers a slightly stronger Eulerian contribution than the strict 1% weight, but still within a rigorously bounded range.

## Implementation Notes (as reflected in the run log)

- **Anchor transformer configuration**: 1024 anchors, KNN binding \(k=10\), transformer dim 64, 4 heads, 2 layers.
- **Auxiliary losses**: physics completion loss (\(\lambda_{\text{phys}}=0.1\)) and anchor smoothness (\(\lambda_{\text{anchor\_smooth}}=0.01\)), with a warmup of 2000 steps before applying the physics term.

## Summary

M1.2 (g005) is a conservative but effective refinement over V5 NoMask: it **does not replace** the V5 blend with a fully adaptive gate; instead, it **perturbs** the optimal 99:1 mixing ratio by a strictly bounded \(\gamma\) predicted from Eulerian uncertainty. This preserves the favorable optimization dynamics of V5 while enabling demand-driven corrections from the Eulerian branch when it is confident.
