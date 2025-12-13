"""
PhysX-Gaussian / PhysX-Hybrid: Anchor-based Spacetime Transformer Deformation Module

This module replaces the HexPlane + MLP deformation field with an anchor-based
transformer architecture that learns physical traction relationships between
anatomical structures via masked modeling (BERT-style).

PhysX-Hybrid extends this with a lightweight HexPlane residual network:
  Δx_total = Δx_anchor (skeleton) + Δx_residual (skin)
  - Anchor: 95% macro motion via Transformer + KNN (topology-preserving)
  - Residual: 5% micro details via lightweight HexPlane (high-frequency)

Key Components:
1. FPS Sampling: Select num_anchors points as physical anchors
2. KNN Binding: Each Gaussian binds to k nearest anchors (skinning weights)
3. Spacetime Transformer: Anchors attend to each other with time encoding
4. Masked Modeling: Randomly mask anchor features during training
5. Interpolation: Gaussian displacement = weighted sum of anchor displacements
6. (Hybrid) HexPlane Residual: Lightweight network for micro-corrections
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .hexplane import HexPlaneField
from .deformation import deform_network  # For PhysX-Boosted (full HexPlane baseline)


def farthest_point_sampling(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """
    Farthest Point Sampling (FPS) to select representative anchor points.
    
    Args:
        points: Point cloud [N, 3]
        num_samples: Number of points to sample (num_anchors)
    
    Returns:
        indices: Indices of sampled points [num_samples]
    """
    device = points.device
    N = points.shape[0]
    
    if num_samples >= N:
        # Return all indices if we want more samples than available
        return torch.arange(N, device=device)
    
    # Initialize with a random point
    indices = torch.zeros(num_samples, dtype=torch.long, device=device)
    distances = torch.full((N,), float('inf'), device=device)
    
    # Start from a random point
    farthest = torch.randint(0, N, (1,), device=device).item()
    
    for i in range(num_samples):
        indices[i] = farthest
        centroid = points[farthest].unsqueeze(0)  # [1, 3]
        
        # Update distances
        dist = torch.sum((points - centroid) ** 2, dim=-1)  # [N]
        distances = torch.min(distances, dist)
        
        # Select the farthest point from the current set
        farthest = torch.argmax(distances).item()
    
    return indices


def compute_knn_weights(
    query_points: torch.Tensor,
    anchor_points: torch.Tensor,
    k: int,
    temperature: float = 1.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute KNN indices and distance-based skinning weights.
    
    Args:
        query_points: Points to query [N, 3] (Gaussian centers)
        anchor_points: Anchor points [M, 3]
        k: Number of nearest neighbors
        temperature: Softmax temperature for weight computation
    
    Returns:
        knn_indices: Indices of k nearest anchors [N, k]
        knn_weights: Normalized distance-based weights [N, k]
    """
    # Compute pairwise distances [N, M]
    dist_sq = torch.cdist(query_points, anchor_points, p=2) ** 2
    
    # Get k nearest neighbors
    k = min(k, anchor_points.shape[0])
    neg_dist_sq = -dist_sq / temperature
    _, knn_indices = torch.topk(neg_dist_sq, k, dim=-1)  # [N, k]
    
    # Gather distances for KNN
    knn_dist_sq = torch.gather(dist_sq, 1, knn_indices)  # [N, k]
    
    # Compute softmax weights (closer anchors have higher weights)
    knn_weights = F.softmax(-knn_dist_sq / temperature, dim=-1)  # [N, k]
    
    return knn_indices, knn_weights


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for 3D positions.
    """
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        self.d_model = d_model
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of positions [batch_size, seq_len] or [seq_len]
        """
        return self.pe[:x.size(0)]


class TimeEncoding(nn.Module):
    """
    Fourier time encoding for temporal information.
    """
    def __init__(self, d_model: int, num_freqs: int = 8):
        super().__init__()
        self.d_model = d_model
        self.num_freqs = num_freqs
        
        # Learnable frequency bands
        self.freq_bands = nn.Parameter(
            torch.linspace(1.0, num_freqs, num_freqs) * math.pi
        )
        
        # Project to d_model
        self.proj = nn.Linear(num_freqs * 2, d_model)
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time values [batch_size] or scalar
        
        Returns:
            time_embed: Time embedding [batch_size, d_model] or [1, d_model]
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # [B, 1]
        
        # Fourier features
        freq_t = t * self.freq_bands  # [B, num_freqs]
        fourier = torch.cat([torch.sin(freq_t), torch.cos(freq_t)], dim=-1)  # [B, 2*num_freqs]
        
        return self.proj(fourier)  # [B, d_model]


class AnchorEmbedding(nn.Module):
    """
    Embed anchor positions into a feature space.
    """
    def __init__(self, pos_dim: int = 3, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.mlp = nn.Sequential(
            nn.Linear(pos_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, embed_dim),
        )
    
    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pos: Anchor positions [M, 3]
        
        Returns:
            embed: Position embeddings [M, embed_dim]
        """
        return self.mlp(pos)


class SpacetimeTransformerEncoder(nn.Module):
    """
    Transformer encoder for spacetime anchor interactions.
    
    The transformer learns how anchors influence each other's motion based on
    their spatial relationships and temporal context (breathing phase).
    """
    def __init__(
        self,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: Input features [batch_size, num_anchors, d_model]
            mask: Optional attention mask [batch_size, num_anchors]
        
        Returns:
            output: Encoded features [batch_size, num_anchors, d_model]
        """
        # TransformerEncoder expects src_key_padding_mask for masking
        return self.encoder(x, src_key_padding_mask=mask)


class AnchorDeformationNet(nn.Module):
    """
    PhysX-Gaussian: Anchor-based Spacetime Transformer for Deformation.
    
    This module replaces HexPlane + MLP by:
    1. Using FPS-sampled anchors as physical control points
    2. Learning anchor interactions via self-attention
    3. Masking anchors during training for robust deformation inference
    4. Interpolating anchor displacements to Gaussian positions via skinning
    
    The key insight is that respiratory motion is governed by physical
    constraints (rib cage, diaphragm, lung tissue), so learning these
    relationships allows generalization to irregular breathing patterns.
    """
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        
        # Hyperparameters
        self.num_anchors = getattr(args, 'num_anchors', 1024)
        self.anchor_k = getattr(args, 'anchor_k', 10)
        self.mask_ratio = getattr(args, 'mask_ratio', 0.25)
        self.d_model = getattr(args, 'transformer_dim', 64)
        self.nhead = getattr(args, 'transformer_heads', 4)
        self.num_layers = getattr(args, 'transformer_layers', 2)
        self.time_embed_dim = getattr(args, 'anchor_time_embed_dim', 16)
        self.pos_embed_dim = getattr(args, 'anchor_pos_embed_dim', 32)
        
        # Mask decay scheduler (v2 feature)
        # When enabled, mask_ratio decays linearly from mask_decay_start to 0
        self.use_mask_decay = getattr(args, 'use_mask_decay', False)
        self.mask_decay_start = getattr(args, 'mask_decay_start', 0.5)
        
        # Anchor state (will be initialized from point cloud)
        self.register_buffer('anchor_positions', torch.zeros(self.num_anchors, 3))
        self.register_buffer('anchor_indices', torch.zeros(self.num_anchors, dtype=torch.long))
        self.register_buffer('initialized', torch.tensor(False))
        
        # KNN cache (updated when Gaussians change)
        self.register_buffer('knn_indices', torch.zeros(1, self.anchor_k, dtype=torch.long))
        self.register_buffer('knn_weights', torch.zeros(1, self.anchor_k))
        self.register_buffer('knn_valid', torch.tensor(False))
        
        # Embeddings
        self.anchor_embed = AnchorEmbedding(pos_dim=3, embed_dim=self.pos_embed_dim)
        self.time_encode = TimeEncoding(d_model=self.time_embed_dim, num_freqs=8)
        
        # Input projection: [pos_embed + time_embed] -> d_model
        input_dim = self.pos_embed_dim + self.time_embed_dim
        self.input_proj = nn.Linear(input_dim, self.d_model)
        
        # Learnable [MASK] token for masked modeling
        self.mask_token = nn.Parameter(torch.randn(1, 1, self.d_model) * 0.02)
        
        # Spacetime Transformer
        self.transformer = SpacetimeTransformerEncoder(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=self.num_layers,
            dim_feedforward=self.d_model * 4,
            dropout=0.1
        )
        
        # Output head: predict anchor displacement
        self.displacement_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 3)  # 3D displacement
        )
        
        # Backward displacement head (for inverse consistency with original pipeline)
        self.displacement_head_backward = nn.Sequential(
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 3)
        )
        
        # Scale/rotation heads for full compatibility
        self.scale_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 3)
        )
        
        self.rotation_head = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.GELU(),
            nn.Linear(self.d_model // 2, 4)
        )
        
        # Cache for masked anchor info (for physics completion loss)
        self._last_masked_indices = None
        self._last_anchor_displacements = None
        self._last_unmasked_displacements = None
        
        # ================================================================
        # PhysX-Hybrid: Lightweight HexPlane residual network ("neural skin")
        # ================================================================
        self.use_hybrid = getattr(args, 'use_hybrid', False)
        self.residual_net = None
        self._last_residual_magnitude = None  # For L1 regularization
        
        if self.use_hybrid:
            residual_dim = getattr(args, 'residual_dim', 8)
            residual_resolution = getattr(args, 'residual_resolution', [64, 64, 64, 50])
            
            # Create lightweight HexPlane for residual displacement
            self.residual_hexplane = HexPlaneField(
                bounds=1.5,  # Will be updated with set_aabb
                planeconfig={
                    'grid_dimensions': 2,
                    'input_coordinate_dim': 4,  # x, y, z, t
                    'output_coordinate_dim': residual_dim,
                    'resolution': residual_resolution,
                },
                multires=[1]  # Single resolution for efficiency
            )
            
            # MLP to decode residual displacement from HexPlane features
            self.residual_mlp = nn.Sequential(
                nn.Linear(residual_dim, residual_dim * 2),
                nn.GELU(),
                nn.Linear(residual_dim * 2, 3)  # 3D displacement
            )
            
            # Initialize residual MLP with small weights (start near zero)
            for m in self.residual_mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)
        
        # ================================================================
        # PhysX-Taylor: First-Order Taylor Expansion (Neural Taylor Anchors)
        # ================================================================
        # Upgrade from zero-order (t only) to first-order Taylor expansion:
        #   Δx = Σ w_pk * (t_k + A_k · (x_point - x_anchor_k))
        #
        # Each anchor predicts:
        #   - t_k ∈ R³: Translation vector
        #   - A_k ∈ R³ˣ³: Local affine deformation gradient (rotation/scale/shear)
        #
        # This allows precise description of complex sharp deformations
        # that zero-order KNN interpolation would smooth out.
        self.use_taylor = getattr(args, 'use_taylor', False)
        self._last_affine_magnitude = None  # For L1 regularization
        
        if self.use_taylor:
            # Affine head: predict 3x3 affine matrix (9 elements)
            # Output dim = 12 (3 translation + 9 affine) but we keep displacement_head
            # for translation and add separate affine_head for the 3x3 matrix
            self.affine_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.GELU(),
                nn.Linear(self.d_model, 9)  # 3x3 affine matrix flattened
            )
            
            # Initialize affine head with very small weights
            # (start near identity transformation, i.e., A ≈ 0)
            for m in self.affine_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.001)
                    nn.init.zeros_(m.bias)
        
        # ================================================================
        # PhysX-Boosted: Full HexPlane Baseline + Anchor Physical Correction
        # ================================================================
        # Strategy: "100% Baseline capability + 10% physical robustness"
        #   Δμ_total = Δμ_hexplane(t) + Δμ_anchor(t)
        #   - HexPlane: "Paint the skin" (high-frequency texture, micro-deformations)
        #   - Anchor: "Draw the skeleton" (anatomical structure, physical consistency)
        #
        # HexPlane is driven by L_render (detail), Anchor is driven by L_render + L_phys
        self.use_boosted = getattr(args, 'use_boosted', False)
        self.original_deformation = None
        
        # V5: Learnable balance parameter between HexPlane and Anchor
        # Δx_total = (1 - α) · Δx_hexplane + α · Δx_anchor
        # α = sigmoid(τ), τ is learnable, initialized to achieve α_init
        self.use_learnable_balance = getattr(args, 'use_learnable_balance', False)
        balance_alpha_init = getattr(args, 'balance_alpha_init', 0.5)
        self._balance_alpha_init = balance_alpha_init  # Store original for reference
        
        # Handle extreme cases: α=0 (pure HexPlane) and α=1 (pure Anchor)
        self.use_pure_hexplane = (balance_alpha_init == 0.0)  # α=0: only HexPlane
        self.use_pure_anchor = (balance_alpha_init == 1.0)    # α=1: only Anchor
        
        # Convert α_init to logit τ: τ = log(α / (1-α))
        # For extreme values, clamp to avoid numerical issues but store the flag
        if balance_alpha_init <= 0:
            tau_init = -10.0  # sigmoid(-10) ≈ 0.00005
        elif balance_alpha_init >= 1:
            tau_init = 10.0   # sigmoid(10) ≈ 0.99995
        else:
            tau_init = np.log(balance_alpha_init / (1 - balance_alpha_init))
        self.balance_logit = nn.Parameter(torch.tensor([tau_init], dtype=torch.float32))
        
        # V6: Orthogonal Gradient Projection
        # Core idea: HexPlane (A) is the "base", Anchor (B) learns the residual
        # Forward: Δx_total = Δx_hex + Δx_anchor (direct sum)
        # Backward: Modify Anchor's gradient to be orthogonal to HexPlane's gradient
        self.use_orthogonal_projection = getattr(args, 'use_orthogonal_projection', False)
        self.ortho_projection_strength = getattr(args, 'ortho_projection_strength', 1.0)
        # Cache for gradient projection
        self._cached_dx_hex_for_grad = None
        self._ortho_hook_handle = None
        
        # V8: Reverse Orthogonal Gradient Projection (swap A and B)
        # Core idea: Anchor (A) is the "base", HexPlane (B) learns the residual
        # Forward: Δx_total = Δx_hex + Δx_anchor (direct sum)
        # Backward: Modify HexPlane's gradient to be orthogonal to Anchor's gradient
        self.use_reverse_orthogonal_projection = getattr(args, 'use_reverse_orthogonal_projection', False)
        
        # V7: Uncertainty-Aware Fusion (Aleatoric Uncertainty)
        # Both HexPlane and Anchor output: displacement [N, 3] + log_var [N, 1]
        # Fusion uses inverse variance weighting
        self.use_uncertainty_fusion = getattr(args, 'use_uncertainty_fusion', False)
        self.uncertainty_eps = getattr(args, 'uncertainty_eps', 1e-6)
        self.lambda_uncertainty = getattr(args, 'lambda_uncertainty', 0.5)
        uncertainty_init = getattr(args, 'uncertainty_init', 0.0)
        
        if self.use_uncertainty_fusion:
            # Anchor uncertainty head: outputs log(σ²) for anchor displacement
            self.anchor_uncertainty_head = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.GELU(),
                nn.Linear(self.d_model // 2, 1)  # log(σ²) scalar per anchor
            )
            # Initialize to output uncertainty_init (σ²=1 when init=0)
            nn.init.zeros_(self.anchor_uncertainty_head[-1].weight)
            nn.init.constant_(self.anchor_uncertainty_head[-1].bias, uncertainty_init)
            
            # HexPlane uncertainty head: takes dx_hex [N, 3] and outputs log(σ²) [N, 1]
            self.hex_uncertainty_head = nn.Sequential(
                nn.Linear(3, 32),
                nn.GELU(),
                nn.Linear(32, 1)  # log(σ²) per Gaussian
            )
            # Initialize to output uncertainty_init
            nn.init.zeros_(self.hex_uncertainty_head[-1].weight)
            nn.init.constant_(self.hex_uncertainty_head[-1].bias, uncertainty_init)
            
            # Cache for uncertainty values (for loss computation)
            self._last_log_var_hex = None
            self._last_log_var_anchor = None
            self._last_weight_hex = None
            self._last_weight_anchor = None
            self._last_anchor_features = None  # Cache anchor features for uncertainty
        
        # V10: Decoupled Masked Modeling
        # Core idea: Decouple rendering from mask training
        # - Rendering uses UNMASKED output (full power)
        # - L_phys separately supervises masked prediction
        self.use_decoupled_mask = getattr(args, 'use_decoupled_mask', False)
        
        # V11: Pretrain-Finetune Masked Modeling
        # Stage 1: Only L_phys with high mask ratio
        # Stage 2: Normal rendering with low anchor LR
        self.use_pretrain_finetune = getattr(args, 'use_pretrain_finetune', False)
        self.pretrain_steps = getattr(args, 'pretrain_steps', 3000)
        self.pretrain_mask_ratio = getattr(args, 'pretrain_mask_ratio', 0.5)
        self.pretrain_only_anchor = getattr(args, 'pretrain_only_anchor', True)
        self.finetune_anchor_lr_scale = getattr(args, 'finetune_anchor_lr_scale', 0.1)
        self._in_pretrain_stage = False  # Runtime flag
        
        # V12: Temporal Mask (Time-step Masking)
        # Mask entire time steps instead of random spatial anchors
        self.use_temporal_mask = getattr(args, 'use_temporal_mask', False)
        self.temporal_mask_ratio = getattr(args, 'temporal_mask_ratio', 0.25)
        # Track which time steps are masked (for batch processing)
        self._temporal_masked_times = set()
        
        # V13: Consistency Regularization
        # Mask as data augmentation, not pretraining
        # L_consist = ||masked_out - unmasked_out.detach()||
        self.use_consistency_mask = getattr(args, 'use_consistency_mask', False)
        self.lambda_consist = getattr(args, 'lambda_consist', 0.1)
        self._last_unmasked_for_consist = None  # Cache for consistency loss
        
        # V14: Temporal Interpolation
        # Predict intermediate frames given context
        self.use_temporal_interp = getattr(args, 'use_temporal_interp', False)
        self.lambda_interp = getattr(args, 'lambda_interp', 0.1)
        self.interp_context_range = getattr(args, 'interp_context_range', 0.2)
        
        # ================================================================
        # V16: Lagrangian Spatio-Temporal Masked Anchor Modeling
        # ================================================================
        # Core idea: BERT-style masking on (anchor, time) tokens
        # Key difference: mask_flag embedding preserves positional info
        self.use_spatiotemporal_mask = getattr(args, 'use_spatiotemporal_mask', False)
        self.lambda_lagbert = getattr(args, 'lambda_lagbert', 0.5)
        self.st_window_size = getattr(args, 'st_window_size', 3)
        self.st_time_delta = getattr(args, 'st_time_delta', 0.1)
        self.st_mask_ratio = getattr(args, 'st_mask_ratio', 0.3)
        
        # V16 Fix 1: mask_embed scale factor (default 1.0 = original behavior)
        self.st_mask_embed_scale = getattr(args, 'st_mask_embed_scale', 1.0)
        
        # V16 Fix 2: Couple render with L_lagbert (default False = original behavior)
        self.st_coupled_render = getattr(args, 'st_coupled_render', False)
        
        if self.use_spatiotemporal_mask:
            # Mask flag embedding: {0: unmasked, 1: masked} -> d_model
            # This is ADDED to token embedding, NOT replacing it
            self.mask_flag_embed = nn.Embedding(2, self.d_model)
            nn.init.normal_(self.mask_flag_embed.weight, std=0.02)
            
            # Cache for L_lagbert computation
            self._last_st_full_out = None  # [K, M, 3] from full pass
            self._last_st_masked_out = None  # [K, M, 3] from masked pass
            self._last_st_mask_flags = None  # [K, M] binary mask
        
        # ================================================================
        # M1: Uncertainty-Gated Residual Fusion
        # ================================================================
        # Replace fixed α with adaptive β(x,t) based on Eulerian uncertainty
        #
        # M1 Fusion formula (paper notation):
        #   Φ(x,t) = Φ_L(x,t) + β(x,t) · Φ_E(x,t)
        #
        # β(x,t) gating modes:
        #   - Bayes: β = σ_L² / (σ_L² + σ_E²(x,t))
        #   - Sigmoid: β = sigmoid((τ - s_E(x,t)) / λ)
        #
        # Where s_E = log(σ_E²) is output by the Eulerian HexPlane uncertainty head
        self.fusion_mode = getattr(args, 'fusion_mode', 'fixed_alpha')
        self.gate_mode = getattr(args, 'gate_mode', 'bayes')
        self.sigma_L2 = getattr(args, 'sigma_L2', 1e-4)
        self.gate_tau = getattr(args, 'gate_tau', 0.0)
        self.gate_lambda = getattr(args, 'gate_lambda', 1.0)
        self.beta_min = getattr(args, 'beta_min', 0.0)
        self.beta_max = getattr(args, 'beta_max', 1.0)
        self.m1_lambda_gate = getattr(args, 'm1_lambda_gate', 0.0)
        
        # M1.2: Small perturbation around V5's optimal 99:1 ratio
        # γ_max controls the maximum deviation from V5's 1% HexPlane weight
        # With γ_max=0.005: HexPlane weight can vary from 0.5% to 1.5%
        # With γ_max=0.01: HexPlane weight can vary from 0% to 2%
        self.gamma_max = getattr(args, 'gamma_max', 0.005)
        
        # Cache for gamma
        self._last_gamma = None
        
        # Cache for M1: store β and s_E for loss computation and logging
        self._last_beta = None
        self._last_beta_mean = None
        self._last_s_E = None
        
        # ================================================================
        # M2: Bounded Learnable Perturbation (ICML-style formulation)
        # ================================================================
        # Formula: Φ = Φ_L + ε * tanh(Φ_E)
        #
        # Key insight from V5 experiments:
        #   - α=0.99 means Lagrangian dominates, Eulerian is small correction
        #   - Instead of weighted average, use "Base + Bounded Perturbation"
        #
        # ε parameterization (bounded to prevent shortcut learning):
        #   ε = ε_max * sigmoid(ρ)
        #   where ρ is a learnable scalar nn.Parameter
        #
        # Initialization to match V5's α=0.99:
        #   ε_init ≈ 0.01 (Eulerian contribution is ~1% scale)
        #   ρ_init = logit(ε_init / ε_max)
        #
        # tanh(Φ_E) bounds the perturbation magnitude, preventing explosions
        # ================================================================
        self.eps_max = getattr(args, 'eps_max', 0.02)
        self.eps_init = getattr(args, 'eps_init', 0.01)
        self.use_tanh = getattr(args, 'use_tanh', True)
        
        if self.fusion_mode == 'bounded_perturb':
            # Initialize ρ such that ε = ε_init
            # ε = ε_max * sigmoid(ρ) → ρ = logit(ε / ε_max)
            eps_ratio = min(max(self.eps_init / self.eps_max, 1e-6), 1 - 1e-6)
            rho_init = math.log(eps_ratio / (1 - eps_ratio))  # logit
            self.rho = nn.Parameter(torch.tensor(rho_init, dtype=torch.float32))
        
        # ================================================================
        # M2.1: Trust-Region Schedule Parameters
        # ================================================================
        self.schedule_mode = getattr(args, 'schedule_mode', 'none')
        self.freeze_steps = getattr(args, 'freeze_steps', 2000)
        self.warmup_steps = getattr(args, 'warmup_steps', 5000)
        
        # ================================================================
        # M2.2: Residual Normalization Mode
        # ================================================================
        # "Residual normalization makes ε a true trust-region radius by
        #  preventing magnitude leakage from the Eulerian stream."
        self.residual_mode = getattr(args, 'residual_mode', 'tanh')
        self.norm_eps = getattr(args, 'norm_eps', 1e-6)
        
        # Cache for M2/M2.1/M2.2 logging
        self._last_eps_raw = None   # ε = ε_max * sigmoid(ρ)
        self._last_eps_eff = None   # ε_eff after schedule
        self._last_warmup_ratio = None
        self._is_frozen = False
        self._last_mean_norm_E = None  # M2.2: mean ||Δ|| before normalization
        self._last_mean_norm_H = None  # M2.2: mean ||H(Δ)|| after normalization
        
        # ================================================================
        # M3: Low-Frequency Leakage Penalty Cache
        # ================================================================
        # "Low-frequency leakage regularization prevents the Eulerian stream
        #  from explaining global motion, reserving it for high-frequency
        #  corrective details around the Lagrangian manifold."
        self.lp_enable = getattr(args, 'lp_enable', False)
        self.lambda_lp = getattr(args, 'lambda_lp', 0.01)
        self.lp_mode = getattr(args, 'lp_mode', 'knn_mean')
        self.lp_k = getattr(args, 'lp_k', 8)
        self.lp_subsample = getattr(args, 'lp_subsample', 2048)
        
        # Cache for LP computation (raw Δ before H(·))
        self._last_delta_raw = None  # Raw Eulerian residual [N, 3]
        self._last_positions = None  # Gaussian positions [N, 3]
        self._last_lp_loss = None    # L_LP value
        self._last_lp_mean = None    # mean ||LP(Δ)||
        self._last_lp_ratio = None   # ratio = ||LP(Δ)|| / ||Δ||
        
        # ================================================================
        # M4: Subspace Decoupling Regularization
        # ================================================================
        # "Subspace decoupling regularization discourages the Eulerian residual
        #  from aligning with the Lagrangian deformation responses, forcing it
        #  to model complementary details rather than shortcuts."
        self.decouple_enable = getattr(args, 'decouple_enable', False)
        self.lambda_decouple = getattr(args, 'lambda_decouple', 0.01)
        self.decouple_mode = getattr(args, 'decouple_mode', 'velocity_corr')
        self.decouple_subsample = getattr(args, 'decouple_subsample', 2048)
        self.decouple_stopgrad_L = getattr(args, 'decouple_stopgrad_L', True)
        self.decouple_dt = getattr(args, 'decouple_dt', 0.02)
        self.decouple_use_squared_cos = getattr(args, 'decouple_use_squared_cos', True)
        self.decouple_num_dirs = getattr(args, 'decouple_num_dirs', 1)
        
        # Cache for decoupling computation
        self._last_decouple_loss = None
        self._last_corr_mean = None
        self._last_grad_L_norm = None
        self._last_grad_E_norm = None
        
        # Cache for velocity computation (reuse deformation outputs)
        self._last_dx_anchor = None  # Lagrangian deformation at t
        self._last_dx_hex = None     # Eulerian deformation at t
        self._last_time = None       # Current time t
        
        if self.use_boosted:
            # Instantiate the FULL-POWER HexPlane baseline (not lightweight)
            # Use original args directly - preserves all baseline configurations
            self.original_deformation = deform_network(args)
            print(f"[PhysX-Boosted] Full HexPlane baseline instantiated")
        
        print(f"[PhysX-Gaussian] Initialized AnchorDeformationNet:")
        print(f"  - num_anchors: {self.num_anchors}")
        print(f"  - anchor_k: {self.anchor_k}")
        print(f"  - mask_ratio: {self.mask_ratio}")
        print(f"  - transformer_dim: {self.d_model}")
        print(f"  - transformer_heads: {self.nhead}")
        print(f"  - transformer_layers: {self.num_layers}")
        if self.use_mask_decay:
            print(f"  - mask_decay: ENABLED (start={self.mask_decay_start} -> 0)")
        if self.use_hybrid:
            print(f"  - HYBRID MODE: Anchor (skeleton) + HexPlane (skin)")
            print(f"    - residual_dim: {residual_dim}")
            print(f"    - residual_resolution: {residual_resolution}")
        if self.use_taylor:
            print(f"  - TAYLOR MODE: First-order affine deformation (t + A·δ)")
            print(f"    - Output: 3 (translation) + 9 (affine matrix) = 12 dims")
        if self.use_boosted:
            print(f"  - BOOSTED MODE: Full HexPlane baseline + Anchor correction")
            print(f"    - HexPlane: Full-power (net_width={getattr(args, 'net_width', 64)})")
            print(f"    - Anchor: V1 lightweight (transformer_dim={self.d_model})")
            if self.use_learnable_balance:
                if self.use_pure_hexplane:
                    print(f"  - V5 FIXED BALANCE: α = 0.0 (PURE HEXPLANE)")
                    print(f"    - Formula: Δx = Δx_hex (Anchor disabled)")
                elif self.use_pure_anchor:
                    print(f"  - V5 FIXED BALANCE: α = 1.0 (PURE ANCHOR)")
                    print(f"    - Formula: Δx = Δx_anchor (HexPlane disabled)")
                else:
                    print(f"  - V5 LEARNABLE BALANCE: α = sigmoid(τ), τ_init={tau_init:.3f} → α_init={balance_alpha_init:.2f}")
                    print(f"    - Formula: Δx = (1-α)·Δx_hex + α·Δx_anchor")
            if self.use_orthogonal_projection:
                print(f"  - V6 ORTHOGONAL GRADIENT PROJECTION: Anchor learns residual only")
                print(f"    - Forward: Δx = Δx_hex + Δx_anchor (direct sum)")
                print(f"    - Backward: grad_anchor ⊥ grad_hex (orthogonal projection)")
                print(f"    - Projection strength: {self.ortho_projection_strength}")
            if self.use_reverse_orthogonal_projection:
                if self.use_learnable_balance:
                    print(f"  - V8.1 REVERSE ORTHOGONAL + WEIGHTED: HexPlane learns residual only")
                    print(f"    - Forward: Δx = (1-α)·Δx_hex + α·Δx_anchor")
                    print(f"    - α_init = {self._balance_alpha_init:.2f}")
                else:
                    print(f"  - V8 REVERSE ORTHOGONAL PROJECTION: HexPlane learns residual only")
                    print(f"    - Forward: Δx = Δx_hex + Δx_anchor (direct sum)")
                print(f"    - Backward: grad_hex ⊥ grad_anchor (orthogonal projection)")
                print(f"    - Projection strength: {self.ortho_projection_strength}")
            if self.use_uncertainty_fusion:
                print(f"  - V7 UNCERTAINTY-AWARE FUSION: Inverse variance weighting")
                print(f"    - Both branches output: Δx + log(σ²)")
                print(f"    - Fusion: Δx_final = (w_A·Δx_hex + w_B·Δx_anchor) / (w_A + w_B)")
                print(f"    - where w = 1/(σ² + ε), ε = {self.uncertainty_eps}")
                print(f"    - Kendall Loss: L/(2Σ) + λ·log(Σ), λ = {self.lambda_uncertainty}")
            if self.use_decoupled_mask:
                print(f"  - V10 DECOUPLED MASKED MODELING: Separate rendering from mask training")
                print(f"    - Rendering: Uses UNMASKED output (full power)")
                print(f"    - L_phys: Separately supervises masked prediction")
                print(f"    - mask_ratio for L_phys: {self.mask_ratio}")
            if self.use_pretrain_finetune:
                print(f"  - V11 PRETRAIN-FINETUNE: True BERT-style two-stage training")
                print(f"    - Stage 1 (Pretrain): {self.pretrain_steps} steps, mask_ratio={self.pretrain_mask_ratio}")
                print(f"    - Only L_phys, no rendering (forces physical relationship learning)")
                print(f"    - Stage 2 (Finetune): Normal rendering, anchor LR *= {self.finetune_anchor_lr_scale}")
            if self.use_temporal_mask:
                print(f"  - V12 TEMPORAL MASK: Mask entire time steps")
                print(f"    - Given t=0,1,2,4,5, predict t=3 (unseen time)")
                print(f"    - temporal_mask_ratio: {self.temporal_mask_ratio}")
                print(f"    - Learns temporal continuity and physical dynamics")
            if self.use_consistency_mask:
                print(f"  - V13 CONSISTENCY REGULARIZATION: Mask as data augmentation")
                print(f"    - unmasked_out → render, masked_out → consistency")
                print(f"    - L_consist = ||masked - unmasked.detach()||")
                print(f"    - lambda_consist: {self.lambda_consist}")
                print(f"    - Teaches robustness, not representation")
            if self.use_temporal_interp:
                print(f"  - V14 TEMPORAL INTERPOLATION: Predict intermediate frames")
                print(f"    - Given t1, t2 context, predict t_mid")
                print(f"    - lambda_interp: {self.lambda_interp}")
                print(f"    - context_range: {self.interp_context_range}")
            if self.use_spatiotemporal_mask:
                print(f"  - V16 LAGRANGIAN SPATIO-TEMPORAL MASKED MODELING:")
                print(f"    - Tokens: (anchor, time) pairs in K={self.st_window_size} time window")
                print(f"    - Mask flag embedding (preserves pos/time info)")
                print(f"    - st_mask_ratio: {self.st_mask_ratio}")
                print(f"    - lambda_lagbert: {self.lambda_lagbert} (MAJOR objective)")
                print(f"    - st_time_delta: {self.st_time_delta}")
                print(f"    - st_mask_embed_scale: {self.st_mask_embed_scale} (1.0=original, <1=reduced interference)")
                print(f"    - st_coupled_render: {self.st_coupled_render} (False=separate, True=shared forward)")
            if self.fusion_mode == 'uncertainty_gated':
                print(f"  - M1.2 SMALL PERTURBATION FUSION (preserves V5's 99:1 ratio):")
                print(f"    - Formula: Φ = (0.99-γ)·Φ_L + (0.01+γ)·Φ_E")
                print(f"    - γ = γ_max * tanh((τ - s_E) / λ)")
                print(f"    - γ_max: {self.gamma_max} (HexPlane weight range: [{0.01-self.gamma_max:.3f}, {0.01+self.gamma_max:.3f}])")
                print(f"    - τ (gate_tau): {self.gate_tau}, λ (gate_lambda): {self.gate_lambda}")
                print(f"    - m1_lambda_gate: {self.m1_lambda_gate}")
            if self.fusion_mode == 'bounded_perturb':
                print(f"  - M2.2 LEARNABLE WEIGHTED AVERAGE + TRUST-REGION + RESIDUAL NORM:")
                print(f"    - Formula: Φ = (1-ε_eff)·Φ_L + ε_eff·H(Φ_E)")
                print(f"    - ε_raw = ε_max·sigmoid(ρ), ρ is learnable")
                print(f"    - ε_max: {self.eps_max}, ε_init: {self.eps_init}")
                print(f"    - ρ_init: {self.rho.item():.4f} → ε_init: {self.eps_max * torch.sigmoid(self.rho).item():.4f}")
                print(f"    - schedule_mode: {self.schedule_mode}")
                if self.schedule_mode == 'freeze_rho':
                    print(f"    - freeze_steps: {self.freeze_steps} (ρ frozen for first N steps)")
                elif self.schedule_mode == 'warmup_cap':
                    print(f"    - warmup_steps: {self.warmup_steps} (ε_eff = min(ε_raw, ε_max * s/warmup_steps))")
                print(f"    - residual_mode: {self.residual_mode} (H(Δ) normalization)")
                print(f"    - norm_eps: {self.norm_eps}")
                if self.lp_enable:
                    print(f"  - M3 LOW-FREQUENCY LEAKAGE PENALTY:")
                    print(f"    - lp_mode: {self.lp_mode}")
                    print(f"    - lambda_lp: {self.lambda_lp}")
                    print(f"    - lp_k: {self.lp_k}")
                    print(f"    - lp_subsample: {self.lp_subsample}")
                if self.decouple_enable:
                    print(f"  - M4 SUBSPACE DECOUPLING:")
                    print(f"    - decouple_mode: {self.decouple_mode}")
                    print(f"    - lambda_decouple: {self.lambda_decouple}")
                    print(f"    - stopgrad_L: {self.decouple_stopgrad_L}")
                    print(f"    - subsample: {self.decouple_subsample}")
                    if self.decouple_mode == 'velocity_corr':
                        print(f"    - dt: {self.decouple_dt}")
                    else:
                        print(f"    - num_dirs: {self.decouple_num_dirs}")
    
    def initialize_anchors(self, points: torch.Tensor) -> None:
        """
        Initialize anchors from point cloud using FPS.
        
        Args:
            points: Initial Gaussian centers [N, 3]
        """
        num_points = points.shape[0]
        actual_num_anchors = min(self.num_anchors, num_points)
        
        # FPS sampling
        indices = farthest_point_sampling(points.detach(), actual_num_anchors)
        
        # Store anchor positions and indices
        # IMPORTANT: detach() to ensure no computation graph from Gaussian parameters
        self.anchor_indices = indices
        self.anchor_positions = points[indices].detach().clone()
        self.initialized.fill_(True)
        
        print(f"[PhysX-Gaussian] Initialized {actual_num_anchors} anchors via FPS from {num_points} points")
    
    def update_knn_binding(self, gaussian_positions: torch.Tensor, temperature: float = 0.01) -> None:
        """
        Update KNN binding between Gaussians and anchors.
        
        This should be called when:
        1. Anchors are first initialized
        2. After densification/pruning (Gaussian count changes)
        
        Args:
            gaussian_positions: Current Gaussian centers [N, 3]
            temperature: Softmax temperature for weight computation
        """
        if not self.initialized:
            raise RuntimeError("Anchors not initialized. Call initialize_anchors first.")
        
        knn_indices, knn_weights = compute_knn_weights(
            gaussian_positions.detach(),  # Detach to avoid computation graph issues
            self.anchor_positions,
            k=self.anchor_k,
            temperature=temperature
        )
        
        # Detach weights to avoid backward through KNN computation
        self.knn_indices = knn_indices.detach()
        self.knn_weights = knn_weights.detach()
        self.knn_valid.fill_(True)
    
    def forward_anchors(
        self,
        time_emb: torch.Tensor,
        is_training: bool = True,
        return_all_info: bool = False,
        iteration_ratio: float = 0.0
    ) -> torch.Tensor:
        """
        Compute anchor displacements at given time.
        
        Args:
            time_emb: Time value [1] or [N, 1]
            is_training: If True, apply masking for physics completion
            return_all_info: If True, return additional info for loss computation
            iteration_ratio: Current iteration / total iterations (0.0 to 1.0)
                            Used for mask decay scheduler
        
        Returns:
            anchor_displacements: Displacement for each anchor [M, 3]
        """
        # V16 Fix 2: If st_coupled_render=True and we have cached dx_center from
        # compute_lagbert_loss(), return it directly instead of recomputing.
        # This ensures rendering uses the same forward pass as L_lagbert.
        if self.st_coupled_render and self.use_spatiotemporal_mask:
            if hasattr(self, '_st_coupled_dx_center') and self._st_coupled_dx_center is not None:
                # Return cached dx_center and clear the cache for next iteration
                dx_center = self._st_coupled_dx_center
                self._st_coupled_dx_center = None  # Clear after use
                return dx_center
        
        # Clear cached tensors from previous iteration to avoid graph conflicts
        self._last_anchor_displacements = None
        self._last_unmasked_displacements = None
        self._last_masked_indices = None
        
        device = self.anchor_positions.device
        M = self.anchor_positions.shape[0]
        
        # Get time value (scalar)
        if time_emb.dim() > 0:
            t = time_emb[0, 0] if time_emb.dim() == 2 else time_emb[0]
        else:
            t = time_emb
        
        # Embed anchor positions (detach to ensure no graph from initialization)
        anchor_pos = self.anchor_positions.detach()
        pos_embed = self.anchor_embed(anchor_pos)  # [M, pos_embed_dim]
        
        # Time encoding (broadcast to all anchors)
        time_embed = self.time_encode(t.unsqueeze(0))  # [1, time_embed_dim]
        time_embed = time_embed.expand(M, -1)  # [M, time_embed_dim]
        
        # Concatenate and project
        anchor_input = torch.cat([pos_embed, time_embed], dim=-1)  # [M, pos_embed_dim + time_embed_dim]
        anchor_features = self.input_proj(anchor_input)  # [M, d_model]
        
        # Add batch dimension
        anchor_features = anchor_features.unsqueeze(0)  # [1, M, d_model]
        
        # Compute effective mask ratio (with optional decay)
        if self.use_mask_decay:
            # Linear decay: start at mask_decay_start, end at 0
            effective_mask_ratio = self.mask_decay_start * (1.0 - iteration_ratio)
        elif self.use_pretrain_finetune and self._in_pretrain_stage:
            # V11: Use higher mask ratio during pretrain stage
            effective_mask_ratio = self.pretrain_mask_ratio
        else:
            # Use fixed mask_ratio (v1 behavior)
            effective_mask_ratio = self.mask_ratio
        
        # Masking for BERT-style training
        # V10: When use_decoupled_mask=True, skip masking in main forward (render path)
        # Masking will be done separately in forward_anchors_masked() for L_phys
        masked_indices = None
        should_mask = is_training and effective_mask_ratio > 0 and not self.use_decoupled_mask
        
        # V12: Temporal Mask - mask all anchors if this time step is masked
        if self.use_temporal_mask and is_training:
            # Discretize time to 10 bins (0-9) for phase-based masking
            time_bin = int(t.item() * 10) % 10
            # Randomly decide if this time bin should be masked
            # Use a deterministic hash based on time_bin for consistency within epoch
            should_temporal_mask = (hash(time_bin) % 100) < (self.temporal_mask_ratio * 100)
            if should_temporal_mask:
                # Mask ALL anchors at this time step
                masked_indices = torch.arange(M, device=device)
                mask_tokens = self.mask_token.expand(1, M, -1)
                anchor_features[0, :] = mask_tokens.squeeze(0)
                self._last_masked_indices = masked_indices
                should_mask = False  # Already handled
        
        if should_mask:
            num_mask = int(M * effective_mask_ratio)
            if num_mask > 0:
                # Random mask selection
                perm = torch.randperm(M, device=device)
                masked_indices = perm[:num_mask]
                
                # Replace masked anchor features with [MASK] token
                mask_tokens = self.mask_token.expand(1, num_mask, -1)
                anchor_features[0, masked_indices] = mask_tokens.squeeze(0)
                
                self._last_masked_indices = masked_indices
        
        # Transformer encoding
        anchor_features = self.transformer(anchor_features)  # [1, M, d_model]
        
        # Displacement prediction (translation t_k)
        anchor_displacements = self.displacement_head(anchor_features).squeeze(0)  # [M, 3]
        
        self._last_anchor_displacements = anchor_displacements
        
        # V7: Cache anchor features for uncertainty computation
        if self.use_uncertainty_fusion:
            self._last_anchor_features = anchor_features.squeeze(0)  # [M, d_model]
        
        # ================================================================
        # PhysX-Taylor: Affine matrix prediction (A_k ∈ R³ˣ³)
        # ================================================================
        anchor_affines = None
        if self.use_taylor and hasattr(self, 'affine_head'):
            # Predict 9 affine matrix elements per anchor
            affine_flat = self.affine_head(anchor_features).squeeze(0)  # [M, 9]
            # Reshape to [M, 3, 3]
            anchor_affines = affine_flat.view(M, 3, 3)
            
            # Cache for L1 regularization
            self._last_affine_magnitude = anchor_affines.abs().mean()
        
        # Store affines for interpolation
        self._last_anchor_affines = anchor_affines
        
        if return_all_info:
            return anchor_displacements, masked_indices, anchor_features.squeeze(0)
        
        return anchor_displacements
    
    def forward_anchors_unmasked(self, time_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute anchor displacements WITHOUT masking (for teacher forcing / GT).
        
        Args:
            time_emb: Time value [1] or [N, 1]
        
        Returns:
            anchor_displacements: Displacement for each anchor [M, 3]
        """
        device = self.anchor_positions.device
        M = self.anchor_positions.shape[0]
        
        # Get time value
        if time_emb.dim() > 0:
            t = time_emb[0, 0] if time_emb.dim() == 2 else time_emb[0]
        else:
            t = time_emb
        
        # Embed anchor positions
        pos_embed = self.anchor_embed(self.anchor_positions)  # [M, pos_embed_dim]
        
        # Time encoding
        time_embed = self.time_encode(t.unsqueeze(0))  # [1, time_embed_dim]
        time_embed = time_embed.expand(M, -1)  # [M, time_embed_dim]
        
        # Concatenate and project
        anchor_input = torch.cat([pos_embed, time_embed], dim=-1)
        anchor_features = self.input_proj(anchor_input).unsqueeze(0)  # [1, M, d_model]
        
        # Transformer encoding (no masking)
        anchor_features = self.transformer(anchor_features)  # [1, M, d_model]
        
        # Displacement prediction
        anchor_displacements = self.displacement_head(anchor_features).squeeze(0)  # [M, 3]
        
        self._last_unmasked_displacements = anchor_displacements
        
        return anchor_displacements
    
    def forward_anchors_masked(self, time_emb: torch.Tensor, iteration_ratio: float = 0.0) -> torch.Tensor:
        """
        V10: Compute anchor displacements WITH masking for L_phys supervision.
        
        This is called separately from the main forward pass when use_decoupled_mask=True.
        The masked predictions are compared against unmasked predictions to compute L_phys.
        
        Args:
            time_emb: Time value [1] or [N, 1]
            iteration_ratio: Current iteration / total iterations (for mask decay)
        
        Returns:
            anchor_displacements: Displacement for each anchor [M, 3] (with some masked)
        """
        device = self.anchor_positions.device
        M = self.anchor_positions.shape[0]
        
        # Get time value
        if time_emb.dim() > 0:
            t = time_emb[0, 0] if time_emb.dim() == 2 else time_emb[0]
        else:
            t = time_emb
        
        # Embed anchor positions
        anchor_pos = self.anchor_positions.detach()
        pos_embed = self.anchor_embed(anchor_pos)  # [M, pos_embed_dim]
        
        # Time encoding
        time_embed = self.time_encode(t.unsqueeze(0))  # [1, time_embed_dim]
        time_embed = time_embed.expand(M, -1)  # [M, time_embed_dim]
        
        # Concatenate and project
        anchor_input = torch.cat([pos_embed, time_embed], dim=-1)
        anchor_features = self.input_proj(anchor_input).unsqueeze(0)  # [1, M, d_model]
        
        # Compute effective mask ratio
        # V11: Use higher mask ratio during pretrain stage
        if self.use_mask_decay:
            effective_mask_ratio = self.mask_decay_start * (1.0 - iteration_ratio)
        elif self.use_pretrain_finetune and self._in_pretrain_stage:
            effective_mask_ratio = self.pretrain_mask_ratio
        else:
            effective_mask_ratio = self.mask_ratio
        
        # Apply masking
        masked_indices = None
        
        # V12: Temporal mask - mask all anchors for certain time steps
        if self.use_temporal_mask:
            time_bin = int(t.item() * 10) % 10
            should_temporal_mask = (hash(time_bin) % 100) < (self.temporal_mask_ratio * 100)
            if should_temporal_mask:
                # Mask ALL anchors at this time step
                masked_indices = torch.arange(M, device=device)
                mask_tokens = self.mask_token.expand(1, M, -1)
                anchor_features[0, :] = mask_tokens.squeeze(0)
                self._last_masked_indices = masked_indices
        elif effective_mask_ratio > 0:
            num_mask = int(M * effective_mask_ratio)
            if num_mask > 0:
                perm = torch.randperm(M, device=device)
                masked_indices = perm[:num_mask]
                
                # Replace masked anchor features with [MASK] token
                mask_tokens = self.mask_token.expand(1, num_mask, -1)
                anchor_features[0, masked_indices] = mask_tokens.squeeze(0)
                
                self._last_masked_indices = masked_indices
        
        # Transformer encoding
        anchor_features = self.transformer(anchor_features)  # [1, M, d_model]
        
        # Displacement prediction
        anchor_displacements = self.displacement_head(anchor_features).squeeze(0)  # [M, 3]
        
        # Cache for L_phys computation
        self._last_masked_displacements = anchor_displacements
        
        return anchor_displacements
    
    def interpolate_displacements(
        self,
        anchor_displacements: torch.Tensor,
        gaussian_positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Interpolate anchor displacements to Gaussian positions using skinning weights.
        
        For PhysX-Taylor, implements first-order Taylor expansion:
            Δx_point = Σ w_pk * (t_k + A_k · (x_point - x_anchor_k))
        
        Args:
            anchor_displacements: Translation for each anchor [M, 3]
            gaussian_positions: Gaussian centers [N, 3] (for KNN update if needed)
        
        Returns:
            gaussian_displacements: Interpolated displacement for each Gaussian [N, 3]
        """
        N = gaussian_positions.shape[0]
        K = self.anchor_k
        
        # Update KNN if Gaussian count changed
        if not self.knn_valid or self.knn_indices.shape[0] != N:
            self.update_knn_binding(gaussian_positions)
        
        # Gather anchor translations for each Gaussian's k neighbors
        # knn_indices: [N, k], anchor_displacements: [M, 3]
        neighbor_translations = anchor_displacements[self.knn_indices]  # [N, K, 3]
        
        # ================================================================
        # PhysX-Taylor: First-Order Taylor Expansion
        # ================================================================
        if self.use_taylor and self._last_anchor_affines is not None:
            # Get affine matrices: [M, 3, 3] -> [N, K, 3, 3]
            anchor_affines = self._last_anchor_affines
            neighbor_affines = anchor_affines[self.knn_indices]  # [N, K, 3, 3]
            
            # Get anchor positions: [M, 3] -> [N, K, 3]
            anchor_pos = self.anchor_positions.detach()
            neighbor_anchor_pos = anchor_pos[self.knn_indices]  # [N, K, 3]
            
            # Compute relative coordinates: δ = x_point - x_anchor_k
            # gaussian_positions: [N, 3] -> [N, 1, 3] -> broadcast to [N, K, 3]
            delta = gaussian_positions.unsqueeze(1) - neighbor_anchor_pos  # [N, K, 3]
            
            # Apply affine transformation: A_k · δ
            # neighbor_affines: [N, K, 3, 3], delta: [N, K, 3]
            # We need: [N, K, 3, 3] @ [N, K, 3, 1] -> [N, K, 3, 1] -> [N, K, 3]
            delta_affine = torch.matmul(
                neighbor_affines,  # [N, K, 3, 3]
                delta.unsqueeze(-1)  # [N, K, 3, 1]
            ).squeeze(-1)  # [N, K, 3]
            
            # Total per-neighbor contribution: t_k + A_k · δ
            neighbor_total = neighbor_translations + delta_affine  # [N, K, 3]
            
            # Weighted sum: Δx = Σ w_pk * (t_k + A_k · δ)
            gaussian_displacements = torch.sum(
                neighbor_total * self.knn_weights.unsqueeze(-1),
                dim=1
            )  # [N, 3]
        else:
            # Zero-order approximation (original): Δx = Σ w_pk * t_k
            gaussian_displacements = torch.sum(
                neighbor_translations * self.knn_weights.unsqueeze(-1),
                dim=1
            )  # [N, 3]
        
        return gaussian_displacements
    
    def forward(
        self,
        gaussian_positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        density: torch.Tensor,
        time_emb: torch.Tensor,
        is_training: bool = True,
        iteration_ratio: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass: compute deformed Gaussian positions.
        
        This interface is compatible with the original Deformation class.
        
        PhysX-Boosted Mode:
            Δμ_total = Δμ_hexplane(t) + Δμ_anchor(t)
            - HexPlane: "Paint the skin" (baseline capability, driven by L_render)
            - Anchor: "Draw the skeleton" (physical robustness, driven by L_render + L_phys)
        
        Args:
            gaussian_positions: Canonical Gaussian centers [N, 3]
            scales: Gaussian scales [N, 3]
            rotations: Gaussian rotations [N, 4]
            density: Gaussian densities [N, 1]
            time_emb: Time value [N, 1]
            is_training: If True, apply masking
            iteration_ratio: Current iteration / total iterations (0.0 to 1.0)
                            Used for mask decay scheduler
        
        Returns:
            deformed_positions: Deformed Gaussian centers [N, 3]
            deformed_scales: Deformed scales [N, 3]
            deformed_rotations: Deformed rotations [N, 4]
        """
        if not self.initialized:
            # Fallback: return original positions if not initialized
            print("[PhysX-Gaussian] Warning: Anchors not initialized, returning original positions")
            return gaussian_positions, scales, rotations
        
        # ================================================================
        # PhysX-Boosted: Full HexPlane Baseline + Anchor Correction
        # "100% Baseline capability + 10% physical robustness = SOTA"
        # ================================================================
        if self.use_boosted and self.original_deformation is not None:
            # Step 1: HexPlane baseline deformation (full-power)
            # deform_network.forward returns (means3D, scales, rotations)
            # where means3D = positions + dx_hex (already added inside)
            means3D_hex, scales_hex, rotations_hex = self.original_deformation(
                gaussian_positions,  # [N, 3]
                scales,  # [N, 3]
                rotations,  # [N, 4]
                density,  # [N, 1]
                time_emb  # [N, 1]
            )
            
            # Extract HexPlane displacement: dx_hex = means3D_hex - positions
            dx_hex = means3D_hex - gaussian_positions
            ds_hex = scales_hex - scales
            dr_hex = rotations_hex - rotations
            
            # Cache HexPlane outputs for loss computation
            self._last_dx_hex = dx_hex
            self._last_ds_hex = ds_hex
            self._last_dr_hex = dr_hex
            
            # Step 2: Anchor displacement (physical skeleton correction)
            anchor_displacements = self.forward_anchors(
                time_emb, is_training=is_training, iteration_ratio=iteration_ratio
            )
            dx_anchor = self.interpolate_displacements(anchor_displacements, gaussian_positions)
            
            # Cache anchor displacement
            self._last_dx_anchor = dx_anchor
            
            # Step 3: Combine displacements
            if self.use_uncertainty_fusion:
                # ================================================================
                # V7: Uncertainty-Aware Fusion (Aleatoric Uncertainty)
                # Both branches output displacement + log(σ²)
                # Fusion uses inverse variance weighting
                # ================================================================
                
                # Get anchor features from cache (computed in forward_anchors)
                anchor_features = self._last_anchor_features  # [M, d_model]
                
                # Compute anchor uncertainty: log(σ²_anchor) per anchor
                anchor_log_var = self.anchor_uncertainty_head(anchor_features)  # [M, 1]
                # Interpolate to Gaussians
                log_var_anchor = self.interpolate_displacements(
                    anchor_log_var, gaussian_positions
                ).squeeze(-1)  # [N]
                
                # HexPlane uncertainty: use learned head on dx_hex magnitude
                # Simple proxy: larger displacement -> more uncertainty
                log_var_hex = self.hex_uncertainty_head(dx_hex)  # [N, 1]
                log_var_hex = log_var_hex.squeeze(-1)  # [N]
                
                # Convert log(σ²) to σ² and compute weights
                var_hex = torch.exp(log_var_hex) + self.uncertainty_eps  # [N]
                var_anchor = torch.exp(log_var_anchor) + self.uncertainty_eps  # [N]
                
                w_hex = 1.0 / var_hex  # [N]
                w_anchor = 1.0 / var_anchor  # [N]
                w_total = w_hex + w_anchor  # [N]
                
                # Normalize weights
                w_hex_norm = (w_hex / w_total).unsqueeze(-1)  # [N, 1]
                w_anchor_norm = (w_anchor / w_total).unsqueeze(-1)  # [N, 1]
                
                # Inverse variance weighted fusion
                dx_combined = w_hex_norm * dx_hex + w_anchor_norm * dx_anchor
                ds_combined = w_hex_norm * ds_hex  # Anchor doesn't modify scales
                dr_combined = w_hex_norm * dr_hex  # Anchor doesn't modify rotations
                
                # Cache for loss computation
                self._last_log_var_hex = log_var_hex
                self._last_log_var_anchor = log_var_anchor
                self._last_weight_hex = w_hex_norm.mean().item()
                self._last_weight_anchor = w_anchor_norm.mean().item()
                self._last_balance_alpha = None
                
            elif self.fusion_mode == 'uncertainty_gated':
                # ================================================================
                # M1.2: Small Perturbation around V5's Optimal 99:1 Ratio
                # ================================================================
                # KEY INSIGHT from experiments:
                #   - V5 with α=0.99 (99% Anchor, 1% HexPlane) is OPTIMAL
                #   - Any other ratio is worse
                #   - This means 99:1 gradient ratio is also optimal for training!
                #
                # M1.0/M1.1 FAILURES:
                #   - M1.0: β stayed at 0.01, no adaptation (just V5)
                #   - M1.1: Gradient decoupling broke training dynamics
                #
                # M1.2 SOLUTION: Keep V5's gradient flow, add TINY perturbation γ
                #   dx = (0.99 - γ) * dx_anchor + (0.01 + γ) * dx_hex
                #   where |γ| ≤ γ_max (default 0.005, i.e., ±0.5% adjustment)
                #
                # γ is computed from uncertainty s_E:
                #   γ = γ_max * tanh((τ - s_E) / λ)
                #   - s_E high (uncertain) → γ < 0 → reduce HexPlane to 0.5%
                #   - s_E low (confident) → γ > 0 → increase HexPlane to 1.5%
                # ================================================================
                
                # Get s_E from HexPlane uncertainty head
                s_E = self.original_deformation.get_last_s_E()  # [N, 1]
                
                if s_E is None:
                    # Fallback: if s_E not computed, use pure V5 (γ=0)
                    gamma = torch.zeros_like(dx_hex[:, :1])
                else:
                    # Compute γ from s_E: small perturbation around 0
                    # γ = γ_max * tanh((τ - s_E) / λ)
                    gamma = self.gamma_max * torch.tanh((self.gate_tau - s_E) / (self.gate_lambda + 1e-8))
                
                # Cache for logging
                self._last_s_E = s_E
                self._last_gamma = gamma
                hex_weight_effective = 0.01 + gamma
                self._last_beta = hex_weight_effective  # For compatibility with logging
                self._last_beta_mean = hex_weight_effective.mean().item()
                
                # M1.2 Fusion: V5 baseline (α=0.99) + small perturbation γ
                # dx = (0.99 - γ) * dx_anchor + (0.01 + γ) * dx_hex
                alpha_base = 0.99  # V5's optimal ratio - DO NOT CHANGE
                hex_weight = (1 - alpha_base) + gamma    # 0.01 + γ
                anchor_weight = alpha_base - gamma       # 0.99 - γ
                
                dx_combined = anchor_weight * dx_anchor + hex_weight * dx_hex
                ds_combined = hex_weight * ds_hex  # Scale from HexPlane only
                dr_combined = hex_weight * dr_hex  # Rotation from HexPlane only
                
                self._last_balance_alpha = None
            
            elif self.fusion_mode == 'bounded_perturb':
                # ================================================================
                # M2.1: Learnable Weighted Average + Trust-Region Schedule
                # ================================================================
                # M2.05 formula: dx = (1-ε)*dx_anchor + ε*dx_hex
                #
                # M2.1 adds trust-region schedule to prevent early shortcuts:
                #   schedule_mode="none"       → M2.05 behavior
                #   schedule_mode="freeze_rho" → ρ frozen for first N steps
                #   schedule_mode="warmup_cap" → ε_eff = min(ε_raw, ε_max * warmup_ratio)
                # ================================================================
                
                # Step 1: Compute raw ε = ε_max * sigmoid(ρ)
                eps_raw = self.eps_max * torch.sigmoid(self.rho)
                self._last_eps_raw = eps_raw.item()
                
                # Step 2: Apply trust-region schedule to get ε_eff
                # Note: global_step is passed via self._current_step (set by train.py)
                current_step = getattr(self, '_current_step', 0)
                
                if self.schedule_mode == 'none':
                    # M2.05 behavior: no schedule
                    eps_eff = eps_raw
                    self._last_warmup_ratio = 1.0
                    self._is_frozen = False
                    
                elif self.schedule_mode == 'freeze_rho':
                    # Hard freeze: ε stays at eps_raw, but ρ gradients are zeroed in train.py
                    # Here we just use eps_raw as-is
                    eps_eff = eps_raw
                    self._is_frozen = (current_step < self.freeze_steps)
                    self._last_warmup_ratio = 0.0 if self._is_frozen else 1.0
                    
                elif self.schedule_mode == 'warmup_cap':
                    # Soft cap: ε_eff = min(ε_raw, ε_max * warmup_ratio)
                    warmup_ratio = min(current_step / max(self.warmup_steps, 1), 1.0)
                    eps_cap = self.eps_max * warmup_ratio
                    eps_eff = torch.min(eps_raw, torch.tensor(eps_cap, device=eps_raw.device))
                    self._last_warmup_ratio = warmup_ratio
                    self._is_frozen = False
                else:
                    # Unknown mode, fallback to no schedule
                    eps_eff = eps_raw
                    self._last_warmup_ratio = 1.0
                    self._is_frozen = False
                
                self._last_eps_eff = eps_eff.item() if isinstance(eps_eff, torch.Tensor) else eps_eff
                self._last_eps = self._last_eps_eff  # Backward compat
                
                # ================================================================
                # M2.2: Apply H(Δ) normalization to Eulerian residuals
                # ================================================================
                # "Residual normalization makes ε a true trust-region radius by
                #  preventing magnitude leakage from the Eulerian stream."
                
                # Compute norms before normalization (for logging)
                with torch.no_grad():
                    norm_E_dx = torch.norm(dx_hex, dim=-1).mean().item()
                    self._last_mean_norm_E = norm_E_dx
                
                # M3: Cache raw Δ and positions for LP regularization
                # Note: We cache before H(·) to preserve frequency structure
                self._last_delta_raw = dx_hex  # [N, 3]
                self._last_positions = gaussian_positions.detach()  # [N, 3]
                
                # M4: Cache deformations for decoupling computation
                self._last_dx_anchor = dx_anchor  # Lagrangian [N, 3]
                self._last_dx_hex = dx_hex        # Eulerian raw [N, 3]
                self._last_time = time_emb if time_emb is not None else None
                
                # Apply H(·) based on residual_mode
                if self.residual_mode == 'tanh':
                    # M2/M2.1 baseline: H(Δ) = tanh(Δ)
                    dx_H = torch.tanh(dx_hex)
                    ds_H = torch.tanh(ds_hex)
                    dr_H = torch.tanh(dr_hex)
                    
                elif self.residual_mode == 'rmsnorm':
                    # M2.2: RMS normalization per point
                    # rms = sqrt(mean(Δ^2, dim=-1) + eps)
                    # H(Δ) = Δ / rms
                    rms_dx = torch.sqrt(torch.mean(dx_hex ** 2, dim=-1, keepdim=True) + self.norm_eps)
                    dx_H = dx_hex / rms_dx
                    
                    rms_ds = torch.sqrt(torch.mean(ds_hex ** 2, dim=-1, keepdim=True) + self.norm_eps)
                    ds_H = ds_hex / rms_ds
                    
                    rms_dr = torch.sqrt(torch.mean(dr_hex ** 2, dim=-1, keepdim=True) + self.norm_eps)
                    dr_H = dr_hex / rms_dr
                    
                elif self.residual_mode == 'unitnorm':
                    # M2.2: L2 unit normalization per point
                    # n = sqrt(sum(Δ^2, dim=-1) + eps)
                    # H(Δ) = Δ / n
                    norm_dx = torch.sqrt(torch.sum(dx_hex ** 2, dim=-1, keepdim=True) + self.norm_eps)
                    dx_H = dx_hex / norm_dx
                    
                    norm_ds = torch.sqrt(torch.sum(ds_hex ** 2, dim=-1, keepdim=True) + self.norm_eps)
                    ds_H = ds_hex / norm_ds
                    
                    norm_dr = torch.sqrt(torch.sum(dr_hex ** 2, dim=-1, keepdim=True) + self.norm_eps)
                    dr_H = dr_hex / norm_dr
                    
                else:
                    # Unknown mode, fallback to tanh
                    dx_H = torch.tanh(dx_hex)
                    ds_H = torch.tanh(ds_hex)
                    dr_H = torch.tanh(dr_hex)
                
                # Compute norms after normalization (for logging)
                with torch.no_grad():
                    norm_H_dx = torch.norm(dx_H, dim=-1).mean().item()
                    self._last_mean_norm_H = norm_H_dx
                
                # Step 3: Weighted average fusion with normalized residuals
                alpha = 1.0 - eps_eff
                
                dx_combined = alpha * dx_anchor + eps_eff * dx_H
                ds_combined = eps_eff * ds_H  # Scale from HexPlane only
                dr_combined = eps_eff * dr_H  # Rotation from HexPlane only
                
                self._last_balance_alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
            
            elif self.use_orthogonal_projection:
                # ================================================================
                # V6: Orthogonal Gradient Projection
                # Forward: Δx_total = Δx_hex + Δx_anchor (direct sum)
                # Backward: Modify Anchor's gradient to be orthogonal to HexPlane's gradient
                # ================================================================
                # Cache dx_hex for the gradient hook (detached to avoid graph issues)
                self._cached_dx_hex_for_grad = dx_hex.detach().clone()
                
                # Create a custom autograd function to apply orthogonal projection
                dx_anchor_projected = self._apply_orthogonal_projection_hook(dx_anchor, dx_hex)
                
                dx_combined = dx_hex + dx_anchor_projected
                ds_combined = ds_hex
                dr_combined = dr_hex
                self._last_balance_alpha = None  # Not using alpha in V6
            
            elif self.use_reverse_orthogonal_projection:
                # ================================================================
                # V8/V8.1: Reverse Orthogonal Gradient Projection (swap A and B)
                # - Anchor (A) is the "base" that learns easily-captured patterns
                # - HexPlane (B) is constrained to learn only the residual
                # V8:   Forward: Δx_total = Δx_hex + Δx_anchor (direct sum)
                # V8.1: Forward: Δx_total = (1-α)·Δx_hex + α·Δx_anchor (weighted sum)
                # Backward: Modify HexPlane's gradient to be orthogonal to Anchor's gradient
                # ================================================================
                # Apply reverse orthogonal projection: HexPlane learns residual of Anchor
                dx_hex_projected = ReverseOrthogonalGradientProjection.apply(
                    dx_hex, 
                    dx_anchor.detach(),  # Use Anchor's displacement as projection direction
                    self.ortho_projection_strength
                )
                
                # V8.1: If learnable_balance is also enabled, use alpha weighting
                if self.use_learnable_balance and not self.use_pure_hexplane and not self.use_pure_anchor:
                    alpha = torch.sigmoid(self.balance_logit)
                    self._last_balance_alpha = alpha.item()
                    dx_combined = (1 - alpha) * dx_hex_projected + alpha * dx_anchor
                    ds_combined = (1 - alpha) * ds_hex
                    dr_combined = (1 - alpha) * dr_hex
                else:
                    # V8: Direct sum (no weighting)
                    dx_combined = dx_hex_projected + dx_anchor
                    ds_combined = ds_hex
                    dr_combined = dr_hex
                    self._last_balance_alpha = None
                
            elif self.use_learnable_balance:
                # V5: Learnable balance - Δx_total = (1-α)·Δx_hex + α·Δx_anchor
                # Handle extreme cases for exact α=0 or α=1
                if self.use_pure_hexplane:
                    # α=0: Pure HexPlane, no Anchor contribution
                    alpha = 0.0
                    dx_combined = dx_hex
                    ds_combined = ds_hex
                    dr_combined = dr_hex
                elif self.use_pure_anchor:
                    # α=1: Pure Anchor, no HexPlane contribution
                    alpha = 1.0
                    dx_combined = dx_anchor
                    ds_combined = torch.zeros_like(ds_hex)  # Anchor doesn't modify scales
                    dr_combined = torch.zeros_like(dr_hex)  # Anchor doesn't modify rotations
                else:
                    # Normal case: use sigmoid for smooth interpolation
                    alpha = torch.sigmoid(self.balance_logit)  # α ∈ (0, 1)
                    dx_combined = (1 - alpha) * dx_hex + alpha * dx_anchor
                    ds_combined = (1 - alpha) * ds_hex
                    dr_combined = (1 - alpha) * dr_hex
                    alpha = alpha.item()
                self._last_balance_alpha = alpha  # Cache for logging
            else:
                # Original: Δμ_total = Δμ_hexplane + Δμ_anchor (direct sum)
                dx_combined = dx_hex + dx_anchor
                ds_combined = ds_hex
                dr_combined = dr_hex
            
            deformed_positions = gaussian_positions + dx_combined
            deformed_scales = scales + ds_combined
            deformed_rotations = rotations + dr_combined
            
            return deformed_positions, deformed_scales, deformed_rotations
        
        # ================================================================
        # Non-Boosted Mode: Anchor-only (original PhysX-Gaussian behavior)
        # ================================================================
        
        # Step 1: Anchor displacement (physical skeleton - 95% of motion)
        anchor_displacements = self.forward_anchors(time_emb, is_training=is_training, iteration_ratio=iteration_ratio)
        
        # Interpolate to Gaussian positions via KNN skinning
        gaussian_dx_anchor = self.interpolate_displacements(anchor_displacements, gaussian_positions)
        
        # ================================================================
        # Step 2: Residual displacement (neural skin - 5% micro-corrections)
        # Only active in Hybrid mode
        # ================================================================
        gaussian_dx_residual = None
        self._last_residual_magnitude = None
        
        if self.use_hybrid and hasattr(self, 'residual_hexplane'):
            # Get time value (scalar) for HexPlane query
            if time_emb.dim() > 1:
                t = time_emb[0, 0] if time_emb.shape[1] > 0 else time_emb[0]
            elif time_emb.dim() == 1:
                t = time_emb[0]
            else:
                t = time_emb
            
            # Prepare time tensor for all Gaussians
            N = gaussian_positions.shape[0]
            time_tensor = t.expand(N, 1)  # [N, 1]
            
            # Query HexPlane features at Gaussian positions + time
            # HexPlane expects normalized coordinates in [-1, 1]
            residual_features = self.residual_hexplane(
                gaussian_positions,  # [N, 3]
                time_tensor  # [N, 1]
            )  # [N, residual_dim]
            
            # Decode to displacement
            gaussian_dx_residual = self.residual_mlp(residual_features)  # [N, 3]
            
            # Cache residual magnitude for L1 regularization loss
            self._last_residual_magnitude = gaussian_dx_residual.abs().mean()
        
        # ================================================================
        # Step 3: Combine displacements
        # Δx_total = Δx_anchor + Δx_residual (if hybrid mode)
        # ================================================================
        if gaussian_dx_residual is not None:
            gaussian_dx_total = gaussian_dx_anchor + gaussian_dx_residual
        else:
            gaussian_dx_total = gaussian_dx_anchor
        
        # Apply total displacement
        deformed_positions = gaussian_positions + gaussian_dx_total
        
        # For now, scales and rotations use simple interpolation from anchor features
        # This can be extended to full anchor-based deformation later
        deformed_scales = scales  # Keep original for now
        deformed_rotations = rotations  # Keep original for now
        
        return deformed_positions, deformed_scales, deformed_rotations
    
    def forward_backward_position(
        self,
        deformed_pts: torch.Tensor,
        time_emb: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute backward deformation (for inverse consistency).
        
        Args:
            deformed_pts: Deformed positions [N, 3]
            time_emb: Time value [N, 1]
        
        Returns:
            reconstructed_pts: Reconstructed canonical positions [N, 3]
            backward_deform: Backward displacement [N, 3]
        """
        if not self.initialized:
            zeros = torch.zeros_like(deformed_pts)
            return deformed_pts, zeros
        
        # Compute anchor displacements using backward head
        device = self.anchor_positions.device
        M = self.anchor_positions.shape[0]
        
        # Get time value
        if time_emb.dim() > 0:
            t = time_emb[0, 0] if time_emb.dim() == 2 else time_emb[0]
        else:
            t = time_emb
        
        # Embed anchor positions (using deformed positions for backward)
        pos_embed = self.anchor_embed(self.anchor_positions)
        time_embed = self.time_encode(t.unsqueeze(0)).expand(M, -1)
        
        anchor_input = torch.cat([pos_embed, time_embed], dim=-1)
        anchor_features = self.input_proj(anchor_input).unsqueeze(0)
        anchor_features = self.transformer(anchor_features)
        
        # Backward displacement
        backward_anchor_dx = self.displacement_head_backward(anchor_features).squeeze(0)
        
        # Interpolate to Gaussian positions
        backward_dx = self.interpolate_displacements(backward_anchor_dx, deformed_pts)
        
        reconstructed_pts = deformed_pts + backward_dx
        
        return reconstructed_pts, backward_dx
    
    def compute_physics_completion_loss(self) -> torch.Tensor:
        """
        Compute physics completion loss L_phys.
        
        This loss encourages the network to predict correct displacements
        for masked anchors by comparing with teacher-forced (unmasked) predictions.
        
        L_phys = || D_masked - D_teacher ||_1
        
        where D_masked are displacements predicted for masked anchors,
        and D_teacher are displacements from unmasked forward pass.
        
        V10 (use_decoupled_mask=True):
        - Uses _last_masked_displacements from forward_anchors_masked()
        - Uses _last_unmasked_displacements from forward_anchors_unmasked()
        
        Original mode:
        - Uses _last_anchor_displacements (which was masked during render)
        - Uses _last_unmasked_displacements from forward_anchors_unmasked()
        
        Returns:
            loss: Physics completion loss (scalar)
        """
        if self._last_masked_indices is None:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        if self._last_unmasked_displacements is None:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        masked_idx = self._last_masked_indices
        
        # V10: Use separately computed masked displacements
        if self.use_decoupled_mask:
            if not hasattr(self, '_last_masked_displacements') or self._last_masked_displacements is None:
                return torch.tensor(0.0, device=self.anchor_positions.device)
            masked_pred = self._last_masked_displacements[masked_idx]  # [num_mask, 3]
        else:
            # Original mode: masked predictions come from main forward (render path)
            if self._last_anchor_displacements is None:
                return torch.tensor(0.0, device=self.anchor_positions.device)
            masked_pred = self._last_anchor_displacements[masked_idx]  # [num_mask, 3]
        
        # Get teacher displacements from unmasked forward
        teacher_pred = self._last_unmasked_displacements[masked_idx].detach()  # [num_mask, 3]
        
        # L1 loss
        loss = F.l1_loss(masked_pred, teacher_pred)
        
        return loss
    
    def compute_anchor_smoothness_loss(self) -> torch.Tensor:
        """
        Compute anchor motion smoothness loss.
        
        This regularizes anchor displacements to be spatially smooth
        by penalizing large differences between neighboring anchors.
        
        Returns:
            loss: Anchor smoothness loss (scalar)
        """
        if self._last_anchor_displacements is None:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        dx = self._last_anchor_displacements  # [M, 3]
        M = dx.shape[0]
        
        if M < 2:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        # Compute pairwise distances between anchors (detach to avoid graph issues)
        anchor_pos = self.anchor_positions.detach()
        anchor_dists = torch.cdist(anchor_pos, anchor_pos)  # [M, M]
        
        # Get k nearest neighbors for each anchor
        k = min(8, M - 1)
        _, neighbor_idx = torch.topk(-anchor_dists, k + 1, dim=-1)  # [M, k+1]
        neighbor_idx = neighbor_idx[:, 1:]  # Exclude self, [M, k]
        
        # Compute displacement differences to neighbors
        neighbor_dx = dx[neighbor_idx]  # [M, k, 3]
        dx_diff = dx.unsqueeze(1) - neighbor_dx  # [M, k, 3]
        
        # Smoothness loss: penalize large displacement differences
        loss = (dx_diff ** 2).sum(dim=-1).mean()
        
        return loss
    
    def compute_consistency_loss(self, time_emb: torch.Tensor) -> torch.Tensor:
        """
        V13: Compute consistency regularization loss.
        
        This loss encourages the model to give consistent outputs
        even when some anchors are masked.
        
        L_consist = ||masked_out - unmasked_out.detach()||
        
        Key insight: The gradient only flows through the masked branch.
        The unmasked branch provides a stable target.
        This teaches the model to be ROBUST to missing information.
        
        Args:
            time_emb: Time value tensor
            
        Returns:
            loss: Consistency loss (scalar)
        """
        if not self.use_consistency_mask:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        if not self.initialized:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        device = self.anchor_positions.device
        M = self.anchor_positions.shape[0]
        
        # Get time value
        if time_emb.dim() > 0:
            t = time_emb[0, 0] if time_emb.dim() == 2 else time_emb[0]
        else:
            t = time_emb
        
        # Step 1: Forward WITHOUT mask (for consistency target)
        # Reuse cached unmasked output if available
        if self._last_unmasked_displacements is not None:
            unmasked_out = self._last_unmasked_displacements.detach()
        else:
            unmasked_out = self.forward_anchors_unmasked(time_emb).detach()
        
        # Step 2: Forward WITH mask (for consistency training)
        anchor_pos = self.anchor_positions.detach()
        pos_embed = self.anchor_embed(anchor_pos)
        time_embed = self.time_encode(t.unsqueeze(0)).expand(M, -1)
        
        anchor_input = torch.cat([pos_embed, time_embed], dim=-1)
        anchor_features = self.input_proj(anchor_input).unsqueeze(0)
        
        # Apply random masking
        num_mask = int(M * self.mask_ratio)
        if num_mask > 0:
            perm = torch.randperm(M, device=device)
            masked_indices = perm[:num_mask]
            mask_tokens = self.mask_token.expand(1, num_mask, -1)
            anchor_features[0, masked_indices] = mask_tokens.squeeze(0)
        
        # Transformer and prediction
        anchor_features = self.transformer(anchor_features)
        masked_out = self.displacement_head(anchor_features).squeeze(0)
        
        # Consistency loss: masked output should match unmasked output
        loss = F.l1_loss(masked_out, unmasked_out)
        
        return loss
    
    def compute_temporal_interp_loss(self, time_emb: torch.Tensor) -> torch.Tensor:
        """
        V14: Compute temporal smoothness loss (acceleration penalty).
        
        This loss encourages temporally smooth anchor motions by penalizing
        large accelerations (second-order derivative).
        
        L_temporal = ||dx(t+ε) - 2*dx(t) + dx(t-ε)||²
        
        This is equivalent to minimizing:
          acceleration = (dx(t+ε) - dx(t))/ε - (dx(t) - dx(t-ε))/ε
        
        Physical meaning: Anchors should move smoothly, not with sudden jerks.
        This is a strong physical prior for breathing motion.
        
        Args:
            time_emb: Current time value tensor
            
        Returns:
            loss: Temporal smoothness loss (scalar)
        """
        if not self.use_temporal_interp:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        if not self.initialized:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        device = self.anchor_positions.device
        
        # Get current time
        if time_emb.dim() > 0:
            t = time_emb[0, 0] if time_emb.dim() == 2 else time_emb[0]
        else:
            t = time_emb
        
        t_val = t.item()
        
        # Time step for finite difference
        epsilon = self.interp_context_range / 2
        t_prev_val = max(0.0, t_val - epsilon)
        t_next_val = min(1.0, t_val + epsilon)
        
        # Skip boundary cases where we can't compute acceleration
        if t_prev_val == t_val or t_next_val == t_val:
            return torch.tensor(0.0, device=device)
        
        t_prev = torch.tensor(t_prev_val, device=device)
        t_next = torch.tensor(t_next_val, device=device)
        
        # Get anchor motions at three time points
        # Current time uses cached value if available (with gradient)
        if self._last_anchor_displacements is not None:
            dx_t = self._last_anchor_displacements
        else:
            dx_t = self.forward_anchors_unmasked(time_emb)
        
        # Neighboring times: detach to prevent gradient explosion
        with torch.no_grad():
            dx_prev = self.forward_anchors_unmasked(t_prev.unsqueeze(0))
            dx_next = self.forward_anchors_unmasked(t_next.unsqueeze(0))
        
        # Second-order finite difference (acceleration)
        # If motion is linear: dx_next - 2*dx_t + dx_prev = 0
        # Penalize deviation from linear motion (i.e., acceleration)
        acceleration = dx_next - 2 * dx_t + dx_prev
        
        # L2 loss on acceleration
        loss = (acceleration ** 2).mean()
        
        return loss
    
    # ================================================================
    # V16: Lagrangian Spatio-Temporal Masked Anchor Modeling
    # ================================================================
    
    def forward_anchors_st(
        self,
        anchor_pos: torch.Tensor,
        t_vec: torch.Tensor,
        mask_flags: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        V16: Spatio-temporal forward pass over (anchor, time) tokens.
        
        This is the core of Lagrangian-BERT: process multiple time steps
        simultaneously, with optional mask flag embedding.
        
        Args:
            anchor_pos: Anchor positions [M, 3]
            t_vec: Time steps [K] (e.g., [t-Δ, t, t+Δ])
            mask_flags: Optional binary mask [K, M] where 1=masked, 0=unmasked
                       If None, all tokens are unmasked.
        
        Returns:
            displacements: [K, M, 3] displacements for each (time, anchor) pair
        """
        if not self.initialized:
            K = t_vec.shape[0]
            M = anchor_pos.shape[0]
            return torch.zeros(K, M, 3, device=anchor_pos.device)
        
        device = anchor_pos.device
        M = anchor_pos.shape[0]
        K = t_vec.shape[0]
        
        # Build spatio-temporal tokens: (anchor_j, time_k) for all j, k
        # Total tokens: K * M
        
        # 1. Position embedding for each anchor (shared across time)
        pos_embed = self.anchor_embed(anchor_pos.detach())  # [M, pos_dim]
        
        # 2. Time embedding for each time step
        time_embeds = []
        for k in range(K):
            t_k = t_vec[k]
            time_embed_k = self.time_encode(t_k.unsqueeze(0))  # [1, time_dim]
            time_embeds.append(time_embed_k)
        time_embeds = torch.cat(time_embeds, dim=0)  # [K, time_dim]
        
        # 3. Build token embeddings for all (anchor, time) pairs
        # Shape: [K, M, pos_dim + time_dim]
        tokens = []
        for k in range(K):
            # Combine pos + time for this time step
            time_k = time_embeds[k:k+1].expand(M, -1)  # [M, time_dim]
            token_k = torch.cat([pos_embed, time_k], dim=-1)  # [M, pos_dim + time_dim]
            tokens.append(token_k)
        tokens = torch.stack(tokens, dim=0)  # [K, M, pos_dim + time_dim]
        
        # 4. Project to d_model
        # Reshape for linear: [K*M, input_dim] -> [K*M, d_model]
        tokens_flat = tokens.reshape(K * M, -1)
        features_flat = self.input_proj(tokens_flat)  # [K*M, d_model]
        
        # 5. Add mask flag embedding (V16 key innovation!)
        # This ADDS to the token embedding, NOT replacing it
        # Fix 1: Apply scale factor to reduce interference (default 1.0 = original)
        if mask_flags is not None and self.use_spatiotemporal_mask:
            mask_flags_flat = mask_flags.reshape(K * M).long()  # [K*M]
            mask_embed = self.mask_flag_embed(mask_flags_flat)  # [K*M, d_model]
            # Scale down mask_embed to reduce its dominance over original features
            features_flat = features_flat + self.st_mask_embed_scale * mask_embed
        
        # 6. Transformer attention across all (anchor, time) tokens
        # Reshape for transformer: [1, K*M, d_model]
        features = features_flat.unsqueeze(0)  # [1, K*M, d_model]
        features = self.transformer(features)  # [1, K*M, d_model]
        features = features.squeeze(0)  # [K*M, d_model]
        
        # 7. Predict displacements
        displacements_flat = self.displacement_head(features)  # [K*M, 3]
        
        # 8. Reshape to [K, M, 3]
        displacements = displacements_flat.reshape(K, M, 3)
        
        return displacements
    
    def sample_time_window(self, t_center: float) -> torch.Tensor:
        """
        Sample a time window around the center time for V16.
        
        Args:
            t_center: Center time value
            
        Returns:
            t_vec: [K] time steps in the window
        """
        device = self.anchor_positions.device
        K = self.st_window_size
        delta = self.st_time_delta
        
        # Generate time steps: [t_center - (K//2)*δ, ..., t_center, ..., t_center + (K//2)*δ]
        half_K = K // 2
        t_vec = []
        for i in range(-half_K, K - half_K):
            t_i = t_center + i * delta
            t_i = max(0.0, min(1.0, t_i))  # Clamp to [0, 1]
            t_vec.append(t_i)
        
        return torch.tensor(t_vec, device=device, dtype=torch.float32)
    
    def sample_st_mask(self, K: int, M: int, device: torch.device) -> torch.Tensor:
        """
        Sample spatio-temporal mask for V16.
        
        Args:
            K: Number of time steps
            M: Number of anchors
            device: Device for tensor
            
        Returns:
            mask_flags: [K, M] binary mask (1=masked, 0=unmasked)
        """
        total_tokens = K * M
        num_mask = int(total_tokens * self.st_mask_ratio)
        
        # Random selection of (anchor, time) tokens to mask
        perm = torch.randperm(total_tokens, device=device)
        mask_flat = torch.zeros(total_tokens, device=device, dtype=torch.long)
        mask_flat[perm[:num_mask]] = 1
        
        mask_flags = mask_flat.reshape(K, M)
        return mask_flags
    
    def compute_lagbert_loss(
        self,
        t_center: torch.Tensor,
        is_training: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        V16: Compute Lagrangian-BERT loss and return center displacement.
        
        This method:
        1. Samples a time window around t_center
        2. Runs full pass (no mask) to get teacher displacements
        3. Runs masked pass to get student displacements
        4. Computes L_lagbert on masked tokens
        5. Returns center time displacement for rendering
        
        Args:
            t_center: Center time for this training step
            is_training: Whether in training mode
            
        Returns:
            dx_center: [M, 3] displacement at center time (for rendering)
            L_lagbert: Lagrangian-BERT loss (scalar)
        """
        if not self.use_spatiotemporal_mask or not self.initialized:
            # Fallback to simple forward
            dx = self.forward_anchors_unmasked(t_center)
            return dx, torch.tensor(0.0, device=self.anchor_positions.device)
        
        device = self.anchor_positions.device
        M = self.anchor_positions.shape[0]
        
        # Get center time value
        if t_center.dim() > 0:
            t_val = t_center[0, 0].item() if t_center.dim() == 2 else t_center[0].item()
        else:
            t_val = t_center.item()
        
        # 1. Sample time window
        t_vec = self.sample_time_window(t_val)  # [K]
        K = t_vec.shape[0]
        center_idx = K // 2  # Center time index
        
        # 2. Full pass (no mask) - teacher
        mask_full = torch.zeros(K, M, device=device, dtype=torch.long)
        dx_full = self.forward_anchors_st(self.anchor_positions, t_vec, mask_full)  # [K, M, 3]
        
        if not is_training:
            # Inference: just return center displacement
            dx_center = dx_full[center_idx]  # [M, 3]
            return dx_center, torch.tensor(0.0, device=device)
        
        # 3. Sample mask and run masked pass - student
        mask_flags = self.sample_st_mask(K, M, device)  # [K, M]
        dx_masked = self.forward_anchors_st(self.anchor_positions, t_vec, mask_flags)  # [K, M, 3]
        
        # 4. Cache for potential debugging
        self._last_st_full_out = dx_full.detach()
        self._last_st_masked_out = dx_masked
        self._last_st_mask_flags = mask_flags
        
        # 5. Compute L_lagbert: L1 loss only on masked positions
        # mask_flags: [K, M], dx: [K, M, 3]
        mask_3d = mask_flags.unsqueeze(-1).expand_as(dx_masked)  # [K, M, 3]
        masked_count = mask_flags.sum()
        
        if masked_count > 0:
            # Extract masked predictions and targets
            masked_pred = dx_masked[mask_3d == 1]  # [num_masked * 3]
            masked_target = dx_full.detach()[mask_3d == 1]  # [num_masked * 3]
            L_lagbert = F.l1_loss(masked_pred, masked_target)
        else:
            L_lagbert = torch.tensor(0.0, device=device)
        
        # 6. Return center displacement for rendering (from full pass)
        dx_center = dx_full[center_idx]  # [M, 3]
        
        # Also cache for other loss computations
        self._last_anchor_displacements = dx_center
        self._last_unmasked_displacements = dx_center.detach()
        
        # V16 Fix 2: Cache dx_center for coupled rendering
        # When st_coupled_render=True, forward_anchors() will return this instead of recomputing
        if self.st_coupled_render:
            self._st_coupled_dx_center = dx_center
        
        return dx_center, L_lagbert
    
    def get_mlp_parameters(self):
        """Return MLP parameters for optimizer (compatibility with original)."""
        params = []
        params.extend(self.anchor_embed.parameters())
        params.extend(self.time_encode.parameters())
        params.extend(self.input_proj.parameters())
        params.extend(self.displacement_head.parameters())
        params.extend(self.displacement_head_backward.parameters())
        params.extend(self.scale_head.parameters())
        params.extend(self.rotation_head.parameters())
        params.append(self.mask_token)
        # V16: Include mask flag embedding if enabled
        if self.use_spatiotemporal_mask:
            params.extend(self.mask_flag_embed.parameters())
        return params
    
    def get_grid_parameters(self):
        """Return transformer parameters (equivalent to 'grid' in original)."""
        return self.transformer.parameters()
    
    # ================================================================
    # PhysX-Hybrid: Residual network methods
    # ================================================================
    
    def get_residual_magnitude(self) -> torch.Tensor:
        """
        Get the cached residual displacement magnitude for L1 regularization.
        
        Returns:
            magnitude: Mean absolute residual displacement (scalar)
        """
        if self._last_residual_magnitude is not None:
            return self._last_residual_magnitude
        return torch.tensor(0.0, device=self.anchor_positions.device)
    
    def get_residual_parameters(self):
        """
        Return residual network parameters for optimizer.
        
        These are separated so they can have their own learning rate schedule
        and warmup behavior.
        
        Returns:
            List of parameters (or empty list if hybrid not enabled)
        """
        if not self.use_hybrid:
            return []
        
        params = []
        if hasattr(self, 'residual_hexplane'):
            params.extend(self.residual_hexplane.parameters())
        if hasattr(self, 'residual_mlp'):
            params.extend(self.residual_mlp.parameters())
        return params
    
    def set_residual_aabb(self, xyz_max, xyz_min):
        """
        Set the AABB (axis-aligned bounding box) for residual HexPlane.
        
        This should be called after scene initialization to match the scene bounds.
        
        Args:
            xyz_max: Maximum coordinates [3]
            xyz_min: Minimum coordinates [3]
        """
        if self.use_hybrid and hasattr(self, 'residual_hexplane'):
            self.residual_hexplane.set_aabb(xyz_max, xyz_min)
            print(f"[PhysX-Hybrid] Residual HexPlane AABB set to [{xyz_min}, {xyz_max}]")
    
    # ================================================================
    # M1: Uncertainty-Gated Fusion Methods
    # ================================================================
    
    def compute_gate_sparsity_loss(self) -> torch.Tensor:
        """
        M1: Compute gate sparsity loss L_gate = E[|β(x,t)|].
        
        This MDL-style regularization encourages the model to prefer
        Lagrangian when possible (minimize Eulerian contribution).
        
        Paper notation:
            L_gate = E_{x,t}[|β(x,t)|₁]
        
        Returns:
            loss: Mean β value (scalar), or 0 if β not computed
        """
        if self.fusion_mode != 'uncertainty_gated':
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        if self._last_beta is None:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        # L1 sparsity: encourage β → 0 (prefer Lagrangian)
        loss = self._last_beta.abs().mean()
        return loss
    
    def compute_uncertainty_supervision_loss(self) -> torch.Tensor:
        """
        M1.1: Compute uncertainty supervision loss (NLL-style).
        
        This loss encourages the uncertainty s_E to be calibrated:
        - High HexPlane residual → high s_E (uncertain)
        - Low HexPlane residual → low s_E (confident)
        
        Formula:
            L_unc = 0.5 * exp(-s_E) * ||dx_hex||² + 0.5 * s_E
        
        This is the negative log-likelihood of a Gaussian with variance exp(s_E).
        
        Returns:
            loss: Uncertainty supervision loss (scalar)
        """
        if self.fusion_mode != 'uncertainty_gated':
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        if self._last_s_E is None or self._last_dx_hex is None:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        s_E = self._last_s_E  # [N, 1]
        dx_hex = self._last_dx_hex  # [N, 3]
        
        # Compute squared magnitude of HexPlane displacement
        dx_hex_sq = (dx_hex ** 2).sum(dim=-1, keepdim=True)  # [N, 1]
        
        # NLL loss: 0.5 * exp(-s_E) * ||dx||² + 0.5 * s_E
        # This encourages: large ||dx|| → large s_E, small ||dx|| → small s_E
        loss = 0.5 * torch.exp(-s_E) * dx_hex_sq + 0.5 * s_E
        
        return loss.mean()
    
    def get_last_beta(self) -> Optional[torch.Tensor]:
        """
        Get the last computed gate value β(x,t).
        
        Returns:
            beta: Tensor [N, 1] of gate values, or None if not computed
        """
        return self._last_beta
    
    def get_last_beta_mean(self) -> Optional[float]:
        """
        Get the mean of the last computed gate value β.
        
        Useful for logging without holding onto large tensors.
        
        Returns:
            beta_mean: Scalar mean β value, or None if not computed
        """
        return self._last_beta_mean
    
    def set_current_step(self, step: int) -> None:
        """
        M2.1: Set the current training step for trust-region schedule.
        Called by train.py at each iteration.
        
        Args:
            step: Global training step (iteration number)
        """
        self._current_step = step
    
    def get_last_eps(self) -> Optional[float]:
        """
        M2: Get the last computed ε value (effective, after schedule).
        
        Returns:
            eps: Scalar ε value, or None if not computed
        """
        return self._last_eps
    
    def get_m2_statistics(self) -> dict:
        """
        M2.2: Get bounded perturbation statistics for logging.
        
        Returns:
            Dictionary with:
                - eps_raw: ε before schedule (ε_max * sigmoid(ρ))
                - eps_eff: ε after schedule (actual value used)
                - eps_max: Maximum ε bound
                - rho: Current ρ parameter value
                - warmup_ratio: Current warmup ratio (for warmup_cap mode)
                - is_frozen: Whether ρ is currently frozen
                - schedule_mode: Current schedule mode
                - residual_mode: H(Δ) normalization mode (M2.2)
                - mean_norm_E: Mean ||Δ|| before normalization (M2.2)
                - mean_norm_H: Mean ||H(Δ)|| after normalization (M2.2)
        """
        if self.fusion_mode != 'bounded_perturb' or not hasattr(self, 'rho'):
            return {}
        
        return {
            'eps_raw': self._last_eps_raw,
            'eps_eff': self._last_eps_eff,
            'eps_max': self.eps_max,
            'rho': self.rho.item() if self.rho is not None else None,
            'warmup_ratio': self._last_warmup_ratio,
            'is_frozen': self._is_frozen,
            'schedule_mode': self.schedule_mode,
            'current_step': getattr(self, '_current_step', 0),
            'residual_mode': self.residual_mode,
            'mean_norm_E': self._last_mean_norm_E,
            'mean_norm_H': self._last_mean_norm_H
        }
    
    def should_freeze_rho(self) -> bool:
        """
        M2.1: Check if ρ should be frozen at current step.
        Used by train.py to zero gradients.
        
        Returns:
            True if ρ should be frozen, False otherwise
        """
        if self.fusion_mode != 'bounded_perturb':
            return False
        if self.schedule_mode != 'freeze_rho':
            return False
        current_step = getattr(self, '_current_step', 0)
        return current_step < self.freeze_steps
    
    def get_m1_statistics(self) -> dict:
        """
        Get M1 fusion statistics for logging/visualization.
        
        Returns:
            Dictionary with:
                - beta_mean: Mean gate value
                - beta_min: Min gate value
                - beta_max: Max gate value  
                - beta_std: Std of gate values
                - s_E_mean: Mean Eulerian log-variance (if available)
        """
        stats = {}
        
        if self._last_beta is not None:
            beta = self._last_beta.detach()
            stats['beta_mean'] = beta.mean().item()
            stats['beta_min'] = beta.min().item()
            stats['beta_max'] = beta.max().item()
            stats['beta_std'] = beta.std().item()
        else:
            stats['beta_mean'] = None
            stats['beta_min'] = None
            stats['beta_max'] = None
            stats['beta_std'] = None
        
        # Get s_E from HexPlane if available
        if self.use_boosted and self.original_deformation is not None:
            s_E = self.original_deformation.get_last_s_E()
            if s_E is not None:
                stats['s_E_mean'] = s_E.detach().mean().item()
            else:
                stats['s_E_mean'] = None
        else:
            stats['s_E_mean'] = None
        
        return stats
    
    def get_uncertainty_parameters(self):
        """
        Get uncertainty head parameters from the Eulerian network.
        
        These should be added to optimizer with appropriate learning rate.
        
        Returns:
            List of parameters (or empty list if not in boosted mode)
        """
        if not self.use_boosted or self.original_deformation is None:
            return []
        return self.original_deformation.get_uncertainty_parameters()
    
    # ================================================================
    # M3: Low-Frequency Leakage Penalty Methods
    # ================================================================
    
    def compute_lp_loss(self) -> torch.Tensor:
        """
        M3: Compute Low-Frequency Leakage Penalty.
        
        "Low-frequency leakage regularization prevents the Eulerian stream
         from explaining global motion, reserving it for high-frequency
         corrective details around the Lagrangian manifold."
        
        L_LP = mean_i || LP(Δ_i) ||^2
        
        Returns:
            L_LP loss tensor (scalar)
        """
        if not self.lp_enable:
            return torch.tensor(0.0, device='cuda')
        
        if self._last_delta_raw is None or self._last_positions is None:
            return torch.tensor(0.0, device='cuda')
        
        delta = self._last_delta_raw  # [N, 3]
        positions = self._last_positions  # [N, 3]
        
        N = delta.shape[0]
        device = delta.device
        
        # Subsample for efficiency
        if self.lp_subsample > 0 and N > self.lp_subsample:
            indices = torch.randperm(N, device=device)[:self.lp_subsample]
            delta_sub = delta[indices]  # [M, 3]
            positions_sub = positions[indices]  # [M, 3]
        else:
            delta_sub = delta
            positions_sub = positions
        
        M = delta_sub.shape[0]
        
        if self.lp_mode == 'knn_mean':
            # LP-1: kNN mean
            # LP(Δ_i) = mean_{j in N_k(i)} Δ_j
            lp_delta = self._compute_knn_mean(delta_sub, positions_sub)
            
        elif self.lp_mode == 'graph_laplacian':
            # LP-2: Graph Laplacian
            # LP(Δ_i) = Δ_i - mean_{j in N(i)} Δ_j
            knn_mean = self._compute_knn_mean(delta_sub, positions_sub)
            lp_delta = delta_sub - knn_mean
            
        else:
            # Fallback to knn_mean
            lp_delta = self._compute_knn_mean(delta_sub, positions_sub)
        
        # L_LP = mean_i || LP(Δ_i) ||^2
        lp_norms = torch.norm(lp_delta, dim=-1)  # [M]
        L_LP = (lp_norms ** 2).mean()
        
        # Cache statistics for logging
        with torch.no_grad():
            delta_norms = torch.norm(delta_sub, dim=-1)
            self._last_lp_loss = L_LP.item()
            self._last_lp_mean = lp_norms.mean().item()
            delta_mean = delta_norms.mean().item()
            self._last_lp_ratio = self._last_lp_mean / (delta_mean + 1e-8)
        
        return L_LP
    
    def _compute_knn_mean(self, delta: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """
        Compute kNN mean of delta based on spatial positions.
        
        Args:
            delta: Residual displacements [M, 3]
            positions: 3D positions [M, 3]
            
        Returns:
            knn_mean: Mean of k-nearest neighbors' delta [M, 3]
        """
        M = delta.shape[0]
        k = min(self.lp_k, M - 1)
        
        if k <= 0:
            return delta.clone()
        
        # Compute pairwise distances (O(M^2) but acceptable for M=2048)
        dists = torch.cdist(positions, positions)  # [M, M]
        
        # Get k+1 nearest neighbors (including self)
        _, knn_indices = torch.topk(dists, k + 1, largest=False, dim=-1)  # [M, k+1]
        
        # Exclude self (first neighbor is always self with dist=0)
        knn_indices = knn_indices[:, 1:]  # [M, k]
        
        # Gather neighbor deltas and compute mean
        knn_delta = delta[knn_indices]  # [M, k, 3]
        knn_mean = knn_delta.mean(dim=1)  # [M, 3]
        
        return knn_mean
    
    def get_lp_statistics(self) -> dict:
        """
        M3: Get LP regularization statistics for logging.
        
        Returns:
            Dictionary with lp_loss, lp_mean, lp_ratio, lp_mode, lp_enable
        """
        return {
            'lp_loss': self._last_lp_loss,
            'lp_mean': self._last_lp_mean,
            'lp_ratio': self._last_lp_ratio,
            'lp_mode': self.lp_mode,
            'lp_enable': self.lp_enable
        }
    
    # ================================================================
    # M4: Subspace Decoupling Regularization Methods
    # ================================================================
    
    def compute_decouple_loss(self, means3D: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        M4: Compute Subspace Decoupling Regularization Loss.
        
        "Subspace decoupling regularization discourages the Eulerian residual
         from aligning with the Lagrangian deformation responses, forcing it
         to model complementary details rather than shortcuts."
        
        Args:
            means3D: Gaussian positions [N, 3]
            times: Time values [N, 1] or scalar
            
        Returns:
            L_decouple loss tensor (scalar)
        """
        if not self.decouple_enable:
            return torch.tensor(0.0, device='cuda')
        
        if not self.use_boosted:
            return torch.tensor(0.0, device='cuda')
        
        if self.decouple_mode == 'velocity_corr':
            return self._compute_velocity_corr_loss(means3D, times)
        elif self.decouple_mode == 'stochastic_jacobian_corr':
            return self._compute_jacobian_corr_loss(means3D, times)
        else:
            return torch.tensor(0.0, device='cuda')
    
    def _compute_velocity_corr_loss(self, means3D: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        M4 Mode 1: Velocity correlation loss.
        
        Computes cosine similarity between Lagrangian and Eulerian velocities:
        v_L = deform_L(x, t+dt) - deform_L(x, t)
        v_E = deform_E(x, t+dt) - deform_E(x, t)
        L = mean(cos^2(v_L, v_E))
        """
        N = means3D.shape[0]
        device = means3D.device
        
        # Subsample for efficiency
        if self.decouple_subsample > 0 and N > self.decouple_subsample:
            indices = torch.randperm(N, device=device)[:self.decouple_subsample]
            means3D_sub = means3D[indices]
            if times.dim() > 0 and times.shape[0] == N:
                times_sub = times[indices]
            else:
                times_sub = times
        else:
            means3D_sub = means3D
            times_sub = times
            indices = None
        
        M = means3D_sub.shape[0]
        
        # Get current time value
        if times_sub.dim() == 0:
            t = times_sub.item()
        elif times_sub.numel() == 1:
            t = times_sub.item()
        else:
            t = times_sub[0].item() if times_sub.dim() > 0 else times_sub.item()
        
        t_dt = t + self.decouple_dt
        
        # Clamp to [0, 1]
        t_dt = min(t_dt, 1.0)
        
        # Create time tensors
        times_t = torch.full((M, 1), t, device=device, dtype=means3D_sub.dtype)
        times_tdt = torch.full((M, 1), t_dt, device=device, dtype=means3D_sub.dtype)
        
        # Get Lagrangian deformation at t and t+dt
        with torch.set_grad_enabled(not self.decouple_stopgrad_L):
            dx_L_t = self._get_anchor_deformation(means3D_sub.detach(), times_t)
            dx_L_tdt = self._get_anchor_deformation(means3D_sub.detach(), times_tdt)
            v_L = dx_L_tdt - dx_L_t  # [M, 3]
            
            if self.decouple_stopgrad_L:
                v_L = v_L.detach()
        
        # Get Eulerian deformation at t and t+dt
        dx_E_t = self._get_eulerian_deformation(means3D_sub.detach(), times_t)
        dx_E_tdt = self._get_eulerian_deformation(means3D_sub.detach(), times_tdt)
        v_E = dx_E_tdt - dx_E_t  # [M, 3]
        
        # Compute cosine similarity
        eps = 1e-8
        v_L_norm = torch.norm(v_L, dim=-1, keepdim=True) + eps
        v_E_norm = torch.norm(v_E, dim=-1, keepdim=True) + eps
        
        cos_sim = torch.sum(v_L * v_E, dim=-1) / (v_L_norm.squeeze() * v_E_norm.squeeze())
        
        # Penalize correlation
        if self.decouple_use_squared_cos:
            L_decouple = (cos_sim ** 2).mean()
        else:
            L_decouple = torch.abs(cos_sim).mean()
        
        # Cache statistics
        with torch.no_grad():
            self._last_decouple_loss = L_decouple.item()
            self._last_corr_mean = cos_sim.abs().mean().item()
        
        return L_decouple
    
    def _compute_jacobian_corr_loss(self, means3D: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """
        M4 Mode 2: Stochastic Jacobian correlation loss using finite differences.
        
        Uses spatial finite differences to approximate Jacobian directional derivatives:
        g_L = (deform_L(x + eps*w, t) - deform_L(x, t)) / eps  (projected onto w)
        g_E = (deform_E(x + eps*w, t) - deform_E(x, t)) / eps  (projected onto w)
        L = mean(cos^2(g_L, g_E))
        
        This avoids second-order derivatives which grid_sampler doesn't support.
        """
        N = means3D.shape[0]
        device = means3D.device
        
        # Subsample for efficiency
        if self.decouple_subsample > 0 and N > self.decouple_subsample:
            indices = torch.randperm(N, device=device)[:self.decouple_subsample]
            means3D_sub = means3D[indices].detach()
            if times.dim() > 0 and times.shape[0] == N:
                times_sub = times[indices]
            else:
                times_sub = times
        else:
            means3D_sub = means3D.detach()
            times_sub = times
        
        M = means3D_sub.shape[0]
        
        # Spatial perturbation step size
        eps_spatial = 0.001
        
        # Generate random directions
        K = self.decouple_num_dirs
        w = torch.randn(K, 3, device=device, dtype=means3D_sub.dtype)
        w = w / (torch.norm(w, dim=-1, keepdim=True) + 1e-8)
        
        total_loss = torch.tensor(0.0, device=device)
        total_corr = 0.0
        total_grad_L_norm = 0.0
        total_grad_E_norm = 0.0
        
        for k in range(K):
            wk = w[k]  # [3]
            
            # Perturbed positions
            means3D_plus = means3D_sub + eps_spatial * wk.unsqueeze(0)
            
            # Get Lagrangian Jacobian direction via finite difference
            if self.decouple_stopgrad_L:
                with torch.no_grad():
                    dx_L_0 = self._get_anchor_deformation(means3D_sub, times_sub)
                    dx_L_plus = self._get_anchor_deformation(means3D_plus, times_sub)
                g_L = (dx_L_plus - dx_L_0) / eps_spatial  # [M, 3]
                g_L = g_L.detach()
            else:
                dx_L_0 = self._get_anchor_deformation(means3D_sub, times_sub)
                dx_L_plus = self._get_anchor_deformation(means3D_plus, times_sub)
                g_L = (dx_L_plus - dx_L_0) / eps_spatial  # [M, 3]
            
            # Get Eulerian Jacobian direction via finite difference
            dx_E_0 = self._get_eulerian_deformation(means3D_sub, times_sub)
            dx_E_plus = self._get_eulerian_deformation(means3D_plus, times_sub)
            g_E = (dx_E_plus - dx_E_0) / eps_spatial  # [M, 3]
            
            # Compute cosine similarity (with numerical stability)
            eps = 1e-8
            g_L_norm = torch.norm(g_L, dim=-1, keepdim=True) + eps
            g_E_norm = torch.norm(g_E, dim=-1, keepdim=True) + eps
            
            cos_sim = torch.sum(g_L * g_E, dim=-1) / (g_L_norm.squeeze() * g_E_norm.squeeze())
            
            if self.decouple_use_squared_cos:
                loss_k = (cos_sim ** 2).mean()
            else:
                loss_k = torch.abs(cos_sim).mean()
            
            total_loss = total_loss + loss_k
            
            with torch.no_grad():
                total_corr += cos_sim.abs().mean().item()
                total_grad_L_norm += g_L_norm.mean().item()
                total_grad_E_norm += g_E_norm.mean().item()
        
        L_decouple = total_loss / K
        
        # Cache statistics
        with torch.no_grad():
            self._last_decouple_loss = L_decouple.item()
            self._last_corr_mean = total_corr / K
            self._last_grad_L_norm = total_grad_L_norm / K
            self._last_grad_E_norm = total_grad_E_norm / K
        
        return L_decouple
    
    def _get_anchor_deformation(self, means3D: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Helper to get Lagrangian (Anchor) deformation only."""
        if not self.initialized:
            return torch.zeros_like(means3D)
        
        N = means3D.shape[0]
        
        # Get anchor deformations (forward_anchors returns single tensor by default)
        anchor_dx = self.forward_anchors(times, is_training=False)
        
        # Interpolate to Gaussian positions using cached KNN
        if self.knn_indices is not None and self.knn_indices.shape[0] >= N:
            knn_dx = anchor_dx[self.knn_indices[:N]]
            weights = self.knn_weights[:N].unsqueeze(-1)
            dx_anchor = (knn_dx * weights).sum(dim=1)
        else:
            dx_anchor = torch.zeros(N, 3, device=means3D.device)
        
        return dx_anchor
    
    def _get_eulerian_deformation(self, means3D: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Helper to get Eulerian (HexPlane) deformation only."""
        if self.original_deformation is None:
            return torch.zeros_like(means3D)
        
        N = means3D.shape[0]
        device = means3D.device
        dtype = means3D.dtype
        
        # Create dummy scales, rotations, density for HexPlane query
        # HexPlane forward signature: (positions, scales, rotations, density, time_emb)
        dummy_scales = torch.ones(N, 3, device=device, dtype=dtype)
        dummy_rotations = torch.zeros(N, 4, device=device, dtype=dtype)
        dummy_rotations[:, 0] = 1.0  # Unit quaternion [1, 0, 0, 0]
        dummy_density = torch.ones(N, 1, device=device, dtype=dtype)
        
        # Ensure times has correct shape [N, 1]
        if times.dim() == 0:
            times = times.unsqueeze(0).unsqueeze(1).expand(N, 1)
        elif times.dim() == 1:
            times = times.unsqueeze(1).expand(N, 1)
        elif times.shape[0] == 1:
            times = times.expand(N, 1)
        
        # Query HexPlane - returns (means3D_deformed, scales_deformed, rotations_deformed)
        means3D_hex, _, _ = self.original_deformation(
            means3D, dummy_scales, dummy_rotations, dummy_density, times
        )
        
        # Extract displacement
        dx_hex = means3D_hex - means3D
        
        return dx_hex
    
    def get_decouple_statistics(self) -> dict:
        """
        M4: Get decoupling regularization statistics for logging.
        
        Returns:
            Dictionary with decouple stats
        """
        return {
            'decouple_loss': self._last_decouple_loss,
            'corr_mean': self._last_corr_mean,
            'grad_L_norm': self._last_grad_L_norm,
            'grad_E_norm': self._last_grad_E_norm,
            'decouple_mode': self.decouple_mode,
            'decouple_enable': self.decouple_enable,
            'stopgrad_L': self.decouple_stopgrad_L
        }
    
    # ================================================================
    # PhysX-Taylor: Affine deformation methods
    # ================================================================
    
    def get_affine_magnitude(self) -> torch.Tensor:
        """
        Get the cached affine matrix magnitude for L1 regularization.
        
        This forces the affine matrices to stay sparse - most regions should
        have only rigid translation (A ≈ 0), with complex affine only at
        sharp boundaries like blood vessel edges.
        
        Returns:
            magnitude: Mean absolute affine matrix elements (scalar)
        """
        if self._last_affine_magnitude is not None:
            return self._last_affine_magnitude
        return torch.tensor(0.0, device=self.anchor_positions.device)
    
    def get_affine_parameters(self):
        """
        Return affine head parameters for optimizer.
        
        Returns:
            List of parameters (or empty list if Taylor not enabled)
        """
        if not self.use_taylor:
            return []
        
        params = []
        if hasattr(self, 'affine_head'):
            params.extend(self.affine_head.parameters())
        return params
    
    # ================================================================
    # PhysX-Boosted: Full HexPlane baseline methods
    # ================================================================
    
    def get_hexplane_mlp_parameters(self):
        """
        Return HexPlane MLP parameters for optimizer (boosted mode).
        
        Returns:
            List of parameters (or empty list if boosted not enabled)
        """
        if not self.use_boosted or self.original_deformation is None:
            return []
        return self.original_deformation.get_mlp_parameters()
    
    def get_hexplane_grid_parameters(self):
        """
        Return HexPlane grid parameters for optimizer (boosted mode).
        
        Returns:
            List of parameters (or empty list if boosted not enabled)
        """
        if not self.use_boosted or self.original_deformation is None:
            return []
        return self.original_deformation.get_grid_parameters()
    
    def get_hexplane_grid(self):
        """
        Return the HexPlane grid for TV loss computation (boosted mode).
        
        Returns:
            HexPlaneField or None if boosted not enabled
        """
        if not self.use_boosted or self.original_deformation is None:
            return None
        return self.original_deformation.deformation_net.grid
    
    def set_hexplane_aabb(self, xyz_max, xyz_min):
        """
        Set AABB for the HexPlane in boosted mode.
        
        Args:
            xyz_max: Maximum coordinates
            xyz_min: Minimum coordinates
        """
        if self.use_boosted and self.original_deformation is not None:
            self.original_deformation.deformation_net.set_aabb(xyz_max, xyz_min)
            print(f"[PhysX-Boosted] HexPlane AABB set to [{xyz_min}, {xyz_max}]")
    
    def forward_backward_hexplane(self, deformed_pts, time_emb):
        """
        Forward backward deformation using HexPlane for inverse consistency loss.
        
        Args:
            deformed_pts: Deformed positions [N, 3]
            time_emb: Time values [N, 1]
            
        Returns:
            reconstructed_pts: Reconstructed positions [N, 3]
            backward_deform: Backward deformation [N, 3]
        """
        if not self.use_boosted or self.original_deformation is None:
            return deformed_pts, torch.zeros_like(deformed_pts)
        return self.original_deformation.forward_backward_position(deformed_pts, time_emb)
    
    # ================================================================
    # V5: Learnable Balance Parameter Methods
    # ================================================================
    
    def get_balance_parameter(self):
        """
        Return balance logit parameter for optimizer (V5 mode).
        
        Returns:
            List containing balance_logit parameter, or empty list if not enabled
        """
        if self.use_learnable_balance:
            return [self.balance_logit]
        return []
    
    def get_balance_alpha(self):
        """
        Get current balance alpha value α = sigmoid(τ).
        
        Returns:
            float: Current alpha value (0 to 1), or 0.5 if not enabled
        """
        if self.use_learnable_balance:
            # Handle extreme cases
            if self.use_pure_hexplane:
                return 0.0
            elif self.use_pure_anchor:
                return 1.0
            else:
                return torch.sigmoid(self.balance_logit).item()
        return 0.5
    
    def compute_balance_regularization_loss(self, alpha_target=0.5):
        """
        Compute regularization loss to prevent alpha from going extreme.
        
        L_balance = (α - α_target)^2
        
        This encourages a balanced use of both HexPlane and Anchor.
        
        Args:
            alpha_target: Target alpha value (default 0.5 for equal balance)
            
        Returns:
            Scalar loss value
        """
        if not self.use_learnable_balance:
            return torch.tensor(0.0, device=self.balance_logit.device)
        
        alpha = torch.sigmoid(self.balance_logit)
        return (alpha - alpha_target) ** 2
    
    # ================================================================
    # V6: Orthogonal Gradient Projection Methods
    # ================================================================
    
    def _apply_orthogonal_projection_hook(self, dx_anchor: torch.Tensor, dx_hex: torch.Tensor) -> torch.Tensor:
        """
        Apply orthogonal gradient projection to dx_anchor during backward pass.
        
        Forward: Returns dx_anchor unchanged
        Backward: Projects out the component of grad_anchor along grad_hex direction
        
        This forces Anchor to only learn what HexPlane cannot capture (residual).
        
        Args:
            dx_anchor: Anchor displacement [N, 3]
            dx_hex: HexPlane displacement [N, 3] (used for gradient direction)
            
        Returns:
            dx_anchor with modified backward gradient
        """
        # Use a custom autograd function for gradient modification
        return OrthogonalGradientProjection.apply(
            dx_anchor, 
            dx_hex.detach(),  # Detach dx_hex - we only need it for gradient direction
            self.ortho_projection_strength
        )
    
    def get_ortho_projection_stats(self):
        """
        Get statistics about orthogonal projection for logging.
        
        Returns:
            dict with projection stats, or None if V6 not enabled
        """
        if not self.use_orthogonal_projection:
            return None
        
        # Return last cached values if available
        if hasattr(self, '_last_projection_ratio'):
            return {
                'projection_ratio': self._last_projection_ratio,
                'grad_hex_norm': self._last_grad_hex_norm,
                'grad_anchor_norm': self._last_grad_anchor_norm,
            }
        return None
    
    # ================================================================
    # V7: Uncertainty-Aware Fusion Methods
    # ================================================================
    
    def get_uncertainty_parameters(self):
        """
        Return uncertainty head parameters for optimizer (V7 mode).
        
        Returns:
            List of parameters for both anchor and hex uncertainty heads
        """
        if not self.use_uncertainty_fusion:
            return []
        
        params = []
        if hasattr(self, 'anchor_uncertainty_head'):
            params.extend(self.anchor_uncertainty_head.parameters())
        if hasattr(self, 'hex_uncertainty_head'):
            params.extend(self.hex_uncertainty_head.parameters())
        return params
    
    def compute_kendall_loss(self, render_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute Kendall-style uncertainty loss for V7.
        
        L_total = L_render / (2 * Σ) + λ * log(Σ)
        
        where Σ = σ²_hex + σ²_anchor (total variance)
        
        This prevents "blind confidence":
        - If networks output high uncertainty (large σ), L_render is down-weighted
        - But log(Σ) penalizes large uncertainty, forcing honest estimation
        
        Args:
            render_loss: The raw render loss (L1 + DSSIM)
            
        Returns:
            Modified loss incorporating uncertainty
        """
        if not self.use_uncertainty_fusion:
            return render_loss
        
        if self._last_log_var_hex is None or self._last_log_var_anchor is None:
            return render_loss
        
        # Compute total variance Σ = σ²_hex + σ²_anchor
        # log_var = log(σ²), so σ² = exp(log_var)
        var_hex = torch.exp(self._last_log_var_hex)  # [N]
        var_anchor = torch.exp(self._last_log_var_anchor)  # [N]
        
        # Mean variance across all Gaussians
        sigma_total = (var_hex + var_anchor).mean() + self.uncertainty_eps
        
        # Kendall loss: L_render / (2Σ) + λ * log(Σ)
        kendall_loss = render_loss / (2 * sigma_total) + self.lambda_uncertainty * torch.log(sigma_total)
        
        return kendall_loss
    
    def get_uncertainty_stats(self):
        """
        Get uncertainty statistics for logging.
        
        Returns:
            dict with uncertainty stats, or None if V7 not enabled
        """
        if not self.use_uncertainty_fusion:
            return None
        
        stats = {}
        if self._last_log_var_hex is not None:
            var_hex = torch.exp(self._last_log_var_hex).mean().item()
            stats['var_hex'] = var_hex
            stats['sigma_hex'] = var_hex ** 0.5
        if self._last_log_var_anchor is not None:
            var_anchor = torch.exp(self._last_log_var_anchor).mean().item()
            stats['var_anchor'] = var_anchor
            stats['sigma_anchor'] = var_anchor ** 0.5
        if self._last_weight_hex is not None:
            stats['weight_hex'] = self._last_weight_hex
        if self._last_weight_anchor is not None:
            stats['weight_anchor'] = self._last_weight_anchor
        
        return stats if stats else None


class OrthogonalGradientProjection(torch.autograd.Function):
    """
    Custom autograd function for V6 Orthogonal Gradient Projection.
    
    Forward: Identity (returns input unchanged)
    Backward: Projects out the component of incoming gradient along dx_hex direction
    
    This implements the core V6 idea:
    - HexPlane (A) is the "base" that learns easily-captured patterns
    - Anchor (B) is constrained to learn only the residual (orthogonal direction)
    
    grad_B_orthogonal = grad_B - proj_{dx_hex}(grad_B)
                      = grad_B - (grad_B · unit_hex) * unit_hex
    """
    
    @staticmethod
    def forward(ctx, dx_anchor: torch.Tensor, dx_hex: torch.Tensor, strength: float):
        """
        Forward pass: Just return dx_anchor unchanged.
        
        Args:
            dx_anchor: Anchor displacement [N, 3]
            dx_hex: HexPlane displacement [N, 3] (for gradient projection direction)
            strength: How much to project out (1.0 = full projection)
        """
        # Save dx_hex for backward (we'll use it as the projection direction)
        ctx.save_for_backward(dx_hex)
        ctx.strength = strength
        return dx_anchor
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: Project out the component along dx_hex direction.
        
        The key insight: We use dx_hex (the HexPlane output) as a proxy for
        "what HexPlane is learning". By projecting out this direction from
        Anchor's gradient, we force Anchor to only update in orthogonal directions.
        
        grad_anchor_orth = grad_anchor - strength * proj_{dx_hex}(grad_anchor)
        """
        dx_hex, = ctx.saved_tensors
        strength = ctx.strength
        
        # grad_output is the gradient w.r.t. dx_anchor (same as grad w.r.t. dx_combined
        # since dx_combined = dx_hex + dx_anchor, and d(dx_combined)/d(dx_anchor) = I)
        grad_anchor = grad_output  # [N, 3]
        
        # Compute unit vector in dx_hex direction (per-Gaussian)
        # This represents "the direction HexPlane is deforming"
        norm_hex = torch.norm(dx_hex, dim=-1, keepdim=True) + 1e-8  # [N, 1]
        unit_hex = dx_hex / norm_hex  # [N, 3]
        
        # Compute projection of grad_anchor onto unit_hex direction
        # proj = (grad_anchor · unit_hex) * unit_hex
        dot_product = torch.sum(grad_anchor * unit_hex, dim=-1, keepdim=True)  # [N, 1]
        projection = dot_product * unit_hex  # [N, 3]
        
        # Remove the projection component (make gradient orthogonal)
        grad_anchor_orthogonal = grad_anchor - strength * projection
        
        # Return gradients: (grad for dx_anchor, None for dx_hex, None for strength)
        return grad_anchor_orthogonal, None, None


class ReverseOrthogonalGradientProjection(torch.autograd.Function):
    """
    Custom autograd function for V8 Reverse Orthogonal Gradient Projection.
    
    This is the reverse of V6:
    - V6: HexPlane (A) is base, Anchor (B) learns residual
    - V8: Anchor (A) is base, HexPlane (B) learns residual
    
    Forward: Identity (returns input unchanged)
    Backward: Projects out the component of incoming gradient along dx_anchor direction
    
    grad_hex_orthogonal = grad_hex - proj_{dx_anchor}(grad_hex)
                        = grad_hex - (grad_hex · unit_anchor) * unit_anchor
    """
    
    @staticmethod
    def forward(ctx, dx_hex: torch.Tensor, dx_anchor: torch.Tensor, strength: float):
        """
        Forward pass: Just return dx_hex unchanged.
        
        Args:
            dx_hex: HexPlane displacement [N, 3]
            dx_anchor: Anchor displacement [N, 3] (for gradient projection direction)
            strength: How much to project out (1.0 = full projection)
        """
        # Save dx_anchor for backward (we'll use it as the projection direction)
        ctx.save_for_backward(dx_anchor)
        ctx.strength = strength
        return dx_hex
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass: Project out the component along dx_anchor direction.
        
        The key insight: We use dx_anchor (the Anchor output) as a proxy for
        "what Anchor is learning". By projecting out this direction from
        HexPlane's gradient, we force HexPlane to only update in orthogonal directions.
        
        grad_hex_orth = grad_hex - strength * proj_{dx_anchor}(grad_hex)
        """
        dx_anchor, = ctx.saved_tensors
        strength = ctx.strength
        
        # grad_output is the gradient w.r.t. dx_hex
        grad_hex = grad_output  # [N, 3]
        
        # Compute unit vector in dx_anchor direction (per-Gaussian)
        # This represents "the direction Anchor is deforming"
        norm_anchor = torch.norm(dx_anchor, dim=-1, keepdim=True) + 1e-8  # [N, 1]
        unit_anchor = dx_anchor / norm_anchor  # [N, 3]
        
        # Compute projection of grad_hex onto unit_anchor direction
        # proj = (grad_hex · unit_anchor) * unit_anchor
        dot_product = torch.sum(grad_hex * unit_anchor, dim=-1, keepdim=True)  # [N, 1]
        projection = dot_product * unit_anchor  # [N, 3]
        
        # Remove the projection component (make gradient orthogonal)
        grad_hex_orthogonal = grad_hex - strength * projection
        
        # Return gradients: (grad for dx_hex, None for dx_anchor, None for strength)
        return grad_hex_orthogonal, None, None


def anchor_deform_network(args):
    """
    Factory function to create anchor deformation network.
    
    This follows the same pattern as deform_network in deformation.py.
    """
    return AnchorDeformationNet(args)
