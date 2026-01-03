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
from .graphics_utils import batch_quaternion_multiply


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


class PhaseEpsilon(nn.Module):
    """
    M5: Phase-Aware Trust-Region ε(t) Module.
    
    "Phase-aware trust-region allocates a bounded residual budget across
     respiratory phases, preserving Lagrangian dominance while enabling
     demand-driven corrections."
    
    Computes time-conditioned epsilon: ε(t) = ε_max * sigmoid(g(t))
    where g(t) is a low-capacity function.
    
    Two modes:
      - per_frame: g_k is a learnable vector [T], one scalar per discrete phase
      - tiny_mlp:  g(t) is a small MLP with Fourier time encoding
    """
    
    def __init__(
        self,
        mode: str = "per_frame",
        num_frames: int = 10,
        mlp_hidden: int = 32,
        mlp_layers: int = 2,
        eps_init: float = 0.015,
        eps_max: float = 0.03,
        num_fourier_freqs: int = 4
    ):
        super().__init__()
        self.mode = mode
        self.num_frames = num_frames
        self.eps_max = eps_max
        self.eps_init = eps_init
        
        # Compute initial rho such that sigmoid(rho) = eps_init / eps_max
        eps_ratio = min(max(eps_init / eps_max, 1e-6), 1 - 1e-6)
        rho_init = math.log(eps_ratio / (1 - eps_ratio))  # logit
        
        if mode == "per_frame":
            # Per-frame learnable parameters g_k
            # Initialize all to same value (reproduces baseline initially)
            self.g = nn.Parameter(torch.full((num_frames,), rho_init))
            
        elif mode == "tiny_mlp":
            # Tiny MLP with Fourier time encoding
            self.num_fourier_freqs = num_fourier_freqs
            input_dim = num_fourier_freqs * 2  # sin + cos for each freq
            
            # Build MLP layers
            layers = []
            in_dim = input_dim
            for i in range(mlp_layers - 1):
                layers.extend([
                    nn.Linear(in_dim, mlp_hidden),
                    nn.ReLU(inplace=True)
                ])
                in_dim = mlp_hidden
            # Final layer outputs scalar g(t)
            layers.append(nn.Linear(in_dim, 1))
            self.mlp = nn.Sequential(*layers)
            
            # Initialize MLP with small weights to output near rho_init
            # Last layer bias = rho_init, weights = small
            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std=0.01)
                    nn.init.zeros_(m.bias)
            # Set last layer bias to rho_init
            self.mlp[-1].bias.data.fill_(rho_init)
            
            # Fixed Fourier frequency bands for time encoding
            self.register_buffer(
                'freq_bands',
                torch.linspace(1.0, num_fourier_freqs, num_fourier_freqs) * math.pi
            )
        else:
            raise ValueError(f"Unknown phase_eps mode: {mode}")
        
        # Cache for logging
        self._last_eps_values = None  # All ε values for logging
        self._last_g_values = None    # Raw g values before sigmoid
        
        print(f"[M5] PhaseEpsilon initialized:")
        print(f"     mode={mode}, eps_init={eps_init:.4f}, eps_max={eps_max:.4f}")
        if mode == "per_frame":
            print(f"     num_frames={num_frames}, g_init={rho_init:.4f}")
        else:
            print(f"     mlp_hidden={mlp_hidden}, mlp_layers={mlp_layers}")
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Compute ε(t) for given time values.
        
        Args:
            t: Time values in [0, 1], shape [] (scalar), [1], or [N, 1]
        
        Returns:
            eps: Epsilon value(s), same shape as input or scalar
        """
        # Normalize t to scalar if needed
        if t.dim() == 2:
            t_scalar = t[0, 0]
        elif t.dim() == 1:
            t_scalar = t[0]
        else:
            t_scalar = t
        
        if self.mode == "per_frame":
            # Map t to discrete frame index
            # t in [0, 1] -> frame_idx in [0, num_frames-1]
            t_clamped = torch.clamp(t_scalar, 0.0, 1.0 - 1e-6)
            frame_idx = (t_clamped * self.num_frames).long()
            frame_idx = torch.clamp(frame_idx, 0, self.num_frames - 1)
            
            # Get g value for this frame
            g_t = self.g[frame_idx]
            
        elif self.mode == "tiny_mlp":
            # Fourier encoding of time
            if t_scalar.dim() == 0:
                t_in = t_scalar.unsqueeze(0)  # [1]
            else:
                t_in = t_scalar
            t_in = t_in.unsqueeze(-1)  # [1, 1]
            
            # Fourier features: [sin(f1*t), cos(f1*t), ...]
            freq_t = t_in * self.freq_bands  # [1, num_freqs]
            fourier = torch.cat([torch.sin(freq_t), torch.cos(freq_t)], dim=-1)  # [1, 2*num_freqs]
            
            # MLP forward
            g_t = self.mlp(fourier).squeeze()  # scalar
        
        # Compute ε(t) = ε_max * sigmoid(g(t))
        eps_t = self.eps_max * torch.sigmoid(g_t)
        
        # Cache for logging
        self._last_g_values = g_t.detach()
        self._last_eps_values = eps_t.detach()
        
        return eps_t
    
    def get_all_eps_values(self, num_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get ε values across all time points for visualization.
        
        Args:
            num_samples: Number of time samples (for tiny_mlp mode)
        
        Returns:
            t_values: Time values [T] or [num_samples]
            eps_values: Corresponding ε values
        """
        device = next(self.parameters()).device
        
        if self.mode == "per_frame":
            # Return all per-frame eps values
            eps_values = self.eps_max * torch.sigmoid(self.g)  # [num_frames]
            t_values = torch.linspace(0, 1, self.num_frames, device=device)
            return t_values.detach(), eps_values.detach()
        
        else:  # tiny_mlp
            # Sample at num_samples points
            t_values = torch.linspace(0, 1, num_samples, device=device)
            eps_values = []
            for t in t_values:
                eps_t = self.forward(t)
                eps_values.append(eps_t.item())
            eps_values = torch.tensor(eps_values, device=device)
            return t_values.detach(), eps_values.detach()
    
    def compute_smooth_loss(self) -> torch.Tensor:
        """
        Compute temporal smoothness prior L_smooth.
        
        For per_frame: L_smooth = mean_k (ε_{k+1} - ε_k)^2
        For tiny_mlp:  L_smooth = mean (ε(t+dt) - ε(t))^2 over sampled t
        
        Returns:
            L_smooth: Smoothness loss (scalar)
        """
        if self.mode == "per_frame":
            # Compute all eps values
            eps_all = self.eps_max * torch.sigmoid(self.g)  # [num_frames]
            
            # First-order difference
            eps_diff = eps_all[1:] - eps_all[:-1]  # [num_frames-1]
            
            # MSE of differences
            L_smooth = (eps_diff ** 2).mean()
            
        else:  # tiny_mlp
            # Sample at a few points and compute differences
            device = next(self.parameters()).device
            num_samples = 10
            dt = 1.0 / num_samples
            
            diffs = []
            for i in range(num_samples):
                t1 = torch.tensor(i * dt, device=device)
                t2 = torch.tensor((i + 1) * dt, device=device)
                eps1 = self.forward(t1)
                eps2 = self.forward(t2)
                diffs.append((eps2 - eps1) ** 2)
            
            L_smooth = torch.stack(diffs).mean()
        
        return L_smooth
    
    def get_stats(self) -> dict:
        """
        Get statistics for logging.
        
        Returns:
            dict with mean_eps, min_eps, max_eps, std_eps
        """
        t_vals, eps_vals = self.get_all_eps_values()
        return {
            'mean_eps': eps_vals.mean().item(),
            'min_eps': eps_vals.min().item(),
            'max_eps': eps_vals.max().item(),
            'std_eps': eps_vals.std().item(),
        }


class LowPassOperator(nn.Module):
    """
    M6: Low-Pass Operator for Structural Frequency Decomposition
    
    "Unlike penalty-based regularization, we enforce a structural frequency
     split of the Eulerian residual in the forward pass, allocating a bounded
     correction budget to the high-frequency component to prevent shortcut
     learning."
    
    Computes r_low = LP(r) via neighbor averaging:
        r_low[i] = mean_{j in N(i)} r[j]
    
    Supports two modes:
        - "graph": Uses existing anchor graph/adjacency
        - "knn_cached": Pre-computes kNN in canonical space, caches indices
    """
    
    def __init__(self, mode: str = "graph", k: int = 8):
        """
        Args:
            mode: "graph" or "knn_cached"
            k: Number of neighbors for LP
        """
        super().__init__()
        self.mode = mode
        self.k = k
        
        # Cached Anchor kNN indices (only used for knn_cached mode)
        # NOTE: anchors are small (e.g. 1024), so O(M^2) is acceptable.
        self._anchor_knn_indices: Optional[torch.Tensor] = None
        self._cached_for_n_anchors: int = 0

    def build_anchor_knn_cache(self, anchor_positions: torch.Tensor) -> None:
        """
        Pre-compute and cache kNN indices among anchors.

        Args:
            anchor_positions: Anchor positions [M, 3]
        """
        M = anchor_positions.shape[0]
        if self._anchor_knn_indices is not None and self._cached_for_n_anchors == M:
            return

        k = min(self.k + 1, M)
        with torch.no_grad():
            dists = torch.cdist(anchor_positions, anchor_positions)  # [M, M]
            _, knn = torch.topk(dists, k, largest=False, dim=-1)     # [M, k]
            # Exclude self (first neighbor is always self with dist=0)
            knn = knn[:, 1:]
            self._anchor_knn_indices = knn.contiguous()
            self._cached_for_n_anchors = M
            
    def forward(
        self,
        r: torch.Tensor,
        knn_indices: torch.Tensor,
        knn_weights: torch.Tensor,
        anchor_positions: torch.Tensor,
        anchor_graph: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Apply low-pass filter to residual field.
        
        Args:
            r: Residual field [N, 3] or [N, D]
            knn_indices: Gaussian -> anchor indices [N, K]
            knn_weights: Gaussian -> anchor weights [N, K]
            anchor_positions: Anchor positions [M, 3]
            anchor_graph: Optional anchor adjacency [M, k] (only used when mode=="graph")
        
        Returns:
            r_low: Low-frequency component [N, 3] or [N, D]
        """
        # ------------------------------------------------
        # Efficient low-pass using the existing Gaussian->Anchor binding graph.
        # Steps:
        #   1) Scatter weighted Gaussian residuals to anchors (aggregate)
        #   2) Optional anchor-space neighbor average (knn_cached)
        #   3) Gather back to Gaussians via the same skinning weights
        # Complexity: O(N*K + M*k_anchor)
        # ------------------------------------------------

        N, K = knn_indices.shape
        M = anchor_positions.shape[0]
        D = r.shape[-1]

        idx = knn_indices.reshape(-1)  # [N*K]
        w = knn_weights.reshape(-1, 1)  # [N*K, 1]
        r_rep = r.unsqueeze(1).expand(-1, K, -1).reshape(-1, D)  # [N*K, D]

        # Scatter Gaussian residuals to anchors
        r_sum = torch.zeros((M, D), device=r.device, dtype=r.dtype)
        w_sum = torch.zeros((M, 1), device=r.device, dtype=r.dtype)
        r_sum.index_add_(0, idx, r_rep * w)
        w_sum.index_add_(0, idx, w)
        r_anchor = r_sum / (w_sum + 1e-8)  # [M, D]

        # Optional: anchor-space smoothing
        if self.mode == "knn_cached":
            if self._anchor_knn_indices is None or self._cached_for_n_anchors != M:
                self.build_anchor_knn_cache(anchor_positions)
            a_knn = self._anchor_knn_indices.to(r.device)  # [M, k]
            r_anchor = r_anchor[a_knn].mean(dim=1)  # [M, D]
        elif self.mode == "graph" and anchor_graph is not None:
            a_knn = anchor_graph.to(r.device)
            r_anchor = r_anchor[a_knn].mean(dim=1)

        # Gather back to Gaussians
        r_low = torch.sum(r_anchor[knn_indices] * knn_weights.unsqueeze(-1), dim=1)  # [N, D]
        return r_low
    
    def get_high_pass(
        self, 
        r: torch.Tensor,
        knn_indices: torch.Tensor,
        knn_weights: torch.Tensor,
        anchor_positions: torch.Tensor,
        anchor_graph: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decompose residual into low and high frequency components.
        
        Args:
            r: Residual field [N, 3]
            knn_indices: Gaussian -> anchor indices [N, K]
            knn_weights: Gaussian -> anchor weights [N, K]
            anchor_positions: Anchor positions [M, 3]
            anchor_graph: Optional pre-computed anchor adjacency [M, k]
        
        Returns:
            r_low: Low-frequency component [N, 3]
            r_high: High-frequency component [N, 3]
        """
        r_low = self.forward(
            r=r,
            knn_indices=knn_indices,
            knn_weights=knn_weights,
            anchor_positions=anchor_positions,
            anchor_graph=anchor_graph,
        )
        r_high = r - r_low
        return r_low, r_high


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

        self.lambda_anchor_distortion = getattr(args, 'lambda_anchor_distortion', 0.0)
        self.lambda_anchor_time = getattr(args, 'lambda_anchor_time', 0.0)
        self.anchor_time_delta = getattr(args, 'anchor_time_delta', 0.05)
        self.anchor_time_eps = getattr(args, 'anchor_time_eps', 1e-8)
        self.anchor_time_stopgrad_neighbors = getattr(args, 'anchor_time_stopgrad_neighbors', True)
        self.anchor_distortion_k = getattr(args, 'anchor_distortion_k', 8)
        self.anchor_distortion_r_min = getattr(args, 'anchor_distortion_r_min', 0.6)
        self.anchor_distortion_r_max = getattr(args, 'anchor_distortion_r_max', 1.6)
        self.anchor_distortion_eps = getattr(args, 'anchor_distortion_eps', 1e-6)
        self.anchor_distortion_sigma = getattr(args, 'anchor_distortion_sigma', 0.0)

        self._anchor_graph_edges = None
        self._anchor_graph_d0 = None
        self._anchor_graph_w = None
        self._anchor_mass = None
        
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

        # a3: Ray-coverage masking for L_phys (mask under-observed anchors instead of random anchors)
        self.phys_mask_mode = getattr(args, 'phys_mask_mode', 'random')
        self.phys_ray_mask_ratio = float(getattr(args, 'phys_ray_mask_ratio', -1.0) or -1.0)
        self.phys_ray_max_cams = int(getattr(args, 'phys_ray_max_cams', 128) or 128)
        self.phys_ray_ndc_z_thresh = float(getattr(args, 'phys_ray_ndc_z_thresh', 1.0) or 1.0)
        self._phys_ray_mask_indices = None
        self._phys_ray_mask_order = None
        self._phys_ray_coverage = None
        
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

        self.a1_reg_enable = bool(getattr(args, 'a1_reg_enable', False))
        self.a1_reg_lambda = float(getattr(args, 'a1_reg_lambda', 0.0) or 0.0)
        self.a1_reg_beta = float(getattr(args, 'a1_reg_beta', 0.0) or 0.0)
        self.a1_reg_g1_weight = float(getattr(args, 'a1_reg_g1_weight', 1.0) or 1.0)
        self.a1_reg_g2_weight = float(getattr(args, 'a1_reg_g2_weight', 0.0) or 0.0)
        self.a1_reg_k = int(getattr(args, 'a1_reg_k', 8) or 8)
        self.a1_reg_weight_mode = getattr(args, 'a1_reg_weight_mode', 'power')
        self.a1_reg_weight_power = float(getattr(args, 'a1_reg_weight_power', 1.0) or 1.0)
        self.a1_reg_c_thresh = float(getattr(args, 'a1_reg_c_thresh', 0.5) or 0.5)
        self.a1_reg_mask_ratio = float(getattr(args, 'a1_reg_mask_ratio', -1.0) or -1.0)
        self.a1_reg_use_mask_decay = bool(getattr(args, 'a1_reg_use_mask_decay', False))
        self.a1_reg_mask_decay_start = float(getattr(args, 'a1_reg_mask_decay_start', 0.5) or 0.5)
        self.a1_reg_ema_decay = float(getattr(args, 'a1_reg_ema_decay', 0.99) or 0.99)

        self._a1_c_ema = None
        self._a1_anchor_knn_cached_k = 0
        self._a1_anchor_knn_indices = None
        self._last_a1_reg_loss = None
        self._last_a1_reg_g1 = None
        self._last_a1_reg_g2 = None
        self._last_a1_c_mean = None
        self._last_a1_c_min = None
        self._last_a1_c_max = None
        self._last_a1_mask_ratio_eff = None
        self._last_a1_masked_indices = None
        
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
        
        # s1: Per-Anchor Small-Perturbation (spatially-varying γ)
        self.per_anchor_gamma = getattr(args, 'per_anchor_gamma', False)
        # s1.1: Anchor Graph spatial smoothness
        self.lambda_gamma_graph = getattr(args, 'lambda_gamma_graph', 0.0)
        # s1.2: Temporal smoothness
        self.lambda_gamma_temp = getattr(args, 'lambda_gamma_temp', 0.0)
        self.gamma_temp_dt = getattr(args, 'gamma_temp_dt', 0.1)
        
        # s2: Extend anchor fusion to scale/rotation
        self.s2_anchor_to_scale = getattr(args, 's2_anchor_to_scale', False)
        self.s2_anchor_to_rotation = getattr(args, 's2_anchor_to_rotation', False)
        
        # s3: Release scale/rotation from (1-α) multiplier
        self.s3_release_scale = getattr(args, 's3_release_scale', False)
        self.s3_release_rotation = getattr(args, 's3_release_rotation', False)
        self.s3_zero_rotation = getattr(args, 's3_zero_rotation', False)

        # s4.1: Anchor-only position field (dx = α * dx_anchor)
        self.s4_1_anchor_only_position = getattr(args, 's4_1_anchor_only_position', False)
        self.s4_dx_anchor_weight = getattr(args, 's4_dx_anchor_weight', -1.0)
        self.s4_ds_hex_weight = getattr(args, 's4_ds_hex_weight', -1.0)
        self.s4_dr_hex_weight = getattr(args, 's4_dr_hex_weight', -1.0)

        self.s6_trust_region = getattr(args, 's6_trust_region', False)
        self.s6_tau_pos = float(getattr(args, 's6_tau_pos', 0.0) or 0.0)
        self.s6_tau_scale = float(getattr(args, 's6_tau_scale', 0.0) or 0.0)
        self.s6_tau_rot = float(getattr(args, 's6_tau_rot', 0.0) or 0.0)
        self.s6_trust_region_start_ratio = float(getattr(args, 's6_trust_region_start_ratio', 0.0) or 0.0)
        self.s6_tau_pos_start = float(getattr(args, 's6_tau_pos_start', 0.0) or 0.0)
        self.s6_tau_pos_end = float(getattr(args, 's6_tau_pos_end', 0.0) or 0.0)
        self.s6_tau_scale_start = float(getattr(args, 's6_tau_scale_start', 0.0) or 0.0)
        self.s6_tau_scale_end = float(getattr(args, 's6_tau_scale_end', 0.0) or 0.0)
        self.s6_tau_rot_start = float(getattr(args, 's6_tau_rot_start', 0.0) or 0.0)
        self.s6_tau_rot_end = float(getattr(args, 's6_tau_rot_end', 0.0) or 0.0)
        self.s6_trust_region_log = bool(getattr(args, 's6_trust_region_log', False))
        self.s6_trust_region_log_interval = int(getattr(args, 's6_trust_region_log_interval', 1000) or 1000)
        self.s6_eps = float(getattr(args, 's6_eps', 1e-8) or 1e-8)
        self._s6_step = 0

        self.s7_per_anchor_wA = bool(getattr(args, 's7_per_anchor_wA', False))
        self.s7_wA_base = float(getattr(args, 's7_wA_base', -1.0) or -1.0)
        self.s7_wA_delta_max = float(getattr(args, 's7_wA_delta_max', 0.02) or 0.02)
        self.s7_wA_only_up = bool(getattr(args, 's7_wA_only_up', False))
        self.s7_lambda_wA_graph = float(getattr(args, 's7_lambda_wA_graph', 0.0) or 0.0)
        self.s7_lambda_wA_temp = float(getattr(args, 's7_lambda_wA_temp', 0.0) or 0.0)
        self.s7_wA_temp_dt = float(getattr(args, 's7_wA_temp_dt', 0.1) or 0.1)
        self.s7_wA_graph_k = int(getattr(args, 's7_wA_graph_k', 8) or 8)
        self.s7_wA_head = None
        if self.s7_per_anchor_wA:
            hidden = max(8, int(self.d_model // 2))
            self.s7_wA_head = nn.Sequential(
                nn.Linear(self.d_model, hidden),
                nn.GELU(),
                nn.Linear(hidden, 1),
            )
            nn.init.zeros_(self.s7_wA_head[-1].weight)
            nn.init.zeros_(self.s7_wA_head[-1].bias)
        self._last_s7_wA_anchor = None
        self._last_s7_wA_anchor_prev = None
        self._last_s7_wA_graph_loss = None
        self._last_s7_wA_temp_loss = None
        self._s7_anchor_knn_cached_k = 0
        self._s7_anchor_knn_indices = None

        self.s5_rot_nlerp = getattr(args, 's5_rot_nlerp', False)
        self.s5_scale_log_fusion = getattr(args, 's5_scale_log_fusion', False)
        self.s5_jacobian_sr = getattr(args, 's5_jacobian_sr', False)
        self.s5_jacobian_k = getattr(args, 's5_jacobian_k', 8)
        self.s5_eps = getattr(args, 's5_eps', 1e-8)
        
        # s0: Gate function variants for M1.2
        self.s0_gate_type = getattr(args, 's0_gate_type', 'tanh')
        self.s0_normalize_se = getattr(args, 's0_normalize_se', False)
        self.s0_se_ema_decay = getattr(args, 's0_se_ema_decay', 0.99)
        self.s0_residual_mode = getattr(args, 's0_residual_mode', False)
        self.s0_beta_min = getattr(args, 's0_beta_min', 0.005)
        self.s0_beta_max = getattr(args, 's0_beta_max', 0.015)

        # s1.4 preset: s0.1b (sigmoid_bipolar gate) + s1.1 (graph smoothness)
        self.s1_4 = getattr(args, 's1_4', False)
        if self.s1_4:
            self.per_anchor_gamma = True
            self.s0_gate_type = 'sigmoid_bipolar'
            if self.lambda_gamma_graph <= 0:
                self.lambda_gamma_graph = 0.05
        
        # s0.2: EMA statistics for s_E normalization
        self._se_ema_mean = None
        self._se_ema_var = None
        
        # Cache for gamma
        self._last_gamma = None
        self._last_gamma_anchor = None  # s1: per-anchor γᵢ
        self._last_gamma_anchor_prev = None  # s1.2: previous frame γᵢ for temporal smoothness
        self._last_gamma_graph_loss = None  # s1.1: spatial smoothness loss
        self._last_gamma_temp_loss = None   # s1.2: temporal smoothness loss
        
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
        self.residual_mode = getattr(args, 'residual_mode', 'none')
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
        
        # ================================================================
        # M5: Phase-Aware Trust-Region ε(t)
        # ================================================================
        # "Phase-aware trust-region allocates a bounded residual budget across
        #  respiratory phases, preserving Lagrangian dominance while enabling
        #  demand-driven corrections."
        self.phase_eps_enable = getattr(args, 'phase_eps_enable', False)
        self.phase_eps_smooth_lambda = getattr(args, 'phase_eps_smooth_lambda', 1e-4)
        self.phase_epsilon = None
        self._last_phase_eps_smooth_loss = None
        
        if self.phase_eps_enable and self.fusion_mode == 'bounded_perturb':
            # Get M5 parameters, fallback to M2 parameters if not specified
            phase_eps_mode = getattr(args, 'phase_eps_mode', 'per_frame')
            phase_eps_num_frames = getattr(args, 'phase_eps_num_frames', 10)
            phase_eps_mlp_hidden = getattr(args, 'phase_eps_mlp_hidden', 32)
            phase_eps_mlp_layers = getattr(args, 'phase_eps_mlp_layers', 2)
            phase_eps_init = getattr(args, 'phase_eps_init_eps', None)
            phase_eps_max = getattr(args, 'phase_eps_eps_max', None)
            
            # Fallback to M2's eps_init/eps_max if not specified
            if phase_eps_init is None:
                phase_eps_init = self.eps_init
            if phase_eps_max is None:
                phase_eps_max = self.eps_max
            
            self.phase_epsilon = PhaseEpsilon(
                mode=phase_eps_mode,
                num_frames=phase_eps_num_frames,
                mlp_hidden=phase_eps_mlp_hidden,
                mlp_layers=phase_eps_mlp_layers,
                eps_init=phase_eps_init,
                eps_max=phase_eps_max
            )
        
        # ================================================================
        # M6: High-Pass Structural Decomposition of Eulerian Residual
        # ================================================================
        # "Unlike penalty-based regularization, we enforce a structural frequency
        #  split of the Eulerian residual in the forward pass, allocating a bounded
        #  correction budget to the high-frequency component to prevent shortcut
        #  learning."
        #
        # r = Φ_E - Φ_L  (Eulerian residual)
        # r_low = LP(r)  (low-frequency via neighbor average)
        # r_high = r - r_low  (high-frequency)
        # Φ = Φ_L + ε_high * r_high + ε_low * r_low
        self.hpass_enable = getattr(args, 'hpass_enable', False)
        self.hpass_lp_mode = getattr(args, 'hpass_lp_mode', 'knn_cached')
        self.hpass_k = getattr(args, 'hpass_k', 8)
        self.hpass_eps_low_mode = getattr(args, 'hpass_eps_low_mode', 'zero')
        
        # Initialize LP operator
        self.lp_operator = None
        self.rho_high = None  # ε_high = eps_high_max * sigmoid(ρ_high)
        self.rho_low = None   # ε_low = eps_low_max * sigmoid(ρ_low) (bounded_small mode)
        
        # M6 cache for logging
        self._last_eps_high = None
        self._last_eps_low = None
        self._last_E_low = None   # mean ||r_low||
        self._last_E_high = None  # mean ||r_high||
        self._last_E_ratio = None # E_low / (E_high + 1e-8)
        
        if self.hpass_enable and self.fusion_mode in ['bounded_perturb', 'uncertainty_gated']:
            # Initialize LP operator
            self.lp_operator = LowPassOperator(
                mode=self.hpass_lp_mode,
                k=self.hpass_k
            )
            
            # Get eps_high parameters (fallback to M2's eps_max/eps_init)
            eps_high_max = getattr(args, 'hpass_eps_high_max', None)
            eps_high_init = getattr(args, 'hpass_eps_high_init', None)
            if eps_high_max is None:
                eps_high_max = self.eps_max
            if eps_high_init is None:
                eps_high_init = self.eps_init
            self.hpass_eps_high_max = eps_high_max
            self.hpass_eps_high_init = eps_high_init
            
            # Initialize ρ_high such that ε_high = eps_high_init
            eps_ratio = min(max(eps_high_init / eps_high_max, 1e-6), 1 - 1e-6)
            rho_high_init = math.log(eps_ratio / (1 - eps_ratio))  # logit
            self.rho_high = nn.Parameter(torch.tensor(rho_high_init, dtype=torch.float32))
            
            # Initialize ε_low based on mode
            if self.hpass_eps_low_mode == 'bounded_small':
                eps_low_max = getattr(args, 'hpass_eps_low_max', 0.005)
                eps_low_init = getattr(args, 'hpass_eps_low_init', 0.001)
                self.hpass_eps_low_max = eps_low_max
                self.hpass_eps_low_init = eps_low_init
                
                eps_ratio_low = min(max(eps_low_init / eps_low_max, 1e-6), 1 - 1e-6)
                rho_low_init = math.log(eps_ratio_low / (1 - eps_ratio_low))
                self.rho_low = nn.Parameter(torch.tensor(rho_low_init, dtype=torch.float32))
            else:
                self.hpass_eps_low_max = 0.0
                self.hpass_eps_low_init = 0.0
        
        # ================================================================
        # M8: Transport-Correction Decomposition (Predictor-Corrector)
        # ================================================================
        # Serial composition: Lagrangian transport → Eulerian closure
        #   1. Predictor:  x' = x + Φ_L(x,t)
        #   2. Corrector:  Δ = Φ_E(x',t)  [comoving frame]
        #   3. Update:     x(t) = x' + ε·Δ
        # ================================================================
        self.transport_correct_enable = getattr(args, 'transport_correct_enable', False)
        self.transport_correct_eps = getattr(args, 'transport_correct_eps', 0.01)
        self.transport_correct_comoving = getattr(args, 'transport_correct_comoving', True)
        self.transport_correct_learnable_beta = getattr(args, 'transport_correct_learnable_beta', False)
        self.transport_correct_beta_max = getattr(args, 'transport_correct_beta_max', 0.03)
        self.transport_correct_beta_init = getattr(args, 'transport_correct_beta_init', 0.01)
        self.transport_correct_beta_budget = getattr(args, 'transport_correct_beta_budget', 0.01)
        self.transport_correct_lambda_budget = getattr(args, 'transport_correct_lambda_budget', 0.1)
        
        # M8 learnable β network (if enabled)
        self.beta_net = None
        if self.transport_correct_enable and self.transport_correct_learnable_beta:
            # Small MLP: (x', t) → β(x',t)
            # Input: 3 (position) + 1 (time) = 4
            # Output: 1 (scalar β)
            self.beta_net = nn.Sequential(
                nn.Linear(4, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
                nn.Sigmoid()  # β ∈ [0, 1], then scaled by beta_max
            )
        
        # M8 cache for logging
        self._last_tc_eps = None       # effective ε or mean β
        self._last_tc_delta_norm = None  # ||Δ|| mean
        self._last_tc_transport_norm = None  # ||x' - x|| = ||Φ_L|| mean
        self._last_tc_budget_loss = None  # budget penalty if learnable
        
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
                    if self.s2_anchor_to_scale or self.s2_anchor_to_rotation:
                        s2_mode = []
                        if self.s2_anchor_to_scale:
                            s2_mode.append("scale")
                        if self.s2_anchor_to_rotation:
                            s2_mode.append("rotation")
                        print(f"  - s2 ANCHOR FUSION EXTENDED to: {', '.join(s2_mode)}")
                        if self.s2_anchor_to_scale:
                            print(f"    - Δs = (1-α)·Δs_hex + α·Δx_anchor")
                        if self.s2_anchor_to_rotation:
                            print(f"    - Δr = (1-α)·Δr_hex + α·Δx_anchor")
                    if self.s3_release_scale or self.s3_release_rotation or self.s3_zero_rotation:
                        s3_mode = []
                        if self.s3_release_scale:
                            s3_mode.append("scale_released")
                        if self.s3_release_rotation:
                            s3_mode.append("rotation_released")
                        if self.s3_zero_rotation:
                            s3_mode.append("rotation_zeroed")
                        print(f"  - s3 MODE: {', '.join(s3_mode)}")
                        if self.s3_release_scale:
                            print(f"    - Δs = Δs_hex (full HexPlane scale)")
                        if self.s3_release_rotation:
                            print(f"    - Δr = Δr_hex (full HexPlane rotation)")
                        if self.s3_zero_rotation:
                            print(f"    - Δr = 0 (HexPlane rotation completely disabled)")
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
                # s0.3: Residual mode (different fusion formula)
                if self.s0_residual_mode:
                    print(f"  - s0.3 RESIDUAL MODE FUSION (Φ = Φ_L + β·Φ_E):")
                    print(f"    - Formula: Φ = Φ_L + β·Φ_E (base + residual)")
                    print(f"    - β = β_min + (β_max - β_min) * sigmoid((τ - s_E) / λ)")
                    print(f"    - β_min: {self.s0_beta_min}, β_max: {self.s0_beta_max}")
                    print(f"    - τ (gate_tau): {self.gate_tau}, λ (gate_lambda): {self.gate_lambda}")
                elif self.per_anchor_gamma:
                    version = "s1"
                    if self.lambda_gamma_graph > 0 and self.lambda_gamma_temp > 0:
                        version = "s1.1+s1.2"
                    elif self.lambda_gamma_graph > 0:
                        version = "s1.1"
                    elif self.lambda_gamma_temp > 0:
                        version = "s1.2"
                    print(f"  - {version} PER-ANCHOR SMALL PERTURBATION (spatially-varying γᵢ):")
                    print(f"    - Formula: Φ = (0.99-γ(x,t))·Φ_L + (0.01+γ(x,t))·Φ_E")
                    gate_func = "tanh" if self.s0_gate_type == 'tanh' else self.s0_gate_type
                    print(f"    - γ(x,t) = Σ wᵢ(x)·γᵢ(t), γᵢ = γ_max * {gate_func}((τ - s_E(i,t)) / λ)")
                    print(f"    - s_E(i,t) aggregated from Gaussians via KNN weights")
                    if self.lambda_gamma_graph > 0:
                        print(f"    - s1.1 SPATIAL SMOOTHNESS: L_graph = λ·Σ(γᵢ-γⱼ)², λ={self.lambda_gamma_graph}")
                    if self.lambda_gamma_temp > 0:
                        print(f"    - s1.2 TEMPORAL SMOOTHNESS: L_temp = λ·Σ|γᵢ(t)-γᵢ(t-Δt)|², λ={self.lambda_gamma_temp}")
                else:
                    print(f"  - M1.2 SMALL PERTURBATION FUSION (preserves V5's 99:1 ratio):")
                    print(f"    - Formula: Φ = (0.99-γ)·Φ_L + (0.01+γ)·Φ_E")
                    gate_func = "tanh" if self.s0_gate_type == 'tanh' else self.s0_gate_type
                    print(f"    - γ = γ_max * {gate_func}((τ - s_E) / λ)")
                # s0 gate variants logging
                if self.s0_gate_type != 'tanh':
                    if self.s0_gate_type == 'sigmoid':
                        print(f"    - s0.1a GATE: sigmoid (γ ∈ (0, γ_max), positive only)")
                    elif self.s0_gate_type == 'sigmoid_bipolar':
                        print(f"    - s0.1b GATE: sigmoid_bipolar (γ = γ_max*(2σ-1), softer than tanh)")
                if self.s0_normalize_se:
                    print(f"    - s0.2 NORMALIZE s_E: EMA normalization (decay={self.s0_se_ema_decay})")
                if not self.s0_residual_mode:
                    print(f"    - γ_max: {self.gamma_max} (HexPlane weight range: [{0.01-self.gamma_max:.3f}, {0.01+self.gamma_max:.3f}])")
                    print(f"    - τ (gate_tau): {self.gate_tau}, λ (gate_lambda): {self.gate_lambda}")
                    print(f"    - m1_lambda_gate: {self.m1_lambda_gate}")
                if self.hpass_enable and self.lp_operator is not None:
                    print(f"  - M7 HIGH-PASS STRUCTURAL DECOMPOSITION ON M1.2:")
                    print(f"    - Formula: Φ = Φ_L + hex_weight·(r_high + tied_factor·r_low)")
                    print(f"    - r = Φ_E - Φ_L, decompose into r_low (LP) + r_high")
                    print(f"    - lp_mode: {self.hpass_lp_mode}, k: {self.hpass_k}")
                    print(f"    - eps_low_mode: {self.hpass_eps_low_mode}")
                    if self.hpass_eps_low_mode == 'zero':
                        print(f"    - tied_factor: 0 (hard high-pass)")
                    elif self.hpass_eps_low_mode == 'tied':
                        print(f"    - tied_factor: 1.0 (sanity check, = M1.2 baseline)")
                    else:
                        print(f"    - tied_factor: learnable via rho_low")
            if self.fusion_mode == 'bounded_perturb':
                if self.residual_mode == 'none':
                    print(f"  - M2.1a LEARNABLE WEIGHTED AVERAGE + TRUST-REGION:")
                    print(f"    - Formula: Φ = (1-ε_eff)·Φ_L + ε_eff·Φ_E")
                else:
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
                if self.phase_eps_enable and self.phase_epsilon is not None:
                    print(f"  - M5 PHASE-AWARE TRUST-REGION ε(t):")
                    print(f"    - 'Phase-aware trust-region allocates a bounded residual budget")
                    print(f"       across respiratory phases, preserving Lagrangian dominance")
                    print(f"       while enabling demand-driven corrections.'")
                    print(f"    - mode: {self.phase_epsilon.mode}")
                    print(f"    - eps_init: {self.phase_epsilon.eps_init:.4f}")
                    print(f"    - eps_max: {self.phase_epsilon.eps_max:.4f}")
                    if self.phase_epsilon.mode == 'per_frame':
                        print(f"    - num_frames: {self.phase_epsilon.num_frames}")
                    else:
                        print(f"    - mlp_hidden: {getattr(self.args, 'phase_eps_mlp_hidden', 32)}")
                        print(f"    - mlp_layers: {getattr(self.args, 'phase_eps_mlp_layers', 2)}")
                    print(f"    - smooth_lambda: {self.phase_eps_smooth_lambda}")
                    print(f"    - freeze_steps: {self.freeze_steps} (inherits M2.1a schedule)")
                if self.hpass_enable and self.lp_operator is not None:
                    print(f"  - M6 HIGH-PASS STRUCTURAL DECOMPOSITION:")
                    print(f"    - 'Unlike penalty-based regularization, we enforce a structural")
                    print(f"       frequency split of the Eulerian residual in the forward pass,")
                    print(f"       allocating a bounded correction budget to the high-frequency")
                    print(f"       component to prevent shortcut learning.'")
                    print(f"    - Formula: Φ = Φ_L + ε_high·r_high + ε_low·r_low")
                    print(f"    - lp_mode: {self.hpass_lp_mode}, k: {self.hpass_k}")
                    print(f"    - eps_low_mode: {self.hpass_eps_low_mode}")
                    print(f"    - eps_high_max: {self.hpass_eps_high_max:.4f}, eps_high_init: {self.hpass_eps_high_init:.4f}")
                    if self.hpass_eps_low_mode == 'bounded_small':
                        print(f"    - eps_low_max: {self.hpass_eps_low_max:.4f}, eps_low_init: {self.hpass_eps_low_init:.4f}")
                    elif self.hpass_eps_low_mode == 'zero':
                        print(f"    - eps_low: 0 (hard high-pass, only r_high contributes)")
                    else:  # tied
                        print(f"    - eps_low: tied to eps_high (sanity check, = baseline)")
                    print(f"    - freeze_steps: {self.freeze_steps} (inherits M2.1a schedule)")
            # M8 print moved outside bounded_perturb block
            if self.transport_correct_enable:
                print(f"  - M8 TRANSPORT-CORRECTION DECOMPOSITION (Predictor-Corrector):")
                print(f"    - 'Serial composition: Lagrangian transport followed by")
                print(f"       Eulerian closure in the comoving frame.'")
                print(f"    - Step 1 (Predictor): x' = x + Φ_L(x,t)")
                if self.transport_correct_comoving:
                    print(f"    - Step 2 (Corrector): Δ = Φ_E(x',t)  [COMOVING FRAME]")
                else:
                    print(f"    - Step 2 (Corrector): Δ = Φ_E(x,t)   [ORIGINAL FRAME - ablation]")
                if self.transport_correct_learnable_beta:
                    print(f"    - Step 3 (Update): x(t) = x' + β(x',t)·Δ  [LEARNABLE β]")
                    print(f"    - β_max: {self.transport_correct_beta_max}")
                    print(f"    - β_init: {self.transport_correct_beta_init}")
                    print(f"    - β_budget: E[β] ≤ {self.transport_correct_beta_budget}")
                    print(f"    - λ_budget: {self.transport_correct_lambda_budget}")
                else:
                    print(f"    - Step 3 (Update): x(t) = x' + ε·Δ  [FIXED ε]")
                    print(f"    - ε: {self.transport_correct_eps}")
    
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

        self._build_anchor_graph()
        
        print(f"[PhysX-Gaussian] Initialized {actual_num_anchors} anchors via FPS from {num_points} points")

    def _build_anchor_graph(self) -> None:
        if not self.initialized or self.anchor_positions is None:
            self._anchor_graph_edges = None
            self._anchor_graph_d0 = None
            self._anchor_graph_w = None
            return

        anchor_pos = self.anchor_positions.detach()
        M = anchor_pos.shape[0]
        if M < 2:
            self._anchor_graph_edges = None
            self._anchor_graph_d0 = None
            self._anchor_graph_w = None
            return

        dists = torch.cdist(anchor_pos, anchor_pos, p=2)
        k = int(max(1, min(self.anchor_distortion_k, M - 1)))
        _, nn_idx = torch.topk(-dists, k + 1, dim=-1)
        nn_idx = nn_idx[:, 1:]

        src = torch.arange(M, device=anchor_pos.device).unsqueeze(1).expand(M, k).reshape(-1)
        dst = nn_idx.reshape(-1)
        edges = torch.stack([src, dst], dim=-1)

        d0 = dists[src, dst]
        w = None
        sigma = float(self.anchor_distortion_sigma)
        if sigma > 0:
            w = torch.exp(-(d0 ** 2) / (sigma ** 2))

        self._anchor_graph_edges = edges.detach()
        self._anchor_graph_d0 = d0.detach()
        self._anchor_graph_w = w.detach() if w is not None else None
    
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

        M = int(self.anchor_positions.shape[0])
        if M > 0:
            idx = self.knn_indices.reshape(-1)
            w = self.knn_weights.reshape(-1)
            mass = torch.zeros((M,), device=idx.device, dtype=w.dtype)
            mass.scatter_add_(0, idx, w)
            self._anchor_mass = mass.detach()
        else:
            self._anchor_mass = None
    
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
        self._last_a1_mask_ratio_eff = None
        
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

        if self.a1_reg_use_mask_decay:
            a1_mask_ratio_eff = self.a1_reg_mask_decay_start * (1.0 - iteration_ratio)
        elif self.a1_reg_mask_ratio >= 0:
            a1_mask_ratio_eff = self.a1_reg_mask_ratio
        else:
            a1_mask_ratio_eff = effective_mask_ratio
        self._last_a1_mask_ratio_eff = float(a1_mask_ratio_eff)
        
        # Masking for BERT-style training
        # V10: When use_decoupled_mask=True, skip masking in main forward (render path)
        # Masking will be done separately in forward_anchors_masked() for L_phys
        masked_indices = None

        # a3: For ray-coverage masking, allow forcing a nonzero mask ratio even if mask_ratio=0.
        # When phys_ray_mask_ratio>=0 we interpret it as an explicit override.
        phys_mask_ratio_override = None
        if self.phys_mask_mode == 'ray_coverage' and self.phys_ray_mask_ratio >= 0:
            phys_mask_ratio_override = float(max(0.0, min(1.0, float(self.phys_ray_mask_ratio))))

        should_mask = (
            is_training
            and (not self.use_decoupled_mask)
            and (
                (effective_mask_ratio > 0)
                or (phys_mask_ratio_override is not None and phys_mask_ratio_override > 0)
            )
        )
        
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

        if self.a1_reg_enable:
            a1_masked_indices = None
            if is_training and a1_mask_ratio_eff > 0:
                if self.use_temporal_mask and is_training and masked_indices is not None and masked_indices.numel() == M:
                    a1_masked_indices = masked_indices
                else:
                    num_mask_a1 = int(M * a1_mask_ratio_eff)
                    if num_mask_a1 > 0:
                        perm_a1 = torch.randperm(M, device=device)
                        a1_masked_indices = perm_a1[:num_mask_a1]
            self._last_a1_masked_indices = a1_masked_indices

            c = torch.ones(M, device=device)
            if a1_masked_indices is not None:
                c[a1_masked_indices] = 0.0

            if a1_mask_ratio_eff <= 0:
                c.fill_(1.0)

            if self._a1_c_ema is None or (not torch.is_tensor(self._a1_c_ema)) or self._a1_c_ema.numel() != M:
                self._a1_c_ema = c.detach()
            else:
                self._a1_c_ema = (self.a1_reg_ema_decay * self._a1_c_ema + (1.0 - self.a1_reg_ema_decay) * c.detach())

            self._last_a1_c_mean = float(self._a1_c_ema.mean().item())
            self._last_a1_c_min = float(self._a1_c_ema.min().item())
            self._last_a1_c_max = float(self._a1_c_ema.max().item())
        
        if should_mask:
            # Choose how many anchors to mask.
            mask_ratio_eff = float(phys_mask_ratio_override) if phys_mask_ratio_override is not None else float(effective_mask_ratio)
            num_mask = int(M * mask_ratio_eff)
            if num_mask > 0:
                if self.phys_mask_mode == 'ray_coverage' and self._phys_ray_mask_order is not None and self._phys_ray_mask_order.numel() > 0:
                    masked_indices = self._phys_ray_mask_order[:num_mask].to(device)
                else:
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

    def _a1_get_anchor_knn(self, k: int) -> torch.Tensor:
        M = int(self.anchor_positions.shape[0])
        k = int(max(1, min(k, M - 1)))
        if self._a1_anchor_knn_indices is not None and self._a1_anchor_knn_cached_k == k:
            return self._a1_anchor_knn_indices
        with torch.no_grad():
            anchor_pos = self.anchor_positions.detach()
            dist_sq = torch.cdist(anchor_pos, anchor_pos, p=2) ** 2
            dist_sq.fill_diagonal_(float('inf'))
            _, knn = torch.topk(dist_sq, k=k, largest=False)
            self._a1_anchor_knn_indices = knn.contiguous()
            self._a1_anchor_knn_cached_k = k
        return self._a1_anchor_knn_indices

    def compute_a1_regularization_loss(self) -> torch.Tensor:
        if not self.a1_reg_enable:
            return None
        if self._last_anchor_displacements is None:
            return None
        if self._a1_c_ema is None:
            return None

        device = self.anchor_positions.device
        dx = self._last_anchor_displacements
        M = int(dx.shape[0])
        if M < 2:
            return None

        a_knn = self._a1_get_anchor_knn(self.a1_reg_k).to(device)  # [M, k]

        c = self._a1_c_ema.to(device)
        if self.a1_reg_weight_mode == 'hard':
            w = (1.0 - (c >= self.a1_reg_c_thresh).float())
        else:
            w = (1.0 - c).clamp(0.0, 1.0)
            if self.a1_reg_weight_mode == 'power':
                w = w ** self.a1_reg_weight_power
            elif self.a1_reg_weight_mode == 'square':
                w = w ** 2
        w = w.detach()

        dx_i = dx.unsqueeze(1)              # [M, 1, 3]
        dx_j = dx[a_knn]                    # [M, k, 3]
        g1 = ((dx_i - dx_j) ** 2).sum(dim=-1).mean(dim=-1)  # [M]

        loss_g1 = (w * g1).mean()
        loss_g2 = None
        g2_weight_eff = self.a1_reg_g2_weight if self.a1_reg_g2_weight != 0 else self.a1_reg_beta
        if g2_weight_eff > 0:
            lap = (dx_j.mean(dim=1) - dx)                # [M, 3]
            dx_lap_j = lap[a_knn]                        # [M, k, 3]
            g2 = ((lap.unsqueeze(1) - dx_lap_j) ** 2).sum(dim=-1).mean(dim=-1)  # [M]
            loss_g2 = (w * g2).mean()

        if self.a1_reg_g1_weight == 0 and g2_weight_eff == 0:
            return None

        reg = self.a1_reg_g1_weight * loss_g1
        if loss_g2 is not None and g2_weight_eff != 0:
            reg = reg + g2_weight_eff * loss_g2

        self._last_a1_reg_loss = float(reg.detach().item())
        self._last_a1_reg_g1 = float(loss_g1.detach().item())
        self._last_a1_reg_g2 = float(loss_g2.detach().item()) if loss_g2 is not None else None
        return reg

    def get_a1_stats(self) -> dict:
        if not self.a1_reg_enable:
            return {}
        return {
            'a1_L_reg': self._last_a1_reg_loss,
            'a1_L_g1': self._last_a1_reg_g1,
            'a1_L_g2': self._last_a1_reg_g2,
            'a1_c_mean': self._last_a1_c_mean,
            'a1_c_min': self._last_a1_c_min,
            'a1_c_max': self._last_a1_c_max,
            'a1_mask_ratio_eff': self._last_a1_mask_ratio_eff,
        }

    def _s7_get_anchor_knn(self, k: int) -> torch.Tensor:
        M = int(self.anchor_positions.shape[0])
        k = int(max(1, min(k, M - 1)))
        if self._s7_anchor_knn_indices is not None and self._s7_anchor_knn_cached_k == k:
            return self._s7_anchor_knn_indices
        with torch.no_grad():
            anchor_pos = self.anchor_positions.detach()
            dist_sq = torch.cdist(anchor_pos, anchor_pos, p=2) ** 2
            dist_sq.fill_diagonal_(float('inf'))
            _, knn = torch.topk(dist_sq, k=k, largest=False)
            self._s7_anchor_knn_indices = knn.contiguous()
            self._s7_anchor_knn_cached_k = k
        return self._s7_anchor_knn_indices

    def _s7_compute_wA(self, anchor_features: torch.Tensor, N: int, wA_base: float):
        if (not self.s7_per_anchor_wA) or (self.s7_wA_head is None) or (anchor_features is None):
            return None, None
        M = int(anchor_features.shape[0])
        knn_idx = self.knn_indices[:N]
        knn_w = self.knn_weights[:N]
        raw = self.s7_wA_head(anchor_features)  # [M, 1]
        if self.s7_wA_only_up:
            # Constrain wA to only increase from base: wA ∈ [wA_base, wA_base + delta_max]
            # This prevents the optimizer from collapsing wA below the "sweet spot" base.
            delta = self.s7_wA_delta_max * torch.sigmoid(raw)
        else:
            delta = self.s7_wA_delta_max * torch.tanh(raw)
        wA_anchor = torch.clamp(delta + float(wA_base), 0.0, 1.0)  # [M, 1]
        wA_neighbors = wA_anchor[knn_idx]  # [N, K, 1]
        wA_gauss = (wA_neighbors.squeeze(-1) * knn_w).sum(dim=1, keepdim=True)  # [N, 1]
        self._last_s7_wA_anchor = wA_anchor.squeeze(-1)

        if self.s7_lambda_wA_graph > 0:
            a_knn = self._s7_get_anchor_knn(self.s7_wA_graph_k)
            wA_i = wA_anchor.squeeze(-1).unsqueeze(1)
            wA_j = wA_anchor.squeeze(-1)[a_knn]
            graph_loss = ((wA_i - wA_j) ** 2).mean()
            self._last_s7_wA_graph_loss = self.s7_lambda_wA_graph * graph_loss
        else:
            self._last_s7_wA_graph_loss = None

        if self.s7_lambda_wA_temp > 0 and self._last_s7_wA_anchor_prev is not None:
            temp_loss = ((wA_anchor.squeeze(-1) - self._last_s7_wA_anchor_prev) ** 2).mean()
            self._last_s7_wA_temp_loss = self.s7_lambda_wA_temp * temp_loss
        else:
            self._last_s7_wA_temp_loss = None

        self._last_s7_wA_anchor_prev = wA_anchor.squeeze(-1).detach().clone()
        return wA_gauss, wA_anchor
    
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

    def build_phys_ray_coverage_mask(self, train_cameras, max_cams: Optional[int] = None) -> None:
        if self.phys_mask_mode != 'ray_coverage':
            return
        if train_cameras is None or len(train_cameras) == 0:
            return
        device = self.anchor_positions.device
        M = int(self.anchor_positions.shape[0])
        if M <= 0:
            return

        max_cams = int(max_cams or self.phys_ray_max_cams)
        cams = list(train_cameras)
        if max_cams > 0 and len(cams) > max_cams:
            stride = max(1, len(cams) // max_cams)
            cams = cams[::stride][:max_cams]

        with torch.no_grad():
            anchor_pos = self.anchor_positions.detach()
            ones = torch.ones((M, 1), dtype=anchor_pos.dtype, device=device)
            X = torch.cat([anchor_pos, ones], dim=1)  # [M, 4]
            coverage = torch.zeros((M,), dtype=torch.float32, device=device)
            z_th = float(self.phys_ray_ndc_z_thresh)
            for cam in cams:
                P = cam.full_proj_transform.to(device)
                # NOTE: Use row-vector convention consistent with dataset.project_point():
                # clip = X @ world_view_transform^T @ projection_matrix^T.
                # Since full_proj_transform = world_view_transform @ projection_matrix,
                # we apply the transpose here.
                clip = X @ P.T  # [M, 4]
                w = clip[:, 3:4]
                w_safe = torch.clamp(w, min=1e-8)
                ndc = clip[:, :3] / w_safe
                in_view = (
                    (w[:, 0] > 0)
                    & (ndc[:, 0].abs() <= 1.0)
                    & (ndc[:, 1].abs() <= 1.0)
                    & (ndc[:, 2].abs() <= z_th)
                )
                coverage += in_view.float()

            self._phys_ray_coverage = coverage
            _, order = torch.sort(coverage, descending=False)
            self._phys_ray_mask_order = order.contiguous()
            self._phys_ray_mask_indices = None
    
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
        else:
            if self.phys_mask_mode == 'ray_coverage' and self._phys_ray_mask_order is not None and self._phys_ray_mask_order.numel() > 0:
                if self.phys_ray_mask_ratio >= 0:
                    ratio = float(self.phys_ray_mask_ratio)
                    ratio = float(max(0.0, min(1.0, ratio)))
                    num_mask = int(M * ratio)
                else:
                    num_mask = int(M * effective_mask_ratio)
                if num_mask > 0:
                    masked_indices = self._phys_ray_mask_order[:num_mask].to(device)
                    mask_tokens = self.mask_token.expand(1, num_mask, -1)
                    anchor_features[0, masked_indices] = mask_tokens.squeeze(0)
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

    def _quat_normalize(self, q: torch.Tensor) -> torch.Tensor:
        return q / (torch.norm(q, dim=-1, keepdim=True) + self.s5_eps)

    def _clamp_norm(self, x: torch.Tensor, tau: float, dim: int = -1) -> torch.Tensor:
        if tau is None or tau <= 0:
            return x
        n = torch.norm(x, dim=dim, keepdim=True)
        # scale = min(1, tau / (n + eps))
        scale = torch.clamp(tau / (n + self.s6_eps), max=1.0)
        return x * scale

    def _clamp_norm_with_stats(self, x: torch.Tensor, tau: float, dim: int = -1):
        if tau is None or tau <= 0:
            stats = {
                'tau': float(tau or 0.0),
                'clamp_ratio': 0.0,
                'mean_scale': 1.0,
                'mean_norm': 0.0,
                'mean_norm_clamped': 0.0,
            }
            return x, stats
        n = torch.norm(x, dim=dim, keepdim=True)
        scale = torch.clamp(tau / (n + self.s6_eps), max=1.0)
        x_clamped = x * scale
        scale_flat = scale.detach().view(-1)
        n_flat = n.detach().view(-1)
        stats = {
            'tau': float(tau),
            'clamp_ratio': float((scale_flat < 1.0).float().mean().item()) if scale_flat.numel() > 0 else 0.0,
            'mean_scale': float(scale_flat.mean().item()) if scale_flat.numel() > 0 else 1.0,
            'mean_norm': float(n_flat.mean().item()) if n_flat.numel() > 0 else 0.0,
            'mean_norm_clamped': float((n_flat * scale_flat).mean().item()) if n_flat.numel() > 0 else 0.0,
        }
        return x_clamped, stats

    def _s6_tau(self, tau_fixed: float, tau_start: float, tau_end: float, iteration_ratio: float) -> float:
        if tau_fixed is not None and tau_fixed > 0:
            return float(tau_fixed)
        if (tau_start is None or tau_start <= 0) and (tau_end is None or tau_end <= 0):
            return 0.0
        t = float(iteration_ratio)
        t = 0.0 if t < 0.0 else (1.0 if t > 1.0 else t)
        return float((1.0 - t) * float(tau_start) + t * float(tau_end))

    def _trust_region_scale_delta(self, scales: torch.Tensor, ds: torch.Tensor, tau_scale: float) -> torch.Tensor:
        # Note: in this codebase `scales` are *raw* Gaussian scale parameters (logits if scale_bound is enabled).
        # We therefore implement a safe trust-region in the raw-delta space by limiting ||ds||.
        if tau_scale is None or tau_scale <= 0:
            return ds
        return self._clamp_norm(ds, tau_scale, dim=-1)

    def _quat_from_matrix(self, R: torch.Tensor) -> torch.Tensor:
        t = R[..., 0, 0] + R[..., 1, 1] + R[..., 2, 2]
        r = torch.sqrt(torch.clamp(1.0 + t, min=self.s5_eps))
        w = 0.5 * r
        inv4w = 0.25 / (w + self.s5_eps)
        x = (R[..., 2, 1] - R[..., 1, 2]) * inv4w
        y = (R[..., 0, 2] - R[..., 2, 0]) * inv4w
        z = (R[..., 1, 0] - R[..., 0, 1]) * inv4w
        q = torch.stack([w, x, y, z], dim=-1)
        return self._quat_normalize(q)

    def _jacobian_sr_reference(
        self,
        gaussian_positions: torch.Tensor,
        scales: torch.Tensor,
        rotations: torch.Tensor,
        anchor_positions: torch.Tensor,
        anchor_displacements: torch.Tensor,
        knn_idx: torch.Tensor,
        knn_w: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        N, K = knn_idx.shape
        pos_i = anchor_positions[knn_idx]  # [N, K, 3]
        u_i = anchor_displacements[knn_idx]  # [N, K, 3]
        w = knn_w.unsqueeze(-1)  # [N, K, 1]

        x = gaussian_positions.unsqueeze(1)  # [N, 1, 3]
        p = pos_i - x  # [N, K, 3]
        u0 = (w * u_i).sum(dim=1, keepdim=True) / (w.sum(dim=1, keepdim=True) + self.s5_eps)  # [N, 1, 3]
        du = u_i - u0  # [N, K, 3]

        P = p.unsqueeze(-1)  # [N, K, 3, 1]
        PT = p.unsqueeze(-2)  # [N, K, 1, 3]
        W = knn_w.view(N, K, 1, 1)
        A = (W * P @ PT).sum(dim=1)  # [N, 3, 3]
        B = (W * du.unsqueeze(-1) @ PT).sum(dim=1)  # [N, 3, 3]

        I = torch.eye(3, device=gaussian_positions.device, dtype=gaussian_positions.dtype).unsqueeze(0)
        A_reg = A + (self.s5_eps * I)
        J = torch.linalg.solve(A_reg, B).transpose(-1, -2)  # [N, 3, 3]
        Fm = I + J

        U, S, Vh = torch.linalg.svd(Fm)
        Rm = U @ Vh
        det = torch.det(Rm)
        mask = det < 0
        if mask.any():
            U2 = U.clone()
            U2[mask, :, 2] *= -1
            Rm = U2 @ Vh
            U = U2

        log_s_iso = torch.log(torch.clamp(S, min=self.s5_eps)).mean(dim=-1, keepdim=True)  # [N, 1]
        s_ref = scales * torch.exp(log_s_iso.expand_as(scales))

        q_delta = self._quat_from_matrix(Rm)
        q_ref = batch_quaternion_multiply(rotations, q_delta)
        q_ref = self._quat_normalize(q_ref)
        return s_ref, q_ref
    
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
            anchor_features_s7 = None
            if self.s7_per_anchor_wA:
                anchor_displacements, _, anchor_features_s7 = self.forward_anchors(
                    time_emb, is_training=is_training, return_all_info=True, iteration_ratio=iteration_ratio
                )
            else:
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
                # s1: Per-Anchor extension (spatially-varying γᵢ)
                # ================================================================
                # KEY INSIGHT from experiments:
                #   - V5 with α=0.99 (99% Anchor, 1% HexPlane) is OPTIMAL
                #   - Any other ratio is worse
                #   - This means 99:1 gradient ratio is also optimal for training!
                #
                # M1.2: Global γ from uncertainty s_E
                #   γ = γ_max * tanh((τ - s_E) / λ)
                #
                # s1: Per-anchor γᵢ(t) with spatial interpolation
                #   1. Aggregate s_E(x,t) to anchors: s_E(i,t) = Σ wᵢ(x)·s_E(x,t) / Σ wᵢ(x)
                #   2. Compute γᵢ = γ_max * tanh((τ - s_E(i,t)) / λ)
                #   3. Interpolate to Gaussians: γ(x,t) = Σ wᵢ(x)·γᵢ
                # ================================================================
                
                # Get s_E from HexPlane uncertainty head
                s_E = self.original_deformation.get_last_s_E()  # [N, 1]
                
                # s0.2: Normalize s_E with EMA statistics
                if s_E is not None and self.s0_normalize_se:
                    s_E_mean = s_E.mean()
                    s_E_var = s_E.var() + 1e-8
                    
                    # Update EMA
                    if self._se_ema_mean is None:
                        self._se_ema_mean = s_E_mean.detach()
                        self._se_ema_var = s_E_var.detach()
                    else:
                        decay = self.s0_se_ema_decay
                        self._se_ema_mean = decay * self._se_ema_mean + (1 - decay) * s_E_mean.detach()
                        self._se_ema_var = decay * self._se_ema_var + (1 - decay) * s_E_var.detach()
                    
                    # Normalize
                    s_E = (s_E - self._se_ema_mean) / (torch.sqrt(self._se_ema_var) + 1e-8)
                
                if s_E is None:
                    # Fallback: if s_E not computed, use pure V5 (γ=0)
                    gamma = torch.zeros_like(dx_hex[:, :1])
                    self._last_gamma_anchor = None
                elif self.per_anchor_gamma:
                    # ============================================================
                    # s1: Per-Anchor Small-Perturbation
                    # ============================================================
                    N = s_E.shape[0]
                    M = self.num_anchors
                    K = self.anchor_k
                    
                    # Step 1: Aggregate s_E(x,t) to anchors via scatter
                    # s_E(i,t) = Σ_x wᵢ(x)·s_E(x,t) / (Σ_x wᵢ(x) + ε)
                    s_E_flat = s_E.view(-1)  # [N]
                    
                    # Expand weights and indices for scatter
                    knn_idx = self.knn_indices[:N]  # [N, K]
                    knn_w = self.knn_weights[:N]    # [N, K]
                    
                    # Weighted s_E per (Gaussian, anchor) pair
                    weighted_s_E = (s_E_flat.unsqueeze(1) * knn_w).view(-1)  # [N*K]
                    flat_idx = knn_idx.view(-1)  # [N*K]
                    flat_w = knn_w.view(-1)  # [N*K]
                    
                    # Scatter sum to anchors
                    s_E_anchor_sum = torch.zeros(M, device=s_E.device, dtype=s_E.dtype)
                    w_anchor_sum = torch.zeros(M, device=s_E.device, dtype=s_E.dtype)
                    s_E_anchor_sum.scatter_add_(0, flat_idx, weighted_s_E)
                    w_anchor_sum.scatter_add_(0, flat_idx, flat_w)
                    
                    # Normalize: s_E(i,t) = sum / (weight_sum + eps)
                    s_E_anchor = s_E_anchor_sum / (w_anchor_sum + 1e-8)  # [M]
                    
                    # Step 2: Compute γᵢ using s0 gate type
                    gate_input = (self.gate_tau - s_E_anchor) / (self.gate_lambda + 1e-8)
                    if self.s0_gate_type == 'sigmoid':
                        # s0.1a: γ = γ_max * sigmoid(...)  ∈ (0, γ_max)
                        gamma_anchor = self.gamma_max * torch.sigmoid(gate_input)
                    elif self.s0_gate_type == 'sigmoid_bipolar':
                        # s0.1b: γ = γ_max * (2*sigmoid(...) - 1)  ∈ (-γ_max, γ_max)
                        gamma_anchor = self.gamma_max * (2 * torch.sigmoid(gate_input) - 1)
                    else:
                        # Default (tanh): γ = γ_max * tanh(...)  ∈ (-γ_max, γ_max)
                        gamma_anchor = self.gamma_max * torch.tanh(gate_input)
                    # [M]
                    
                    # Step 3: Interpolate back to Gaussians: γ(x,t) = Σ wᵢ(x)·γᵢ
                    gamma_neighbors = gamma_anchor[knn_idx]  # [N, K]
                    gamma = (gamma_neighbors * knn_w).sum(dim=1, keepdim=True)  # [N, 1]
                    
                    # ============================================================
                    # s1.1: Anchor Graph spatial smoothness
                    # L_graph = Σ_{(i,j)∈E} (γᵢ - γⱼ)²
                    # ============================================================
                    if self.lambda_gamma_graph > 0:
                        # Use anchor KNN graph as edge set E
                        # anchor_knn_indices: [M, k] - each anchor's k nearest anchor neighbors
                        if hasattr(self, 'anchor_knn_indices') and self.anchor_knn_indices is not None:
                            anchor_knn = self.anchor_knn_indices  # [M, k]
                            gamma_i = gamma_anchor.unsqueeze(1)  # [M, 1]
                            gamma_j = gamma_anchor[anchor_knn]   # [M, k]
                            graph_loss = ((gamma_i - gamma_j) ** 2).mean()
                            self._last_gamma_graph_loss = self.lambda_gamma_graph * graph_loss
                        else:
                            # Fallback: compute anchor KNN on-the-fly
                            anchor_pos = self.anchor_positions.detach()  # [M, 3]
                            dist_sq = torch.cdist(anchor_pos, anchor_pos, p=2) ** 2  # [M, M]
                            # Exclude self (set diagonal to inf)
                            dist_sq.fill_diagonal_(float('inf'))
                            _, anchor_knn = torch.topk(dist_sq, k=min(8, M-1), largest=False)  # [M, k]
                            gamma_i = gamma_anchor.unsqueeze(1)  # [M, 1]
                            gamma_j = gamma_anchor[anchor_knn]   # [M, k]
                            graph_loss = ((gamma_i - gamma_j) ** 2).mean()
                            self._last_gamma_graph_loss = self.lambda_gamma_graph * graph_loss
                    else:
                        self._last_gamma_graph_loss = None
                    
                    # ============================================================
                    # s1.2: Temporal smoothness
                    # L_temp = Σᵢ |γᵢ(t) - γᵢ(t-Δt)|²
                    # ============================================================
                    if self.lambda_gamma_temp > 0 and self._last_gamma_anchor_prev is not None:
                        temp_loss = ((gamma_anchor - self._last_gamma_anchor_prev) ** 2).mean()
                        self._last_gamma_temp_loss = self.lambda_gamma_temp * temp_loss
                    else:
                        self._last_gamma_temp_loss = None
                    
                    # Update previous gamma for next iteration
                    self._last_gamma_anchor_prev = gamma_anchor.detach().clone()
                    
                    # Cache for logging
                    self._last_gamma_anchor = gamma_anchor
                else:
                    # Original M1.2: Global γ from s_E with s0 gate type
                    gate_input = (self.gate_tau - s_E) / (self.gate_lambda + 1e-8)
                    if self.s0_gate_type == 'sigmoid':
                        # s0.1a: γ = γ_max * sigmoid(...)  ∈ (0, γ_max)
                        gamma = self.gamma_max * torch.sigmoid(gate_input)
                    elif self.s0_gate_type == 'sigmoid_bipolar':
                        # s0.1b: γ = γ_max * (2*sigmoid(...) - 1)  ∈ (-γ_max, γ_max)
                        gamma = self.gamma_max * (2 * torch.sigmoid(gate_input) - 1)
                    else:
                        # Default (tanh): γ = γ_max * tanh(...)  ∈ (-γ_max, γ_max)
                        gamma = self.gamma_max * torch.tanh(gate_input)
                    self._last_gamma_anchor = None
                
                # Cache for logging
                self._last_s_E = s_E
                self._last_gamma = gamma
                
                # s0.3: Residual mode - Φ = Φ_L + β·Φ_E
                if self.s0_residual_mode:
                    # β = β_min + (β_max - β_min) * sigmoid((τ - s_E) / λ)
                    gate_input = (self.gate_tau - s_E) / (self.gate_lambda + 1e-8)
                    beta = self.s0_beta_min + (self.s0_beta_max - self.s0_beta_min) * torch.sigmoid(gate_input)
                    hex_weight_effective = beta
                    self._last_beta = beta
                    self._last_beta_mean = beta.mean().item()
                    
                    # s0.3 Fusion: Φ = Φ_L + β·Φ_E (base + residual)
                    hex_weight = beta
                    anchor_weight = torch.ones_like(beta)  # Anchor is always 1.0
                else:
                    hex_weight_effective = 0.01 + gamma
                    self._last_beta = hex_weight_effective  # For compatibility with logging
                    self._last_beta_mean = hex_weight_effective.mean().item()
                    
                    # M1.2 Fusion: V5 baseline (α=0.99) + small perturbation γ
                    # dx = (0.99 - γ) * dx_anchor + (0.01 + γ) * dx_hex
                    alpha_base = 0.99  # V5's optimal ratio - DO NOT CHANGE
                    hex_weight = (1 - alpha_base) + gamma    # 0.01 + γ
                    anchor_weight = alpha_base - gamma       # 0.99 - γ
                
                # ================================================================
                # M7: High-Pass Structural Decomposition on M1.2 (when hpass_enable=True)
                # ================================================================
                # Same as M6 but applied to M1.2's uncertainty-gated fusion:
                # r = Φ_E - Φ_L, decompose into r_low + r_high
                # Φ = Φ_L + (hex_weight) * (r_high + tied_factor * r_low)
                # When tied: tied_factor = 1.0 → degenerates to original M1.2
                # ================================================================
                
                if self.hpass_enable and self.lp_operator is not None:
                    # Compute residual r = Φ_E - Φ_L
                    r = dx_hex - dx_anchor  # [N, 3]
                    
                    # Decompose into low/high frequency
                    r_low, r_high = self.lp_operator.get_high_pass(
                        r=r,
                        knn_indices=self.knn_indices,
                        knn_weights=self.knn_weights,
                        anchor_positions=self.anchor_positions.detach(),
                        anchor_graph=None
                    )
                    
                    # Compute tied_factor based on mode
                    if self.hpass_eps_low_mode == 'zero':
                        # Hard high-pass: only high-frequency contributes
                        tied_factor = 0.0
                    elif self.hpass_eps_low_mode == 'tied':
                        # Sanity check: same weight for both (degenerates to M1.2)
                        tied_factor = 1.0
                    else:  # bounded_small
                        # Small learnable budget for low-frequency (relative to high)
                        # Use rho_low to compute a ratio
                        tied_factor = torch.sigmoid(self.rho_low).item() if self.rho_low is not None else 0.5
                    
                    # M7 Fusion: Φ = Φ_L + hex_weight * (r_high + tied_factor * r_low)
                    # When tied_factor=1.0: Φ = Φ_L + hex_weight * r = Φ_L + hex_weight * (Φ_E - Φ_L)
                    #                         = (1 - hex_weight) * Φ_L + hex_weight * Φ_E  (exactly M1.2)
                    dx_combined = dx_anchor + hex_weight * (r_high + tied_factor * r_low)
                    
                    # For scale/rotation, use original M1.2 logic
                    ds_combined = hex_weight * ds_hex
                    dr_combined = hex_weight * dr_hex
                    
                    # Cache for logging (reuse M6 logging fields)
                    self._last_eps_high = hex_weight.mean().item() if isinstance(hex_weight, torch.Tensor) else hex_weight
                    self._last_eps_low = (hex_weight * tied_factor).mean().item() if isinstance(hex_weight, torch.Tensor) else (hex_weight * tied_factor)
                    
                    with torch.no_grad():
                        self._last_E_low = torch.norm(r_low, dim=-1).mean().item()
                        self._last_E_high = torch.norm(r_high, dim=-1).mean().item()
                        self._last_E_ratio = self._last_E_low / (self._last_E_high + 1e-8)
                else:
                    # Original M1.2 fusion without hpass
                    dx_combined = anchor_weight * dx_anchor + hex_weight * dx_hex
                    ds_combined = hex_weight * ds_hex  # Scale from HexPlane only
                    dr_combined = hex_weight * dr_hex  # Rotation from HexPlane only
                
                self._last_balance_alpha = None
            
            elif self.fusion_mode == 'bounded_perturb':
                # ================================================================
                # M2.1: Learnable Weighted Average + Trust-Region Schedule
                # M5: Phase-Aware Trust-Region ε(t) (when phase_eps_enable=True)
                # ================================================================
                # M2.05 formula: dx = (1-ε)*dx_anchor + ε*dx_hex
                #
                # M2.1 adds trust-region schedule to prevent early shortcuts:
                #   schedule_mode="none"       → M2.05 behavior
                #   schedule_mode="freeze_rho" → ρ frozen for first N steps
                #   schedule_mode="warmup_cap" → ε_eff = min(ε_raw, ε_max * warmup_ratio)
                #
                # M5 upgrades ε from scalar to time-conditioned ε(t):
                #   ε(t) = ε_max * sigmoid(g(t))
                # ================================================================
                
                # Step 1: Compute raw ε (M5: time-conditioned, else: scalar)
                current_step = getattr(self, '_current_step', 0)
                
                if self.phase_eps_enable and self.phase_epsilon is not None:
                    # M5: Phase-aware trust-region ε(t)
                    eps_raw = self.phase_epsilon(time_emb)
                    self._last_eps_raw = eps_raw.item() if isinstance(eps_raw, torch.Tensor) else eps_raw
                else:
                    # M2.1: Scalar ε = ε_max * sigmoid(ρ)
                    eps_raw = self.eps_max * torch.sigmoid(self.rho)
                    self._last_eps_raw = eps_raw.item()
                
                # Step 2: Apply trust-region schedule to get ε_eff
                # M5 inherits M2.1a freeze logic: freeze g parameters for first freeze_steps
                if self.schedule_mode == 'none':
                    # M2.05 behavior: no schedule
                    eps_eff = eps_raw
                    self._last_warmup_ratio = 1.0
                    self._is_frozen = False
                    
                elif self.schedule_mode == 'freeze_rho':
                    # Hard freeze: ε stays at eps_raw, but ρ/g gradients are zeroed in train.py
                    # For M5: PhaseEpsilon parameters are also frozen during freeze_steps
                    eps_eff = eps_raw
                    self._is_frozen = (current_step < self.freeze_steps)
                    self._last_warmup_ratio = 0.0 if self._is_frozen else 1.0
                    
                elif self.schedule_mode == 'warmup_cap':
                    # Soft cap: ε_eff = min(ε_raw, ε_max * warmup_ratio)
                    warmup_ratio = min(current_step / max(self.warmup_steps, 1), 1.0)
                    eps_max_ref = self.phase_epsilon.eps_max if self.phase_eps_enable and self.phase_epsilon else self.eps_max
                    eps_cap = eps_max_ref * warmup_ratio
                    eps_eff = torch.min(eps_raw, torch.tensor(eps_cap, device=eps_raw.device) if not isinstance(eps_raw, torch.Tensor) else torch.tensor(eps_cap, device=eps_raw.device))
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
                if self.residual_mode == 'none':
                    # M2.1a baseline: NO normalization (original working formula)
                    # Φ = (1-ε)·Φ_L + ε·Φ_E
                    dx_H = dx_hex
                    ds_H = ds_hex
                    dr_H = dr_hex
                    
                elif self.residual_mode == 'tanh':
                    # M2.2 variant: H(Δ) = tanh(Δ) - bounds to [-1, 1]
                    # WARNING: This destroys magnitude information!
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
                    # Unknown mode, fallback to none (safe default)
                    dx_H = dx_hex
                    ds_H = ds_hex
                    dr_H = dr_hex
                
                # Compute norms after normalization (for logging)
                with torch.no_grad():
                    norm_H_dx = torch.norm(dx_H, dim=-1).mean().item()
                    self._last_mean_norm_H = norm_H_dx
                
                # Step 3: Weighted average fusion with normalized residuals
                # ================================================================
                # M6: High-Pass Structural Decomposition (when hpass_enable=True)
                # ================================================================
                # "Unlike penalty-based regularization, we enforce a structural
                #  frequency split of the Eulerian residual in the forward pass,
                #  allocating a bounded correction budget to the high-frequency
                #  component to prevent shortcut learning."
                #
                # r = H(Φ_E) - Φ_L  (normalized Eulerian residual)
                # Note: We work with normalized residuals (dx_H) after H(·) to preserve M2.2
                # r_low = LP(r), r_high = r - r_low
                # Φ = Φ_L + ε_high * r_high + ε_low * r_low
                # ================================================================
                
                if self.hpass_enable and self.lp_operator is not None:
                    # Compute residual r = Φ_E - Φ_L (Eulerian over Lagrangian)
                    # This definition guarantees exact degeneration to M2.1a when:
                    #   eps_low == eps_high == eps_eff
                    r = dx_H - dx_anchor  # [N, 3]
                    
                    # Decompose into low/high frequency
                    r_low, r_high = self.lp_operator.get_high_pass(
                        r=r,
                        knn_indices=self.knn_indices,
                        knn_weights=self.knn_weights,
                        anchor_positions=self.anchor_positions.detach(),
                        anchor_graph=None
                    )
                    
                    # Compute ε_high (use rho_high instead of rho)
                    eps_high_raw = self.hpass_eps_high_max * torch.sigmoid(self.rho_high)
                    
                    # Apply freeze schedule to ε_high
                    if self._is_frozen:
                        eps_high = torch.tensor(self.hpass_eps_high_init, device=eps_high_raw.device)
                    else:
                        eps_high = eps_high_raw
                    
                    # Compute ε_low based on mode
                    if self.hpass_eps_low_mode == 'zero':
                        # Hard high-pass: only high-frequency contributes
                        eps_low = torch.tensor(0.0, device=r.device)
                    elif self.hpass_eps_low_mode == 'tied':
                        # Sanity check: same as baseline (should give identical results)
                        eps_low = eps_high
                    else:  # bounded_small
                        # Small learnable budget for low-frequency
                        eps_low_raw = self.hpass_eps_low_max * torch.sigmoid(self.rho_low)
                        if self._is_frozen:
                            eps_low = torch.tensor(self.hpass_eps_low_init, device=eps_low_raw.device)
                        else:
                            eps_low = eps_low_raw
                    
                    # Dual-budget fusion: Φ = Φ_L + ε_high * r_high + ε_low * r_low
                    dx_combined = dx_anchor + eps_high * r_high + eps_low * r_low
                    
                    # For scale/rotation, use same eps_high (simplification)
                    ds_combined = eps_high * ds_H
                    dr_combined = eps_high * dr_H
                    
                    # Cache for logging
                    self._last_eps_high = eps_high.item() if isinstance(eps_high, torch.Tensor) else eps_high
                    self._last_eps_low = eps_low.item() if isinstance(eps_low, torch.Tensor) else eps_low
                    
                    with torch.no_grad():
                        self._last_E_low = torch.norm(r_low, dim=-1).mean().item()
                        self._last_E_high = torch.norm(r_high, dim=-1).mean().item()
                        self._last_E_ratio = self._last_E_low / (self._last_E_high + 1e-8)
                    
                    # For compatibility with existing logging
                    self._last_balance_alpha = (1.0 - eps_high).item() if isinstance(eps_high, torch.Tensor) else (1.0 - eps_high)
                    
                else:
                    # Original M2.1/M2.2/M5 fusion without M6
                    alpha = 1.0 - eps_eff
                    
                    dx_combined = alpha * dx_anchor + eps_eff * dx_H
                    ds_combined = eps_eff * ds_H  # Scale from HexPlane only
                    dr_combined = eps_eff * dr_H  # Rotation from HexPlane only
                    
                    self._last_balance_alpha = alpha.item() if isinstance(alpha, torch.Tensor) else alpha
            
            elif self.transport_correct_enable:
                # ================================================================
                # M8: Transport-Correction Decomposition (Predictor-Corrector)
                # ================================================================
                # Serial composition instead of parallel blending:
                #   1. Predictor (Lagrangian transport):  x' = x + Φ_L(x,t)
                #   2. Corrector (Eulerian at x'):        Δ = Φ_E(x',t)
                #   3. Update (budgeted residual):        x(t) = x' + ε·Δ
                #
                # Key insight: Residual evaluated in comoving frame (at x') cannot
                # learn large-scale transport already captured by Φ_L.
                # ================================================================
                
                # Step 1: Lagrangian transport (already computed: dx_anchor)
                # x' = x + dx_anchor (transported position)
                x_transported = gaussian_positions + dx_anchor
                
                # Step 2: Compute Eulerian corrector
                if self.transport_correct_comoving:
                    # M8: Query HexPlane at transported position x' (COMOVING FRAME)
                    # This is the key innovation - residual is evaluated where the
                    # Lagrangian predictor has already moved the point
                    means3D_corrector, scales_corrector, rotations_corrector = self.original_deformation(
                        x_transported,  # Query at transported position x'
                        scales,
                        rotations,
                        density,
                        time_emb
                    )
                    # Δ = Φ_E(x') - x' (corrector displacement relative to transported)
                    delta_corrector = means3D_corrector - x_transported
                else:
                    # Ablation: Query HexPlane at original position x (ORIGINAL FRAME)
                    # This should be worse - included for ablation study
                    delta_corrector = dx_hex  # Already computed earlier
                
                ds_corrector = scales_corrector - scales if self.transport_correct_comoving else ds_hex
                dr_corrector = rotations_corrector - rotations if self.transport_correct_comoving else dr_hex
                
                # Step 3: Apply budgeted residual
                if self.transport_correct_learnable_beta and self.beta_net is not None:
                    # Learnable β(x',t) with budget constraint
                    # Input: concatenate transported position and time
                    t_expanded = time_emb[:, 0:1] if time_emb.dim() > 1 else time_emb.unsqueeze(-1)
                    beta_input = torch.cat([x_transported, t_expanded.expand(x_transported.shape[0], 1)], dim=-1)
                    beta_raw = self.beta_net(beta_input)  # [N, 1], in [0, 1] due to sigmoid
                    beta = self.transport_correct_beta_max * beta_raw  # Scale to [0, beta_max]
                    
                    # Compute budget penalty: L_budget = λ * max(0, E[β] - budget)^2
                    beta_mean = beta.mean()
                    budget_violation = torch.relu(beta_mean - self.transport_correct_beta_budget)
                    self._last_tc_budget_loss = self.transport_correct_lambda_budget * budget_violation ** 2
                    
                    eps_effective = beta
                    self._last_tc_eps = beta_mean.item()
                else:
                    # Fixed ε (matches V5's 0.01 finding)
                    eps_effective = self.transport_correct_eps
                    self._last_tc_eps = self.transport_correct_eps
                    self._last_tc_budget_loss = None
                
                # Final update: x(t) = x' + ε·Δ = x + dx_anchor + ε·Δ
                dx_combined = dx_anchor + eps_effective * delta_corrector
                ds_combined = eps_effective * ds_corrector
                dr_combined = eps_effective * dr_corrector
                
                # Cache for logging
                with torch.no_grad():
                    self._last_tc_delta_norm = torch.norm(delta_corrector, dim=-1).mean().item()
                    self._last_tc_transport_norm = torch.norm(dx_anchor, dim=-1).mean().item()
                
                self._last_balance_alpha = None
            
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
                    wA_used = None
                    # NOTE: S4 fixed fusion is implemented as a dedicated branch (independent of learnable balance).
                    # Here we keep V5 semantics: dx fusion follows alpha unless s4_1 is enabled.
                    if self.s4_1_anchor_only_position:
                        # s4.1: dx = α * dx_anchor (remove HexPlane position contribution)
                        wA_used = alpha
                        dx_combined = alpha * dx_anchor
                    else:
                        wA_used = alpha
                        dx_combined = (1 - alpha) * dx_hex + alpha * dx_anchor
                    # s2 series: optionally extend anchor fusion to scale/rotation
                    # s3 series: optionally release scale/rotation from (1-α) multiplier
                    if self.s3_release_scale:
                        # s3.1/s3.3: ds = ds_hex (not multiplied by 1-α)
                        ds_combined = ds_hex
                    elif self.s2_anchor_to_scale:
                        ds_combined = (1 - alpha) * ds_hex + alpha * dx_anchor
                    else:
                        ds_combined = (1 - alpha) * ds_hex
                    
                    if self.s3_zero_rotation:
                        # s3.4+: dr = 0 (completely disable HexPlane rotation)
                        dr_combined = torch.zeros_like(dr_hex)
                    elif self.s3_release_rotation:
                        # s3.2/s3.3: dr = dr_hex (not multiplied by 1-α)
                        dr_combined = dr_hex
                    elif self.s2_anchor_to_rotation:
                        # dr_hex is quaternion [N, 4], dx_anchor is [N, 3], pad with 0
                        dx_anchor_4d = torch.cat([dx_anchor, torch.zeros_like(dx_anchor[:, :1])], dim=1)
                        dr_combined = (1 - alpha) * dr_hex + alpha * dx_anchor_4d
                    else:
                        dr_combined = (1 - alpha) * dr_hex

                    if self.s5_rot_nlerp or self.s5_scale_log_fusion:
                        wA = float(wA_used) if wA_used is not None else alpha
                        s_ref = scales
                        q_ref = rotations
                        if self.s5_jacobian_sr:
                            k = min(int(self.s5_jacobian_k), self.knn_indices.shape[1])
                            s_ref, q_ref = self._jacobian_sr_reference(
                                gaussian_positions=gaussian_positions,
                                scales=scales,
                                rotations=rotations,
                                anchor_positions=self.anchor_positions,
                                anchor_displacements=anchor_displacements,
                                knn_idx=self.knn_indices[:gaussian_positions.shape[0], :k],
                                knn_w=self.knn_weights[:gaussian_positions.shape[0], :k],
                            )

                        if self.s5_scale_log_fusion:
                            log_s_hex = torch.log(torch.clamp(scales_hex, min=self.s5_eps))
                            log_s_ref = torch.log(torch.clamp(s_ref, min=self.s5_eps))
                            scales_fused = torch.exp((1 - wA) * log_s_hex + wA * log_s_ref)
                            ds_combined = scales_fused - scales

                        if self.s5_rot_nlerp:
                            q_hex = self._quat_normalize(rotations_hex)
                            q_ref = self._quat_normalize(q_ref)
                            q_fused = self._quat_normalize((1 - wA) * q_hex + wA * q_ref)
                            dr_combined = q_fused - rotations

                    # s4.2/s4.3: optional override for rotation weight
                    # Only apply when rotation is not explicitly disabled/released/extended.
                    if (
                        self.s4_dr_hex_weight is not None and self.s4_dr_hex_weight >= 0
                        and (not self.s3_zero_rotation)
                        and (not self.s3_release_rotation)
                        and (not self.s2_anchor_to_rotation)
                    ):
                        k = float(self.s4_dr_hex_weight)
                        dr_combined = k * dr_hex
                    alpha = alpha.item()
                self._last_balance_alpha = alpha  # Cache for logging
            else:
                # Original: Δμ_total = Δμ_hexplane + Δμ_anchor (direct sum)
                dx_combined = dx_hex + dx_anchor
                ds_combined = ds_hex
                dr_combined = dr_hex

            # ================================================================
            # S4 (independent fixed fusion): explicit (wA, ds_weight, k)
            # Trigger when any s4_* weight is set, regardless of use_learnable_balance.
            # This is designed to remove reliance on learnable balance/alpha.
            # ================================================================
            s4_use_wA = (self.s4_dx_anchor_weight is not None and self.s4_dx_anchor_weight >= 0)
            s4_use_ds = (self.s4_ds_hex_weight is not None and self.s4_ds_hex_weight >= 0)
            s4_use_k = (self.s4_dr_hex_weight is not None and self.s4_dr_hex_weight >= 0)
            if s4_use_wA or s4_use_ds or s4_use_k:
                if s4_use_wA:
                    wA_base = float(self.s4_dx_anchor_weight)
                    if self.s7_per_anchor_wA and self.s7_wA_base is not None and self.s7_wA_base >= 0:
                        wA_base = float(self.s7_wA_base)
                    if self.s7_per_anchor_wA and anchor_features_s7 is not None:
                        wA_gauss, _ = self._s7_compute_wA(anchor_features_s7, dx_hex.shape[0], wA_base)
                        if wA_gauss is not None:
                            dx_combined = (1 - wA_gauss) * dx_hex + wA_gauss * dx_anchor
                        else:
                            dx_combined = (1 - wA_base) * dx_hex + wA_base * dx_anchor
                    else:
                        dx_combined = (1 - wA_base) * dx_hex + wA_base * dx_anchor
                # If not set, keep whatever dx_combined was computed by the active mode.

                if s4_use_ds:
                    ds_weight = float(self.s4_ds_hex_weight)
                    ds_combined = ds_weight * ds_hex
                # If not set, keep ds_combined as computed by the active mode.

                if (
                    s4_use_k
                    and (not self.s3_zero_rotation)
                    and (not self.s3_release_rotation)
                    and (not self.s2_anchor_to_rotation)
                ):
                    k = float(self.s4_dr_hex_weight)
                    dr_combined = k * dr_hex
                # If not set, keep dr_combined as computed by the active mode.

            # ================================================================
            # S6 (Trust-Region Geometric Fusion): stabilize dx/ds/dr updates
            # ================================================================
            if self.s6_trust_region and (float(iteration_ratio) >= float(self.s6_trust_region_start_ratio)):
                tau_pos = self._s6_tau(self.s6_tau_pos, self.s6_tau_pos_start, self.s6_tau_pos_end, iteration_ratio)
                tau_scale = self._s6_tau(self.s6_tau_scale, self.s6_tau_scale_start, self.s6_tau_scale_end, iteration_ratio)
                tau_rot = self._s6_tau(self.s6_tau_rot, self.s6_tau_rot_start, self.s6_tau_rot_end, iteration_ratio)

                do_log = bool(getattr(self.args, 'debug', False)) and self.s6_trust_region_log
                if do_log:
                    self._s6_step += 1

                if tau_pos > 0:
                    if do_log:
                        dx_combined, dx_stats = self._clamp_norm_with_stats(dx_combined, tau_pos, dim=-1)
                    else:
                        dx_combined = self._clamp_norm(dx_combined, tau_pos, dim=-1)

                if tau_scale > 0:
                    if do_log:
                        ds_combined, ds_stats = self._clamp_norm_with_stats(ds_combined, tau_scale, dim=-1)
                    else:
                        ds_combined = self._clamp_norm(ds_combined, tau_scale, dim=-1)

                if tau_rot > 0:
                    if do_log:
                        dr_combined, dr_stats = self._clamp_norm_with_stats(dr_combined, tau_rot, dim=-1)
                    else:
                        dr_combined = self._clamp_norm(dr_combined, tau_rot, dim=-1)

                if do_log and (self._s6_step % max(int(self.s6_trust_region_log_interval), 1) == 0):
                    parts = [
                        f"[TRGF] step={self._s6_step} ratio={float(iteration_ratio):.4f}",
                    ]
                    if tau_pos > 0:
                        parts.append(
                            f"dx(tau={dx_stats['tau']:.4g}) clamp={dx_stats['clamp_ratio']:.3f} mean_scale={dx_stats['mean_scale']:.3f} "
                            f"mean_norm={dx_stats['mean_norm']:.4g} mean_norm_clamped={dx_stats['mean_norm_clamped']:.4g}"
                        )
                    if tau_scale > 0:
                        parts.append(
                            f"ds(tau={ds_stats['tau']:.4g}) clamp={ds_stats['clamp_ratio']:.3f} mean_scale={ds_stats['mean_scale']:.3f} "
                            f"mean_norm={ds_stats['mean_norm']:.4g} mean_norm_clamped={ds_stats['mean_norm_clamped']:.4g}"
                        )
                    if tau_rot > 0:
                        parts.append(
                            f"dr(tau={dr_stats['tau']:.4g}) clamp={dr_stats['clamp_ratio']:.3f} mean_scale={dr_stats['mean_scale']:.3f} "
                            f"mean_norm={dr_stats['mean_norm']:.4g} mean_norm_clamped={dr_stats['mean_norm_clamped']:.4g}"
                        )
                    print(' | '.join(parts), flush=True)

            deformed_positions = gaussian_positions + dx_combined
            deformed_scales = scales + ds_combined
            deformed_rotations = rotations + dr_combined
            
            return deformed_positions, deformed_scales, deformed_rotations

    def get_s7_statistics(self) -> dict:
        if (not self.s7_per_anchor_wA) or (self._last_s7_wA_anchor is None):
            return None
        wA = self._last_s7_wA_anchor
        return {
            'wA_mean': wA.mean().item(),
            'wA_std': wA.std().item(),
            'wA_min': wA.min().item(),
            'wA_max': wA.max().item(),
            'graph_loss': self._last_s7_wA_graph_loss.item() if self._last_s7_wA_graph_loss is not None else None,
            'temp_loss': self._last_s7_wA_temp_loss.item() if self._last_s7_wA_temp_loss is not None else None,
            'lambda_graph': self.s7_lambda_wA_graph,
            'lambda_temp': self.s7_lambda_wA_temp,
        }

    def get_s7_loss(self) -> torch.Tensor:
        if not self.s7_per_anchor_wA:
            return None
        total_loss = None
        if self._last_s7_wA_graph_loss is not None:
            total_loss = self._last_s7_wA_graph_loss if total_loss is None else total_loss + self._last_s7_wA_graph_loss
        if self._last_s7_wA_temp_loss is not None:
            total_loss = self._last_s7_wA_temp_loss if total_loss is None else total_loss + self._last_s7_wA_temp_loss
        return total_loss
        
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

    def compute_anchor_time_smooth_loss(self, time_emb: torch.Tensor) -> torch.Tensor:
        if self.lambda_anchor_time <= 0:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        if not self.initialized:
            return torch.tensor(0.0, device=self.anchor_positions.device)

        device = self.anchor_positions.device
        if time_emb.dim() > 0:
            t = time_emb[0, 0] if time_emb.dim() == 2 else time_emb[0]
        else:
            t = time_emb

        dt = float(self.anchor_time_delta)
        if dt <= 0:
            return torch.tensor(0.0, device=device)

        t_val = float(t.item())
        t_prev_val = max(0.0, t_val - dt)
        t_next_val = min(1.0, t_val + dt)
        if t_prev_val == t_val or t_next_val == t_val:
            return torch.tensor(0.0, device=device)

        t_prev = torch.tensor(t_prev_val, device=device, dtype=t.dtype)
        t_next = torch.tensor(t_next_val, device=device, dtype=t.dtype)

        dx_t = self.forward_anchors_unmasked(time_emb)

        if bool(self.anchor_time_stopgrad_neighbors):
            with torch.no_grad():
                dx_prev = self.forward_anchors_unmasked(t_prev.unsqueeze(0))
                dx_next = self.forward_anchors_unmasked(t_next.unsqueeze(0))
        else:
            dx_prev = self.forward_anchors_unmasked(t_prev.unsqueeze(0))
            dx_next = self.forward_anchors_unmasked(t_next.unsqueeze(0))

        acc = (dx_next - 2.0 * dx_t + dx_prev)
        acc_sq = (acc ** 2).sum(dim=-1)

        if self._anchor_mass is not None and torch.is_tensor(self._anchor_mass) and self._anchor_mass.numel() == acc_sq.numel():
            m = self._anchor_mass.to(device=device, dtype=acc_sq.dtype)
            denom = (m.sum() + float(self.anchor_time_eps))
            loss = (m * acc_sq).sum() / denom
        else:
            loss = acc_sq.mean()

        return loss

    def compute_anchor_distortion_loss(self) -> torch.Tensor:
        if self._last_anchor_displacements is None:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        if self._anchor_graph_edges is None or self._anchor_graph_d0 is None:
            return torch.tensor(0.0, device=self.anchor_positions.device)

        edges = self._anchor_graph_edges
        src = edges[:, 0]
        dst = edges[:, 1]

        a = self.anchor_positions.detach()
        dx = self._last_anchor_displacements
        p = a + dx

        d = torch.norm(p[src] - p[dst], dim=-1)  # [E]
        d0 = self._anchor_graph_d0
        eps = float(self.anchor_distortion_eps)
        r = d / (d0 + eps)

        r_min = float(self.anchor_distortion_r_min)
        r_max = float(self.anchor_distortion_r_max)
        hi = F.relu(r - r_max) ** 2
        lo = F.relu(r_min - r) ** 2
        per_edge = hi + lo

        if self._anchor_graph_w is not None:
            per_edge = self._anchor_graph_w * per_edge

        return per_edge.mean()
    
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
    
    def compute_phase_eps_smooth_loss(self) -> torch.Tensor:
        """
        M5: Compute temporal smoothness prior L_smooth for phase-aware ε(t).
        
        L_smooth = mean_k (ε_{k+1} - ε_k)^2
        
        This prevents per-frame overfitting and encourages smooth ε(t) curves
        across respiratory phases.
        
        Returns:
            loss: Phase epsilon smoothness loss (scalar)
        """
        if not self.phase_eps_enable or self.phase_epsilon is None:
            return torch.tensor(0.0, device=self.anchor_positions.device)
        
        L_smooth = self.phase_epsilon.compute_smooth_loss()
        self._last_phase_eps_smooth_loss = L_smooth.item()
        
        return L_smooth
    
    def get_phase_eps_stats(self) -> dict:
        """
        M5: Get phase epsilon statistics for logging.
        
        Returns:
            dict with mode, mean_eps, min_eps, max_eps, std_eps, L_smooth
        """
        if not self.phase_eps_enable or self.phase_epsilon is None:
            return {}
        
        stats = self.phase_epsilon.get_stats()
        stats['mode'] = self.phase_epsilon.mode
        stats['L_smooth'] = self._last_phase_eps_smooth_loss or 0.0
        stats['is_frozen'] = self._is_frozen
        
        return stats
    
    def get_phase_eps_curve(self, num_samples: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        M5: Get ε(t) curve for visualization.
        
        Args:
            num_samples: Number of time samples (for tiny_mlp mode)
        
        Returns:
            t_values: Time values [T] or [num_samples]
            eps_values: Corresponding ε values
        """
        if not self.phase_eps_enable or self.phase_epsilon is None:
            return None, None
        
        return self.phase_epsilon.get_all_eps_values(num_samples)
    
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
        if self.s7_wA_head is not None:
            params.extend(self.s7_wA_head.parameters())
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
    
    def get_m6_statistics(self) -> dict:
        """
        M6: Get high-pass structural decomposition statistics for logging.
        
        Returns:
            Dictionary with:
                - eps_high: High-frequency budget
                - eps_low: Low-frequency budget
                - E_low: Mean ||r_low||
                - E_high: Mean ||r_high||
                - E_ratio: E_low / (E_high + 1e-8)
                - eps_low_mode: Current eps_low mode
                - hpass_k: Number of neighbors for LP
        """
        if not self.hpass_enable or self.lp_operator is None:
            return {}
        
        return {
            'eps_high': self._last_eps_high,
            'eps_low': self._last_eps_low,
            'E_low': self._last_E_low,
            'E_high': self._last_E_high,
            'E_ratio': self._last_E_ratio,
            'eps_low_mode': self.hpass_eps_low_mode,
            'hpass_k': self.hpass_k,
            'rho_high': self.rho_high.item() if self.rho_high is not None else None,
            'rho_low': self.rho_low.item() if self.rho_low is not None else None,
            'is_frozen': self._is_frozen
        }
    
    def get_m8_statistics(self) -> dict:
        """
        M8: Get Transport-Correction statistics for logging.
        
        Returns:
            Dictionary with:
                - eps: Effective ε or mean β
                - delta_norm: Mean ||Δ|| (corrector magnitude)
                - transport_norm: Mean ||Φ_L|| (predictor magnitude)
                - comoving: Whether using comoving frame
                - learnable_beta: Whether β is learnable
                - budget_loss: Budget violation penalty (if learnable)
        """
        if not self.transport_correct_enable:
            return {}
        
        return {
            'eps': self._last_tc_eps,
            'delta_norm': self._last_tc_delta_norm,
            'transport_norm': self._last_tc_transport_norm,
            'comoving': self.transport_correct_comoving,
            'learnable_beta': self.transport_correct_learnable_beta,
            'budget_loss': self._last_tc_budget_loss.item() if self._last_tc_budget_loss is not None else None,
            'beta_budget': self.transport_correct_beta_budget,
            'beta_max': self.transport_correct_beta_max
        }
    
    def get_s1_statistics(self) -> dict:
        """
        s1/s1.1/s1.2: Get per-anchor gamma statistics for logging.
        
        Returns:
            Dictionary with:
                - gamma_mean: Mean γ across anchors
                - gamma_std: Std of γ across anchors
                - gamma_min: Min γ
                - gamma_max: Max γ
                - graph_loss: s1.1 spatial smoothness loss (if enabled)
                - temp_loss: s1.2 temporal smoothness loss (if enabled)
        """
        if not self.per_anchor_gamma or self._last_gamma_anchor is None:
            return {}
        
        gamma = self._last_gamma_anchor
        return {
            'gamma_mean': gamma.mean().item(),
            'gamma_std': gamma.std().item(),
            'gamma_min': gamma.min().item(),
            'gamma_max': gamma.max().item(),
            'graph_loss': self._last_gamma_graph_loss.item() if self._last_gamma_graph_loss is not None else None,
            'temp_loss': self._last_gamma_temp_loss.item() if self._last_gamma_temp_loss is not None else None,
            'lambda_graph': self.lambda_gamma_graph,
            'lambda_temp': self.lambda_gamma_temp
        }
    
    def get_s1_loss(self) -> torch.Tensor:
        """
        s1.1/s1.2: Get combined regularization loss for per-anchor gamma.
        
        Returns:
            Total regularization loss (graph + temporal), or None if not enabled
        """
        if not self.per_anchor_gamma:
            return None
        
        total_loss = None
        if self._last_gamma_graph_loss is not None:
            total_loss = self._last_gamma_graph_loss if total_loss is None else total_loss + self._last_gamma_graph_loss
        if self._last_gamma_temp_loss is not None:
            total_loss = self._last_gamma_temp_loss if total_loss is None else total_loss + self._last_gamma_temp_loss
        
        return total_loss
    
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
