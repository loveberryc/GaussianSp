#!/usr/bin/env python
"""
PhysX-Gaussian Integration Tests

This script tests the integration with GaussianModel and the full render pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project to path
sys.path.insert(0, '/root/autodl-tmp/4dctgs/x2-gaussian-main-origin')

from x2_gaussian.gaussian.gaussian_model import GaussianModel
from x2_gaussian.arguments import ModelHiddenParams, OptimizationParams


def create_mock_gaussian_model():
    """Create a mock GaussianModel with anchor deformation enabled"""
    
    class MockModelParams:
        def __init__(self):
            self.sh_degree = 3
            # PhysX-Gaussian params
            self.use_anchor_deformation = True
            self.num_anchors = 64
            self.anchor_k = 5
            self.mask_ratio = 0.25
            self.transformer_dim = 32
            self.transformer_heads = 2
            self.transformer_layers = 1
            self.anchor_time_embed_dim = 8
            self.anchor_pos_embed_dim = 16
            # Other required params
            self.use_v7_2_consistency = False
            self.v7_2_alpha_init = 0.3
            self.v7_2_alpha_learnable = True
            self.v7_2_use_time_dependent_alpha = False
    
    return GaussianModel(MockModelParams())


def test_gaussian_model_deformed_centers():
    """Test get_deformed_centers with anchor deformation"""
    print("\n" + "="*60)
    print("TEST: GaussianModel get_deformed_centers")
    print("="*60)
    
    gaussians = create_mock_gaussian_model()
    
    # Create fake point cloud
    N = 1000
    from plyfile import PlyData, PlyElement
    import numpy as np
    import tempfile
    
    # Create temporary PLY file
    points = np.random.randn(N, 3).astype(np.float32)
    vertex = np.array([(p[0], p[1], p[2], 0, 0, 0) for p in points],
                      dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')])
    el = PlyElement.describe(vertex, 'vertex')
    
    with tempfile.NamedTemporaryFile(suffix='.ply', delete=False) as f:
        temp_ply = f.name
        PlyData([el]).write(f)
    
    # Initialize from PLY
    from x2_gaussian.utils.general_utils import build_scaling_rotation
    
    class MockPCD:
        def __init__(self, points):
            self.points = torch.tensor(points, dtype=torch.float32)
            self.colors = torch.zeros(len(points), 3)
    
    pcd = MockPCD(points)
    
    # Manually initialize
    fused_point_cloud = pcd.points.float().cuda()
    N = fused_point_cloud.shape[0]
    
    # Set up Gaussian parameters
    gaussians._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    gaussians._features_dc = nn.Parameter(torch.zeros(N, 1, 3).cuda().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(torch.zeros(N, 15, 3).cuda().requires_grad_(True))
    gaussians._scaling = nn.Parameter(torch.zeros(N, 3).cuda().requires_grad_(True))
    gaussians._rotation = nn.Parameter(torch.zeros(N, 4).cuda().requires_grad_(True))
    gaussians._rotation.data[:, 0] = 1.0
    gaussians._density = nn.Parameter(torch.ones(N, 1).cuda().requires_grad_(True))
    
    # Initialize anchor deformation
    if gaussians.use_anchor_deformation and gaussians._deformation_anchor is not None:
        gaussians._deformation_anchor = gaussians._deformation_anchor.to("cuda")
        gaussians._deformation_anchor.initialize_anchors(fused_point_cloud)
        gaussians._deformation_anchor.update_knn_binding(fused_point_cloud)
        print("  Anchor deformation initialized")
    
    # Test forward
    time = torch.rand(1).cuda()
    
    try:
        deformed, scales, rotations = gaussians.get_deformed_centers(time, is_training=True)
        print(f"  Deformed shape: {deformed.shape}")
        print(f"  Deformed requires_grad: {deformed.requires_grad}")
        
        # Test backward
        loss = deformed.sum()
        loss.backward()
        print("✓ PASSED: get_deformed_centers forward-backward works")
        
        # Clean up
        os.unlink(temp_ply)
        return True
    except Exception as e:
        print(f"✗ FAILED: {e}")
        os.unlink(temp_ply)
        return False


def test_physics_loss_after_render():
    """Test physics completion loss computed after a simulated render"""
    print("\n" + "="*60)
    print("TEST: Physics loss after render simulation")
    print("="*60)
    
    gaussians = create_mock_gaussian_model()
    
    # Set up Gaussian
    N = 1000
    fused_point_cloud = torch.randn(N, 3).float().cuda()
    
    gaussians._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    gaussians._features_dc = nn.Parameter(torch.zeros(N, 1, 3).cuda().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(torch.zeros(N, 15, 3).cuda().requires_grad_(True))
    gaussians._scaling = nn.Parameter(torch.zeros(N, 3).cuda().requires_grad_(True))
    gaussians._rotation = nn.Parameter(torch.zeros(N, 4).cuda().requires_grad_(True))
    gaussians._rotation.data[:, 0] = 1.0
    gaussians._density = nn.Parameter(torch.ones(N, 1).cuda().requires_grad_(True))
    
    if gaussians.use_anchor_deformation and gaussians._deformation_anchor is not None:
        gaussians._deformation_anchor = gaussians._deformation_anchor.to("cuda")
        gaussians._deformation_anchor.initialize_anchors(fused_point_cloud.detach())
        gaussians._deformation_anchor.update_knn_binding(fused_point_cloud.detach())
    
    time = torch.rand(1).cuda()
    
    try:
        # Step 1: Simulate render (get deformed centers)
        deformed, scales, rotations = gaussians.get_deformed_centers(time, is_training=True)
        
        # Step 2: Simulate render loss
        render_loss = deformed.sum()
        
        # Step 3: Compute physics completion loss
        L_phys = gaussians.compute_physics_completion_loss(time)
        
        # Step 4: Compute anchor smoothness loss
        L_smooth = gaussians.compute_anchor_smoothness_loss()
        
        print(f"  Render loss: {render_loss.item():.4f}")
        print(f"  Physics loss: {L_phys.item():.4f}")
        print(f"  Smoothness loss: {L_smooth.item():.4f}")
        
        # Total loss
        total_loss = render_loss + 0.1 * L_phys + 0.01 * L_smooth
        
        # Backward
        total_loss.backward()
        print("✓ PASSED: Physics loss after render works")
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


def test_repeated_iterations():
    """Test multiple training iterations"""
    print("\n" + "="*60)
    print("TEST: Multiple training iterations")
    print("="*60)
    
    gaussians = create_mock_gaussian_model()
    
    N = 1000
    fused_point_cloud = torch.randn(N, 3).float().cuda()
    
    gaussians._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    gaussians._features_dc = nn.Parameter(torch.zeros(N, 1, 3).cuda().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(torch.zeros(N, 15, 3).cuda().requires_grad_(True))
    gaussians._scaling = nn.Parameter(torch.zeros(N, 3).cuda().requires_grad_(True))
    gaussians._rotation = nn.Parameter(torch.zeros(N, 4).cuda().requires_grad_(True))
    gaussians._rotation.data[:, 0] = 1.0
    gaussians._density = nn.Parameter(torch.ones(N, 1).cuda().requires_grad_(True))
    
    if gaussians.use_anchor_deformation and gaussians._deformation_anchor is not None:
        gaussians._deformation_anchor = gaussians._deformation_anchor.to("cuda")
        gaussians._deformation_anchor.initialize_anchors(fused_point_cloud.detach())
        gaussians._deformation_anchor.update_knn_binding(fused_point_cloud.detach())
    
    # Create optimizer
    params = [gaussians._xyz]
    if gaussians._deformation_anchor is not None:
        params.extend(gaussians._deformation_anchor.parameters())
    
    optimizer = torch.optim.Adam(params, lr=1e-4)
    
    passed = True
    for i in range(5):
        optimizer.zero_grad()
        
        time = torch.rand(1).cuda()
        
        # Simulate render
        deformed, scales, rotations = gaussians.get_deformed_centers(time, is_training=True)
        render_loss = deformed.sum()
        
        # PhysX losses
        L_phys = gaussians.compute_physics_completion_loss(time)
        L_smooth = gaussians.compute_anchor_smoothness_loss()
        
        total_loss = render_loss + 0.1 * L_phys + 0.01 * L_smooth
        
        try:
            total_loss.backward()
            optimizer.step()
            print(f"  Iter {i+1}: Loss = {total_loss.item():.4f}")
        except Exception as e:
            print(f"  Iter {i+1}: FAILED - {e}")
            passed = False
            break
    
    if passed:
        print("✓ PASSED: Multiple iterations work")
    else:
        print("✗ FAILED: Multiple iterations failed")
    return passed


def test_problem_scenario():
    """
    Test the exact scenario that might be causing the issue:
    - Render uses get_deformed_centers (is_training=True)
    - Render loss computed
    - Other losses (regulation, etc.) computed
    - PhysX losses computed
    - Single backward
    """
    print("\n" + "="*60)
    print("TEST: Problem scenario simulation")
    print("="*60)
    
    gaussians = create_mock_gaussian_model()
    
    N = 1000
    fused_point_cloud = torch.randn(N, 3).float().cuda()
    
    gaussians._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
    gaussians._features_dc = nn.Parameter(torch.zeros(N, 1, 3).cuda().requires_grad_(True))
    gaussians._features_rest = nn.Parameter(torch.zeros(N, 15, 3).cuda().requires_grad_(True))
    gaussians._scaling = nn.Parameter(torch.zeros(N, 3).cuda().requires_grad_(True))
    gaussians._rotation = nn.Parameter(torch.zeros(N, 4).cuda().requires_grad_(True))
    gaussians._rotation.data[:, 0] = 1.0
    gaussians._density = nn.Parameter(torch.ones(N, 1).cuda().requires_grad_(True))
    
    if gaussians.use_anchor_deformation and gaussians._deformation_anchor is not None:
        gaussians._deformation_anchor = gaussians._deformation_anchor.to("cuda")
        gaussians._deformation_anchor.initialize_anchors(fused_point_cloud.detach())
        gaussians._deformation_anchor.update_knn_binding(fused_point_cloud.detach())
    
    time = torch.rand(1).cuda()
    
    loss = {}
    
    try:
        # === RENDER PHASE ===
        deformed, scales, rotations = gaussians.get_deformed_centers(time, is_training=True)
        
        # Simulate rasterizer output
        fake_image = deformed.mean(dim=0)  # [3]
        gt_image = torch.rand(3).cuda()
        
        # === LOSSES ===
        # 1. Render loss (L1)
        loss["render"] = F.l1_loss(fake_image, gt_image)
        loss["total"] = loss["render"]
        
        # 2. Some regularization (simulating compute_regulation)
        # In actual code, this uses _deformation, not _deformation_anchor
        # So it shouldn't cause issues
        loss["reg"] = (scales ** 2).mean() if scales is not None else torch.tensor(0.0).cuda()
        loss["total"] = loss["total"] + 0.01 * loss["reg"]
        
        # 3. PhysX-Gaussian losses
        L_phys = gaussians.compute_physics_completion_loss(time)
        loss["phys"] = L_phys
        loss["total"] = loss["total"] + 0.1 * L_phys
        
        L_smooth = gaussians.compute_anchor_smoothness_loss()
        loss["smooth"] = L_smooth
        loss["total"] = loss["total"] + 0.01 * L_smooth
        
        print(f"  Render loss: {loss['render'].item():.4f}")
        print(f"  Reg loss: {loss['reg'].item():.4f}")
        print(f"  Phys loss: {loss['phys'].item():.4f}")
        print(f"  Smooth loss: {loss['smooth'].item():.4f}")
        print(f"  Total loss: {loss['total'].item():.4f}")
        
        # === BACKWARD ===
        loss["total"].backward()
        
        print("✓ PASSED: Problem scenario works in isolation")
        return True
        
    except Exception as e:
        import traceback
        print(f"✗ FAILED: {e}")
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("PhysX-Gaussian Integration Tests")
    print("=" * 60)
    
    results = {}
    
    # results['gaussian_deformed_centers'] = test_gaussian_model_deformed_centers()
    results['physics_loss_after_render'] = test_physics_loss_after_render()
    results['repeated_iterations'] = test_repeated_iterations()
    results['problem_scenario'] = test_problem_scenario()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    if all(results.values()):
        print("\n✓ All integration tests passed")
        print("  The issue is likely in interaction with OTHER parts of train.py")
        print("  such as: compute_regulation, cycle_motion_loss, etc.")
    else:
        print("\n✗ Some integration tests failed")
        print("  The issue is in the PhysX-Gaussian implementation itself")
