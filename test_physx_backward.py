#!/usr/bin/env python
"""
PhysX-Gaussian Backward Pass Diagnostic Tests

This script tests the backward pass to identify where the 
"backward through graph twice" error occurs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Add project to path
sys.path.insert(0, '/root/autodl-tmp/4dctgs/x2-gaussian-main-origin')

from x2_gaussian.gaussian.anchor_module import AnchorDeformationNet

class MockArgs:
    """Mock arguments for testing"""
    def __init__(self):
        self.use_anchor_deformation = True
        self.num_anchors = 64  # Small for testing
        self.anchor_k = 5
        self.mask_ratio = 0.25
        self.transformer_dim = 32
        self.transformer_heads = 2
        self.transformer_layers = 1
        self.anchor_time_embed_dim = 8
        self.anchor_pos_embed_dim = 16


def test_1_basic_forward_backward():
    """Test 1: Basic forward and backward through anchor deformation"""
    print("\n" + "="*60)
    print("TEST 1: Basic forward-backward")
    print("="*60)
    
    args = MockArgs()
    net = AnchorDeformationNet(args).cuda()
    
    # Create dummy data
    N = 1000  # Number of Gaussians
    gaussian_pos = torch.randn(N, 3, device='cuda', requires_grad=True)
    scales = torch.randn(N, 3, device='cuda')
    rotations = torch.randn(N, 4, device='cuda')
    density = torch.randn(N, 1, device='cuda')
    time = torch.rand(N, 1, device='cuda')
    
    # Initialize
    net.initialize_anchors(gaussian_pos.detach())
    net.update_knn_binding(gaussian_pos.detach())
    
    # Forward
    deformed_pos, deformed_scales, deformed_rotations = net(
        gaussian_pos, scales, rotations, density, time, is_training=True
    )
    
    # Simple loss
    loss = deformed_pos.sum()
    
    # Backward
    try:
        loss.backward()
        print("✓ PASSED: Basic forward-backward works")
        return True
    except RuntimeError as e:
        print(f"✗ FAILED: {e}")
        return False


def test_2_forward_with_smoothness_loss():
    """Test 2: Forward + anchor smoothness loss"""
    print("\n" + "="*60)
    print("TEST 2: Forward + smoothness loss")
    print("="*60)
    
    args = MockArgs()
    net = AnchorDeformationNet(args).cuda()
    
    N = 1000
    gaussian_pos = torch.randn(N, 3, device='cuda', requires_grad=True)
    scales = torch.randn(N, 3, device='cuda')
    rotations = torch.randn(N, 4, device='cuda')
    density = torch.randn(N, 1, device='cuda')
    time = torch.rand(N, 1, device='cuda')
    
    net.initialize_anchors(gaussian_pos.detach())
    net.update_knn_binding(gaussian_pos.detach())
    
    # Forward (simulates render)
    deformed_pos, _, _ = net(gaussian_pos, scales, rotations, density, time, is_training=True)
    
    # Main loss (simulates render loss)
    loss = deformed_pos.sum()
    
    # Add smoothness loss (uses cached _last_anchor_displacements)
    smooth_loss = net.compute_anchor_smoothness_loss()
    total_loss = loss + 0.01 * smooth_loss
    
    print(f"  Main loss: {loss.item():.4f}")
    print(f"  Smooth loss: {smooth_loss.item():.4f}")
    
    try:
        total_loss.backward()
        print("✓ PASSED: Forward + smoothness loss works")
        return True
    except RuntimeError as e:
        print(f"✗ FAILED: {e}")
        return False


def test_3_forward_with_physics_completion_loss():
    """Test 3: Forward + physics completion loss (the problematic one)"""
    print("\n" + "="*60)
    print("TEST 3: Forward + physics completion loss")
    print("="*60)
    
    args = MockArgs()
    net = AnchorDeformationNet(args).cuda()
    
    N = 1000
    gaussian_pos = torch.randn(N, 3, device='cuda', requires_grad=True)
    scales = torch.randn(N, 3, device='cuda')
    rotations = torch.randn(N, 4, device='cuda')
    density = torch.randn(N, 1, device='cuda')
    time = torch.rand(N, 1, device='cuda')
    
    net.initialize_anchors(gaussian_pos.detach())
    net.update_knn_binding(gaussian_pos.detach())
    
    # Forward (simulates render)
    deformed_pos, _, _ = net(gaussian_pos, scales, rotations, density, time, is_training=True)
    
    # Main loss (simulates render loss)
    loss = deformed_pos.sum()
    
    # Physics completion loss
    # First get teacher predictions with no_grad
    with torch.no_grad():
        _ = net.forward_anchors_unmasked(time)
    
    phys_loss = net.compute_physics_completion_loss()
    total_loss = loss + 0.1 * phys_loss
    
    print(f"  Main loss: {loss.item():.4f}")
    print(f"  Phys loss: {phys_loss.item():.4f}")
    
    try:
        total_loss.backward()
        print("✓ PASSED: Forward + physics completion loss works")
        return True
    except RuntimeError as e:
        print(f"✗ FAILED: {e}")
        return False


def test_4_all_losses_combined():
    """Test 4: All losses combined (simulates full training loop)"""
    print("\n" + "="*60)
    print("TEST 4: All losses combined")
    print("="*60)
    
    args = MockArgs()
    net = AnchorDeformationNet(args).cuda()
    
    N = 1000
    gaussian_pos = torch.randn(N, 3, device='cuda', requires_grad=True)
    scales = torch.randn(N, 3, device='cuda')
    rotations = torch.randn(N, 4, device='cuda')
    density = torch.randn(N, 1, device='cuda')
    time = torch.rand(N, 1, device='cuda')
    
    net.initialize_anchors(gaussian_pos.detach())
    net.update_knn_binding(gaussian_pos.detach())
    
    # Forward (simulates render)
    deformed_pos, _, _ = net(gaussian_pos, scales, rotations, density, time, is_training=True)
    
    # Main loss
    loss = deformed_pos.sum()
    
    # Smoothness loss
    smooth_loss = net.compute_anchor_smoothness_loss()
    
    # Physics completion loss
    with torch.no_grad():
        _ = net.forward_anchors_unmasked(time)
    phys_loss = net.compute_physics_completion_loss()
    
    total_loss = loss + 0.01 * smooth_loss + 0.1 * phys_loss
    
    print(f"  Main loss: {loss.item():.4f}")
    print(f"  Smooth loss: {smooth_loss.item():.4f}")
    print(f"  Phys loss: {phys_loss.item():.4f}")
    print(f"  Total loss: {total_loss.item():.4f}")
    
    try:
        total_loss.backward()
        print("✓ PASSED: All losses combined works")
        return True
    except RuntimeError as e:
        print(f"✗ FAILED: {e}")
        return False


def test_5_multiple_forward_passes():
    """Test 5: Multiple forward passes (potential issue source)"""
    print("\n" + "="*60)
    print("TEST 5: Multiple forward passes")
    print("="*60)
    
    args = MockArgs()
    net = AnchorDeformationNet(args).cuda()
    
    N = 1000
    gaussian_pos = torch.randn(N, 3, device='cuda', requires_grad=True)
    scales = torch.randn(N, 3, device='cuda')
    rotations = torch.randn(N, 4, device='cuda')
    density = torch.randn(N, 1, device='cuda')
    time = torch.rand(N, 1, device='cuda')
    
    net.initialize_anchors(gaussian_pos.detach())
    net.update_knn_binding(gaussian_pos.detach())
    
    # First forward
    deformed_pos1, _, _ = net(gaussian_pos, scales, rotations, density, time, is_training=True)
    loss1 = deformed_pos1.sum()
    
    # Second forward (this might cause issues!)
    deformed_pos2, _, _ = net(gaussian_pos, scales, rotations, density, time, is_training=True)
    loss2 = deformed_pos2.sum()
    
    total_loss = loss1 + loss2
    
    print(f"  Loss 1: {loss1.item():.4f}")
    print(f"  Loss 2: {loss2.item():.4f}")
    
    try:
        total_loss.backward()
        print("✓ PASSED: Multiple forward passes work")
        return True
    except RuntimeError as e:
        print(f"✗ FAILED: {e}")
        return False


def test_6_cached_tensor_reuse():
    """Test 6: Test if cached tensor is properly maintaining gradients"""
    print("\n" + "="*60)
    print("TEST 6: Cached tensor gradient flow")
    print("="*60)
    
    args = MockArgs()
    net = AnchorDeformationNet(args).cuda()
    
    N = 1000
    gaussian_pos = torch.randn(N, 3, device='cuda', requires_grad=True)
    scales = torch.randn(N, 3, device='cuda')
    rotations = torch.randn(N, 4, device='cuda')
    density = torch.randn(N, 1, device='cuda')
    time = torch.rand(N, 1, device='cuda')
    
    net.initialize_anchors(gaussian_pos.detach())
    net.update_knn_binding(gaussian_pos.detach())
    
    # Forward
    deformed_pos, _, _ = net(gaussian_pos, scales, rotations, density, time, is_training=True)
    
    # Check cached tensor
    cached = net._last_anchor_displacements
    print(f"  Cached tensor shape: {cached.shape}")
    print(f"  Cached tensor requires_grad: {cached.requires_grad}")
    print(f"  Cached tensor grad_fn: {cached.grad_fn}")
    
    # Use cached tensor in two different ways
    loss1 = deformed_pos.sum()  # Uses through interpolation
    loss2 = cached.sum()  # Direct use
    
    total_loss = loss1 + loss2
    
    try:
        total_loss.backward()
        print("✓ PASSED: Cached tensor gradient flow works")
        return True
    except RuntimeError as e:
        print(f"✗ FAILED: {e}")
        return False


def test_7_simulate_training_loop():
    """Test 7: Simulate actual training loop structure"""
    print("\n" + "="*60)
    print("TEST 7: Simulated training loop")
    print("="*60)
    
    args = MockArgs()
    net = AnchorDeformationNet(args).cuda()
    
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    
    N = 1000
    gaussian_pos = torch.randn(N, 3, device='cuda', requires_grad=True)
    
    net.initialize_anchors(gaussian_pos.detach())
    net.update_knn_binding(gaussian_pos.detach())
    
    passed = True
    for i in range(3):
        optimizer.zero_grad()
        
        scales = torch.randn(N, 3, device='cuda')
        rotations = torch.randn(N, 4, device='cuda')
        density = torch.randn(N, 1, device='cuda')
        time = torch.rand(N, 1, device='cuda')
        
        # Forward
        deformed_pos, _, _ = net(gaussian_pos, scales, rotations, density, time, is_training=True)
        
        # Losses
        main_loss = deformed_pos.sum()
        smooth_loss = net.compute_anchor_smoothness_loss()
        
        with torch.no_grad():
            _ = net.forward_anchors_unmasked(time)
        phys_loss = net.compute_physics_completion_loss()
        
        total_loss = main_loss + 0.01 * smooth_loss + 0.1 * phys_loss
        
        try:
            total_loss.backward()
            optimizer.step()
            print(f"  Iteration {i+1}: Loss = {total_loss.item():.4f}")
        except RuntimeError as e:
            print(f"  Iteration {i+1}: FAILED - {e}")
            passed = False
            break
    
    if passed:
        print("✓ PASSED: Simulated training loop works")
    else:
        print("✗ FAILED: Simulated training loop failed")
    return passed


if __name__ == "__main__":
    print("PhysX-Gaussian Backward Pass Diagnostic Tests")
    print("=" * 60)
    
    results = {}
    
    results['test_1'] = test_1_basic_forward_backward()
    results['test_2'] = test_2_forward_with_smoothness_loss()
    results['test_3'] = test_3_forward_with_physics_completion_loss()
    results['test_4'] = test_4_all_losses_combined()
    results['test_5'] = test_5_multiple_forward_passes()
    results['test_6'] = test_6_cached_tensor_reuse()
    results['test_7'] = test_7_simulate_training_loop()
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    for name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
    
    if all_passed:
        print("\n✓ All tests passed - issue is likely in integration with GaussianModel")
        print("  Need to check how get_deformed_centers interacts with other losses")
    else:
        print("\n✗ Some tests failed - issue is in AnchorDeformationNet itself")
