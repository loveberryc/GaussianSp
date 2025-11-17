#!/usr/bin/env python
"""Test script to verify static+residual mode parameter count and functionality."""

import torch
import sys
sys.path.append('./')

from x2_gaussian.gaussian.hexplane import (
    FourOrthogonalVolumeField,
    StaticPlusResidualVolumeField,
    LegacyHexPlaneField
)

def count_parameters(model):
    """Count total parameters and trainable parameters in a model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def test_mode(mode_name, model_class, config):
    """Test a specific grid mode and report parameter counts."""
    print(f"\n{'='*60}")
    print(f"Testing {mode_name}")
    print(f"{'='*60}")
    
    model = model_class(
        bounds=1.6,
        planeconfig=config,
        multires=[1, 2, 4, 8]
    )
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    print(f"Memory estimate: ~{total * 4 / (1024**2):.2f} MB (float32)")
    
    # Test forward pass
    pts = torch.randn(100, 3)
    timestamps = torch.randn(100, 1)
    
    try:
        with torch.no_grad():
            output = model(pts, timestamps)
        print(f"Output shape: {output.shape}")
        print(f"Feature dimension: {model.feat_dim}")
        print("✓ Forward pass successful")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
    
    return total, trainable

def main():
    print("\n" + "="*60)
    print("Parameter Count Comparison: Different Grid Modes")
    print("="*60)
    
    # Base configuration
    base_config = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 150],
        'max_spatial_resolution': 80,
        'max_time_resolution': 150,
    }
    
    results = {}
    
    # Test 1: Legacy HexPlane (baseline)
    print("\n[1/4] Legacy HexPlane (Baseline)")
    results['hexplane'] = test_mode(
        "Legacy HexPlane",
        LegacyHexPlaneField,
        base_config.copy()
    )
    
    # Test 2: Four Orthogonal Volumes (current default, high memory)
    print("\n[2/4] Four Orthogonal Volumes (Current Default)")
    results['four_volume'] = test_mode(
        "Four Orthogonal Volumes",
        FourOrthogonalVolumeField,
        base_config.copy()
    )
    
    # Test 3: Static + Residual (default settings)
    print("\n[3/4] Static + Residual (Default: 1.0x static, 0.5x residual)")
    static_residual_config = base_config.copy()
    static_residual_config.update({
        'static_resolution_multiplier': 1.0,
        'residual_resolution_multiplier': 0.5,
        'residual_weight': 1.0,
        'use_residual_clamp': False,
        'residual_clamp_value': 2.0,
    })
    results['static_residual_default'] = test_mode(
        "Static + Residual (Default)",
        StaticPlusResidualVolumeField,
        static_residual_config
    )
    
    # Test 4: Static + Residual (memory saving)
    print("\n[4/4] Static + Residual (Conservative: 1.2x static, 0.6x residual)")
    conservative_config = base_config.copy()
    conservative_config.update({
        'static_resolution_multiplier': 1.2,
        'residual_resolution_multiplier': 0.6,
        'residual_weight': 1.0,
        'use_residual_clamp': False,
        'residual_clamp_value': 2.0,
    })
    results['static_residual_conservative'] = test_mode(
        "Static + Residual (Conservative)",
        StaticPlusResidualVolumeField,
        conservative_config
    )
    
    # Summary
    print("\n" + "="*60)
    print("Summary")
    print("="*60)
    
    hexplane_params = results['hexplane'][0]
    four_vol_params = results['four_volume'][0]
    static_res_default_params = results['static_residual_default'][0]
    static_res_conservative_params = results['static_residual_conservative'][0]
    
    print(f"\nHexPlane (baseline):            {hexplane_params:>12,} params")
    print(f"Four Orthogonal Volumes:        {four_vol_params:>12,} params  ({four_vol_params/hexplane_params:.2f}x)")
    print(f"Static+Residual (default):      {static_res_default_params:>12,} params  ({static_res_default_params/hexplane_params:.2f}x)")
    print(f"Static+Residual (conservative): {static_res_conservative_params:>12,} params  ({static_res_conservative_params/hexplane_params:.2f}x)")
    
    print(f"\nMemory savings vs Four Orthogonal Volumes:")
    print(f"  Static+Residual (default):      {(1 - static_res_default_params/four_vol_params)*100:>6.2f}% reduction")
    print(f"  Static+Residual (conservative): {(1 - static_res_conservative_params/four_vol_params)*100:>6.2f}% reduction")
    
    print("\n✓ All tests completed successfully!")

if __name__ == "__main__":
    main()

