#!/usr/bin/env python
"""
Minimal test to find the double-backward issue in actual training loop.
"""

import torch
import sys
sys.path.insert(0, '/root/autodl-tmp/4dctgs/x2-gaussian-main-origin')

# Load actual training components
from train import scene_reconstruction
from x2_gaussian.gaussian.render_query import render
from x2_gaussian.gaussian.gaussian_model import GaussianModel


def trace_training_loss_computation():
    """
    Trace through the actual training code to find where backward is called twice.
    """
    print("Analyzing train.py for potential double backward issues...")
    
    # Read train.py
    with open('/root/autodl-tmp/4dctgs/x2-gaussian-main-origin/train.py', 'r') as f:
        content = f.read()
    
    # Find all .backward() calls
    import re
    backward_calls = [(m.start(), m.group()) for m in re.finditer(r'\.backward\(', content)]
    
    print(f"\nFound {len(backward_calls)} .backward() calls:")
    
    lines = content.split('\n')
    for pos, match in backward_calls:
        # Find line number
        line_num = content[:pos].count('\n') + 1
        line = lines[line_num - 1].strip()
        print(f"  Line {line_num}: {line}")
    
    # Check for losses that might trigger their own backward
    print("\nChecking for potential issues...")
    
    # Look for patterns that might cause double backward
    patterns = [
        (r'\.backward\(.*retain_graph', 'retain_graph=True usage'),
        (r'with torch\.no_grad\(\):', 'torch.no_grad usage'),
        (r'\.detach\(\)', 'detach() usage'),
        (r'compute_.*loss', 'custom loss functions'),
    ]
    
    for pattern, desc in patterns:
        matches = list(re.finditer(pattern, content))
        print(f"  {desc}: {len(matches)} occurrences")


def test_actual_loss_order():
    """
    Check the order of loss computation in scene_reconstruction.
    """
    print("\n" + "="*60)
    print("Analyzing loss computation order in scene_reconstruction")
    print("="*60)
    
    with open('/root/autodl-tmp/4dctgs/x2-gaussian-main-origin/train.py', 'r') as f:
        content = f.read()
    
    # Find scene_reconstruction function
    import re
    
    # Find where PhysX losses are added
    physx_section = re.search(r'PhysX-Gaussian.*?backward\(\)', content, re.DOTALL)
    if physx_section:
        section = physx_section.group()
        print("\nPhysX-Gaussian loss section found:")
        lines = section.split('\n')[:30]  # First 30 lines
        for i, line in enumerate(lines):
            print(f"  {line}")
    
    # Check what losses come before PhysX
    # Find loss["total"] assignments before PhysX
    before_physx = content[:content.find('PhysX-Gaussian')]
    total_assignments = re.findall(r'loss\["total"\]\s*=.*', before_physx)
    print(f"\nloss['total'] assignments before PhysX: {len(total_assignments)}")
    
    # Check if any compute_*_loss functions call backward internally
    loss_functions = re.findall(r'def (compute_\w+loss)\(', content)
    print(f"\nCustom loss functions in train.py: {loss_functions}")


def check_gaussian_model_losses():
    """
    Check GaussianModel loss functions for backward calls.
    """
    print("\n" + "="*60)
    print("Checking GaussianModel loss functions")
    print("="*60)
    
    with open('/root/autodl-tmp/4dctgs/x2-gaussian-main-origin/x2_gaussian/gaussian/gaussian_model.py', 'r') as f:
        content = f.read()
    
    import re
    
    # Find all compute_*_loss methods
    loss_methods = re.findall(r'def (compute_\w+)\(self.*?\n(?:.*?\n)*?        return', content)
    print(f"Found loss methods: {len(loss_methods)}")
    
    # Check for backward calls in gaussian_model.py
    backward_calls = re.findall(r'\.backward\(', content)
    print(f"Backward calls in gaussian_model.py: {len(backward_calls)}")
    
    # Check for potential issues in compute_physics_completion_loss
    phys_loss = re.search(r'def compute_physics_completion_loss\(.*?(?=\n    def |\Z)', content, re.DOTALL)
    if phys_loss:
        func = phys_loss.group()
        print("\ncompute_physics_completion_loss analysis:")
        if '.backward(' in func:
            print("  WARNING: Contains .backward() call!")
        if 'forward_anchors' in func:
            print("  Contains forward_anchors call")
        if 'forward_anchors_unmasked' in func:
            print("  Contains forward_anchors_unmasked call")
        if 'torch.no_grad' in func:
            print("  Uses torch.no_grad()")


def find_issue_in_train_loop():
    """
    Detailed analysis of the training loop around line 1298.
    """
    print("\n" + "="*60)
    print("Detailed analysis around line 1298 (backward call)")
    print("="*60)
    
    with open('/root/autodl-tmp/4dctgs/x2-gaussian-main-origin/train.py', 'r') as f:
        lines = f.readlines()
    
    # Print lines 1260-1310
    print("\nLines 1260-1310:")
    for i in range(1259, min(1310, len(lines))):
        line = lines[i].rstrip()
        marker = " >>> " if i+1 == 1298 else "     "
        print(f"{i+1:4d}{marker}{line}")
    
    # Look for all loss computations that happen before backward
    print("\n\nLosses added to total before backward (line 1298):")
    
    # Find scene_reconstruction function start
    func_start = None
    for i, line in enumerate(lines):
        if 'def scene_reconstruction(' in line:
            func_start = i
            break
    
    if func_start:
        for i in range(func_start, 1298):
            line = lines[i]
            if 'loss["total"]' in line and ('=' in line or '+=' in line):
                print(f"  Line {i+1}: {line.strip()}")


if __name__ == "__main__":
    trace_training_loss_computation()
    test_actual_loss_order()
    check_gaussian_model_losses()
    find_issue_in_train_loop()
    
    print("\n" + "="*60)
    print("DIAGNOSIS")
    print("="*60)
    print("""
The 'backward through graph twice' error occurs when:
1. A tensor is used in multiple computation paths
2. One path's backward frees the intermediate values
3. Another path tries to use those freed values

In PhysX-Gaussian context, the likely issue is:
- Render uses get_deformed_centers() -> builds graph A
- Some OTHER loss also uses _deformation or related tensors
- When backward() is called, both graphs need the same tensors

CHECK: Are there losses that use self._deformation when 
       use_anchor_deformation=True? These might share tensors.
""")
