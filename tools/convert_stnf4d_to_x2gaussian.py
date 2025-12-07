#!/usr/bin/env python3
"""
Convert STNF4D pickle data format to X2-Gaussian format.

Key differences:
- X2-Gaussian requires 'time' field (0.0 - 1.0)
- X2-Gaussian uses 0-based phase (0-9), STNF4D uses 1-based (1-10)
- X2-Gaussian may need 'scanTime' field

Usage:
    python convert_stnf4d_to_x2gaussian.py --input <stnf4d.pickle> --output <x2gaussian.pickle>
    
    # Or convert all files in a directory
    python convert_stnf4d_to_x2gaussian.py --input_dir /path/to/stnf4d/data --output_dir /path/to/x2gaussian/data
"""

import os
import argparse
import pickle
import numpy as np
from tqdm import tqdm


def convert_stnf4d_to_x2gaussian(input_path: str, output_path: str) -> dict:
    """
    Convert a single STNF4D pickle file to X2-Gaussian format.
    
    Args:
        input_path: Path to STNF4D pickle file
        output_path: Path to save converted X2-Gaussian pickle file
        
    Returns:
        dict: Conversion statistics
    """
    print(f"\n{'='*60}")
    print(f"Converting: {os.path.basename(input_path)}")
    print(f"{'='*60}")
    
    # Load STNF4D data
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get number of phases from config or data
    num_phases = len(np.unique(data['train']['phase']))
    print(f"  Detected {num_phases} phases")
    
    # Create output data (copy scanner parameters)
    output_data = {}
    
    # Copy scanner parameters (these are the same)
    scanner_keys = [
        'numTrain', 'numVal', 'DSD', 'DSO', 
        'nDetector', 'dDetector', 'nVoxel', 'dVoxel',
        'offOrigin', 'offDetector', 'accuracy', 'mode', 'filter',
        'totalAngle', 'startAngle', 'randomAngle',
        'convert', 'rescale_slope', 'rescale_intercept', 
        'normalize', 'noise'
    ]
    
    for key in scanner_keys:
        if key in data:
            output_data[key] = data[key]
    
    # Copy image (GT volumes)
    if 'image' in data:
        output_data['image'] = data['image']
        print(f"  GT volume shape: {data['image'].shape}")
    
    # Add scanTime if not present (total scan time in seconds)
    # Assume 1 second per phase cycle for now
    if 'scanTime' not in output_data:
        output_data['scanTime'] = float(num_phases)
    
    # Convert train split
    print(f"  Converting train split...")
    output_data['train'] = convert_split(
        data['train'], 
        num_phases,
        data.get('numTrain', len(data['train']['projections']))
    )
    print(f"    - projections: {output_data['train']['projections'].shape}")
    print(f"    - angles: {output_data['train']['angles'].shape}")
    print(f"    - phase range: {output_data['train']['phase'].min()} - {output_data['train']['phase'].max()}")
    print(f"    - time range: {output_data['train']['time'].min():.4f} - {output_data['train']['time'].max():.4f}")
    
    # Convert val split
    print(f"  Converting val split...")
    output_data['val'] = convert_split(
        data['val'],
        num_phases,
        data.get('numVal', len(data['val']['projections']))
    )
    print(f"    - projections: {output_data['val']['projections'].shape}")
    print(f"    - angles: {output_data['val']['angles'].shape}")
    print(f"    - phase range: {output_data['val']['phase'].min()} - {output_data['val']['phase'].max()}")
    print(f"    - time range: {output_data['val']['time'].min():.4f} - {output_data['val']['time'].max():.4f}")
    
    # Save converted data
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f, pickle.HIGHEST_PROTOCOL)
    
    print(f"  ✓ Saved to: {output_path}")
    
    # Return stats
    return {
        'input': input_path,
        'output': output_path,
        'num_train': output_data['numTrain'],
        'num_val': output_data['numVal'],
        'num_phases': num_phases,
        'image_shape': output_data['image'].shape if 'image' in output_data else None
    }


def convert_split(split_data: dict, num_phases: int, num_samples: int) -> dict:
    """
    Convert a single split (train/val) from STNF4D to X2-Gaussian format.
    
    Args:
        split_data: Dict containing projections, angles, phase
        num_phases: Number of phases
        num_samples: Number of samples in this split
        
    Returns:
        Converted split data with time field added
    """
    output_split = {}
    
    # Copy projections and angles
    output_split['projections'] = split_data['projections'][:num_samples]
    output_split['angles'] = split_data['angles'][:num_samples]
    
    # Convert phase from 1-based to 0-based
    phase = split_data['phase'][:num_samples]
    if phase.min() >= 1:
        # STNF4D uses 1-based phase (1, 2, ..., N)
        # X2-Gaussian uses 0-based phase (0, 1, ..., N-1)
        phase = phase - 1
    output_split['phase'] = phase.astype(np.int32)
    
    # Generate time field from phase
    # time = phase / num_phases, normalized to [0, 1)
    # Add small offset for each sample within same phase to avoid identical times
    time = phase.astype(np.float32) / num_phases
    
    # Add small random offset to differentiate samples with same phase
    # This helps the deformation network learn continuous motion
    phase_counts = {}
    time_with_offset = np.zeros_like(time)
    for i, p in enumerate(phase):
        if p not in phase_counts:
            phase_counts[p] = 0
        else:
            phase_counts[p] += 1
        # Small offset within phase bin
        offset = phase_counts[p] * 0.001  # Very small offset
        time_with_offset[i] = time[i] + offset
    
    output_split['time'] = time_with_offset.astype(np.float32)
    
    return output_split


def main():
    parser = argparse.ArgumentParser(description='Convert STNF4D pickle to X2-Gaussian format')
    parser.add_argument('--input', type=str, help='Input STNF4D pickle file')
    parser.add_argument('--output', type=str, help='Output X2-Gaussian pickle file')
    parser.add_argument('--input_dir', type=str, help='Input directory containing STNF4D pickle files')
    parser.add_argument('--output_dir', type=str, help='Output directory for X2-Gaussian pickle files')
    
    args = parser.parse_args()
    
    if args.input and args.output:
        # Single file conversion
        convert_stnf4d_to_x2gaussian(args.input, args.output)
    elif args.input_dir and args.output_dir:
        # Directory conversion
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Find all pickle files (excluding extract_projections.py artifacts)
        pickle_files = [
            f for f in os.listdir(args.input_dir) 
            if f.endswith('.pickle') and not f.startswith('.')
        ]
        
        print(f"Found {len(pickle_files)} pickle files to convert")
        
        all_stats = []
        for pickle_file in tqdm(pickle_files, desc="Converting"):
            input_path = os.path.join(args.input_dir, pickle_file)
            output_path = os.path.join(args.output_dir, pickle_file)
            
            try:
                stats = convert_stnf4d_to_x2gaussian(input_path, output_path)
                all_stats.append(stats)
            except Exception as e:
                print(f"  ✗ Error converting {pickle_file}: {e}")
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"CONVERSION SUMMARY")
        print(f"{'='*60}")
        print(f"Successfully converted: {len(all_stats)}/{len(pickle_files)} files")
        for stats in all_stats:
            print(f"  - {os.path.basename(stats['output'])}: "
                  f"train={stats['num_train']}, val={stats['num_val']}, "
                  f"phases={stats['num_phases']}")
    else:
        parser.print_help()
        print("\nExamples:")
        print("  # Convert single file:")
        print("  python convert_stnf4d_to_x2gaussian.py --input data/XCAT.pickle --output data/XCAT_converted.pickle")
        print()
        print("  # Convert all files in directory:")
        print("  python convert_stnf4d_to_x2gaussian.py --input_dir /path/to/stnf4d/data --output_dir ./data")


if __name__ == '__main__':
    main()
