import os
import sys
import os.path as osp
import numpy as np

sys.path.append("./")
from x2_gaussian.gaussian.gaussian_model import GaussianModel
from x2_gaussian.arguments import ModelParams
from x2_gaussian.utils.graphics_utils import fetchPly
from x2_gaussian.utils.system_utils import searchForMaxIteration


def initialize_gaussian(gaussians: GaussianModel, args: ModelParams, loaded_iter=None, scene=None):
    if loaded_iter:
        if loaded_iter == -1:
            loaded_iter = searchForMaxIteration(
                osp.join(args.model_path, "point_cloud")
            )
        ply_path = os.path.join(
            args.model_path,
            "point_cloud",
            "iteration_" + str(loaded_iter),
            "point_cloud.pickle",  # Pickle rather than ply
        )
        assert osp.exists(ply_path), f"Cannot find {ply_path} for loading."
        gaussians.load_ply(ply_path)
        print("Loading trained model at iteration {}".format(loaded_iter))


        gaussians.load_model(os.path.join(args.model_path,
                                        "point_cloud",
                                        "iteration_" + str(loaded_iter),
                                                   ))
    else:
        if args.ply_path == "":
            if osp.exists(osp.join(args.source_path, "meta_data.json")):
                ply_path = osp.join(
                    args.source_path, "init_" + osp.basename(args.source_path) + ".npy"
                )
            elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
                ply_path = osp.join(
                    osp.dirname(args.source_path),
                    "init_" + osp.basename(args.source_path).split(".")[0] + ".npy",
                )
            else:
                raise ValueError("Could not recognize scene type!")
        else:
            ply_path = args.ply_path

        assert osp.exists(
            ply_path
        ), f"Cannot find {ply_path} for initialization. Please specify a valid ply_path or generate point cloud with initialize_pcd.py."

        print(f"Initialize Gaussians with {osp.basename(ply_path)}")
        ply_type = ply_path.split(".")[-1]
        if ply_type == "npy":
            point_cloud = np.load(ply_path)
            xyz = point_cloud[:, :3]
            density = point_cloud[:, 3:4]
        elif ply_type == ".ply":
            point_cloud = fetchPly(ply_path)
            xyz = np.asarray(point_cloud.points)
            density = np.asarray(point_cloud.colors[:, :1])

        gaussians.create_from_pcd(xyz, density, 1.0)
    
    # Initialize static prior if requested (only for static_residual mode)
    if hasattr(gaussians, 'args') and gaussians.args is not None:
        if getattr(gaussians.args, 'use_static_prior', False) and gaussians.grid_mode == "static_residual_four_volume":
            if scene is None:
                print("[Warning] use_static_prior=True but no scene provided, skipping static prior initialization")
            else:
                print("\n" + "="*70)
                print("Initializing Static Prior from Training Data (NO TEST DATA LEAKAGE)")
                print("="*70)
                
                from x2_gaussian.utils.static_prior import initialize_static_from_data
                from x2_gaussian.gaussian.deformation import deform_network
                
                # Compute static prior from training data only
                resolution = getattr(gaussians.args, 'static_prior_resolution', 64)
                static_prior_volume = initialize_static_from_data(
                    gaussians, 
                    scene, 
                    resolution=(resolution, resolution, resolution)
                )
                
                # Create multi-scale static priors
                multires = gaussians.args.multires
                static_prior_list = []
                for res_mult in multires:
                    target_res = int(resolution * res_mult * gaussians.args.static_resolution_multiplier)
                    target_res = min(target_res, int(gaussians.args.max_spatial_resolution * gaussians.args.static_resolution_multiplier))
                    target_res = max(target_res, 2)
                    
                    # Resize to target resolution
                    import torch
                    prior_resized = torch.nn.functional.interpolate(
                        static_prior_volume,
                        size=(target_res, target_res, target_res),
                        mode='trilinear',
                        align_corners=True
                    )
                    static_prior_list.append(prior_resized)
                    print(f"  Multi-scale level {len(static_prior_list)}: resolution={target_res}")
                
                # Reinitialize deformation network with static prior
                print("  Reinitializing deformation network with static prior...")
                gaussians._deformation = deform_network(gaussians.args, static_prior=static_prior_list)
                gaussians._deformation.cuda()
                
                print("✓ Static prior initialization complete!")
                print("="*70 + "\n")

    return loaded_iter
