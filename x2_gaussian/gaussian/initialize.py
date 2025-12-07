import os
import sys
import os.path as osp
import numpy as np

sys.path.append("./")
from x2_gaussian.gaussian.gaussian_model import GaussianModel
from x2_gaussian.arguments import ModelParams
from x2_gaussian.utils.graphics_utils import fetchPly
from x2_gaussian.utils.system_utils import searchForMaxIteration


def initialize_gaussian(gaussians: GaussianModel, args: ModelParams, loaded_iter=None,
                        custom_positions=None, custom_densities=None):
    """
    Initialize Gaussians from various sources.
    
    Args:
        gaussians: GaussianModel to initialize
        args: ModelParams with paths
        loaded_iter: Iteration to load from (if resuming)
        custom_positions: Optional custom positions [N, 3] for s4_2 init
        custom_densities: Optional custom densities [N] for s4_2 init
    """
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
    elif custom_positions is not None and custom_densities is not None:
        # s4_2: Initialize from custom positions and densities (from avg CT)
        print(f"Initialize Gaussians from custom positions ({len(custom_positions)} points)")
        xyz = custom_positions
        density = custom_densities.reshape(-1, 1)
        gaussians.create_from_pcd(xyz, density, 1.0)
    else:
        if args.ply_path == "":
            if osp.exists(osp.join(args.source_path, "meta_data.json")):
                ply_path = osp.join(
                    args.source_path, "init_" + osp.basename(args.source_path) + ".npy"
                )
            elif args.source_path.split(".")[-1] in ["pickle", "pkl"]:
                # Use replace instead of split to preserve dots in filename (e.g., noise0.15)
                basename = osp.basename(args.source_path).replace(".pickle", "").replace(".pkl", "")
                ply_path = osp.join(
                    osp.dirname(args.source_path),
                    "init_" + basename + ".npy",
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

    return loaded_iter
