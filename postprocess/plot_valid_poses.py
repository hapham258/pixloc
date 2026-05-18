import argparse
import pickle
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from tqdm import tqdm

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("pose_file")
    parser.add_argument("log_file")
    parser.add_argument(
        "--min_inliers",
        type=int,
        default=30,
        help="Minimum PnP inliers",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every N poses",
    )
    parser.add_argument(
        "--log_type",
        type=str,
        choices=["hloc", "pixloc"],
        default=["hloc"],
        help=("Type of localization log"),
    )
    args = parser.parse_args()

    # Load logs
    print("Loading logs...")
    if args.log_type == "hloc":
        with open(args.log_file, "rb") as f:
            logs = pickle.load(f)["loc"]
    elif args.log_type == "pixloc":
        with open(args.log_file, "rb") as f:
            logs = pickle.load(f)["localization"]
    else:
        raise ValueError(f"Unsupported log_type: {args.log_type}")

    # Load pose file
    poses = []
    validity_log = []
    num_total = 0
    num_missing = 0
    num_failed = 0
    num_valid = 0
    with open(args.pose_file, "r") as f:
        lines = f.readlines()
    print("Processing poses...")
    for line in tqdm(lines):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        num_total += 1
        parts = line.split()
        image_name = parts[0]
        if image_name not in logs:
            num_missing += 1
            validity_log.append((image_name, False))
            continue

        # Process HLoc-style logs
        if args.log_type == "hloc":
            entry = logs[image_name]
            if "PnP_ret" in entry:
                pnp_ret = entry["PnP_ret"]
                num_inliers = pnp_ret.get("num_inliers", 0)
                if num_inliers < args.min_inliers:
                    num_failed += 1
                    validity_log.append((image_name, False))
                    continue

        # Process PixLoc-style logs
        if args.log_type == "pixloc":
            entry = logs[image_name]
            success = entry.get("success", False)
            if not success:
                num_failed += 1
                validity_log.append((image_name, False))
                continue

        # Parse pose
        qw = float(parts[1])
        qx = float(parts[2])
        qy = float(parts[3])
        qz = float(parts[4])
        tx = float(parts[5])
        ty = float(parts[6])
        tz = float(parts[7])
        t = np.array([tx, ty, tz])

        # COLMAP / HLoc convention:
        # X_cam = R * X_world + t
        # camera center = -R^T * t
        R = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
        pos = -R.T @ t
        poses.append(pos)
        num_valid += 1
        validity_log.append((image_name, True))
    print(f"Total poses: {num_total}")
    print(f"Missing logs: {num_missing}")
    print(f"Failed poses: {num_failed}")
    print(f"Valid poses: {num_valid}")
    if len(poses) == 0:
        print("No valid poses remaining after filtering.")
        exit(0)
    positions = np.array(poses)

    # Decimate trajectory
    positions = positions[:: args.stride]
    print(f"Valid poses after filtering: {len(positions)}")

    # Save validity txt log
    log_txt_file = args.log_file + ".txt"
    with open(log_txt_file, "w") as f:
        f.write("#image_file valid\n")
        for image_name, is_valid in validity_log:
            f.write(f"{image_name} {is_valid}\n")
    print(f"Saved text log to: {log_txt_file}")

    # Open3D visualization
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(positions)
    pcd.paint_uniform_color([0.0, 0.0, 1.0])
    start_pcd = o3d.geometry.PointCloud()
    start_pcd.points = o3d.utility.Vector3dVector([positions[0]])
    start_pcd.paint_uniform_color([0.0, 1.0, 0.0])
    end_pcd = o3d.geometry.PointCloud()
    end_pcd.points = o3d.utility.Vector3dVector([positions[-1]])
    end_pcd.paint_uniform_color([1.0, 0.0, 0.0])
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    o3d.visualization.draw_geometries(
        [pcd, start_pcd, end_pcd, frame],
        window_name="Filtered HLoc Trajectory",
    )
