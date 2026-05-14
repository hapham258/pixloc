import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import cv2
import numpy as np
import open3d as o3d
from tqdm import tqdm


def process_file(
    depth_path,
    pcd_dir,
    fx,
    fy,
    cx,
    cy,
    scale_factor,
    voxel_size,
    max_depth,
    u,
    v,
):
    #
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        print(f"Failed to read: {depth_path}")
        return
    depth = depth.astype(np.float32) / scale_factor

    #
    mask = np.isfinite(depth) & (depth > 0.0) & (depth < max_depth)
    if not np.any(mask):
        print(f"No valid depth: {depth_path}")
        return
    z = depth[mask]
    uu = u[mask]
    vv = v[mask]
    x = (uu - cx) * z / fx
    y = (vv - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)

    #
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    if voxel_size > 0.0:
        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    pcd_path = pcd_dir / f"{depth_path.stem}.pcd"
    success = o3d.io.write_point_cloud(
        str(pcd_path), pcd, write_ascii=False, compressed=False
    )
    if not success:
        print(f"Failed to write: {pcd_path}")


if __name__ == "__main__":

    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth_dir", type=Path, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    parser.add_argument("--scale_factor", type=float, default=1000.0)
    parser.add_argument("--voxel_size", type=float, default=0.1)
    parser.add_argument("--max_depth", type=float, default=4.0)
    parser.add_argument("--num_workers", type=int, default=8)
    args = parser.parse_args()

    #
    depth_files = sorted(args.depth_dir.glob("*.png"))
    pcd_dir = Path("outputs/hloc/zedx_mini/clouds")
    pcd_dir.mkdir(parents=True, exist_ok=True)

    #
    u, v = np.meshgrid(
        np.arange(args.w, dtype=np.float32), np.arange(args.h, dtype=np.float32)
    )
    worker_fn = partial(
        process_file,
        pcd_dir=pcd_dir,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        scale_factor=args.scale_factor,
        voxel_size=args.voxel_size,
        max_depth=args.max_depth,
        u=u,
        v=v,
    )
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        list(tqdm(executor.map(worker_fn, depth_files), total=len(depth_files)))
    print("Done.")
