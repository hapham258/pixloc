import argparse
import numpy as np
from scipy.spatial.transform import Rotation as R
from sklearn.cluster import DBSCAN
import open3d as o3d


def load_validity(validity_file):
    valid = {}
    with open(validity_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split()
            if len(parts) != 2:
                continue

            name, flag = parts
            valid[name] = flag.lower() == "true"
    return valid


def load_poses(pose_file, valid_map=None):
    poses = []
    with open(pose_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) != 8:
                continue

            name = parts[0]
            if valid_map is not None and not valid_map.get(name, False):
                continue

            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            poses.append(
                {
                    "name": name,
                    "q": np.array([qw, qx, qy, qz], dtype=np.float64),
                    "t": np.array([tx, ty, tz], dtype=np.float64),
                }
            )
    return poses


def filter_largest_cluster(poses, eps=0.3, min_samples=10):
    centers = np.array([camera_center(p["q"], p["t"]) for p in poses])
    labels = DBSCAN(
        eps=eps,
        min_samples=min_samples,
    ).fit_predict(centers)
    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        print("DBSCAN found no clusters.")
        return poses
    unique, counts = np.unique(
        valid_labels,
        return_counts=True,
    )
    main_label = unique[np.argmax(counts)]
    keep_mask = labels == main_label
    filtered = [p for p, keep in zip(poses, keep_mask) if keep]
    print(f"DBSCAN: kept {len(filtered)} / {len(poses)} poses " f"in largest cluster")
    return filtered


def quat_to_rotmat(q):
    qw, qx, qy, qz = q
    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def camera_center(q, t):
    Rcw = quat_to_rotmat(q)
    return -Rcw.T @ t


def pose_to_world_transform(q, t):
    qw, qx, qy, qz = q
    Rcw = R.from_quat([qx, qy, qz, qw]).as_matrix()
    Twc = np.eye(4)
    Twc[:3, :3] = Rcw.T
    Twc[:3, 3] = -Rcw.T @ t
    return Twc


def create_frustums(poses, indices, scale, color):
    all_points = []
    all_lines = []
    all_colors = []
    offset = 0
    for idx in indices:
        p = poses[idx]
        T = pose_to_world_transform(
            p["q"],
            p["t"],
        )
        pts = np.array(
            [
                [0, 0, 0],
                [-1, -0.75, 1],
                [1, -0.75, 1],
                [1, 0.75, 1],
                [-1, 0.75, 1],
            ],
            dtype=np.float64,
        )
        pts *= scale
        pts_h = np.hstack([pts, np.ones((5, 1))])
        pts_w = (T @ pts_h.T).T[:, :3]
        lines = (
            np.array(
                [
                    [0, 1],
                    [0, 2],
                    [0, 3],
                    [0, 4],
                    [1, 2],
                    [2, 3],
                    [3, 4],
                    [4, 1],
                ]
            )
            + offset
        )
        all_points.append(pts_w)
        all_lines.append(lines)
        all_colors.extend([color] * len(lines))
        offset += 5
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    ls.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(all_colors))
    return ls


def rotation_difference_deg(q1, q2):
    r1 = R.from_quat([q1[1], q1[2], q1[3], q1[0]])
    r2 = R.from_quat([q2[1], q2[2], q2[3], q2[0]])
    rel = r2 * r1.inv()
    return np.degrees(rel.magnitude())


def select_keyframes(poses, trans_thresh=0.15, rot_thresh_deg=8.0):
    n = len(poses)
    if n <= 2:
        return list(range(n))

    centers = [camera_center(p["q"], p["t"]) for p in poses]
    keep = [0]
    acc_trans = 0.0
    acc_rot = 0.0
    for i in range(1, n):
        acc_trans += np.linalg.norm(centers[i] - centers[i - 1])
        acc_rot += rotation_difference_deg(poses[i - 1]["q"], poses[i]["q"])
        if acc_trans >= trans_thresh or acc_rot >= rot_thresh_deg:
            keep.append(i)
            acc_trans = 0.0
            acc_rot = 0.0
    if keep[-1] != n - 1:
        keep.append(n - 1)
    return keep


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("pose_file")
    parser.add_argument("validity_file")
    parser.add_argument("--trans_thresh", type=float, default=1.0)
    parser.add_argument("--rot_thresh", type=float, default=45.0)
    parser.add_argument("--dbscan_eps", type=float, default=0.3)
    parser.add_argument("--dbscan_min_samples", type=int, default=10)
    args = parser.parse_args()

    #
    valid_map = load_validity(args.validity_file)
    all_poses = load_poses(args.pose_file, None)
    poses = load_poses(args.pose_file, valid_map)
    print(
        f"Validity filter: {len(poses)} / {len(all_poses)} poses "
        f"({100.0 * len(poses) / len(all_poses):.1f}%)"
    )
    poses = filter_largest_cluster(
        poses,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples,
    )
    if len(poses) == 0:
        raise RuntimeError("No valid poses found.")
    keep_indices = select_keyframes(
        poses,
        trans_thresh=args.trans_thresh,
        rot_thresh_deg=args.rot_thresh,
    )

    #
    selected_names = {poses[i]["name"] for i in keep_indices}
    output_file = args.validity_file + ".sub.txt"
    with open(output_file, "w") as f:
        f.write("#image_file valid\n")
        for pose in all_poses:
            valid = pose["name"] in selected_names
            f.write(f"{pose['name']} {valid}\n")
    print(
        f"Selected {len(keep_indices)} / {len(poses)} frames "
        f"({100.0 * len(keep_indices) / len(poses):.1f}%)"
    )
    print(f"Saved to {output_file}")

    #
    all_frustums = create_frustums(
        poses,
        range(len(poses)),
        scale=0.02,
        color=(0.6, 0.6, 0.6),
    )
    selected_frustums = create_frustums(
        poses,
        keep_indices,
        scale=0.04,
        color=(0.0, 1.0, 0.0),
    )
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Gray=all poses, Green=selected poses")
    vis.add_geometry(all_frustums)
    vis.add_geometry(selected_frustums)
    opt = vis.get_render_option()
    opt.background_color = np.array([0.0, 0.0, 0.0])
    vis.run()
    vis.destroy_window()
