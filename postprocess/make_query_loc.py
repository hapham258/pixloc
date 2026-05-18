import argparse
import pickle
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_pkl",
        help="Existing query_loc.txt_logs.pkl",
    )
    parser.add_argument(
        "pose_file",
        help="Pose txt file",
    )
    parser.add_argument(
        "--output_pkl",
        default=None,
        help="Output patched pkl",
    )
    args = parser.parse_args()

    #
    pose_path = Path(args.pose_file)
    if args.output_pkl is None:
        output_pkl = pose_path.with_name(pose_path.name + "_logs.pkl")
    else:
        output_pkl = Path(args.output_pkl)

    #
    pose_dict = {}
    with open(args.pose_file, "r") as f:
        lines = f.readlines()
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        timestamp = parts[0]
        pose_dict[timestamp] = {
            "tx": float(parts[1]),
            "ty": float(parts[2]),
            "tz": float(parts[3]),
            "qw": float(parts[4]),
            "qx": float(parts[5]),
            "qy": float(parts[6]),
            "qz": float(parts[7]),
        }

    #
    num_updated = 0
    num_missing = 0
    with open(args.input_pkl, "rb") as f:
        logs = pickle.load(f)
    for image_name in logs["loc"]:
        #
        timestamp = Path(image_name).stem
        if timestamp not in pose_dict:
            num_missing += 1
            continue

        #
        pose = pose_dict[timestamp]
        t_wc = np.array(
            [
                pose["tx"],
                pose["ty"],
                pose["tz"],
            ]
        )
        R_wc = Rotation.from_quat(
            [
                pose["qx"],
                pose["qy"],
                pose["qz"],
                pose["qw"],
            ]
        ).as_matrix()
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        q_cw = Rotation.from_matrix(R_cw).as_quat()

        #
        logs["loc"][image_name]["PnP_ret"]["qvec"] = np.array(
            [
                q_cw[3],
                q_cw[0],
                q_cw[1],
                q_cw[2],
            ]
        )
        logs["loc"][image_name]["PnP_ret"]["tvec"] = t_cw
        logs["loc"][image_name]["PnP_ret"]["success"] = True
        logs["loc"][image_name]["PnP_ret"]["num_inliers"] = 10000
        num_updated += 1

    #
    with open(output_pkl, "wb") as f:
        pickle.dump(logs, f)
    print()
    print(f"Saved: {output_pkl}")
    print(f"Updated poses: {num_updated}")
    print(f"Missing poses: {num_missing}")
