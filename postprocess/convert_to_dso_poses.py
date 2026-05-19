import argparse
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def load_valid_set(log_file):
    valid_set = set()
    with open(log_file, "r") as f:
        for line in f:
            #
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue

            #
            parts = line.split()
            image_name = parts[0]
            is_valid = parts[1] == "True"
            if not is_valid:
                continue

            #
            stem = Path(image_name).stem
            valid_set.add(stem)
    return valid_set


def convert_hloc_to_mapping_format(
    hloc_pose_file,
    log_file,
    output_pose_file,
):
    valid_set = load_valid_set(log_file)
    print(f"Loaded {len(valid_set)} valid poses")
    num_written = 0
    with open(hloc_pose_file, "r") as f_in, open(output_pose_file, "w") as f_out:
        f_out.write("#timestamp[ns] tx[m] ty[m] tz[m] qw[] qx[] qy[] qz[]\n")
        for line in f_in:
            #
            line = line.strip()
            if len(line) == 0 or line.startswith("#"):
                continue

            #
            parts = line.split()
            image_name = parts[0]
            timestamp_ns = Path(image_name).stem
            if timestamp_ns not in valid_set:
                continue

            #
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz = map(float, parts[5:8])
            R_wc = R.from_quat([qx, qy, qz, qw]).as_matrix()
            t_wc = np.array([tx, ty, tz])
            R_cw = R_wc.T
            t_cw = -R_cw @ t_wc
            q_cw = R.from_matrix(R_cw).as_quat()
            qx_cw, qy_cw, qz_cw, qw_cw = q_cw
            f_out.write(
                f"{timestamp_ns} "
                f"{t_cw[0]} {t_cw[1]} {t_cw[2]} "
                f"{qw_cw} {qx_cw} {qy_cw} {qz_cw}\n"
            )
            num_written += 1
    print(f"Saved {num_written} poses to: {output_pose_file}")


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser(
        description="Convert valid HLoc poses to mapping pose format"
    )
    parser.add_argument(
        "input_pose_file",
        type=str,
        help="Input HLoc pose file",
    )
    parser.add_argument(
        "log_file",
        type=str,
        help="Input HLoc pose log file",
    )
    parser.add_argument(
        "output_pose_file",
        type=str,
        help="Output mapping-format (DSO type) pose file",
    )
    args = parser.parse_args()

    #
    convert_hloc_to_mapping_format(
        args.input_pose_file,
        args.log_file,
        args.output_pose_file,
    )
