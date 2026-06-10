import argparse
import os
import h5py
from pathlib import Path

import numpy as np
import pycolmap
from hloc import (
    pairs_from_retrieval,
    match_features,
    triangulation,
)


def load_valid_images(validity_file: Path):
    valid_images = []
    with open(validity_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            image_name, valid = line.split()
            if valid == "True":
                valid_images.append(image_name)
    return valid_images


def load_pose_file(pose_file: Path):
    poses = {}
    with open(pose_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            tokens = line.split()
            image_name = tokens[0]
            qw, qx, qy, qz = map(float, tokens[1:5])
            tx, ty, tz = map(float, tokens[5:8])
            poses[image_name] = {
                "qvec": np.array([qw, qx, qy, qz]),
                "tvec": np.array([tx, ty, tz]),
            }
    return poses


def merge_images(image_dirs, output_dir):
    print("Merging images...")
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for src_id, image_dir in enumerate(image_dirs):
        print(f"  source {src_id}: {image_dir}")
        for image_path in image_dir.iterdir():
            dst = output_dir / f"src{src_id}_{image_path.name}"
            if not dst.exists():
                os.symlink(image_path.resolve(), dst)
                count += 1
    print(f"Merged {count} images")


def load_valid_images_multi(validity_files):
    valid_images = []
    for src_id, validity_file in enumerate(validity_files):
        with open(validity_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                image_name, valid = line.split()
                if valid == "True":
                    valid_images.append(f"src{src_id}_{image_name}")
    return valid_images


def load_pose_files_multi(pose_files):
    poses = {}
    for src_id, pose_file in enumerate(pose_files):
        with open(pose_file) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                tokens = line.split()
                image_name = f"src{src_id}_{tokens[0]}"
                qw, qx, qy, qz = map(float, tokens[1:5])
                tx, ty, tz = map(float, tokens[5:8])
                poses[image_name] = {
                    "qvec": np.array([qw, qx, qy, qz]),
                    "tvec": np.array([tx, ty, tz]),
                }
    return poses


def merge_local_features(input_files, output_file):
    print("Merging local features...")
    count = 0
    with h5py.File(output_file, "w") as fout:
        for src_id, input_file in enumerate(input_files):
            print(f"  source {src_id}: {input_file}")
            with h5py.File(input_file, "r") as fin:
                for image_name in fin.keys():
                    fout.copy(fin[image_name], f"src{src_id}_{image_name}")
                    count += 1
    print(f"Merged {count} local feature entries")


def merge_global_features(input_files, output_file):
    print("Merging global features...")
    count = 0
    with h5py.File(output_file, "w") as fout:
        for src_id, input_file in enumerate(input_files):
            print(f"  source {src_id}: {input_file}")
            with h5py.File(input_file, "r") as fin:
                for image_name in fin.keys():
                    fout.copy(fin[image_name], f"src{src_id}_{image_name}")
                    count += 1
    print(f"Merged {count} global feature entries")


def create_reference_model(
    output_model: Path, image_names, poses, fx, fy, cx, cy, w, h
):
    output_model.mkdir(parents=True, exist_ok=True)

    #
    with open(output_model / "cameras.txt", "w") as f:
        f.write("# Camera list\n" "# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"1 PINHOLE {w} {h} " f"{fx} {fy} {cx} {cy}\n")

    #
    with open(output_model / "images.txt", "w") as f:
        f.write(
            "# Image list\n"
            "# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, "
            "CAMERA_ID, NAME\n"
            "# POINTS2D[]\n"
        )
        for image_id, image_name in enumerate(image_names, start=1):
            pose = poses[image_name]
            qw, qx, qy, qz = pose["qvec"]
            tx, ty, tz = pose["tvec"]
            f.write(
                f"{image_id} "
                f"{qw} {qx} {qy} {qz} "
                f"{tx} {ty} {tz} "
                f"1 {image_name}\n"
            )
            f.write("\n")

    #
    with open(output_model / "points3D.txt", "w") as f:
        f.write("# 3D point list\n" "# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")

    #
    recon = pycolmap.Reconstruction()
    recon.read_text(output_model)
    print(recon.summary())


def run_bundle_adjustment(model_path: Path):
    reconstruction = pycolmap.Reconstruction(model_path)
    ba_options = pycolmap.BundleAdjustmentOptions()
    ba_options.refine_focal_length = True
    ba_options.refine_principal_point = True
    ba_options.solver_options.minimizer_progress_to_stdout = True
    ba_options.solver_options.max_num_iterations = 200
    ba_options.refine_extra_params = False
    pycolmap.bundle_adjustment(reconstruction=reconstruction, options=ba_options)
    reconstruction.write(model_path)


if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pose_files",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--validity_files",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--image_dirs",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--global_features",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--local_features",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    args = parser.parse_args()

    #
    args.output.mkdir(parents=True, exist_ok=True)
    pairs_file = args.output / "pairs.txt"
    matches_file = args.output / "matches.h5"
    reference_model = args.output / "reference_model"
    triangulated_model = args.output / "triangulated_model"

    #
    merged_images = args.output / "merged_images"
    merged_global = args.output / "merged_global.h5"
    merged_local = args.output / "merged_local.h5"
    merge_images(
        args.image_dirs,
        merged_images,
    )
    merge_global_features(
        args.global_features,
        merged_global,
    )
    merge_local_features(
        args.local_features,
        merged_local,
    )
    valid_images = load_valid_images_multi(
        args.validity_files,
    )
    poses = load_pose_files_multi(
        args.pose_files,
    )

    #
    print("Generating pairs from retrieval ...")
    pairs_from_retrieval.main(
        descriptors=merged_global,
        output=pairs_file,
        num_matched=20,
    )

    #
    valid_set = set(valid_images)
    filtered_pairs = args.output / "pairs_filtered.txt"
    with open(pairs_file) as fin, open(filtered_pairs, "w") as fout:
        for line in fin:
            name0, name1 = line.strip().split()
            if name0 in valid_set and name1 in valid_set:
                fout.write(line)
    pairs_file = filtered_pairs

    #
    print("Matching image pairs ...")
    matcher_conf = match_features.confs["superpoint+lightglue"]
    match_features.main(
        matcher_conf,
        pairs=pairs_file,
        features=merged_local,
        matches=matches_file,
    )

    #
    print("Creating a reference model ...")
    create_reference_model(
        output_model=reference_model,
        image_names=valid_images,
        poses=poses,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        w=args.w,
        h=args.h,
    )

    #
    print("Triangulating 3D points ...")
    triangulation.main(
        sfm_dir=triangulated_model,
        reference_model=reference_model,
        image_dir=merged_images,
        pairs=pairs_file,
        features=merged_local,
        matches=matches_file,
    )

    #
    print("Running bundle adjustment...")
    run_bundle_adjustment(triangulated_model)
    print("Done.")
    print(f"Model saved to: {triangulated_model}")
