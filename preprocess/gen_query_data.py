from pathlib import Path
import argparse
import pickle
from tqdm import tqdm
import numpy as np
import yaml
from hloc import (
    extract_features,
    pairs_from_retrieval,
    match_features,
    match_dense,
    localize_sfm,
)
from hloc.utils.io import write_poses as original_write_poses
from hloc.utils import io as hloc_io


def safe_write_poses(poses, path, prepend_camera_name=False):
    filtered = {}
    bad = 0
    for name, t in poses.items():
        if hasattr(t, "rotation") and hasattr(t, "translation") and not callable(t):
            filtered[name] = t
        else:
            bad += 1
            print(f"[WARNING] Skipping bad pose entry: " f"{name} ({type(t)})")
    print(f"write_poses: kept {len(filtered)} poses, " f"removed {bad} bad entries")
    return original_write_poses(
        filtered,
        path,
        prepend_camera_name=prepend_camera_name,
    )


def load_selected_config(config_path: Path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    triangulation_cfg = cfg.get("triangulation", {})
    selected = {
        "feature_conf": triangulation_cfg.get("feature_conf"),
        "matcher_conf": triangulation_cfg.get("matcher_conf"),
        "retrieval_conf": triangulation_cfg.get("retrieval_conf"),
    }
    return selected


def create_merged_image_dir(
    db_dir: Path,
    query_dir: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for src_dir in (db_dir, query_dir):
        for src in src_dir.glob("*.png"):
            dst = output_dir / src.name
            if dst.exists():
                raise RuntimeError(f"Duplicate filename detected: {src.name}")
            dst.symlink_to(src.resolve())


hloc_io.write_poses = safe_write_poses
localize_sfm.write_poses = safe_write_poses

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image_dir", type=Path, default=Path("datasets/zedx_mini/images")
    )
    parser.add_argument("--query_name", type=str, required=True)
    parser.add_argument("--session_name", type=str, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    args = parser.parse_args()

    # Specify paths
    base_dir = Path("outputs/hloc/zedx_mini") / args.session_name
    db_global_feats_path = base_dir / "db_global_feats.h5"
    db_local_feats_path = base_dir / "db_local_feats.h5"
    sfm_dir = base_dir / "sfm_model"
    db_img_dir = Path(args.image_dir) / "db"
    query_img_dir = Path(args.image_dir) / args.query_name
    output_dir = base_dir / args.query_name

    # Load selected configs
    config_path = base_dir / "config.yaml"
    config = load_selected_config(config_path)
    print(config)

    # Extract global descriptors for query images
    query_global_feats_path = output_dir / "query_global_feats.h5"
    query_names = sorted(query_img_dir.glob("*.png"))
    retrieval_conf = extract_features.confs[config["retrieval_conf"]]
    extract_features.main(
        retrieval_conf,
        image_dir=query_img_dir,
        feature_path=query_global_feats_path,
        image_list=[p.name for p in query_names],
    )

    # Retrieve top-k matches
    query_db_pairs = output_dir / "query_db_pairs.txt"
    pairs_from_retrieval.main(
        descriptors=query_global_feats_path,
        output=query_db_pairs,
        num_matched=20,
        db_descriptors=db_global_feats_path,
    )
    print(f"Saved pairs to: {query_db_pairs}")

    # Export intrinsics
    queries_with_intrinsics = output_dir / "queries_with_intrinsics.txt"
    with open(queries_with_intrinsics, "w") as f:
        for img_path in query_names:
            name = img_path.name
            f.write(
                f"{name} PINHOLE "
                f"{args.w} {args.h} "
                f"{args.fx} {args.fy} "
                f"{args.cx} {args.cy}\n"
            )
    print(f"Saved queries with intrinsics to: " f"{queries_with_intrinsics}")

    # Select between sparse and dense matching
    query_local_feats_path = output_dir / "query_local_feats.h5"
    query_db_matches_path = output_dir / "query_db_matches.h5"
    if config["matcher_conf"].startswith("loftr"):
        # Perform dense matching
        query_db_matcher_conf = match_dense.confs[config["matcher_conf"]]
        query_db_image_dir = output_dir / "image_dir"
        create_merged_image_dir(
            db_dir=db_img_dir,
            query_dir=query_img_dir,
            output_dir=query_db_image_dir,
        )
        match_dense.main(
            query_db_matcher_conf,
            pairs=query_db_pairs,
            image_dir=query_db_image_dir,
            export_dir=output_dir,
            features=query_local_feats_path,
            matches=query_db_matches_path,
            features_ref=db_local_feats_path,
        )
    else:
        # Extract local descriptors for query images
        local_feature_conf = extract_features.confs[config["feature_conf"]]
        extract_features.main(
            local_feature_conf,
            image_dir=query_img_dir,
            feature_path=query_local_feats_path,
            image_list=[p.name for p in query_names],
        )

        # Match query and database features
        query_db_matcher_conf = match_features.confs[config["matcher_conf"]]
        match_features.main(
            query_db_matcher_conf,
            pairs=query_db_pairs,
            features=query_local_feats_path,
            matches=query_db_matches_path,
            features_ref=db_local_feats_path,
        )

    # Localize queries
    results_path = output_dir / "query_loc.txt"
    hloc_logs_path = Path(str(results_path) + "_logs.pkl")
    if not results_path.exists() or not hloc_logs_path.exists():
        localize_sfm.main(
            reference_sfm=sfm_dir,
            queries=queries_with_intrinsics,
            retrieval=query_db_pairs,
            features=query_local_feats_path,
            matches=query_db_matches_path,
            results=results_path,
            covisibility_clustering=False,
            prepend_camera_name=False,
        )
        print(f"Saved localization results to: {results_path}")
        print(f"Saved hloc logs to: {hloc_logs_path}")
    else:
        print(f"Skipping localization.")

    # Patch for compatibility with PixLoc
    with open(hloc_logs_path, "rb") as f:
        logs = pickle.load(f)
    for loc in tqdm(
        logs["loc"].values(),
        desc="Patching hloc logs",
    ):
        #
        pnp = loc.get("PnP_ret", None)
        if pnp is None:
            continue
        if not isinstance(pnp, dict):
            continue

        #
        if "success" not in pnp:
            pnp["success"] = pnp.get("num_inliers", 0) > 0
        if "inliers" not in pnp and "inlier_mask" in pnp:
            pnp["inliers"] = pnp["inlier_mask"]
        if "qvec" not in pnp or "tvec" not in pnp:
            T = pnp.get(
                "cam_from_world",
                None,
            )
            if callable(T):
                continue
            if T is None:
                continue
            qxyzw = np.asarray(T.rotation.quat)
            pnp["qvec"] = np.array(
                [
                    qxyzw[3],
                    qxyzw[0],
                    qxyzw[1],
                    qxyzw[2],
                ]
            )
            pnp["tvec"] = np.asarray(T.translation)

    # Get bad keys
    bad_keys = []
    for k, loc in logs["loc"].items():
        pnp = loc.get("PnP_ret", None)
        if pnp is None:
            bad_keys.append(k)
            continue
        if not isinstance(pnp, dict):
            bad_keys.append(k)
            continue
        T = pnp.get("cam_from_world", None)
        if callable(T):
            bad_keys.append(k)

    # Remove bad keys from logs
    for k in bad_keys:
        del logs["loc"][k]
    print(f"Removed {len(bad_keys)} invalid log entries")

    # Also filter queries_with_intrinsics.txt
    with open(queries_with_intrinsics, "r") as f:
        lines = f.readlines()
    bad_set = set(bad_keys)
    filtered_lines = []
    for line in lines:
        name = line.split()[0]
        if name not in bad_set:
            filtered_lines.append(line)
    with open(queries_with_intrinsics, "w") as f:
        f.writelines(filtered_lines)
    print(
        f"Filtered queries_with_intrinsics.txt: "
        f"{len(lines)} -> {len(filtered_lines)}"
    )

    # Save new logs
    with open(hloc_logs_path, "wb") as f:
        pickle.dump(logs, f, protocol=pickle.HIGHEST_PROTOCOL)
