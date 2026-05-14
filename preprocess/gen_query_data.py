from pathlib import Path
import argparse
import pickle
from tqdm import tqdm
import numpy as np
from hloc import (
    extract_features,
    pairs_from_retrieval,
    match_features,
    localize_sfm,
)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dir", type=Path, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    args = parser.parse_args()

    # Specify paths
    db_global_feats_path = Path("outputs/hloc/zedx_mini/db_global_feats.h5")
    db_local_feats_path = Path("outputs/hloc/zedx_mini/db_local_feats.h5")
    sfm_dir = Path("outputs/hloc/zedx_mini/sfm_model")
    query_dir = args.query_dir
    output_dir = Path("outputs/hloc/zedx_mini")

    # Extract global descriptors for query images
    query_global_feats_path = output_dir / "query_global_feats.h5"
    query_names = sorted(query_dir.glob("*.png"))
    retrieval_conf = extract_features.confs["netvlad"]
    extract_features.main(
        retrieval_conf,
        image_dir=query_dir,
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

    # Extract local descriptors for query images
    query_local_feats_path = output_dir / "query_local_feats.h5"
    local_feature_conf = extract_features.confs["superpoint_max"]
    extract_features.main(
        local_feature_conf,
        image_dir=query_dir,
        feature_path=query_local_feats_path,
        image_list=[p.name for p in query_names],
    )

    # Match query and database features
    query_db_matches_path = output_dir / "query_db_matches.h5"
    query_db_matcher_conf = match_features.confs["superpoint+lightglue"]
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
        pnp = loc["PnP_ret"]
        if "success" not in pnp:
            pnp["success"] = pnp["num_inliers"] > 0
        if "inliers" not in pnp and "inlier_mask" in pnp:
            pnp["inliers"] = pnp["inlier_mask"]
        if "qvec" not in pnp or "tvec" not in pnp:
            T = pnp["cam_from_world"]
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
    with open(hloc_logs_path, "wb") as f:
        pickle.dump(logs, f, protocol=pickle.HIGHEST_PROTOCOL)
