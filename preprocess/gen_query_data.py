from pathlib import Path
import argparse
from hloc import extract_features, pairs_from_retrieval

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("--query_dir", type=Path, required=True)
    parser.add_argument("--fx", type=float, required=True)
    parser.add_argument("--fy", type=float, required=True)
    parser.add_argument("--cx", type=float, required=True)
    parser.add_argument("--cy", type=float, required=True)
    parser.add_argument("--w", type=int, required=True)
    parser.add_argument("--h", type=int, required=True)
    args = parser.parse_args()

    #
    db_global_feats_path = Path("outputs/hloc/zedx_mini/db_global_feats.h5")
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
