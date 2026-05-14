import argparse
import pickle
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    #
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", type=Path, help="HLoc logs.pkl file")
    args = parser.parse_args()

    #
    log_file = args.log_file
    out_file = Path(str(log_file).replace(".pkl", "_compact.pkl"))
    print(f"[INFO] Loading: {log_file}")
    with open(log_file, "rb") as f:
        data = pickle.load(f)
    logs = data["loc"]

    #
    print(f"[INFO] Converting {len(logs)} entries...")
    compact_logs = {}
    for k, v in tqdm(logs.items()):
        p = v.get("PnP_ret", {})
        compact_logs[k] = {
            "PnP_ret": {
                "success": p.get("success", False),
                "num_inliers": p.get("num_inliers", 0),
                "num_matches": p.get("num_matches", 0),
                "inlier_reprojection_error": p.get(
                    "inlier_reprojection_error",
                    p.get("reproj_error", None),
                ),
            }
        }

    #
    out = {"loc": compact_logs}
    print(f"[INFO] Saving: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("[DONE] Compact log created.")
