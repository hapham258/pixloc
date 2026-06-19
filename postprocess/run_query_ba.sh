#!/bin/bash
set -e

#
if [ $# -ne 8 ]; then
    echo "usage: $0 {pixloc|hloc} session_name fx fy cx cy w h"
    exit 1
fi
log_type=$1
session_name=$2
fx=$3
fy=$4
cx=$5
cy=$6
w=$7
h=$8

#
result_root="outputs/results/${session_name}"
hloc_root="outputs/hloc/zedx_mini/${session_name}"
image_root="datasets/zedx_mini/images"
pose_files=()
validity_files=()
image_dirs=()
local_features=()
global_features=()
for query_dir in "${result_root}"/query[0-9]*; do
    query_name=$(basename "$query_dir")
    if [ "$log_type" = "pixloc" ]; then
        pose_file="${query_dir}/pixloc_zedx_mini.txt"
        validity_file="${query_dir}/pixloc_zedx_mini.txt_logs.pkl.txt.sub.txt"
    elif [ "$log_type" = "hloc" ]; then
        pose_file="${hloc_root}/${query_name}/query_loc.txt"
        validity_file="${hloc_root}/${query_name}/query_loc.txt_logs_compact.pkl.txt"
    else
        echo "Invalid log type: $log_type (expected pixloc or hloc)"
        exit 1
    fi
    image_dir="${image_root}/${query_name}"
    local_feature="${hloc_root}/${query_name}/query_local_feats.h5"
    global_feature="${hloc_root}/${query_name}/query_global_feats.h5"
    if [[ ! -f "$pose_file" ||
          ! -f "$validity_file" ||
          ! -d "$image_dir" ||
          ! -f "$local_feature" ||
          ! -f "$global_feature" ]]; then
        echo "Skipping ${query_name}: missing files"
        continue
    fi
    pose_files+=("$pose_file")
    validity_files+=("$validity_file")
    image_dirs+=("$image_dir")
    local_features+=("$local_feature")
    global_features+=("$global_feature")
done

#
if [ "$log_type" = "pixloc" ]; then
    output_dir="${result_root}/query_ba"
else
    output_dir="${result_root}/query_ba_no_pixloc"
fi
python postprocess/run_query_ba.py \
    --pose_files "${pose_files[@]}" \
    --validity_files "${validity_files[@]}" \
    --image_dirs "${image_dirs[@]}" \
    --local_features "${local_features[@]}" \
    --global_features "${global_features[@]}" \
    --config_file "${hloc_root}/config.yaml" \
    --output "${output_dir}" \
    --fx "${fx}" --fy "${fy}" --cx "${cx}" --cy "${cy}" --w "${w}" --h "${h}"
