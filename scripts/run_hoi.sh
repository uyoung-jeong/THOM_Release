#!/bin/bash
obj_prompt=$1
hand_prompt=$2
hoi_prompt=$3
t2hoi_prompt=$4
gpu=${5-0}
output_dir=${6-'output/'$hoi_prompt'_vlrefine'}
output_dir="${output_dir// /_}"

set -x 

CUDA_VISIBLE_DEVICES=$gpu python tools/run_obj.py --opt './configs/obj/it7000_obj.yaml' \
    --prompt "$obj_prompt" --initprompt "$obj_prompt" --output_dir "$output_dir/obj"

CUDA_VISIBLE_DEVICES=$gpu python tools/run_t2hoi.py dataset=grab \
    +test_text="[$t2hoi_prompt]" +nsamples=10 \
    +input_obj_path="['$output_dir/obj/meshify_5000_coarse.ply']" \
    hydra.output_subdir=null hydra/job_logging=disabled hydra/hydra_logging=disabled \
    result_dir="$output_dir"

CUDA_VISIBLE_DEVICES=$gpu python tools/run_hand.py --opt './configs/hand/it7000_hand.yaml' \
    --prompt "$hand_prompt" --initprompt "$hand_prompt" --output_dir "$output_dir/hand" \
    --text2hoi_pkl "$output_dir/t2hoi_results/text2hoi_res_min.pkl"

CUDA_VISIBLE_DEVICES=$gpu python tools/run_vlrefine.py \
    --opt configs/hoi/it1000_hoi.yaml \
    --obj_prompt "$obj_prompt" \
    --hand_prompt "$hand_prompt" \
    --hoi_prompt "$hoi_prompt" \
    --text2hoi_pkl "$output_dir/t2hoi_results/text2hoi_res_min.pkl" \
    --obj_pth "$output_dir/obj/chkpnt7000.pth" \
    --obj_ply "$output_dir/obj/point_cloud/iteration_7000/point_cloud.ply" \
    --hand_pth "$output_dir/hand/chkpnt7000.pth" \
    --output_dir "$output_dir/internvl_t"

CUDA_VISIBLE_DEVICES=$gpu python tools/run_hoi.py --opt './configs/hoi/it1000_hoi.yaml' \
    --obj_prompt "$obj_prompt" \
    --hand_prompt "$hand_prompt" \
    --hoi_prompt "$hoi_prompt" \
    --t2hoi_prompt "$t2hoi_prompt" \
    --text2hoi_pkl "$output_dir/t2hoi_results/text2hoi_res_min_vlm_t.pkl" \
    --obj_pth "$output_dir/obj/chkpnt7000.pth" \
    --obj_ply "$output_dir/obj/point_cloud/iteration_7000/point_cloud.ply" \
    --hand_pth "$output_dir/hand/chkpnt7000.pth" \
    --output_dir "$output_dir/hoi"
