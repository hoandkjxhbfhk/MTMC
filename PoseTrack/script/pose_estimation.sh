#!/usr/bin/env bash

# Register a function to be called on exit
function cleanup {
  echo "Cleaning up..."
  pkill -P $$ # Kill all child processes of this script
}

trap cleanup EXIT

cd mmpose

set -x

# --- Configuration ---
# Define the specific scenes you want to process
SCENES_TO_PROCESS=("scene_061" "scene_081")

# Define hardware resources
start_gpu=0
gpu_nums=4          # Total number of GPUs to use
cpu_nums_per_gpu=4  # Number of CPU cores to assign per process

# --- Execution ---
gpu_idx=0

for scene_name in "${SCENES_TO_PROCESS[@]}"; do
    # Assign a GPU for the current process
    current_gpu=$((start_gpu + gpu_idx))
    export CUDA_VISIBLE_DEVICES=$current_gpu

    # Calculate CPU core range based on the assigned GPU to prevent conflicts
    cpu_start=$((current_gpu * cpu_nums_per_gpu))
    cpu_end=$((cpu_start + cpu_nums_per_gpu - 1))

    echo "Processing $scene_name on GPU $current_gpu with CPUs $cpu_start-$cpu_end"

    # Run pose estimation for the specific scene in the background
    # Use the new --scene_name argument
    taskset -c $cpu_start-$cpu_end python demo/save_pose_with_det_multiscene.py \
      demo/mmdetection_cfg/faster_rcnn_r50_fpn_coco.py \
      https://download.openxlab.org.cn/models/mmdetection/FasterR-CNN/weight/faster-rcnn_r50_fpn_1x_coco \
      configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_hrnet-w32_8xb64-210e_coco-256x192.py \
      ../ckpt_weight/td-hm_hrnet-w32_8xb64-210e_coco-256x192-81c58e40_20220909.pth \
      --input examples/88.jpg \
      --output-root vis_results/ \
      --draw-bbox \
      --show-kpt-idx \
      --scene_name "$scene_name" &

    # Cycle to the next GPU
    gpu_idx=$(((gpu_idx + 1) % gpu_nums))
    
    # If we have used all available GPUs, wait for them to finish before starting a new batch
    if [ $gpu_idx -eq 0 ]; then
        wait
    fi
done

# Wait for any remaining background jobs to complete
wait
