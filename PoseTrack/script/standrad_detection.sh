#!/usr/bin/env bash

# Register a function to be called on exit
function cleanup {
  echo "Cleaning up..."
  pkill -P $$ # Kill all child processes of this script
}

trap cleanup EXIT

set -x

# --- Configuration ---
# Define the specific scene numbers you want to process
SCENES_TO_PROCESS=(61)

# Define hardware resources
start_gpu=0
gpu_nums=4          # Total number of GPUs to use
cpu_nums_per_gpu=20  # Number of CPU cores to assign per process

# --- Execution ---
gpu_idx=0

for scene_num in "${SCENES_TO_PROCESS[@]}"; do
    # Assign a GPU for the current process
    current_gpu=$((start_gpu + gpu_idx))
    export CUDA_VISIBLE_DEVICES=$current_gpu

    # Calculate CPU core range based on the assigned GPU to prevent conflicts
    cpu_start=$((gpu_idx * cpu_nums_per_gpu))
    cpu_end=$((cpu_start + cpu_nums_per_gpu - 1))

    echo "Processing scene $scene_num on GPU $current_gpu with CPUs $cpu_start-$cpu_end"

    # Run detection for the specific scene in the background (standard PyTorch)
    taskset -c $cpu_start-$cpu_end python detection/get_detection.py --scene "$scene_num" --ckpt ckpt_weight/bytetrack_x_mot17.pth &

    # Cycle to the next GPU
    gpu_idx=$(((gpu_idx + 1) % gpu_nums))
    
    # If we have used all available GPUs, wait for them to finish before starting a new batch
    if [ $gpu_idx -eq 0 ]; then
        wait
    fi
done

# Wait for any remaining background jobs to complete
wait

echo "All detection jobs finished."