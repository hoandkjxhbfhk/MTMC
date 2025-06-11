#!/usr/bin/env bash

# Register a function to be called on exit
function cleanup {
  echo "Cleaning up..."
  pkill -P $$ # Kill all child processes of this script
}

trap cleanup EXIT

set -x

# --- Configuration ---
# Define the specific scenes you want to process
# SCENES_TO_PROCESS=("scene_061" "scene_081")
SCENES_TO_PROCESS=( "scene_061")
# Create the log directory if it doesn't exist
mkdir -p result/track_log

# --- Execution ---
for scene_name in "${SCENES_TO_PROCESS[@]}"; do
  echo "Running tracking for $scene_name"
  # OMP_NUM_THREADS=1: limits OpenMP to 1 thread to avoid conflicts in parallel runs
  # Run tracking for each scene in the background, redirecting output to a log file
  OMP_NUM_THREADS=20 python track/run_tracking_batch.py "$scene_name" > "result/track_log/${scene_name}.txt" &
done

# Wait for all background tracking jobs to complete
wait

echo "All tracking jobs finished."
