#!/bin/bash

SCRIPT="python3 eval_mcpt.py"

CUDA_VISIBLE_DEVICES=0 $SCRIPT --scene scene_061  
CUDA_VISIBLE_DEVICES=1 $SCRIPT --scene scene_071  &
CUDA_VISIBLE_DEVICES=2 $SCRIPT --scene scene_081  
# Wait for all to finish
wait
echo "Scene061-070 processes have completed."

# merge results for submission
#python3 tools/merge_results.py