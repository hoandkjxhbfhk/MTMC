# export dataset_path="/home/anhttt16/lvm_workspace/MTMC/MTMC_Evaluate_2025/data/F11"
# export predict_path="/home/anhttt16/lvm_workspace/MTMC/MTMC_Evaluate_2025/infer/MODEL_103_restructure"

export dataset_path="/home/hoantv/AIC2024/converted"
export predict_path="/home/hoantv/AIC2024/converted"

# Please specify the metrics to print before proceeding.
# You may only choose from these:

# "HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "CLR_FP", "CLR_FN", 
# "IDSW", "Frag", "IDF1", "IDFP", "IDFN", "Dets", "GT_Dets", "IDs", "GT_IDs"

metric_list=("HOTA" "MOTA" "IDF1" "IDSW" "MOTP" "IDFP" "IDFN" "IDs" "GT_IDs" "CLR_FP" "CLR_FN")

# The old MTMC code is faulty, leading to the frame being 2 higher than what it's supposed to be.
# Set this flag to "yes" to account for this
vtx_model_0_offset="no"

python -m mtmc_eval --data_path $dataset_path \
                    --pred_path $predict_path \
                    --metric_list "${metric_list[@]}" \
                    --vtx_offset $vtx_model_0_offset \
                    --mode "mtmc"
