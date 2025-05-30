import argparse
import json
import shutil
import os
import cv2
import pandas as pd
from prettytable import PrettyTable
from datetime import datetime
import numpy as np

from src.dataset.vtx_dataloader import VTXDataset2024
from src.tools.utils import (convert_mot_to_mtmc,
                             convert_txt_to_csv,
                             merge_dataframe,
                             convert_dataframe_to_txt,
                             convert_mot_txt_to_csv)

from src.scripts.run_mot_challenge import evaluate_mtmc

"""
    Fixed variables.
"""

AVAILABLE_METRICS = ["scene", "HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "CLR_FP", "CLR_FN", "IDSW", 
                     "Frag", "IDF1", "IDFP", "IDFN", "Dets", "GT_Dets", "IDs", "GT_IDs"]

SEQINFO_TEMPLATE = """[Sequence]
name={}
imDir=img1
frameRate={}
seqLength={}
imWidth={}
imHeight={}
imExt=.jpg
"""

"""
    Step 2 functions: Create ground truth and prediction data frames
"""

def get_pred_files(path):
    """Get path of MOT and MTMC prediction files path

    pred_dir:
        F11_01:
            mot:
                111101.txt
                111103.txt
                ..........
            mtmc:
                111101.txt
                111103.txt
                ..........
        F11_02:...........
        ..................
    """
    pred_mot = []
    pred_mtmc = []

    cam_ids = [cam_id for cam_id in sorted(os.listdir(path)) if (os.path.isdir(os.path.join(path, cam_id)) and cam_id != "viz")]

    for cam_id in cam_ids:
        pred_mot.append(os.path.join(path, cam_id, "mot", f"{cam_id}.txt"))

    for cam_id in cam_ids:
        pred_mtmc.append(os.path.join(path, cam_id, "mtmc", f"{cam_id}.txt"))

    # for mot_file in sorted(os.listdir(os.path.join(path, "mot"))):
    #     if mot_file.endswith(".txt"):
    #         pred_mot.append(os.path.join(os.path.join(path, "mot"), mot_file))

    # for mtmc_file in sorted(os.listdir(os.path.join(path, "mtmc"))):
    #     if mtmc_file.endswith(".txt"):
    #         pred_mtmc.append(os.path.join(os.path.join(path, "mtmc"), mtmc_file))
        
    return pred_mot, pred_mtmc

def merge_view_into_multiview(df_gts, df_preds, vtx_offset):
    """Concat all video track in a scene to a single track for MOT evaluation

    Params
    ------
    df_gts : pandas.DataFrame
        Ground truth data of all video in a scene.
    df_preds : pandas.DataFrame
        Prediction/test data of all video in a scene. 
    Returns
    -------
    {}_processed : pandas.DataFrame
        Results after concat all 

    Sample gt dataframe input:
         CameraId  FrameId    Id       X      Y   Width  Height
    0      111101      1.0   1.0  1019.0  586.0   703.0  1574.0
    1      111101      1.0   2.0  2544.0  136.0   346.0   753.0
    2      111101      1.0   3.0   589.0  295.0   397.0  1250.0
    3      111101      1.0   4.0  1781.0   83.0   250.0   629.0
    4      111101      1.0   5.0  1083.0   26.0   282.0   750.0
    ...       ...      ...   ...     ...    ...     ...     ...
    6679   111111    237.0   1.0   928.0  143.0   243.0   661.0
    6680   111111    237.0   3.0  1397.0  248.0   341.0   646.0
    6681   111111    237.0   4.0   756.0  730.0  1177.0  1427.0
    6682   111111    237.0   2.0     0.0  773.0   340.0  1386.0
    6683   111111    237.0  19.0  2607.0  275.0   305.0  1054.0

    """

    # gt: ground truth, ts: infer results

    gtds = []
    tsds = []

    # get all unique camera ids of each set
    gtcams = df_gts['CameraId'].unique()
    tscams = df_preds['CameraId'].unique()

    maxFrameId = 0

    for k in sorted(gtcams):
        gtd = df_gts.loc[df_gts['CameraId'] == k]

        gtd = gtd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
        # max FrameId in gtd only
        mfid = gtd['FrameId'].max()
        gtd['FrameId'] += maxFrameId

        gtds.append(gtd)

        if k in tscams:
            tsd = df_preds.loc[df_preds["CameraId"] == k]
            tsd = tsd[['FrameId', 'Id', 'X', 'Y', 'Width', 'Height']]
            # max FrameId among both gtd and tsd
            mfid = max(mfid, tsd['FrameId'].max())
            tsd['FrameId'] += maxFrameId

            if vtx_offset == "yes":
                tsd['FrameId'] -= 2 # With the VTX's model 0 code, there is a major issue with frame offset
            
            tsds.append(tsd)

            # print(f"Currently doing cam {k}:, mtmc txt pred\n{tsd}")

        maxFrameId += mfid

    try:
        gt_processed = pd.concat(gtds)
        pred_processed = pd.concat(tsds)
    except:
        raise Exception("Wrong concat")
    return gt_processed, pred_processed    

def process_one_scene(gt_path, pred_path, vtx_offset):
    """Convert list of MOT result of each camera into 1 MTMC files

    Args:
        gt_path (List): List of paths to groundtruth files
        pred_path (List): List of paths to prediction files
    Returns:
        Pandas Dataframes: [description]
    """

    gt_cam_ids = [os.path.split(cam_id)[-1][:-4] for cam_id in gt_path]
    pred_cam_ids = [os.path.split(cam_id)[-1][:-4] for cam_id in pred_path]

    scene_name = gt_path[0].split('/')[-3]

    assert gt_cam_ids == pred_cam_ids, f"""Scene {scene_name} contains inconsistent gt-pred data:\n
                                            GT: {gt_cam_ids}
                                            Pred: {pred_cam_ids}
                                            """

    gts = [] 
    preds = []
    
    for idx in range(len(gt_path)):
        
        gt = convert_mot_to_mtmc(gt_path[idx]) # return mot info + cameraid
        gt = convert_txt_to_csv(gt) # convert to dataframe

        pred = convert_mot_to_mtmc(pred_path[idx])
        pred = convert_txt_to_csv(pred)
        
        gts.append(gt)
        preds.append(pred) 
    df_gts = merge_dataframe(gts) 
    df_preds = merge_dataframe(preds)
    
    gt_processed, pred_processed = merge_view_into_multiview(df_gts, df_preds, vtx_offset)
    return gt_processed, pred_processed

"""
    Result presentation functions
"""

def print_and_save_metrics_table(metrics, to_print, output_path=None):
    """
    Print a table of metrics and save it to a TXT file.

    Args:
        metrics (dict): Dictionary containing metrics for different scenes.
        output_file (str): Path to the file where the table should be saved.
    """

    header = ["scene"]
    header.extend(to_print)

    # Initialize the table
    table = PrettyTable()
    table.field_names = header

    # Populate the table with scene data

    idx = 0

    for scene, values in metrics.items():
        row = {key: "-" for key in header}
        row["scene"] = scene

        # Fill the row with metric values
        for category, metrics_data in values.items():
            for metric, value in metrics_data.items():
                if metric in header:
                    row[metric] = value

        
        
        if idx == len(metrics.keys()) - 2:
            table.add_row([row[key] for key in header], divider=True)
        else:
            table.add_row([row[key] for key in header])

        idx += 1

    # Print the table
    print(table)

    if output_path is not None:
        with open(output_path, "w") as result_file:
            result_file.write(str(table))
            

"""
    The main function stuff.
"""

def main(opt):

    # Step 0. Extract configurations

    """
        The ground truth's data structure should look like this:

        gt_dir:
            F11_01:
                111101:
                    111101.mp4
                    111101.txt
                111103:
                    111103.mp4
                    111103.txt
                111105:.......
                ..............
                grid_view.mp4
                grid_view.txt
            F11_02:...........
            ..................
    """
    data_path = opt.data_path

    """
        The prediction's data structure should look like this:
    
        pred_path:
            F11_01:
                mot:
                    111101.txt
                    111103.txt
                    ..........
                mtmc:
                    111101.txt
                    111103.txt
                    ..........
            F11_02:...........
            ..................
    """
    pred_path = opt.pred_path
    vtx_offset = opt.vtx_offset
    mode = opt.mode

    # Step 1. Initiate the VTX Dataset

    vtx_dataset = VTXDataset2024(data_path, (3840, 2160))

    # Step 2. Setup MOT/MTMC Dataframes for both ground truth and prediction
    
    scene_datas = {}

    if mode == "mtmc":
        for scene_name, scene_data in vtx_dataset.scene_info.items():

            gt_txt_files = []
            pred_txt_files = []

            combined_scene_length = 0

            ## Parse ground truth information

            for camera_name, camera_info in scene_data["cameras"].items():
                gt_txt_files.append(camera_info["label"])

                ### Parse information for seqini
                cap = cv2.VideoCapture(camera_info["video"])

                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                combined_scene_length += int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            ## Parse prediction information

            _, pred_txt_files = get_pred_files(f"{pred_path}/{scene_name}")

            # Construct data for Step 3

            gt_mtmc_df, pred_mtmc_df = process_one_scene(gt_txt_files, pred_txt_files, vtx_offset)
            
            scene_datas[scene_name] = {
                "seq_ini": {
                    "frameRate": fps,
                    "seqLength": combined_scene_length,
                    "imWidth": width,
                    "imHeight": height
                },
                "gt_mtmc_df": gt_mtmc_df,
                "pred_mtmc_df": pred_mtmc_df            
            }

    elif mode == "mot":
        for scene_name, scene_data in vtx_dataset.scene_info.items():

            pred_txt_files, _ = get_pred_files(f"{pred_path}/{scene_name}")

            cam_idx = 0
            for camera_name, camera_info in scene_data["cameras"].items():

                mot_name = f"{scene_name}_{camera_info['cam_id']}"

                pred_cam_id = pred_txt_files[cam_idx].split('/')[-1][:-4]
                assert pred_cam_id == camera_info["cam_id"], f'GT-Pred Mismatch! pred: {pred_cam_id}. gt: {camera_info["cam_id"]}'

                gt_mot_lines = []
                pred_mot_lines = []

                with open(camera_info["label"]) as mot_gt:
                    for line in mot_gt:
                        line_data = line.strip().split(',')
                        gt_mot_lines.append(line_data)

                with open(pred_txt_files[cam_idx]) as mot_pred:
                    for line in mot_pred:
                        line_data = line.strip().split()
                        pred_mot_lines.append(line_data)

                gt_mot_df = convert_mot_txt_to_csv(gt_mot_lines)
                pred_mot_df = convert_mot_txt_to_csv(pred_mot_lines)
                
                if vtx_offset == "yes":
                    pred_mot_df['FrameId'] -= 2
                
                pred_mot_df['FrameId'] -= 1
                
                cap = cv2.VideoCapture(camera_info["video"])
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                seq_len = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                scene_datas[mot_name] = {
                    "seq_ini": {
                        "frameRate": fps,
                        "seqLength": seq_len,
                        "imWidth": width,
                        "imHeight": height
                    },
                    "gt_mtmc_df": gt_mot_df,
                    "pred_mtmc_df": pred_mot_df            
                }

                cam_idx += 1

    # Step 3. Setup TrackEval MTMC data

    gt_trackeval_path = "./src/data/gt/mot_challenge"
    gt_files = os.path.join(gt_trackeval_path, "MOT16-test")
    gt_seqmaps = os.path.join(gt_trackeval_path, "seqmaps")

    os.makedirs(gt_files, exist_ok=True)
    os.makedirs(gt_seqmaps, exist_ok=True)

    seqmap_file = open(os.path.join(gt_seqmaps, "MOT16-test.txt"), mode="w+")
    seqmap_file.writelines("name\n")

    pred_files = "./src/data/trackers/mot_challenge/MOT16-test/VTX_MTMC/data"

    os.makedirs(pred_files, exist_ok=True)

    ## Loop through scenes and initiate directories

    for scene_name in scene_datas.keys():

        # Write to seqmap
        seqmap_file.writelines(f"{scene_name}\n")
        
        # Scene directory establishment
        scene_gt_path = os.path.join(gt_files, scene_name)
        os.makedirs(scene_gt_path, exist_ok=True)
        os.makedirs(f"{scene_gt_path}/gt", exist_ok=True)

        # Create seqinfo.ini files for the scene. Code is a bit lmfao. Please ignore.

        max_gt_pred_frame = max(scene_datas[scene_name]["gt_mtmc_df"]["FrameId"].max(),
                                scene_datas[scene_name]["pred_mtmc_df"]["FrameId"].max()) + 1
        max_seq_length = max(scene_datas[scene_name]["seq_ini"]["seqLength"], max_gt_pred_frame)

        with open(f"{scene_gt_path}/seqinfo.ini", mode="w+") as seqini_file:
            seqini_file.write(
                SEQINFO_TEMPLATE.format(scene_name,
                                        scene_datas[scene_name]["seq_ini"]["frameRate"],
                                        int(max_seq_length),
                                        scene_datas[scene_name]["seq_ini"]["imWidth"],
                                        scene_datas[scene_name]["seq_ini"]["imHeight"]
                                        )
            )

        scene_gt_txt_path = f"{scene_gt_path}/gt/gt.txt"

        # Create MTMC gt files
        convert_dataframe_to_txt(scene_datas[scene_name]["gt_mtmc_df"], scene_gt_txt_path, "gt")    
        convert_dataframe_to_txt(scene_datas[scene_name]["pred_mtmc_df"], f"{pred_files}/{scene_name}.txt", "pred")    

"""
    Initialization
"""

def parse_opt():
    parser = argparse.ArgumentParser(description='Specify the path to the dataset and prediction results.')
    parser.add_argument('--mode', type=str, help='Evaluate MOT or MTMC.', default="mtmc")
    parser.add_argument('--data_path', type=str, help='Dataset path.', required=True)
    parser.add_argument('--pred_path', type=str, help='Prediction results path.', required=True)
    parser.add_argument('--metric_list', nargs='+', help='List of metrics to evaluate', required=True)
    parser.add_argument('--vtx_offset', type=str, help="Whether to subtract 2 frames from the count due to broken legacy code", default="no")
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':

    """
        Please specify the metrics to print before proceeding.
        You may only choose from these:

        "HOTA", "DetA", "AssA", "LocA", "MOTA", "MOTP", "CLR_FP", "CLR_FN", 
        "IDSW", "Frag", "IDF1", "IDFP", "IDFN", "Dets", "GT_Dets", "IDs", "GT_IDs"

    """
    # Prepare data
    opt = parse_opt()

    to_print = [metric for metric in opt.metric_list]
    assert set(to_print).issubset(set(AVAILABLE_METRICS)), "There is an unavailable metric!"
    
    if not (opt.mode == "mot" or opt.mode == "mtmc"):
        raise Exception("Invalid mode!")

    main(opt)
    res, msg = evaluate_mtmc()

    # Construct a final summary table

    result_summary = {}
    scenes = res["MotChallenge2DBox"]["VTX_MTMC"]
    
    for scene_name, scene_scores in scenes.items():
        
        result_summary[scene_name] = {
            "HOTA": {
                "HOTA": None,
                "DetA": None,
                "AssA": None,
                "LocA": None
            },
            "CLEAR": {
                "MOTA": None,
                "MOTP": None,
                "CLR_FP": None,
                "CLR_FN": None,
                "IDSW": None,
                "Frag": None
            },
            "IDF1": {
                "IDF1": None,
                "IDFP": None,
                "IDFN": None
            },
            "Count": {
                "Dets": None,
                "GT_Dets": None,
                "IDs": None,
                "GT_IDs": None
            }          
        }

        # HOTA
        result_summary[scene_name]["HOTA"]["HOTA"] = f"{float(np.mean(scene_scores['pedestrian']['HOTA']['HOTA']))*100:.3f}"
        result_summary[scene_name]["HOTA"]["DetA"] = f"{float(np.mean(scene_scores['pedestrian']['HOTA']['DetA']))*100:.3f}"
        result_summary[scene_name]["HOTA"]["AssA"] = f"{float(np.mean(scene_scores['pedestrian']['HOTA']['AssA']))*100:.3f}"
        result_summary[scene_name]["HOTA"]["LocA"] = f"{float(np.mean(scene_scores['pedestrian']['HOTA']['LocA']))*100:.3f}"

        # CLEAR
        result_summary[scene_name]["CLEAR"]["MOTA"] = f"{scene_scores['pedestrian']['CLEAR']['MOTA']*100:.3f}"
        result_summary[scene_name]["CLEAR"]["MOTP"] = f"{scene_scores['pedestrian']['CLEAR']['MOTP']*100:.3f}"
        result_summary[scene_name]["CLEAR"]["CLR_FP"] = scene_scores['pedestrian']['CLEAR']['CLR_FP']
        result_summary[scene_name]["CLEAR"]["CLR_FN"] = scene_scores['pedestrian']['CLEAR']['CLR_FN']
        result_summary[scene_name]["CLEAR"]["IDSW"] = int(scene_scores['pedestrian']['CLEAR']['IDSW'])
        result_summary[scene_name]["CLEAR"]["Frag"] = int(scene_scores['pedestrian']['CLEAR']['Frag'])

        # IDF1
        result_summary[scene_name]["IDF1"]["IDF1"] = f"{scene_scores['pedestrian']['Identity']['IDF1']*100:.3f}"
        result_summary[scene_name]["IDF1"]["IDFP"] = int(scene_scores['pedestrian']['Identity']['IDFP'])
        result_summary[scene_name]["IDF1"]["IDFN"] = int(scene_scores['pedestrian']['Identity']['IDFN'])

        # Count
        result_summary[scene_name]["Count"]["Dets"] = int(scene_scores['pedestrian']['Count']['Dets'])
        result_summary[scene_name]["Count"]["GT_Dets"] = int(scene_scores['pedestrian']['Count']['GT_Dets'])
        result_summary[scene_name]["Count"]["IDs"] = int(scene_scores['pedestrian']['Count']['IDs'])
        result_summary[scene_name]["Count"]["GT_IDs"] = int(scene_scores['pedestrian']['Count']['GT_IDs'])

    # Moves the result to logs folder and outputs it

    current_time = datetime.now()
    log_dir = f"./logs/{current_time:%Y%m%d_%H%M%S}_{str(opt.mode).upper()}_{str(opt.data_path).split('/')[-1]}"
    os.makedirs(log_dir, exist_ok=True)

    shutil.move("./src/data/", log_dir)
    print_and_save_metrics_table(result_summary, to_print, f"{log_dir}/results.txt")
