#!/usr/bin/env python3
"""
convert_to_mot16.py

Given a multi‐camera tracking dataset in which each scene folder contains:
  - calibration_2025_format.json
  - ground_truth.txt        (scene‐wide, format below)
  - ground_truth_2025_format.json
  - camera_<CAM_ID>/        (one folder per camera, each with calibration.json + video.mp4)

And where each line in ground_truth.txt is:
    <camera_id> <obj_id> <frame_id> <xmin> <ymin> <width> <height> <xworld> <world>

This script splits the scene‐level ground_truth.txt into per‐camera text files in MOT16 format:
    <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>

and then copies/renames each camera’s video.mp4 to camera_<CAM_ID>.mp4 under the new output dataset.

Usage:
    python convert_to_mot16.py --input /path/to/test --output /path/to/new_dataset
"""
import os
import argparse
import shutil
from collections import defaultdict
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Convert a multi‐camera dataset’s scene‐level GT into per‐camera MOT16 files.")
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to the root input folder (e.g. /path/to/test)."
    )
    parser.add_argument(
        "--output", "-o",
        required=True,
        help="Path to the root output folder (e.g. /path/to/new_dataset)."
    )
    return parser.parse_args()

def read_scene_ground_truth(gt_path):
    """
    Reads a ground_truth.txt file at scene level.
    Returns a dict:
       camera_id_str → list of lines, where each line is a tuple:
         (camera_id, obj_id, frame_id, xmin, ymin, w, h, xworld, yworld)
    """
    per_cam = defaultdict(list)

    count = 0
    print(f"Reading ground truth from {gt_path}")

    with open(gt_path, "r") as f:
        for line in f:

            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            cam_id, obj_id, frame_id = parts[0], parts[1], parts[2]
            xmin, ymin, w, h = parts[3], parts[4], parts[5], parts[6]
            xworld, yworld = parts[7], parts[8]
            # keep as strings or convert to int/float
            per_cam[cam_id].append((int(frame_id),
                                    int(obj_id),
                                    float(xmin),
                                    float(ymin),
                                    float(w),
                                    float(h),
                                    float(xworld),
                                    float(yworld)))
    
    return per_cam

def write_mot16_for_camera(out_txt_path, records):
    """
    Given a list of records for one camera:
      records: list of tuples (frame_id, obj_id, xmin, ymin, w, h, xworld, yworld)
    write them to out_txt_path in MOT16 format:
      <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<x>,<y>,<z>
    We choose conf=1.0, and z=-1.0.
    """
    with open(out_txt_path, "w") as fout:
        # sort by frame_id, then by obj_id (just for consistency)
        records_sorted = sorted(records, key=lambda x: (x[0], x[1]))
        for rec in records_sorted:
            frame_id, obj_id, xmin, ymin, w, h, xworld, yworld = rec
            conf = 1.0
            z = -1.0
            line = f"{frame_id},{obj_id},{xmin:.2f},{ymin:.2f},{w:.2f},{h:.2f},{conf:.2f},{xworld:.2f},{yworld:.2f},{z:.2f}\n"
            fout.write(line)

from tqdm import tqdm
import os
import shutil
from collections import defaultdict
import argparse

# … (reuse read_scene_ground_truth and write_mot16_for_camera from before) …

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="Path to the root input folder")
    parser.add_argument("--output", "-o", required=True, help="Path to the root output folder")
    args = parser.parse_args()

    input_root = os.path.abspath(args.input)
    output_root = os.path.abspath(args.output)
    os.makedirs(output_root, exist_ok=True)

    scene_names = sorted([d for d in os.listdir(input_root) if os.path.isdir(os.path.join(input_root, d))])
    for scene_name in tqdm(scene_names, desc="Scenes"):
        scene_path = os.path.join(input_root, scene_name)
        gt_path = os.path.join(scene_path, "ground_truth.txt")
        if not os.path.isfile(gt_path):
            continue

        per_cam_records = read_scene_ground_truth(gt_path)
        if not per_cam_records:
            continue

        # print(f"Processing scene: {scene_name} with {len(per_cam_records)} cameras")

        out_scene_dir = os.path.join(output_root, scene_name)
        os.makedirs(out_scene_dir, exist_ok=True)

        cam_folders = sorted([d for d in os.listdir(scene_path) if d.startswith("camera_")])
        for cam_folder in tqdm(cam_folders, desc=f"  {scene_name}", leave=False):
            cam_id = str(int(cam_folder.split("camera_")[-1]))
            if cam_id not in per_cam_records:
                continue


            cam_path = os.path.join(scene_path, cam_folder)
            out_cam_dir = os.path.join(out_scene_dir, cam_folder)
            os.makedirs(out_cam_dir, exist_ok=True)

            # Copy video
            src_video = os.path.join(cam_path, "video.mp4")
            if os.path.isfile(src_video):
                dst_video = os.path.join(out_cam_dir, f"{cam_folder}.mp4")
                shutil.copy2(src_video, dst_video)

            # Write MOT16 TXT
            out_txt = os.path.join(out_cam_dir, f"{cam_folder}.txt")
            write_mot16_for_camera(out_txt, per_cam_records[cam_id])

    print("All scenes processed. Output at:", output_root)

if __name__ == "__main__":
    main()
