import os
import argparse
from collections import defaultdict
from tqdm import tqdm

def read_scene_predictions(pred_file):
    """
    Read MTMC prediction file and group by camera.
    Format: <camera_id> <obj_id> <frame_id> <xmin> <ymin> <width> <height> <xworld> <yworld>
    """
    per_camera_preds = defaultdict(list)
    with open(pred_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 9:
                continue
            cam_id, obj_id, frame_id, xmin, ymin, width, height, xw, yw = parts
            cam_key = f"camera_{int(cam_id):04d}"  # zero-padded camera ID
            mot_line = [
                int(frame_id),
                int(obj_id),
                float(xmin),
                float(ymin),
                float(width),
                float(height),
                -1,    # conf
                -1, -1, -1  # optional x,y,z not available here
            ]
            per_camera_preds[cam_key].append(mot_line)
    return per_camera_preds



def write_mot16_predictions(output_path, records):
    """
    Write MOT16-style predictions to a file.
    """
    with open(output_path, "w") as f:
        for r in sorted(records, key=lambda x: (x[0], x[1])):  # sort by frame, then ID
            f.write(",".join(map(str, r)) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--predictions", required=True, help="Path to folder with scene prediction .txt files")
    parser.add_argument("-d", "--dataset", required=True, help="Path to the previously converted dataset folder")
    args = parser.parse_args()

    prediction_folder = os.path.abspath(args.predictions)
    dataset_folder = os.path.abspath(args.dataset)

    scene_files = sorted([f for f in os.listdir(prediction_folder) if f.endswith(".txt")])

    for scene_file in tqdm(scene_files, desc="Scenes"):
        scene_name = os.path.splitext(scene_file)[0]  # scene_061
        pred_path = os.path.join(prediction_folder, scene_file)

        print(f"Processing scene: {scene_name}")

        per_camera_preds = read_scene_predictions(pred_path)
        if not per_camera_preds:
            continue

        scene_dataset_path = os.path.join(dataset_folder, scene_name)
        if not os.path.isdir(scene_dataset_path):
            print(f"Warning: Scene folder {scene_dataset_path} not found in dataset")
            continue

        for cam_id, mot_records in tqdm(per_camera_preds.items(), desc=f"  {scene_name}", leave=False):
            cam_folder_path = os.path.join(scene_dataset_path, cam_id)
            if not os.path.isdir(cam_folder_path):
                print(f"  Skipping missing camera folder: {cam_folder_path}")
                continue

            mtmc_folder = os.path.join(cam_folder_path, "mtmc")
            os.makedirs(mtmc_folder, exist_ok=True)

            output_pred_file = os.path.join(mtmc_folder, f"{cam_id}.txt")
            write_mot16_predictions(output_pred_file, mot_records)

    print("\nPrediction conversion complete.")

if __name__ == "__main__":
    main()
