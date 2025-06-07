import os
import argparse
import yaml


def scale_and_merge_gt_files(gt_files, output_file, grid_positions):
    scale_factor = 0.25  
    grid_width, grid_height = 960, 540  
    output_annotations = []

    current_max_label = 0

    for idx, gt_file in enumerate(gt_files):
        grid_x_offset, grid_y_offset = grid_positions[idx]
        
        with open(gt_file, 'r') as f:
            for line in f:
                # Parse the line in MOT format
                frame, obj_id, bb_left, bb_top, bb_width, bb_height, conf, obj_class, visibility = map(float, line.split(','))
                
                # Scale the bounding box coordinates
                new_bb_left = bb_left * scale_factor + grid_x_offset
                new_bb_top = bb_top * scale_factor + grid_y_offset
                new_bb_width = bb_width * scale_factor
                new_bb_height = bb_height * scale_factor
                
                # Create the updated line with scaled coordinates
                new_line = f"{int(frame)}, {int(obj_id)}, {new_bb_left:.2f}, {new_bb_top:.2f}, {new_bb_width:.2f}, {new_bb_height:.2f}, {conf}, {int(obj_class)}, {visibility:.2f}\n"
                output_annotations.append(new_line)

    output_annotations = sorted(output_annotations, key=lambda x: int(x.split(',')[0].strip()))

    # Write the output to a new file
    with open(output_file, 'w') as f_out:
        f_out.writelines(output_annotations)


def parse_opt():
    parser = argparse.ArgumentParser(description="MTMC Grid-view Decomposer")
    parser.add_argument('--scene', type=str, default='')
    parser.add_argument('--config_file', type=str, default='config/config.yml')
    opt = parser.parse_args()
    return opt


def main(opt):

    grid_positions = [
        (0, 0), (960, 0), (1920, 0),  # Row 1 (video 1, 2, 3)
        (0, 540), (960, 540), (1920, 540),  # Row 2 (video 4, 5, 6)
        (0, 1080)  # Row 3 (video 7)
    ]

    floor_id = str(opt.scene)[1:3]
    scene_id = str(opt.scene)[4:]

    with open(opt.config_file) as config_file:
        config = yaml.safe_load(config_file)

    if floor_id == "11" and scene_id in set(["02", "08"]):
        layout = config['CAMERA_GRID_LAYOUT']["FLOOR_11_6CAM"]
        grid_positions = [
            (0, 0), (960, 0), (1920, 0),  # Row 1 (video 1, 2, 3)
            (0, 540), (960, 540), (1920, 540)  # Row 2 (video 4, 5, 6)
        ]
    else:
        layout = config['CAMERA_GRID_LAYOUT']["FLOOR_" + floor_id]


    input_files = [f"data/F{floor_id}_{scene_id}/{cam}/{cam}.txt" for cam in layout]
    output_file = f"data/F{floor_id}_{scene_id}/grid_gt.txt"

    scale_and_merge_gt_files(input_files, output_file, grid_positions)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
