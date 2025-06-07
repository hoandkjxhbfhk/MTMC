import os
import argparse
import yaml


def get_bounding_box_area_in_view(bb_left, bb_top, bb_width, bb_height, view_x, view_y, view_w, view_h):
    """Calculate the intersection area of the bounding box with the view (camera)."""
    x_overlap = max(0, min(bb_left + bb_width, view_x + view_w) - max(bb_left, view_x))
    y_overlap = max(0, min(bb_top + bb_height, view_y + view_h) - max(bb_top, view_y))
    return x_overlap * y_overlap

def confine_to_single_view(bb_left, bb_top, bb_width, bb_height, view_x, view_y, view_w, view_h):
    """Adjust bounding box coordinates to be confined within a single view (camera)."""
    new_left = max(bb_left, view_x)
    new_top = max(bb_top, view_y)
    new_right = min(bb_left + bb_width, view_x + view_w)
    new_bottom = min(bb_top + bb_height, view_y + view_h)
    
    return new_left, new_top, new_right - new_left, new_bottom - new_top

def split_and_scale_gt(merged_gt_file, output_files, grid_positions):
    scale_factor = 4  # Reverse the 25% scaling (1 / 0.25)
    grid_width, grid_height = 960, 540  # Size of each scaled video

    label_mapper = {}
    current_max_index = 1

    # Create a dictionary to hold the annotations for each video
    split_annotations = {i: [] for i in range(1, 8)}

    with open(merged_gt_file, 'r') as f:
        for line in f:
            # Parse the line in MOT format
            frame, obj_id, bb_left, bb_top, bb_width, bb_height, conf, obj_class, visibility = map(float, line.split(','))

            # Check which views (cameras) the bounding box overlaps
            max_area = 0
            best_view_idx = None
            confined_bb = None

            obj_class = int(obj_class)
            
            assessed_index = obj_class
            if assessed_index not in label_mapper.keys():
                label_mapper[assessed_index] = current_max_index
                current_max_index += 1

            obj_class = label_mapper[assessed_index]
            obj_id = obj_class
            
            for idx, (grid_x_offset, grid_y_offset) in enumerate(grid_positions):
                # Calculate the area of the bounding box in this view
                area = get_bounding_box_area_in_view(
                    bb_left, bb_top, bb_width, bb_height,
                    grid_x_offset, grid_y_offset, grid_width, grid_height
                )

                if area > max_area:
                    max_area = area
                    best_view_idx = idx
                    confined_bb = confine_to_single_view(
                        bb_left, bb_top, bb_width, bb_height,
                        grid_x_offset, grid_y_offset, grid_width, grid_height
                    )
            
            # If no suitable view was found, skip this bounding box
            if best_view_idx is None or confined_bb is None:
                continue
            
            # Extract confined bounding box and assign to the best view
            confined_bb_left, confined_bb_top, confined_bb_width, confined_bb_height = confined_bb

            # Scale the bounding box back to original resolution
            original_bb_left = (confined_bb_left - grid_positions[best_view_idx][0]) * scale_factor
            original_bb_top = (confined_bb_top - grid_positions[best_view_idx][1]) * scale_factor
            original_bb_width = confined_bb_width * scale_factor
            original_bb_height = confined_bb_height * scale_factor

            # Create the updated line with original coordinates
            new_line = f"{int(frame)}, {int(obj_id)}, {original_bb_left:.2f}, {original_bb_top:.2f}, {original_bb_width:.2f}, {original_bb_height:.2f}, {conf}, {int(obj_class)}, {visibility:.2f}\n"
            split_annotations[best_view_idx + 1].append(new_line)

    # Write the split and scaled annotations to their respective files
    for idx, output_file in enumerate(output_files, start=1):
        print(f"Written to {output_file}")
        with open(output_file, 'w+') as f_out:
            f_out.writelines(split_annotations[idx])


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
    elif floor_id == "11" and scene_id == "09":
        layout = config['CAMERA_GRID_LAYOUT']["FLOOR_11_SCENE9"]
    else:
        layout = config['CAMERA_GRID_LAYOUT']["FLOOR_" + floor_id]

    merged_gt_file = f"data/F{floor_id}_{scene_id}/annotations/gt/gt.txt"
    
    output_files = []
    for camera in layout:
        output_file = f"data/F{floor_id}_{scene_id}/annotations/{camera}.txt"
        output_files.append(output_file)

    split_and_scale_gt(merged_gt_file, output_files, grid_positions)

    

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)


# # Example usage
# merged_gt_file = 'merged_gt.txt'
# output_files = [
#     'video1/gt.txt', 'video2/gt.txt', 'video3/gt.txt',
#     'video4/gt.txt', 'video5/gt.txt', 'video6/gt.txt',
#     'video7/gt.txt'
# ]
# split_and_scale_gt(merged_gt_file, output_files)
