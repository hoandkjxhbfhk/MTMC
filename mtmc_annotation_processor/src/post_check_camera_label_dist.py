import argparse
import os
import yaml
import colorama
import json
from colorama import Fore, Back, Style


def get_bounding_box_area_in_view(bb_left, bb_top, bb_width, bb_height, view_x, view_y, view_w, view_h):
    """Calculate the intersection area of the bounding box with the view (camera)."""
    x_overlap = max(0, min(bb_left + bb_width, view_x + view_w) - max(bb_left, view_x))
    y_overlap = max(0, min(bb_top + bb_height, view_y + view_h) - max(bb_top, view_y))
    return x_overlap * y_overlap


def check_track_fragmentations(nums):
    nums = list(map(int, nums))

    non_consecutive_count = 0
    duplicate_count = 0
    
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1]:  # Duplicate case
            duplicate_count += 1
            print(Back.LIGHTYELLOW_EX + f"Duplicate ID found at frame: {nums[i] - 1}" + Style.RESET_ALL)
        # elif nums[i] != nums[i-1] + 1:  # Non-consecutive case
        #     non_consecutive_count += 1
        #     print(f"Fragmentation found at frame {nums[i-1] - 1} jumping to {nums[i] - 1}!")
        #     if nums[i] < nums[i-1] + 5:
        #         print(Back.LIGHTRED_EX + "Small fragmentation!" + Style.RESET_ALL)
    
    return non_consecutive_count, duplicate_count


def analyze_grid_view_annotations(merged_gt_file, grid_positions):
    grid_width, grid_height = 960, 540  # Size of each scaled video
    view_class_dict = {i: {} for i in range(1, len(grid_positions) + 1)}  # Dictionary to store obj_class for each camera view

    with open(merged_gt_file, 'r') as f:
        for line in f:
            # Parse the line in MOT format
            frame, obj_id, bb_left, bb_top, bb_width, bb_height, conf, obj_class, visibility = map(float, line.split(','))

            # Find the view where the bounding box occupies the most area
            max_area = 0
            best_view_idx = None
            
            for idx, (grid_x_offset, grid_y_offset) in enumerate(grid_positions):
                # Calculate the area of the bounding box in this view
                area = get_bounding_box_area_in_view(
                    bb_left, bb_top, bb_width, bb_height,
                    grid_x_offset, grid_y_offset, grid_width, grid_height
                )
                
                # Track the view with the largest area of overlap
                if area > max_area:
                    max_area = area
                    best_view_idx = idx

            # If a view is found, add the obj_class to that view
            if best_view_idx is not None:
                # view_class_dict[best_view_idx + 1].add(int(obj_class))
                current_id = str(int(obj_class))
                if current_id not in view_class_dict[best_view_idx + 1].keys():
                    view_class_dict[best_view_idx + 1][current_id] = []
                view_class_dict[best_view_idx + 1][current_id].append(str(int(frame)))

    return view_class_dict


def parse_opt():
    parser = argparse.ArgumentParser(description="MTMC Grid-view Analyzer")
    parser.add_argument('--scene', type=str, default='')
    parser.add_argument('--config_file', type=str, default='config/config.yml')
    opt = parser.parse_args()
    return opt


def get_unique_values(dictionary):
    # Flatten all values to a list with keys to count occurrences
    all_values = {}
    for key, values in dictionary.items():
        for value in values:
            if value in all_values:
                all_values[value].append(key)
            else:
                all_values[value] = [key]
                
    # Build the dictionary with unique values
    unique_dict = {key: [] for key in dictionary}
    for value, keys in all_values.items():
        if len(keys) == 1:  # Value appears only in one key
            unique_dict[keys[0]].append(value)
    
    return unique_dict


def main(opt):
    grid_positions = [
        (0, 0), (960, 0), (1920, 0),  # Row 1 (video 1, 2, 3)
        (0, 540), (960, 540), (1920, 540),  # Row 2 (video 4, 5, 6)
        (0, 1080)  # Row 3 (video 7)
    ]

    scene_wise_frags = 0
    scene_wise_doops = 0

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

    merged_gt_file = f"VTX_MTMC_2024/F{floor_id}_{scene_id}/grid_gt.txt"
    view_class_dict = analyze_grid_view_annotations(merged_gt_file, grid_positions)

    for view in view_class_dict.keys():
        view_wise_frags = 0
        view_wise_doops = 0

        print(Fore.MAGENTA + f"Analyzing view {view}, camera {layout[view - 1]} has {sorted(list(map(int,list(set(view_class_dict[view].keys())))))}:" + Style.RESET_ALL)
        for label in view_class_dict[view].keys():
            print(Fore.BLUE + f"ID number {label} appears from {int(view_class_dict[view][label][0]) - 1} to {int(view_class_dict[view][label][-1]) - 1}" + Style.RESET_ALL)
            no_frags, no_doops = check_track_fragmentations(view_class_dict[view][label])
    #         if no_frags or no_doops:
    #             print(Fore.RED + f"Found {no_frags} fragmentations and {no_doops} duplicated labels!" + Style.RESET_ALL)

    #         view_wise_frags += no_frags
    #         view_wise_doops += no_doops

    #     if view_wise_doops or view_wise_frags:
    #         print(Fore.LIGHTRED_EX + f"There are {view_wise_frags} track fragmentations and {view_wise_doops} duplicated labels!" + Style.RESET_ALL)
    #     else:
    #         print(Fore.GREEN + f"View annotated properly!" + Style.RESET_ALL)

    #     scene_wise_frags += view_wise_frags
    #     scene_wise_doops += view_wise_doops

    # if scene_wise_frags or scene_wise_doops:
    #     print(Fore.RED + f"There are {scene_wise_frags} track fragmentations and {scene_wise_doops} duplicated labels throughout the scene!" + Style.RESET_ALL)
    # else:
    #     print(Fore.GREEN + f"Scene annotated properly!" + Style.RESET_ALL)

    for view in view_class_dict:
        id_list = [x for x in view_class_dict[view].keys()]
        view_class_dict[view] = id_list

    print(get_unique_values(view_class_dict))
        

if __name__ == '__main__':
    opt = parse_opt()
    main(opt)



# Example usage
# merged_gt_file = 'data/F09_03/annotations/gt/gt.txt'
# view_class_dict = analyze_grid_view_annotations(merged_gt_file)

# # Output the result
# for view, obj_classes in view_class_dict.items():
#     print(f"View {view}: {obj_classes.keys()}")


