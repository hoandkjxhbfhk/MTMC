import os
from os import path
import argparse
import numpy as np
from moviepy.editor import *
import yaml
import colorama
from colorama import Fore, Back, Style


# def parse_opt():
#     parser = argparse.ArgumentParser(description='Stack video')
#     parser.add_argument('--data_dir', type=str, default='', help='data directory')
#     parser.add_argument('--cam_list', nargs='+', help='Camera list')
#     parser.add_argument('--rs_ratio', type=float, default='', help='resize ratio')
#     opt = parser.parse_args()
#     return opt

def parse_opt():
    parser = argparse.ArgumentParser(description="MTMC Grid-view Decomposer")
    parser.add_argument('--data_dir', type=str, default='')
    parser.add_argument('--config_file', type=str, default='config/config.yml')
    opt = parser.parse_args()
    return opt

def main(opt):
    # Get video list path
    all_vids = os.listdir(opt.data_dir)

    for scene in sorted(all_vids)[37:38]:

        floor_id = str(scene)[1:3]
        scene_id = str(scene)[4:]
        vid_list = []
        scale_factor = 0.25

        with open(opt.config_file) as config_file:
            config = yaml.safe_load(config_file)

        if floor_id == "11" and scene_id in set(["02", "08"]):
            layout = config['CAMERA_GRID_LAYOUT']["FLOOR_11_6CAM"]
            grid_positions = [
                (0, 0), (960, 0), (1920, 0),  # Row 1 (video 1, 2, 3)
                (0, 540), (960, 540), (1920, 540)  # Row 2 (video 4, 5, 6)
            ]
        # elif floor_id == "11" and scene_id == "09":
        #     layout = config['CAMERA_GRID_LAYOUT']["FLOOR_11_SCENE9"]
        else:
            layout = config['CAMERA_GRID_LAYOUT']["FLOOR_" + floor_id]


        for cam_id in layout:
            vid_path = f"{opt.data_dir}/{scene}/{cam_id}/{cam_id}.mp4"
            vid_list.append(vid_path)
    
        vids = [VideoFileClip(v) for v in vid_list]
        rs_vid = [vid.resize(scale_factor) for vid in vids]
        size = rs_vid[0].size
        duration = rs_vid[0].duration
        blank_vid = ColorClip(size=size, duration=duration, color=(0, 0, 0))
        if len(rs_vid) == 1:
            raise Exception(f'number of videos must greater than 1')
        else:
            if len(rs_vid) == 2:
                stack_vid = [[rs_vid[0], rs_vid[1]]]
            elif len(rs_vid) == 3:
                stack_vid = [[rs_vid[0], rs_vid[1]],
                            [rs_vid[2], rs_vid]]
            elif len(rs_vid) == 4:
                stack_vid = [[rs_vid[0], rs_vid[1]],
                            [rs_vid[2], rs_vid[3]]]
            elif len(rs_vid) == 5:
                stack_vid = [[rs_vid[0], rs_vid[1], rs_vid[2]],
                            [rs_vid[3], rs_vid[4], blank_vid]]
            elif len(rs_vid) == 6:
                stack_vid = [[rs_vid[0], rs_vid[1], rs_vid[2]],
                            [rs_vid[3], rs_vid[4], rs_vid[5]]]
            elif len(rs_vid) == 7:
                stack_vid = [[rs_vid[0], rs_vid[1], rs_vid[2]],
                            [rs_vid[3], rs_vid[4], rs_vid[5]],
                            [rs_vid[6], blank_vid, blank_vid]]
            elif len(rs_vid) == 8:
                stack_vid = [[rs_vid[0], rs_vid[1], rs_vid[2]],
                            [rs_vid[3], rs_vid[4], rs_vid[5]],
                            [rs_vid[6], rs_vid[7], blank_vid]]
            elif len(rs_vid) == 9:
                stack_vid = [[rs_vid[0], rs_vid[1], rs_vid[2]],
                            [rs_vid[3], rs_vid[4], rs_vid[5]],
                            [rs_vid[6], rs_vid[7], rs_vid[8]]]
                
        print(Fore.RED + f"Processing scene {scene} with {len(rs_vid)} cameras!" + Style.RESET_ALL)

        # final_vid = clips_array(stack_vid)
        final_vid = clips_array(stack_vid)
        print(final_vid.size)

        # save_path = path.join(opt.data_dir, 'grid_view_' + str(len(rs_vid)) + '.mp4')
        save_path = f"{opt.data_dir}/{scene}/grid_view.mp4"
        final_vid.write_videofile(save_path)


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
