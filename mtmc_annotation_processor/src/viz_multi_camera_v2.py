import os
import cv2
import time
from collections import defaultdict
from typing import Dict, List, Tuple
import json
import argparse
import numpy as np
import pandas as pd
import math
import yaml

color_map = json.load(open("config/color_map.json", mode="r"))
color_map = {int(k): v for k, v in color_map.items()}

def hex_to_bgr(hex_color: str):
    """
        Input: 
            - hex_color (str): RGB hex code of a color (e.g: "#ffffff")
        Output: 
            - tuple: A BGR tuple of that color (e.g: (255, 255, 255))
    """
    hex_color = hex_color.lstrip('#') 
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

    return bgr

def get_text_color(r, g, b):
    r /= 255.0
    g /= 255.0
    b /= 255.0
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

def camera_to_ground_coordinates(homography_matrix: np.array, camera_points: np.array):
    """
        Input:
            - homography_matrix (np.array): A 3x3 homography matrix that maps a point from the original (image) plane to the groud (map) plane
            - camera_points (np.array): A (2,) vector with the coordinates of the image point.
        Output:
            - x, y: The x and y coordinates of the mapped coordinates.
    """
    if homography_matrix is None:
        raise ValueError("Homography matrix has not been calculated or loaded.")

    camera_point_homogeneous = np.append(camera_points, 1) # Add a 1 to the end of the camera point to create a homogeneous coordinate
    topo_point_homogeneous = np.dot(homography_matrix, camera_point_homogeneous) # Transform the point using the homography matrix
    topo_point = (topo_point_homogeneous / topo_point_homogeneous[2])[:2].astype(int) # Divide by the last element to get the final (x, y) coordinates

    return topo_point[0], topo_point[1]

"""
    Annotations processing.
"""

def load_annotations(txt_file, mode):
    if mode == 'gt':
        cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'confidence', 'class', 'visibility']
    else:  # pred
        cols = ['frame', 'id', 'x', 'y', 'w', 'h', 'confidence', 'x_vel', 'y_vel', 'z_vel']
    
    annotations = pd.read_csv(txt_file, header=None, names=cols)
    return annotations[['frame', 'id', 'x', 'y', 'w', 'h']]

"""
    Visualization processing.
"""

def get_camera_grid_order(arr):
    n = len(arr)
    rows = math.ceil(math.sqrt(n)) 
    cols = math.ceil(n / rows) 
    
    matrix = []
    index = 0
    for r in range(rows):
        row = []
        for c in range(cols):
            if index < n:
                row.append(arr[index])
                index += 1
        matrix.append(row)
    
    return matrix

def find_value_index(matrix, value):
    for r, row in enumerate(matrix):
        for c, element in enumerate(row):
            if element == value:
                return (r, c)
    return None

def display_frame(current_frame, scene_id, scene_data: dict, show_topology, topology_map=255*np.ones((1000, 1000, 3), dtype=np.uint8)):
    
    grid_order = get_camera_grid_order(list(scene_data.keys()))
    no_rows, no_cols = len(grid_order), len(grid_order[0])
    cap_width = 1600
    cap_height = 900

    grid = np.zeros((cap_height * no_rows, cap_width * no_cols, 3), dtype=np.uint8)

    for cam_id, cam_val in scene_data.items():
        cap = cam_val["video_cap"]
        box_data = cam_val["box_data"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        font_size = 4
        thickness = 4

        if ret:

            frame_annotations = box_data[box_data['frame'] == current_frame + 1]
            for _, row in frame_annotations.iterrows():
                x1, y1, x2, y2 = int(row['x']), int(row['y']), int(row['x']) + int(row['w']), int(row['y']) + int(row['h'])
                start_point = (x1, y1)
                end_point = (x2, y2)
                if int(row['id']) > 10:
                    rect_point = (x1 + 70, y1 + 50)
                else:
                    rect_point = (x1 + 45, y1 + 50)
                color = hex_to_bgr(color_map[int(row["id"])])
                frame = cv2.rectangle(frame, start_point, end_point, color, 8)
                frame = cv2.rectangle(frame, start_point, rect_point, color, -1)
                frame = cv2.putText(frame, str(int(row["id"])), (x1, y1 + 50), cv2.FONT_HERSHEY_PLAIN, font_size, get_text_color(*color), thickness)

                if show_topology:
                    camera_point = np.array([(x1 + x2) // 2, y2], dtype='float32')
                    ground_point = camera_to_ground_coordinates(cam_val["H_matrix"], camera_point)

                    topology_map = cv2.circle(topology_map, ground_point, 7, color, -1)
                    topology_map = cv2.circle(topology_map, ground_point, 25, color, 2)
                    topology_map = cv2.putText(topology_map, cam_id[-2:], (ground_point[0], ground_point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 1.25, color, 2)

            frame = cv2.putText(frame, f'{current_frame}', (100, 60), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 8)
            frame = cv2.resize(frame, (cap_width, cap_height))

            grid_row, grid_col = find_value_index(grid_order, cam_id)
            grid_x = grid_col * cap_width
            grid_y = grid_row * cap_height

            grid[grid_y:grid_y + cap_height, grid_x:grid_x + cap_width] = frame
    
    # grid = cv2.resize(grid, (960, 1080))
    grid_scale_coeff = 2
    grid = cv2.resize(grid, (int(no_cols * 400 * grid_scale_coeff), int(no_rows * 225 * grid_scale_coeff)))
    # grid = cv2.resize(grid, (1600, 900))
    cv2.imshow(f'MTMC Visualizer - {scene_id}', grid)
    # map_writer.write(grid)
    
    if show_topology:
        topology_map = cv2.resize(topology_map, (1000, 1000))
        cv2.imshow("Topology Map", topology_map)
        # map_writer.write(topology_map)

"""
    Main function stuff.
"""

def parse_opt():
    parser = argparse.ArgumentParser(description='Specify MTMC Scene')
    parser.add_argument('--data_dir', type=str, help='Data Path', default="VTX_MTMC_2024")
    parser.add_argument('--scene', type=str, help='Scene ID')

    parser.add_argument('--show_topology', type=str, help='Show topology', default="Yes")
    parser.add_argument('--topology_map', type=str, help='Path to topology map')
    parser.add_argument('--homography_matrices', type=str, help='Path to homography matrix config file')

    parser.add_argument('')

    parser.add_argument('--camera_order', nargs='+', help='Specify camera order', default=None)
    parser.add_argument("--mode", type=str, choices=['gt', 'pred'], required=True, help="Mode: 'gt' for ground truth, 'pred' for predictions.")
    opt = parser.parse_args()
    return opt

def main(opt):
    scene = opt.scene
    show_topology = True if opt.show_topology == "Yes" else False
    data_root = os.path.join(opt.data_dir, scene)

    camera_views = [cam_id for cam_id in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, cam_id))]
    camera_order = opt.camera_order

    if camera_order is not None:
        order_list = [cam_id for cam_id in opt.camera_order]
        assert set(order_list).issubset(set(camera_views)), f"Camera order does not align with available camera views! {order_list} != {camera_views}"
    else:
        order_list = camera_views

    if show_topology:
        base_map_img = cv2.imread(opt.topology_map)
        homography_json = json.load(open(opt.homography_matrices))
        homography_data = defaultdict(list)

        for cluster, cameras in homography_json["ground_plane"].items():
            for camera_id, homo_data in cameras.items():
                # The homography matrices shall be referred to by camera ID
                homography_data[camera_id] = np.asarray(homo_data["H"])
        
        # global map_writer
        # map_writer = cv2.VideoWriter('map_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (960, 1080))

    """
        Scene data format:
        scene_data = {
            "111101": {
                "video_cap": 111101.mp4,
                "box_data": pd.DataFrame,
                "H_matrix": np.array
            },
            "111103": {
                ...
            },
            ...
        }
    """

    max_length = 0
    scene_data = {}

    for camera in order_list:
        if opt.mode == "gt":
            txt_file = f"{data_root}/{camera}/{camera}.txt"
        elif opt.mode == "pred":
            txt_file = f"{data_root}/{camera}/mtmc/{camera}.txt"

        video_path = f"{data_root}/{camera}/{camera}.mp4"
        box_data = load_annotations(txt_file, opt.mode)

        cap = cv2.VideoCapture(video_path)
        max_length = max(max_length, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        scene_data[camera] = {
            "video_cap": cap,
            "box_data": box_data
        }

        if show_topology:
            scene_data[camera]["H_matrix"] = homography_data[camera]

    current_frame = 0
    paused = False
    reverse = False
    speed = 15

    while True:
        if not paused:
            if reverse:
                current_frame -= 1
                if current_frame < 0:
                    current_frame = max_length - 1 
            else:
                current_frame += 1
                if current_frame >= max_length:
                    break
                    current_frame = 0 

            display_frame(current_frame, scene, scene_data, show_topology, base_map_img.copy() if show_topology else None)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused  # Toggle between play and pause
        elif key == ord('k'):
            reverse = True  # Set to reverse play
            paused = False  # Resume playing if paused
        elif key == ord('l'):
            reverse = False  # Set to normal play
            paused = False  # Resume playing if paused
        elif key == ord('d'):
            paused = True  # Pause the video while seeking
            current_frame -= 1
            if current_frame < 0:
                current_frame = max_length - 1  # Loop back to the last frame
            display_frame(current_frame, scene, scene_data, show_topology, base_map_img.copy() if show_topology else None)
        elif key == ord('f'):
            paused = True  # Pause the video while seeking
            current_frame += 1
            if current_frame >= max_length:
                current_frame = 0  # Loop back to the first frame
            display_frame(current_frame, scene, scene_data, show_topology, base_map_img.copy() if show_topology else None)
        elif key == ord('e'):
            paused = True  # Pause the video while seeking
            current_frame -= 5
            if current_frame < 0:
                current_frame = max_length - 1  # Loop back to the last frame
            display_frame(current_frame, scene, scene_data, show_topology, base_map_img.copy() if show_topology else None)
        elif key == ord('r'):
            paused = True  # Pause the video while seeking
            current_frame += 5
            if current_frame >= max_length:
                current_frame = 0  # Loop back to the first frame
            display_frame(current_frame, scene, scene_data, show_topology, base_map_img.copy() if show_topology else None)
        elif key == ord('m'):
            speed *= 2 # Speed playback up by two
        elif key == ord('n'):
            speed /= 2 # Show playback down by two

        if not paused:
            time.sleep(1/speed)    

    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
