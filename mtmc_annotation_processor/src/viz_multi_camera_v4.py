import os
import cv2
import time
from collections import defaultdict
import json
import argparse
import numpy as np
import pandas as pd
import math
import yaml

from mmpose.apis import init_model, inference_topdown

pose_config = "./mmpose/configs/body_2d_keypoint/rtmpose/crowdpose/rtmpose-m_8xb64-210e_crowdpose-256x192.py"
checkpoint_file = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth'
pose_model = init_model(pose_config, checkpoint_file, device='cuda:0')

"""
    MTMC Visualizer v4:
    - Support MMPose for inferring the foot point.
    - Requires GPU however.
"""

color_map = json.load(open("config/color_map.json", mode="r"))
color_map = {int(k): v for k, v in color_map.items()}

def hex_to_bgr(hex_color: str):
    hex_color = hex_color.lstrip('#') 
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))

    return bgr

def get_text_color(r, g, b):
    r /= 255.0
    g /= 255.0
    b /= 255.0
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

def centroidPutText(img, text, org, fontFace, fontScale, color, thickness, lineType=cv2.LINE_AA):
    (text_width, text_height), baseline = cv2.getTextSize(text, fontFace, fontScale, thickness)
    
    # Compute new text position (centered)
    text_x = org[0] - text_width // 2
    text_y = org[1] + text_height // 2  # Adjust for vertical centering

    # Draw text
    cv2.putText(img, text, (text_x, text_y), fontFace, fontScale, color, thickness, lineType)
    
    return img

def camera_to_ground_coordinates(homography_matrix: np.array, camera_points: np.array):
    if homography_matrix is None:
        raise ValueError("Homography matrix has not been calculated or loaded.")

    camera_point_homogeneous = np.append(camera_points, 1) # Add a 1 to the end of the camera point to create a homogeneous coordinate
    topo_point_homogeneous = np.dot(homography_matrix, camera_point_homogeneous) # Transform the point using the homography matrix
    topo_point = (topo_point_homogeneous / topo_point_homogeneous[2])[:2].astype(int) # Divide by the last element to get the final (x, y) coordinates

    return topo_point[0], topo_point[1]

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

def visualize_pose_estimation(image, detections, pose_results):
    skeleton_connections = [
        (12, 13),  # head to neck
        (13, 0), (13, 1),  # neck to shoulders
        (0, 2), (2, 4),  # left arm
        (1, 3), (3, 5),  # right arm
        (0, 6), (1, 7),  # shoulders to hips
        (6, 8), (8, 10),  # left leg
        (7, 9), (9, 11),  # right hip to right knee
    ]

    for i, (det, keypoints) in enumerate(zip(detections, pose_results)):
        # Draw the bounding box
        x1, y1, x2, y2 = map(int, det)

        keypoint_score_list = keypoints.pred_instances.keypoint_scores[0].tolist()
        keypoints = keypoints.pred_instances.keypoints[0]

        for idx, (x, y) in enumerate(keypoints):
            cv2.circle(image, (int(x), int(y)), 10, (0, 0, 255), -1)  # Red keypoints
            # cv2.putText(image, f"{idx}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if idx in (10, 11):
                cv2.putText(image, f"{keypoint_score_list[idx]:.3f}", (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 6)


        # Draw the skeletal connections
        for kp1, kp2 in skeleton_connections:
            x1, y1 = keypoints[kp1]
            x2, y2 = keypoints[kp2]
            cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)  # Yellow lines

    return image

trail_history = {}

def display_frame(current_frame, scene_id, scene_data: dict, topology_map, config_file):
    grid_order = get_camera_grid_order(list(scene_data.keys()))
    no_rows, no_cols = len(grid_order), len(grid_order[0])
    cap_width = 1600
    cap_height = 900

    grid = np.zeros((cap_height * no_rows, cap_width * no_cols, 3), dtype=np.uint8)

    topology_radius = config_file["TOPOLOGY"]["RADIUS"]
    id_topo_range = {}
    centroid_range = {}

    for cam_id, cam_val in scene_data.items():
        cap = cam_val["video_cap"]
        box_data = cam_val["box_data"]

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        font_size = 4
        thickness = 4

        no_obj = 0

        if ret:
            for_pose = frame.copy()
            pose_input = []

            frame_annotations = box_data[box_data['frame'] == current_frame + 1]

            for _, row in frame_annotations.iterrows():
                no_obj += 1

                x1, y1, x2, y2 = int(row['x']), int(row['y']), int(row['x']) + int(row['w']), int(row['y']) + int(row['h'])
                start_point = (x1, y1)
                end_point = (x2, y2)

                pose_input.append([x1, y1, x2, y2])

                if int(row['id']) >= 10:
                    rect_point = (x1 + 75, y1 + 50)
                else:
                    rect_point = (x1 + 45, y1 + 50)
                visualized_id = int(row["id"])
                color = hex_to_bgr(color_map[visualized_id])
                frame = cv2.rectangle(frame, start_point, end_point, color, 8)
                frame = cv2.rectangle(frame, start_point, rect_point, color, -1)
                frame = cv2.putText(frame, str(visualized_id), (x1, y1 + 50), cv2.FONT_HERSHEY_PLAIN, font_size, get_text_color(*color), thickness)

                if config_file["TOPOLOGY"]["SHOW"] == "DISTRIBUTED":
                    camera_point = np.array([(x1 + x2) // 2, y2], dtype='float32')
                    ground_point = camera_to_ground_coordinates(cam_val["H_matrix"], camera_point)

                    # Draw the smallest rectangular convex hull enclosing the topology points
                    if visualized_id not in id_topo_range:
                        id_topo_range[visualized_id] = {
                            "x1": ground_point[0],
                            "y1": ground_point[1],
                            "x2": ground_point[0],
                            "y2": ground_point[1]
                        }

                    id_topo_range[visualized_id]["x1"] = min(id_topo_range[visualized_id]["x1"], ground_point[0])
                    id_topo_range[visualized_id]["y1"] = min(id_topo_range[visualized_id]["y1"], ground_point[1])
                    id_topo_range[visualized_id]["x2"] = max(id_topo_range[visualized_id]["x2"], ground_point[0])
                    id_topo_range[visualized_id]["y2"] = max(id_topo_range[visualized_id]["y2"], ground_point[1])

                    topology_map = cv2.circle(topology_map, ground_point, 7, color, -1)
                    topology_map = cv2.circle(topology_map, ground_point, topology_radius, color, 2)
                    topology_map = cv2.putText(topology_map, cam_id[-2:], (ground_point[0], ground_point[1] - 27), cv2.FONT_HERSHEY_PLAIN, 1.25, color, 2)

                elif config_file["TOPOLOGY"]["SHOW"] == "CENTROID":
                    camera_point = np.array([(x1 + x2) // 2, y2], dtype='float32')
                    ground_point = camera_to_ground_coordinates(cam_val["H_matrix"], camera_point)
                    
                    if visualized_id not in centroid_range:
                        centroid_range[visualized_id] = []

                    centroid_range[visualized_id].append(ground_point)

            pose_result = inference_topdown(pose_model, for_pose, pose_input, bbox_format='xyxy')
            frame = visualize_pose_estimation(frame, pose_input, pose_result)
            

            frame = cv2.putText(frame, f'{current_frame} - {cam_id}', (100, 60), cv2.FONT_HERSHEY_PLAIN, 4, (0, 255, 255), 6)
            frame = cv2.resize(frame, (cap_width, cap_height))

            grid_row, grid_col = find_value_index(grid_order, cam_id)
            grid_x = grid_col * cap_width
            grid_y = grid_row * cap_height

            grid[grid_y:grid_y + cap_height, grid_x:grid_x + cap_width] = frame

        # print(f"{cam_id} has {no_obj}!")

    grid = cv2.resize(grid, (int(config_file["DISPLAY"]["CELL_WIDTH"] * no_cols * config_file["DISPLAY"]["SCALE_FACTOR"]), 
                             int(config_file["DISPLAY"]["CELL_HEIGHT"] * no_rows * config_file["DISPLAY"]["SCALE_FACTOR"])))
    cv2.imshow(f'MTMC Visualizer - {scene_id}', grid)

    if config_file["SAVE_VIDEO"]["CAMERA_FEED"]["SAVE"]:
        grid = cv2.resize(grid, config_file["SAVE_VIDEO"]["CAMERA_FEED"]["OUTPUT_RESOLUTION"], )
        feed_writer.write(grid)
    
    if config_file["TOPOLOGY"]["SHOW"] == "DISTRIBUTED":
        for id in id_topo_range:
            id = int(id)
            convex_tl = (id_topo_range[id]["x1"] - topology_radius, id_topo_range[id]["y1"] - topology_radius)
            convex_br = (id_topo_range[id]["x2"] + topology_radius, id_topo_range[id]["y2"] + topology_radius)
            if id < 10 and id >= 0:
                id_box = (convex_tl[0] + 20, convex_tl[1] - 20)
            else:
                id_box = (convex_tl[0] + 40, convex_tl[1] - 20)
            color = hex_to_bgr(color_map[id])
            topology_map = cv2.rectangle(topology_map, convex_tl, convex_br, color, 2)
            topology_map = cv2.rectangle(topology_map, convex_tl, id_box, color, -1)
            topology_map = cv2.putText(topology_map, str(id), convex_tl, cv2.FONT_HERSHEY_PLAIN, 1.5, get_text_color(*color), 2)

        if config_file["DISPLAY"]["TOPDOWN_WIDTH"] != -1 and config_file["DISPLAY"]["TOPDOWN_HEIGHT"] != -1:
            topology_map = cv2.resize(topology_map, (config_file["DISPLAY"]["TOPDOWN_WIDTH"], config_file["DISPLAY"]["TOPDOWN_HEIGHT"]))

        cv2.imshow("Topology Map", topology_map)

        if config_file["SAVE_VIDEO"]["CAMERA_FEED"]["SAVE"]:
            map_writer.write(topology_map)
    
    elif config_file["TOPOLOGY"]["SHOW"] == "CENTROID":
        for id in centroid_range:
            id = int(id)
            center_points = centroid_range[id]
            center = tuple(map(lambda x: sum(x) // len(x), zip(*center_points)))
            color = hex_to_bgr(color_map[id])
            edge_color = get_text_color(*color)

            if id in trail_history:
                if len(trail_history[id]["trail"]) >= 2:
                    for pt in range(len(trail_history[id]["trail"]) - 1):
                        start_point = trail_history[id]["trail"][pt]
                        end_point = trail_history[id]["trail"][pt + 1]

                        # print(f"ID: {id}, startpoint {start_point}, endpoint {end_point}")
                        topology_map = cv2.line(topology_map, start_point, end_point, color, thickness)

            topology_map = cv2.circle(topology_map, center, 25, color, -1)
            topology_map = cv2.circle(topology_map, center, 19, edge_color, 2)
            topology_map = centroidPutText(topology_map, str(id), center, cv2.FONT_HERSHEY_PLAIN, 2, edge_color, 2)

            if id not in trail_history:
                trail_history[id] = {
                    "trail": [],
                    "last_update": 0,
                    "updated": True
                }

            trail_history[id]["trail"].append(center)
            trail_history[id]["updated"] = True

            if len(trail_history[id]["trail"]) > config_file["TOPOLOGY"]["TRAIL_MAX_LEN"]:
                trail_history[id]["trail"].pop(0)           
            

        if config_file["DISPLAY"]["TOPDOWN_WIDTH"] != -1 and config_file["DISPLAY"]["TOPDOWN_HEIGHT"] != -1:
            topology_map = cv2.resize(topology_map, (config_file["DISPLAY"]["TOPDOWN_WIDTH"], config_file["DISPLAY"]["TOPDOWN_HEIGHT"]))

        cv2.imshow("Topology Map", topology_map)

        if config_file["SAVE_VIDEO"]["CAMERA_FEED"]["SAVE"]:
            map_writer.write(topology_map)

        removal_ids = set()

        for id in trail_history:
            if trail_history[id]["updated"]:
                trail_history[id]["updated"] = False
                trail_history[id]["last_update"] = 0
            else:
                trail_history[id]["last_update"] += 1
                if trail_history[id]["last_update"] > config_file["TOPOLOGY"]["TRAIL_MAX_AGE"]:
                    removal_ids.add(id)                    

        for id in removal_ids:            
            trail_history.pop(id, None)



"""
    Main function stuff.
"""

def parse_opt():
    parser = argparse.ArgumentParser(description='Provide MTMC Visualization Configuration')
    parser.add_argument('--config_file', type=str, help='Path to config file', default="config/visualizer.yml")
    opt = parser.parse_args()
    return opt

def main(opt):
    config = yaml.safe_load(open(opt.config_file))

    scene = config["DATA"]["SCENE"]
    show_topology = config["TOPOLOGY"]["SHOW"]

    data_root = os.path.join(config["DATA"]["PATH"], config["DATA"]["SCENE"])

    camera_views = [cam_id for cam_id in sorted(os.listdir(data_root)) if os.path.isdir(os.path.join(data_root, cam_id))]
    camera_order = config["CAMERA_ORDER"]

    if camera_order is not None:
        order_list = [cam_id for cam_id in camera_order]
        assert set(order_list).issubset(set(camera_views)), f"Camera order does not align with available camera views! {order_list} != {camera_views}"
    else:
        order_list = camera_views

    if show_topology:
        img_read = cv2.imread(config["TOPOLOGY"]["TOPOLOGY_MAP"]) 
        base_map_img = img_read if img_read is not None \
                                else np.full((config["TOPOLOGY"]["BASE_MAP_HEIGHT"], config["TOPOLOGY"]["BASE_MAP_WIDTH"], 3), 0, dtype=np.uint8)
        
        homography_json = json.load(open(config["TOPOLOGY"]["HOMOGRAPHY_PATH"]))
        homography_data = defaultdict(list)

        # for camera_id, homography_matrix in homography_json.items():
        #     homography_data[camera_id] = np.linalg.inv(np.asarray(homography_matrix))

        for cluster, cameras in homography_json["ground_plane"].items():
            for camera_id, homo_data in cameras.items():
                # The homography matrices shall be referred to by camera ID
                homography_data[camera_id] = np.asarray(homo_data["H"])

        if config["SAVE_VIDEO"]["TOPOLOGY_MAP"]["SAVE"]:
            global map_writer
            os.makedirs(config["SAVE_VIDEO"]["TOPOLOGY_MAP"]["OUTPUT_DIR"], exist_ok=True)

            if config["DISPLAY"]["TOPDOWN_WIDTH"] != -1 and config["DISPLAY"]["TOPDOWN_HEIGHT"] != -1:
                map_writer = cv2.VideoWriter(f'{config["SAVE_VIDEO"]["TOPOLOGY_MAP"]["OUTPUT_DIR"]}/map_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                                             config["DATA"]["FRAME_RATE"], (config["DISPLAY"]["TOPDOWN_WIDTH"], config["DISPLAY"]["TOPDOWN_HEIGHT"]))
            else:
                map_height, map_width, _ = base_map_img.shape
                map_writer = cv2.VideoWriter(f'{config["SAVE_VIDEO"]["TOPOLOGY_MAP"]["OUTPUT_DIR"]}/map_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 
                                             config["DATA"]["FRAME_RATE"], (map_width, map_height))

    if config["SAVE_VIDEO"]["CAMERA_FEED"]["SAVE"]:
        grid_order = get_camera_grid_order(order_list)
        no_rows, no_cols = len(grid_order), len(grid_order[0])

        global feed_writer
        os.makedirs(config["SAVE_VIDEO"]["CAMERA_FEED"]["OUTPUT_DIR"], exist_ok=True)
        config["SAVE_VIDEO"]["CAMERA_FEED"]["OUTPUT_RESOLUTION"] = (800 * no_cols, 480 * no_rows)
        feed_writer = cv2.VideoWriter(f'{config["SAVE_VIDEO"]["CAMERA_FEED"]["OUTPUT_DIR"]}/feed_output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), config["DATA"]["FRAME_RATE"], (800 * no_cols, 480 * no_rows))
    

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
        # if config["MODE"] == "gt":
        #     txt_file = f"{data_root}/{camera}/gt.txt"
        # elif config["MODE"] == "pred":
        #     txt_file = f"{data_root}/{camera}/mtmc/{camera}.txt"

        if config["MODE"] == "gt":
            txt_file = f"{data_root}/{camera}/{camera}.txt"
        elif config["MODE"] == "pred":
            txt_file = f"{data_root}/{camera}/mtmc/{camera}.txt"

        video_path = f"{data_root}/{camera}/{camera}.mp4"
        box_data = load_annotations(txt_file, config["MODE"])

        cap = cv2.VideoCapture(video_path)
        max_length = max(max_length, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))

        scene_data[camera] = {
            "video_cap": cap,
            "txt_file": txt_file,
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
                    trail_history.clear()
            else:
                current_frame += 1
                if current_frame >= max_length:
                    # break
                    current_frame = 0 
                    trail_history.clear()

            display_frame(current_frame, scene, scene_data, base_map_img.copy() if show_topology else None, config)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('p'):
            paused = not paused  # Toggle between play and pause
        elif key == ord('k'):
            reverse = True  # Set to reverse play
            paused = False  # Resume playing if paused
            trail_history.clear()
        elif key == ord('l'):
            reverse = False  # Set to normal play
            paused = False  # Resume playing if paused
            trail_history.clear()
        elif key == ord('d'):
            paused = True  # Pause the video while seeking
            current_frame -= 1
            if current_frame < 0:
                current_frame = max_length - 1  # Loop back to the last frame
            trail_history.clear()
            display_frame(current_frame, scene, scene_data, base_map_img.copy() if show_topology else None, config)
        elif key == ord('f'):
            paused = True  # Pause the video while seeking
            current_frame += 1
            if current_frame >= max_length:
                current_frame = 0  # Loop back to the first frame
            trail_history.clear()
            display_frame(current_frame, scene, scene_data, base_map_img.copy() if show_topology else None, config)
        elif key == ord('e'):
            paused = True  # Pause the video while seeking
            current_frame -= 5
            if current_frame < 0:
                current_frame = max_length - 1  # Loop back to the last frame
            trail_history.clear()
            display_frame(current_frame, scene, scene_data, base_map_img.copy() if show_topology else None, config)
        elif key == ord('r'):
            paused = True  # Pause the video while seeking
            current_frame += 5
            if current_frame >= max_length:
                current_frame = 0  # Loop back to the first frame
            trail_history.clear()
            display_frame(current_frame, scene, scene_data, base_map_img.copy() if show_topology else None, config)
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
