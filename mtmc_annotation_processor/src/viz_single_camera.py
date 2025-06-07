import cv2
import time
import json
import argparse
import colorama
from colorama import Fore, Style

# Topology visualization
from transform_plane import CoordinateTransformer

color_map = json.load(open("config/color_map.json", mode="r"))
color_map = {int(k): v for k, v in color_map.items()}

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#') # Remove the '#' symbol if present
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0)) # Convert the hex string to BGR tuple (note the order of slicing)

    return bgr

def get_text_color(r, g, b):
    r /= 255.0
    g /= 255.0
    b /= 255.0
    luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)


def display_frame(current_frame, scene_id, cap, bboxes, show_topology=True):
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()

    if show_topology:
        topo_map = camera_map.copy()
    
    if ret:

        current_frame += 1

        if str(current_frame) in bboxes.keys():
            for box in bboxes[str(current_frame)]:
                x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                start_point = (x, y)
                if box["label"] > 10:
                    rect_point = (x + 70, y + 50)
                else:
                    rect_point = (x + 45, y + 50)
                end_point = (x + w, y + h)
                color = hex_to_bgr(color_map[box["label"]])
                thickness = 8
                frame = cv2.rectangle(frame, start_point, end_point, color, 8)
                # frame = cv2.rectangle(frame, start_point, end_point, (0, 255, 0), thickness)
                frame = cv2.rectangle(frame, start_point, rect_point, color, -1)
                frame = cv2.putText(frame, str(box["label"]), (x, y + 50), cv2.FONT_HERSHEY_PLAIN, 3.5, get_text_color(*color), thickness)
                
                if show_topology:
                    max_y_coordinate = ((x + x + w) / 2, y + h)
                    transformed_point = plane_transformer.camera_to_map(max_y_coordinate)[0]

                    cv2.circle(topo_map, (int(transformed_point[0]), int(transformed_point[1])), 10, color, -1)

        frame = cv2.putText(frame, f'{current_frame - 1}', (100, 60), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 255), 8)

        current_frame -= 1

        frame = cv2.resize(frame, (1280, 720))   
        cv2.imshow("MOT Annotation Verifier" + scene_id, frame)

        if show_topology:
            topo_map = cv2.resize(topo_map, (topo_map.shape[1] // 2, topo_map.shape[0] // 2))
            cv2.imshow("Map with Point", topo_map)   


def parse_opt():
    parser = argparse.ArgumentParser(description='Specify video')
    parser.add_argument('--data_dir', type=str, help='Data Path', default="VTX_MTMC_2024")
    parser.add_argument('--scene', type=str, help='Scene ID')
    parser.add_argument('--cam_id', type=str, help='Camera ID')
    parser.add_argument('--show_topology', type=str, help='Show topology', default="Yes")
    opt = parser.parse_args()
    return opt


def main(opt):
    colorama.init()

    floor_id = str(opt.scene)[1:3]
    scene_id = str(opt.scene)[4:]
    cam_id = opt.cam_id
    data_path = opt.data_dir
    show_topology = opt.show_topology

    vid_path = f"{data_path}/F{floor_id}_{scene_id}/{cam_id}/{cam_id}.mp4"
    gt_path = f"{data_path}/F{floor_id}_{scene_id}/{cam_id}/{cam_id}.txt"

    cap = cv2.VideoCapture(vid_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
    current_frame = 0
    paused = False  # Variable to track the pause state
    reverse = False  # Variable to track the play direction
    speed = 15

    bboxes = {}
    current_max_index = 1
    label_mapper = {}

    if show_topology == "Yes":
        global camera_map
        camera_map = cv2.imread(f"assets/t{floor_id.lstrip('0')}.png")

        global plane_transformer

        homography_file = json.load(open("config/dataset_VTX2024_homo_info_calc1.json", mode="r"))
        transform_data = {
            "transform_matrix": homography_file["ground_plane"][f"cluster_{floor_id.lstrip('0')}A"][cam_id]["H"],
            "map_points": homography_file["ground_plane"][f"cluster_{floor_id.lstrip('0')}A"][cam_id]["ground"],
            "camera_points": homography_file["ground_plane"][f"cluster_{floor_id.lstrip('0')}A"][cam_id]["image"]
        }

        plane_transformer = CoordinateTransformer.load_matrices(transform_data)

    with open(gt_path, mode='r') as gt:
        for bbox in gt:
            bbox = bbox[:-1]
            bbox = bbox.split(",")

            # bbox[0] here is the frame number
            if bbox[0] not in bboxes.keys():
                bboxes[bbox[0]] = []

            assessed_index = int(bbox[1])
            if assessed_index not in label_mapper.keys():
                label_mapper[assessed_index] = current_max_index
                current_max_index += 1

            box_data = {
                # "label": label_mapper[assessed_index],
                "label": assessed_index,
                "x": int(float(bbox[2])),
                "y": int(float(bbox[3])),
                "w": int(float(bbox[4])),
                "h": int(float(bbox[5])),
            }

            bboxes[bbox[0]].append(box_data)


    while True:
        if not paused:
            if reverse:
                current_frame -= 1
                if current_frame < 0:
                    current_frame = total_frames - 1  # Loop back to the last frame
            else:
                current_frame += 1
                if current_frame >= total_frames:
                    current_frame = 0  # Loop back to the first frame

            display_frame(current_frame, f" - F{floor_id}_{scene_id} - {cam_id}", cap=cap, bboxes=bboxes, show_topology=(show_topology == "Yes"))

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
                current_frame = total_frames - 1  # Loop back to the last frame
            display_frame(current_frame, f" - F{floor_id}_{scene_id} - {cam_id}", cap=cap, bboxes=bboxes, show_topology=(show_topology == "Yes"))
        elif key == ord('f'):
            paused = True  # Pause the video while seeking
            current_frame += 1
            if current_frame >= total_frames:
                current_frame = 0  # Loop back to the first frame
            display_frame(current_frame, f" - F{floor_id}_{scene_id} - {cam_id}", cap=cap, bboxes=bboxes, show_topology=(show_topology == "Yes"))
        elif key == ord('e'):
            paused = True  # Pause the video while seeking
            current_frame -= 5
            if current_frame < 0:
                current_frame = total_frames - 1  # Loop back to the last frame
            display_frame(current_frame, f" - F{floor_id}_{scene_id} - {cam_id}", cap=cap, bboxes=bboxes, show_topology=(show_topology == "Yes"))
        elif key == ord('r'):
            paused = True  # Pause the video while seeking
            current_frame += 5
            if current_frame >= total_frames:
                current_frame = 0  # Loop back to the first frame
            display_frame(current_frame, f" - F{floor_id}_{scene_id} - {cam_id}", cap=cap, bboxes=bboxes, show_topology=(show_topology == "Yes"))
        elif key == ord('m'):
            speed *= 2 # Speed playback up by two
        elif key == ord('n'):
            speed /= 2 # Show playback down by two

        if not paused:
            time.sleep(1/speed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)
