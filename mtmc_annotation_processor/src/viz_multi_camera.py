import cv2
import time
import json
import argparse
import colorama
from colorama import Fore, Style

color_map = json.load(open("config/color_map.json", mode="r"))
color_map = {int(k): v for k, v in color_map.items()}

def hex_to_bgr(hex_color):
    hex_color = hex_color.lstrip('#') # Remove the '#' symbol if present
    bgr = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0)) # Convert the hex string to BGR tuple (note the order of slicing)

    return bgr


def display_frame(current_frame, scene_id):
    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
    ret, frame = cap.read()
    if ret:

        def get_text_color(r, g, b):
            # Normalize RGB values to [0, 1]
            r /= 255.0
            g /= 255.0
            b /= 255.0
            
            # Calculate luminance
            luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
            
            # Choose text color based on luminance
            return (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

        font_size = 1.5
        thickness = 2

        current_frame += 1

        if str(current_frame) in bboxes.keys():
            for box in bboxes[str(current_frame)]:
                x, y, w, h = box["x"], box["y"], box["w"], box["h"]
                start_point = (x, y)
                end_point = (x + w, y + h)

                if box["label"] >= 10:
                    rect_point = (x + 35, y + 22)
                else:
                    rect_point = (x + 17, y + 22)

                color = hex_to_bgr(color_map[box["label"]])
                
                frame = cv2.rectangle(frame, start_point, end_point, color, thickness)
                frame = cv2.rectangle(frame, start_point, rect_point, color, -1)
                frame = cv2.putText(frame, str(box["label"]), (x, y + 20), cv2.FONT_HERSHEY_PLAIN, font_size, get_text_color(*color), thickness)

        frame = cv2.putText(frame, f'{current_frame - 1}', (10, 30), cv2.FONT_HERSHEY_PLAIN, font_size, (0, 255, 255), 3)

        current_frame -= 1

        frame = cv2.resize(frame, (1600, 900))   
        cv2.imshow("MTMC Annotation Verifier" + scene_id, frame)

def check_track_fragmentations(nums):
    nums = list(map(int, nums))

    non_consecutive_count = 0
    duplicate_count = 0
    
    for i in range(1, len(nums)):
        if nums[i] == nums[i-1]:  # Duplicate case
            duplicate_count += 1
            print(f"Mis-labeling found at frame: {nums[i] - 1}")
        # if nums[i] != nums[i-1] + 1:  # Non-consecutive case
        #     non_consecutive_count += 1
        #     print(f"Fragmentation found at frame {nums[i-1] - 1} jumping to {nums[i] - 1}!")
        #     if nums[i] < nums[i-1] + 5:
        #         print(Fore.LIGHTMAGENTA_EX + "Unfortunate MOT fragmentation fault found!!!!!!!!!!!!!!!!" + Style.RESET_ALL)
    
    return non_consecutive_count, duplicate_count

def parse_opt():
    parser = argparse.ArgumentParser(description='Specify video')
    parser.add_argument('--scene', type=str, help='Scene ID')
    opt = parser.parse_args()
    return opt

def main(opt):

    def get_key_from_value(d, value):
        # Iterate through the dictionary items to find the key for the given value
        for key, val in d.items():
            if val == value:
                return key
    # cam_count = "111108"
    # scene_id = "03"
    # floor_id = "11"

    floor_id = str(opt.scene)[1:3]
    scene_id = str(opt.scene)[4:]
    cam_count = 7

    vid_path = f"VTX_MTMC_2024/F{floor_id}_{scene_id}/grid_view.mp4"
    gt_path = f"VTX_MTMC_2024/F{floor_id}_{scene_id}/grid_gt.txt"

    global cap
    cap = cv2.VideoCapture(vid_path)

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))    

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Get the total number of frames
    current_frame = 0
    paused = False  # Variable to track the pause state
    reverse = False  # Variable to track the play direction
    
    global speed
    speed = 15

    global bboxes
    bboxes = {}
    current_max_index = 1
    label_mapper = {}

    appearance_stats = {}

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
                appearance_stats[assessed_index] = []
                current_max_index += 1

            box_data = {
                "label": assessed_index,
                "x": int(float(bbox[2])),
                "y": int(float(bbox[3])),
                "w": int(float(bbox[4])),
                "h": int(float(bbox[5])),
            }

            appearance_stats[assessed_index].append(bbox[0])

            bboxes[bbox[0]].append(box_data)

    # print(appearance_stats)

    total_frags = 0
    total_doops = 0

    for labels in appearance_stats.keys():    
        print(Fore.LIGHTBLUE_EX + f"Checking label no {labels}, appearing from {int(appearance_stats[labels][0] )- 1} to {int(appearance_stats[labels][-1]) - 1}:" + Style.RESET_ALL)
        # no_frags, no_doops = check_track_fragmentations(appearance_stats[labels])

        # if no_frags or no_doops:
        #     print(Fore.RED + f"Found {no_frags} fragmentations and {no_doops} duplicated labels!" + Style.RESET_ALL)
        # else:
        #     print(Fore.GREEN + f"Found {no_frags} fragmentations and {no_doops} duplicated labels!" + Style.RESET_ALL)

        # total_frags += no_frags
        # total_doops += no_doops

    print(f"Found {total_frags} frags and {total_doops} doops in a {total_frames} long video with {current_max_index - 1} people!")

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

            display_frame(current_frame, f" - F{floor_id}_{scene_id} - {cam_count}")

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
            display_frame(current_frame, f" - F{floor_id}_{scene_id} - {cam_count}")  # Display the updated frame
        elif key == ord('f'):
            paused = True  # Pause the video while seeking
            current_frame += 1
            if current_frame >= total_frames:
                current_frame = 0  # Loop back to the first frame
            display_frame(current_frame, f" - F{floor_id}_{scene_id} - {cam_count}")  # Display the updated frame
        elif key == ord('e'):
            paused = True  # Pause the video while seeking
            current_frame -= 5
            if current_frame < 0:
                current_frame = total_frames - 1  # Loop back to the last frame
            display_frame(current_frame, f" - F{floor_id}_{scene_id} - {cam_count}")  # Display the updated frame
        elif key == ord('r'):
            paused = True  # Pause the video while seeking
            current_frame += 5
            if current_frame >= total_frames:
                current_frame = 0  # Loop back to the first frame
            display_frame(current_frame, f" - F{floor_id}_{scene_id} - {cam_count}")  # Display the updated frame
        elif key == ord('m'):
            speed *= 2
        elif key == ord('n'):
            speed /= 2

        if not paused:
            time.sleep(1/speed)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    opt = parse_opt()
    main(opt)