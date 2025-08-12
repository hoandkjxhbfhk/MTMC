import argparse
import os
import os.path as osp
import sys
import numpy as np
from tqdm import tqdm

from util.camera import Camera
from Tracker.PoseTracker import Detection_Sample, TrackState
from Tracker.PoseTracker_cluster import PoseTracker


"""
Script batch để test phiên bản PoseTracker đã cải tiến (cluster refinement nội bộ) trong
`Tracker/PoseTracker_cluster.py`. Không dùng SCT giả lập hay gom cụm liên-camera.
"""


def main():
    scene_name = sys.argv[1]

    current_file_path = os.path.abspath(__file__)
    path_arr = current_file_path.split('/')[:-2]
    root_path = '/'.join(path_arr)

    det_dir = osp.join(root_path, "result/detection", scene_name)
    pose_dir = osp.join(root_path, "result/pose", scene_name)
    reid_dir = osp.join(root_path, "result/reid", scene_name)

    cal_dir = osp.join(root_path, 'dataset/test', scene_name)
    save_dir = os.path.join(root_path, "result/track_cluster")
    os.makedirs(save_dir, exist_ok=True)
    save_path = osp.join(save_dir, scene_name + ".txt")

    all_items = sorted(os.listdir(cal_dir))
    cams = [item for item in all_items if item.startswith('camera_') and osp.isdir(osp.join(cal_dir, item))]
    cals = [Camera(osp.join(cal_dir, cam, "calibration.json")) for cam in cams]

    det_data = []
    files = sorted(os.listdir(det_dir))
    files = [f for f in files if f[0] == 'c']
    for f in files:
        det_data.append(np.loadtxt(osp.join(det_dir, f), delimiter=","))

    pose_data = []
    files = sorted(os.listdir(pose_dir))
    files = [f for f in files if f[0] == 'c']
    for f in files:
        pose_data.append(np.loadtxt(osp.join(pose_dir, f)))

    reid_data = []
    files = sorted(os.listdir(reid_dir))
    files = [f for f in files if f[0] == 'c']
    for f in files:
        reid_data_scene = np.load(osp.join(reid_dir, f), mmap_mode='r')
        if len(reid_data_scene):
            reid_data_scene = reid_data_scene / np.linalg.norm(reid_data_scene, axis=1, keepdims=True)
        reid_data.append(reid_data_scene)

    # Suy ra chiều vector reid (fallback 2048 nếu không có dữ liệu)
    reid_dim = 2048
    for arr in reid_data:
        if hasattr(arr, 'shape') and len(arr.shape) == 2 and arr.shape[1] > 0:
            reid_dim = int(arr.shape[1])
            break

    print("reading finish")

    max_frame = []
    for det_sv in det_data:
        if len(det_sv):
            max_frame.append(np.max(det_sv[:, 0]))
    max_frame = int(np.max(max_frame))

    # Khởi tạo PoseTracker (bản cải tiến trong PoseTracker_cluster)
    tracker = PoseTracker(cals)
    box_thred = 0.3
    results = []

    for frame_id in tqdm(range(max_frame + 1), desc=scene_name):
        detection_sample_mv = []
        for v in range(len(cals)):
            detection_sample_sv = []
            det_sv = det_data[v]
            if len(det_sv) == 0:
                detection_sample_mv.append(detection_sample_sv)
                continue
            idx = det_sv[:, 0] == frame_id
            cur_det = det_sv[idx]
            cur_pose = pose_data[v][idx]
            # Lấy reid theo frame; nếu thiếu file hoặc không đồng bộ với det, dùng zero-vector
            if v < len(reid_data) and hasattr(reid_data[v], 'shape') and reid_data[v].shape[0] == det_sv.shape[0]:
                cur_reid = reid_data[v][idx]
            else:
                cur_reid = np.zeros((int(np.sum(idx)), reid_dim), dtype=np.float32)

            for det, pose, reid in zip(cur_det, cur_pose, cur_reid):
                if len(det) == 0 or det[-1] < box_thred:
                    continue
                new_sample = Detection_Sample(
                    bbox=det[2:],
                    keypoints_2d=pose[6:].reshape(17, 3),
                    reid_feat=reid,
                    cam_id=v,
                    frame_id=frame_id,
                )
                detection_sample_sv.append(new_sample)
            detection_sample_mv.append(detection_sample_sv)

        # Cập nhật trực tiếp PoseTracker cải tiến (có cluster refinement nội bộ)
        tracker.mv_update_wo_pred(detection_sample_mv, frame_id)

        frame_results = tracker.output(frame_id)
        results += frame_results

    if len(results):
        results = np.concatenate(results, axis=0)
        sort_idx = np.lexsort((results[:, 2], results[:, 0]))
        results = np.ascontiguousarray(results[sort_idx])
        np.savetxt(save_path, results)
    else:
        # vẫn tạo file rỗng để đánh dấu đã chạy
        open(save_path, 'a').close()


if __name__ == '__main__':
    main()


