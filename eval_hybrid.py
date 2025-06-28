import argparse
import time
import os
import cv2
import numpy as np

from ultralytics import YOLO
from mmpose.apis import init_model

from trackers.botsort.bot_sort import BoTSORT
from trackers.hybrid.hybrid_track import HybridMCTracker
from trackers.multicam_tracker.clustering import Clustering, ID_Distributor

from perspective_transform.model import PerspectiveTransform
from perspective_transform.calibration import calibration_position

from tools.utils import (_COLORS, get_reader_writer, finalize_cams, write_vids, write_results_testset, 
                    update_result_lists_testset, sources, result_paths, cam_ids)


def make_parser():
    parser = argparse.ArgumentParser("Evaluate MTMC with Hybrid Tracker")
    parser.add_argument("-s", "--scene", type=str, default=None, help="scene name")
    return parser.parse_args()


def run(args_cfg, conf_thres, iou_thres, sources, result_paths, perspective, cam_ids, scene):
    # detection model
    if int(scene.split('_')[1]) < 81:
        detection = YOLO('pretrained/yolov8x_aic.pt')
    else:
        detection = YOLO('yolov8x.pt')

    # pose model
    cfg_file = 'mmpose/configs/body_2d_keypoint/rtmpose/crowdpose/rtmpose-m_8xb64-210e_crowdpose-256x192.py'
    ckpt = 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-crowdpose_pt-aic-coco_210e-256x192-e6192cac_20230224.pth'
    pose = init_model(cfg_file, ckpt, device='cuda:0')

    # single-cam trackers
    trackers = [BoTSORT(track_buffer=args_cfg['track_buffer'], max_batch_size=args_cfg['max_batch_size'], 
                        appearance_thresh=args_cfg['sct_appearance_thresh'], euc_thresh=args_cfg['sct_euclidean_thresh'])
                for _ in range(len(sources))]

    # perspective transforms
    calib = calibration_position[perspective]
    p_transforms = [PerspectiveTransform(c) for c in calib]

    # clustering + hybrid tracker
    clustering = Clustering(appearance_thresh=args_cfg['clt_appearance_thresh'], euc_thresh=args_cfg['clt_euclidean_thresh'])
    mc_tracker = HybridMCTracker(scene=scene, cameras=None)  # cameras sẽ nạp trong perspective transform nếu cần
    id_dist = ID_Distributor()

    src_handlers = [get_reader_writer(s) for s in sources]
    results_lists = [[] for _ in range(len(sources))]

    total_frames = max([len(s[0]) for s in src_handlers])
    frame_id = 1

    while True:
        imgs = []
        stop = False
        for (img_paths, writer), tracker, p_trans, res_list in zip(src_handlers, trackers, p_transforms, results_lists):
            if len(img_paths) == 0:
                stop = True
                break
            img_path = img_paths.pop(0)
            img = cv2.imread(img_path)
            dets = detection(img, conf=conf_thres, iou=iou_thres, classes=0, verbose=False)[0].boxes.data.cpu().numpy()
            online_targets, ratio = tracker.update(dets, img, img_path, pose)
            p_trans.run(tracker, ratio)
            for t in tracker.tracked_stracks:
                t.t_global_id = id_dist.assign_id()
            imgs.append(img)
        if stop:
            break

        groups = clustering.update(trackers, frame_id, scene)
        mc_tracker.update(trackers, groups, scene)
        clustering.update_using_mctracker(trackers, mc_tracker)

        if frame_id % 5 == 0:
            mc_tracker.refinement_clusters()

        update_result_lists_testset(trackers, results_lists, frame_id, cam_ids, scene)

        if args_cfg['write_vid']:
            write_vids(trackers, imgs, src_handlers, pose, _COLORS, mc_tracker, frame_id)

        print(f"frame {frame_id}/{total_frames}")
        frame_id += 1

    finalize_cams(src_handlers)
    write_results_testset(results_lists, result_paths)
    print('Done')


if __name__ == '__main__':
    args_dict = {
        'max_batch_size': 64,
        'track_buffer': 150,
        'with_reid': True,
        'sct_appearance_thresh': 0.4,
        'sct_euclidean_thresh': 0.1,
        'clt_appearance_thresh': 0.35,
        'clt_euclidean_thresh': 0.3,
        'frame_rate': 30,
        'write_vid': False,
    }

    scene_arg = make_parser().scene
    if scene_arg:
        run(args_dict, 0.1, 0.45, sources[scene_arg], result_paths[scene_arg], scene_arg, cam_ids[scene_arg], scene_arg)
    else:
        for scene in sources.keys():
            run(args_dict, 0.1, 0.45, sources[scene], result_paths[scene], scene, cam_ids[scene], scene) 