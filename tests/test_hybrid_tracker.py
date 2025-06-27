import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np

from trackers.hybrid.hybrid_track import HybridMCTracker


class DummyCamera:
    def __init__(self):
        # 3×4 ma trận chiếu đơn giản
        self.project_mat = np.array([[1, 0, 0, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0]], dtype=float)
        self.homo_inv = np.eye(3)
        self.pos = np.zeros(3)
        self.project_inv = np.eye(3,4)


# Tạo dữ liệu giả
num_keypoints = 17

def random_pose(cam_id):
    keypoints = np.random.rand(num_keypoints, 3)
    keypoints[:, 2] = 0.8  # độ tin cậy cao
    return {
        'keypoints': keypoints,
        'cam_id': cam_id
    }

def build_group(global_id, x, y, cam_id):
    rng = np.random.RandomState(global_id)  # đảm bảo cùng ID -> cùng feature
    feat = [rng.rand(2048).astype(np.float32)]
    centroid = (x, y)
    pose = [random_pose(cam_id)]
    path = [f"/tmp/img_{global_id}.jpg"]
    tlbr = [np.array([x, y, x+50, y+150])]
    coords = [(x, y, 0)]
    return [global_id, feat, centroid, pose, path, tlbr, coords]


def test_basic_tracking():
    cameras = [DummyCamera(), DummyCamera()]
    tracker = HybridMCTracker(cameras=cameras, scene="S01_61")

    # Frame 1: hai người A,B
    groups_f1 = np.array([
        build_group(1, 0.0, 0.0, 0),
        build_group(2, 5.0, 0.0, 0)
    ], dtype=object)
    tracker.update([], groups_f1, scene="S01_61")
    assert len(tracker.tracked_mtracks) == 2

    ids_f1 = {t.global_id: t.track_id for t in tracker.tracked_mtracks}

    # Frame 2: di chuyển nhẹ
    groups_f2 = np.array([
        build_group(1, 0.5, 0.1, 1),  # camera 1
        build_group(2, 5.4, -0.1, 1)
    ], dtype=object)
    tracker.update([], groups_f2, scene="S01_61")
    ids_f2_map = {t.global_id: t.track_id for t in tracker.tracked_mtracks}

    # Kiểm tra track_id giữ nguyên cho mỗi global_id
    assert ids_f1 == ids_f2_map


if __name__ == "__main__":
    test_basic_tracking()
    print("Test passed!") 