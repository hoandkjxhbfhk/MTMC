import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import numpy as np
from trackers.hybrid.hybrid_track import HybridTrack

class DummyCamera:
    def __init__(self):
        self.project_mat = np.eye(3,4)
        self.project_inv = np.eye(3,4)
        self.homo_inv = np.eye(3)
        self.pos = np.zeros(3)


def test_reid_bank():
    cam = DummyCamera()
    track = HybridTrack(global_id=1, centroid=(0,0), feat=None, pose=None, img_path=[], tlbr=[], coords=[], cameras=[cam])

    # helper pose & bbox satisfying conditions
    kp = np.zeros((17,3),dtype=np.float32)
    kp[[5,6,11,12],2]=0.8  # upper body confident
    pose_ok = {"cam_id":0, "keypoints":kp}
    tlbr_ok = np.array([0.0,0.0,10.0,10.0,0.95])

    # first feature
    f1 = [np.ones(2048, dtype=np.float32)]
    track.update_features(f1, poses=[pose_ok], img_paths=["img"], tlbrs=[tlbr_ok], coords=[])
    assert track.feat_count == 1

    # duplicate feature (cosine sim 1) should be ignored
    track.update_features(f1, poses=[pose_ok], img_paths=["img"], tlbrs=[tlbr_ok], coords=[])
    assert track.feat_count == 1

    # different feature
    vec = np.zeros(2048, dtype=np.float32)
    vec[0] = 1.0
    f2 = [vec]
    track.update_features(f2, poses=[pose_ok], img_paths=["img"], tlbrs=[tlbr_ok], coords=[])
    assert track.feat_count == 2

    print("ReID bank test passed!")

if __name__ == "__main__":
    test_reid_bank() 