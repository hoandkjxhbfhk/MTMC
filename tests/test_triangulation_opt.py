import numpy as np
from trackers.hybrid.hybrid_track import HybridMCTracker, HybridConfig

class DummyCam:
    def __init__(self):
        self.project_mat = np.hstack((np.eye(3), np.zeros((3,1))))  # 3x4
        self.project_inv = np.vstack((np.eye(3), np.zeros((1,3))))  # 4x3 pseudo
        self.pos = np.zeros(3)
        self.homo_inv = np.eye(3)
        self.homo_feet_inv = np.eye(3)


def _make_pose(cam_id, xy=(0,0)):
    kp = np.zeros((17,3),dtype=np.float32)
    kp[0,:2]=xy
    kp[0,2]=1.0  # confidence
    return {"cam_id":cam_id, "keypoints":kp}


def _create_group(frame_shift:float=0.0):
    # one group with two views
    poses=[_make_pose(0,(10+frame_shift,10)), _make_pose(1,(12+frame_shift,10))]
    features=[np.random.rand(2048).astype(np.float32)]
    centroid=np.array([0,0])
    tlbrs=[np.array([0.0,0.0,10.0,10.0]), np.array([0.0,0.0,10.0,10.0])]
    return [1, features, centroid, poses, ["img","img"], tlbrs, None]


def test_periodic_triangulation():
    cams=[DummyCam(), DummyCam()]
    cfg=HybridConfig(triang_interval=2)
    tracker=HybridMCTracker(cameras=cams, config=cfg, scene="S01_61")

    # frame 1
    groups=np.array([_create_group(0)],dtype=object)
    tracker.update([], groups, scene="S01_61")
    track=tracker.tracked_mtracks[0]
    assert np.all(track.keypoints_3d==0), "Triangulation should not run on frame1"

    # frame2 triggers triangulation
    groups=np.array([_create_group(1)],dtype=object)
    tracker.update([], groups, scene="S01_61")
    track=tracker.tracked_mtracks[0]
    assert track.keypoints_3d[0,3]!=0, "Triangulation expected on frame2" 