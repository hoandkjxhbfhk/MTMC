import numpy as np
from trackers.hybrid.hybrid_track import HybridMCTracker, HybridTrack, HybridConfig

def _dummy_track(update_age:int=0, oc=False):
    kp = np.zeros((17,3),dtype=np.float32)
    kp[0,2]=1.0  # confident joint to pass threshold
    pose = {"cam_id":0, "keypoints":kp}
    tlbr = np.array([0.0,0.0,10.0,10.0])
    t = HybridTrack(0, np.array([0,0]), [np.zeros(2048)], [pose], ["img"], [tlbr], None, 0, scene="S01_61", cameras=None, config=HybridConfig())
    t.update_age = update_age
    t.oc_state = [oc]
    return t

def test_dynamic_weight_adjustment():
    tracker = HybridMCTracker(scene="S01_61", cameras=None)
    base_reid = tracker.cfg.w_reid
    # track with high update_age triggers decrease
    t_old = _dummy_track(update_age=20)
    tracker._adjust_weights([t_old])
    assert tracker.cfg.w_reid < base_reid
    # reset
    tracker.cfg.w_reid = base_reid
    # track with occlusion triggers decrease
    t_occ = _dummy_track(update_age=0, oc=True)
    tracker._adjust_weights([t_occ])
    assert tracker.cfg.w_reid < base_reid 