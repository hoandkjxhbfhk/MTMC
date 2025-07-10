import numpy as np
import pytest

from trackers.hybrid.hybrid_track import HybridMCTracker


# Helper --------------------------------------------------------------

def _create_group(global_id: int, tlbr, feat):
    """Tạo một phần tử group đúng định dạng expectations của HybridMCTracker."""
    centroid = np.array([(tlbr[0] + tlbr[2]) / 2, (tlbr[1] + tlbr[3]) / 2])
    kp = np.zeros((17,3),dtype=np.float32)
    kp[0,2]=1.0
    pose = {"cam_id":0, "keypoints":kp}
    return [global_id, [feat], centroid, [pose], ["img"], [np.array(tlbr)], None]


# Test ----------------------------------------------------------------

def test_occlusion_switch_id_stability():
    rng = np.random.default_rng(0)
    feat1 = rng.random(2048, dtype=np.float32)
    feat2 = rng.random(2048, dtype=np.float32)

    tracker = HybridMCTracker(scene="S01_61", cameras=None)

    # Frame 1 – hai người tách biệt
    g1_f1 = _create_group(1, [10, 10, 60, 100], feat1)
    g2_f1 = _create_group(2, [200, 10, 250, 100], feat2)
    groups = np.array([g1_f1, g2_f1], dtype=object)
    tracker.update([], groups, scene="S01_61")
    id_map = {t.global_id: t.track_id for t in tracker.tracked_mtracks}

    # Frame 2 – chồng bbox (occlusion)
    g1_f2 = _create_group(1, [100, 10, 160, 100], feat1)
    g2_f2 = _create_group(2, [110, 10, 170, 100], feat2)
    groups = np.array([g1_f2, g2_f2], dtype=object)
    tracker.update([], groups, scene="S01_61")

    # Frame 3 – tách trở lại
    g1_f3 = _create_group(1, [10, 10, 60, 100], feat1)
    g2_f3 = _create_group(2, [200, 10, 250, 100], feat2)
    groups = np.array([g1_f3, g2_f3], dtype=object)
    tracker.update([], groups, scene="S01_61")

    final_map = {t.global_id: t.track_id for t in tracker.tracked_mtracks}

    assert final_map == id_map, "Track IDs changed after occlusion" 