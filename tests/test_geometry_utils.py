import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from trackers.hybrid.geometry_utils import triangulate


class DummyCamera:
    def __init__(self, tx=0.0):
        # camera at (tx,0,0) looking along z-axis with focal length 1
        self.project_mat = np.array([[1, 0, -tx, 0],
                                     [0, 1, 0, 0],
                                     [0, 0, 1, 0]], dtype=float)


def test_triangulate_simple():
    cams = [DummyCamera(0.0), DummyCamera(0.2)]
    # world point (0,0,1)
    kp0 = np.array([0.0, 0.0, 1.0])
    kp1 = np.array([-0.2, 0.0, 1.0])  # because of shift
    keypoints_mv = np.zeros((2, 1, 3))
    keypoints_mv[0, 0] = kp0
    keypoints_mv[1, 0] = kp1

    kp3d, valid = triangulate(keypoints_mv, cams)
    assert valid[0]


if __name__ == "__main__":
    test_triangulate_simple()
    print("Triangulate test passed!") 