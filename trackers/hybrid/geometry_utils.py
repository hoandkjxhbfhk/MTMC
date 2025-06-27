import numpy as np

"""geometry_utils.py
Các hàm hình học hỗ trợ Hybrid Multi-Camera Tracker.
Hiện tại chỉ cung cấp các hàm khung (stub). Sau khi có thông tin calib camera cụ thể, hãy
hoàn thiện nội dung.
"""


def triangulate(keypoints_mv, cameras):
    """Triangulate 3D điểm khớp ‑ SVD.

    Parameters
    ----------
    keypoints_mv : np.ndarray (N_cam, N_kp, 3)
        Giá trị (x, y, conf) của từng khớp trên mỗi camera.
    cameras : List[object]
        Đối tượng camera, phải có thuộc tính `project_mat` (3×4 numpy).

    Returns
    -------
    keypoints_3d : np.ndarray (N_kp, 4)
        Toạ độ đồng nhất của khớp (x, y, z, w), đã chuẩn hoá để w = 1.
    valid_mask : np.ndarray (N_kp,) bool
        True nếu khớp được tái tạo bởi ≥2 view."""
    num_cam, num_kp, _ = keypoints_mv.shape
    keypoints_3d = np.zeros((num_kp, 4))
    valid_mask = np.zeros(num_kp, dtype=bool)

    for j in range(num_kp):
        # xác định view hợp lệ (độ tin cậy > 0.5)
        mask = keypoints_mv[:, j, 2] > 0.5
        if np.sum(mask) < 2:
            continue
        A = []
        for v_idx in np.where(mask)[0]:
            kp = keypoints_mv[v_idx, j]
            P = cameras[v_idx].project_mat  # (3, 4)
            x, y, conf = kp
            A.append(conf * (x * P[2] - P[0]))
            A.append(conf * (y * P[2] - P[1]))
        A = np.stack(A)
        _, _, vt = np.linalg.svd(A)
        X = vt[-1]
        X = X / (X[3] + 1e-6)  # chuẩn hoá để w = 1
        keypoints_3d[j] = X
        valid_mask[j] = True
    return keypoints_3d, valid_mask


def feet_homography_to_world(feet_xy, camera):
    """Chuyển vị trí bàn chân (pixel) sang toạ độ phẳng (world X-Y) bằng homography."""
    feet_h = np.array([feet_xy[0], feet_xy[1], 1.0])
    world = camera.homo_inv @ feet_h
    return world[:-1] / world[-1]


def p2l_distance(pt3d, ray_origin, ray_dir):
    """Khoảng cách từ điểm 3D tới đường 3D (point-to-line)."""
    v = pt3d - ray_origin
    cross = np.linalg.norm(np.cross(v, ray_dir))
    denom = np.linalg.norm(ray_dir) + 1e-6
    return cross / denom


# ---------- Hàm bổ sung cho affinity cao cấp -----------------------------

def compute_ray_from_pixel(px: np.ndarray, camera):
    """Tính tia (origin, direction) từ điểm ảnh.

    Parameters
    ----------
    px : np.ndarray (2,)  pixel (x,y)
    camera : object phải có `project_inv` (4×3?) và thuộc tính `pos` (3,)

    Returns (origin, direction) 3-D numpy arrays."""
    # joints_h shape 3
    joints_h = np.array([px[0], px[1], 1.0])
    ray = camera.project_inv @ joints_h  # homogeneous 3D
    ray = ray[:-1] / ray[-1]
    origin = camera.pos
    dir_vec = ray - origin
    dir_vec = dir_vec / (np.linalg.norm(dir_vec) + 1e-6)
    return origin, dir_vec


def p2l_distance_set(track_keypoints_3d: np.ndarray, detection_pose: dict, camera):
    """Tính trung bình point-to-line cho các khớp chung có độ tin cậy."""
    if detection_pose is None:
        return 1e6
    kp2d = detection_pose['keypoints']  # (17,3)
    conf = kp2d[:, 2]
    valid = conf > 0.5
    if not np.any(valid):
        return 1e6
    dists = []
    for j in np.where(valid)[0]:
        if track_keypoints_3d[j, 3] == 0:
            continue
        origin, dir_vec = compute_ray_from_pixel(kp2d[j, :2], camera)
        pt3d = track_keypoints_3d[j, :3]
        d = p2l_distance(pt3d, origin, dir_vec)
        dists.append(d)
    if not dists:
        return 1e6
    return float(np.mean(dists))


def feet_distance(track_world, det_feet_xy, camera):
    """Khoảng cách Euclid trên mặt phẳng giữa feet track (world) và feet detection."""
    det_world = feet_homography_to_world(det_feet_xy, camera)
    return float(np.linalg.norm(track_world - det_world))


# ----------------- Epipolar consistency -------------------

def epipolar_score(pos_i: np.ndarray, ray_i: np.ndarray, pos_j: np.ndarray, ray_j: np.ndarray, thresh: float = 0.2):
    """Tính điểm số epipolar 3D giống PoseTracker: 1 - (khoảng cách hai tia / thresh)."""
    v1 = ray_i / (np.linalg.norm(ray_i) + 1e-6)
    v2 = ray_j / (np.linalg.norm(ray_j) + 1e-6)
    # Khoảng cách min giữa hai đường trong không gian
    cross_v = np.cross(v1, v2)
    cross_norm = np.linalg.norm(cross_v)
    if cross_norm < 1e-6:
        return 1.0
    diff_pos = pos_j - pos_i
    dist = abs(np.dot(diff_pos, cross_v / cross_norm))
    return max(0.0, 1 - dist / thresh) 